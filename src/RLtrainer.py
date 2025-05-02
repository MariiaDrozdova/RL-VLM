# src/rl_trainer.py

import os
import random
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.trainer import BaseTrainer, reward_function_vlm

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ValueNetwork(nn.Module):
    """
    Simple critic network: takes a state embedding and outputs V(s).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 4)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim // 4, 1), std=1.0),
        )

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        # state_emb: [B, D] -> [B]
        return self.net(state_emb).squeeze(-1)

class RLTrainer(BaseTrainer):
    """
    Simple REINFORCE + baseline (actor-critic) trainer.
    """
    def __init__(
        self,
        model,
        processor,
        train_loader,
        val_loader,
        device,
        config,
        use_accelerator: bool = True,
        checkpoint_dir: str = "rl_checkpoint/",
        tb_writer: SummaryWriter = None
    ):
        super().__init__(
            model, processor,
            train_loader, val_loader,
            device, config,
            use_accelerator,
            checkpoint_dir,
            tb_writer
        )

        # hyperparameters
        self.max_new_tokens = config.get("max_new_tokens", 10)
        self.best_val = -float("inf")
        self.global_step = 0

        # critic network
        self.hidden_size = config.get("hidden_size", 768)
        self.critic_warmup_epochs = config.get("critic_warmup_epochs", 3)
        hidden_dim = config.get("critic_hidden_dim", 512)
        self.critic = ValueNetwork(
            input_dim=self.hidden_size,
            hidden_dim=hidden_dim
        ).to(device)

        # Optimizers
        lr_actor  = config.get("actor_lr", 1e-6)
        lr_critic = config.get("critic_lr", 1e-5)
        self.actor_optimizer  = AdamW(self.model.parameters(),  lr=lr_actor)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic)

        # accelerator wrapping
        if use_accelerator and hasattr(self, "accelerator"):
            self.model, self.critic, \
            self.actor_optimizer, self.critic_optimizer = self.accelerator.prepare(
                self.model,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer
            )
        else:
            self.logger.info("No accelerator: running on raw device.")

    @torch.no_grad()
    def generate_one_pass(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        max_new_tokens: int = None
    ):
        """
        Sample a response, returning:
          - seqs      : [B, T] new token IDs (excluding prompt)
          - logps     : [B, T] log-probs under current policy
          - state_emb : [B, D] encoder embedding for baseline
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        B = input_ids.size(0)
        # encoder pass to get state embedding
        enc = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            return_dict=True,
            output_hidden_states=True
        )
        state_emb = enc.encoder_last_hidden_state[:, 0, :]  # [B, D]

        # sample via generate + scores
        gen_out = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        # strip prompt tokens
        seqs = gen_out.sequences[:, input_ids.size(1):]  # [B, T]
        scores = gen_out.scores                           # list of T tensors [B, V]

        # compute log-probs
        logps = []
        for t, logits in enumerate(scores):
            dist  = torch.distributions.Categorical(logits=logits)
            logps.append(dist.log_prob(seqs[:, t]))       # [B]
        logps = torch.stack(logps, dim=1)                 # [B, T]

        return seqs, logps, state_emb

    def train_rl(self, epochs: int = 3):
        """
        REINFORCE with baseline training loop.
        """
        self.model.train()
        self.critic.train()

        for epoch in range(epochs):
            total_reward = 0.0
            total_count = 0
            policy_loss_sum = 0.0
            value_loss_sum  = 0.0

            train_actor = (epoch >= self.critic_warmup_epochs)

            # optionally freeze vision tower
            for p in self.model.vision_tower.parameters():
                p.requires_grad = False

            for inputs, answers in tqdm(self.train_loader, desc=f"RL Epoch {epoch+1}/{epochs}"):
                
                in_ids = inputs["input_ids"].to(self.device)
                pix    = inputs["pixel_values"].to(self.device)

                # sample under no_grad
                seqs, old_logps, state_emb = self.generate_one_pass(in_ids, pix)

                # compute rewards
                texts = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                rewards = torch.tensor(
                    [reward_function_vlm(t, g) for t, g in zip(texts, answers)],
                    dtype=torch.float,
                    device=self.device
                )

                # baseline values
                values = self.critic(state_emb)              # [B]
                advs   = (rewards - values).detach()         # [B]

                # compute losses
                policy_loss = - (old_logps.sum(dim=1) * advs).mean()
                value_loss  = nn.functional.mse_loss(values, rewards)
                loss = policy_loss + value_loss

                # backward & step
                if train_actor:
                    self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if hasattr(self, "accelerator"):
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(),  0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                if train_actor:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                # metrics
                B = in_ids.size(0)
                total_reward += rewards.sum().item()
                total_count  += B
                policy_loss_sum += policy_loss.item() * B
                value_loss_sum  += value_loss.item()  * B

                # free memory
                del in_ids, pix, seqs, old_logps, rewards, values, state_emb
                torch.cuda.empty_cache()

            # epoch metrics
            avg_r  = total_reward / total_count
            avg_pl = policy_loss_sum / total_count
            avg_vl = value_loss_sum  / total_count
            print(f"[RL][Epoch {epoch+1}] reward={avg_r:.3f}, "
                  f"policy_loss={avg_pl:.4f}, value_loss={avg_vl:.4f}")

            if self.tb:
                self.tb.add_scalar("RL/avg_reward",    avg_r,  epoch)
                self.tb.add_scalar("RL/policy_loss",   avg_pl, epoch)
                self.tb.add_scalar("RL/value_loss",    avg_vl, epoch)

            # validation + checkpointing
            if self.val_loader:
                self._validate_rl(epoch)

        torch.cuda.empty_cache()

    def _validate_rl(self, epoch: int):
        self.model.eval()
        self.critic.eval()
        val_r = 0.0
        cnt   = 0

        with torch.no_grad():
            for inputs, answers in tqdm(self.val_loader, desc="RL Validation"):
                in_ids = inputs["input_ids"].to(self.device)
                pix    = inputs["pixel_values"].to(self.device)

                seqs, _, state_emb = self.generate_one_pass(in_ids, pix)
                texts = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)

                for t, g in zip(texts, answers):
                    val_r += reward_function_vlm(t, g)
                    cnt  += 1

                del in_ids, pix, seqs, state_emb
                torch.cuda.empty_cache()

        avg_val = val_r / cnt
        print(f"[RL][Validation] epoch={epoch+1} avg_reward={avg_val:.3f}")

        if self.tb:
            self.tb.add_scalar("RL/val_reward", avg_val, epoch)

        if avg_val > self.best_val:
            self.best_val = avg_val
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            to_save = self.model.module if hasattr(self.model, "module") else self.model
            to_save.save_pretrained(self.checkpoint_dir)
            torch.save(to_save.state_dict(), os.path.join(self.checkpoint_dir, "pytorch_model.bin"))
            self.processor.save_pretrained(self.checkpoint_dir)

