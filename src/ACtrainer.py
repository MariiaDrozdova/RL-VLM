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

class ACTrainer(BaseTrainer):
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
        checkpoint_dir: str = "ac_checkpoint/",
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
        lr_critic = config.get("critic_lr", 1e-4)
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
    def generate_one_pass(self, input_ids, pixel_values, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
    
        B = input_ids.size(0)
        H = self.hidden_size
    
        # accumulators
        sum_state_emb = torch.zeros(B, H, device=self.device)
        num_steps     = 0
    
        generated_ids = []
        all_log_probs = []
    
        bos_token_id = 2
        eos_token_id = 2
        decoder_input_ids = torch.tensor([[2, 0]] * B, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
    
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
    
            # pull out just the [CLS] or last-hidden embedding once
            state_emb = outputs.encoder_last_hidden_state[:, 0, :]  # [B, H]
            sum_state_emb += state_emb
            num_steps    += 1
    
            # sampling
            logits = outputs.logits[:, -1, :]               # [B, V]
            #probs  = torch.softmax(logits, dim=-1)          # [B, V]
            #dist   = torch.distributions.Categorical(probs)
            dist = torch.distributions.Categorical(logits=logits)
            nxt    = dist.sample()                          # [B]
            lp     = dist.log_prob(nxt)                     # [B]

            # advance decoder inputs
            decoder_input_ids = torch.cat([decoder_input_ids, nxt.unsqueeze(-1)], dim=1)
            
            # 1) mask after EOS
            mask = ~finished
            nxt  = torch.where(mask, nxt, eos_token_id)
            lp   = lp  * mask.float()
        
            # 2) append
            generated_ids.append(nxt.unsqueeze(-1))
            all_log_probs.append(lp  .unsqueeze(-1))
        
            # 3) update finished and decoder inputs
            finished |= (nxt == eos_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids, nxt.unsqueeze(-1)], dim=1)
    
        # stack only the things you need to
        generated_ids = torch.cat(generated_ids, dim=1)    # [B, T]
        all_log_probs = torch.stack(all_log_probs, dim=1)  # [B, T]
    
        # now do the mean‐pool:
        avg_state_emb = sum_state_emb / float(num_steps)   # [B, H]
    
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
        return generated_ids, all_log_probs, avg_state_emb

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
            if self.accelerator is not None:
                # get back the underlying HF model
                model_to_save = self.accelerator.unwrap_model(self.model)
            else:
                model_to_save = self.model
                    
            # now it's safe to call HF save
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            model_to_save.save_pretrained(self.checkpoint_dir)
            torch.save(model_to_save.state_dict(),
                os.path.join(self.checkpoint_dir, "pytorch_model.bin"))
            self.processor.save_pretrained(self.checkpoint_dir)

