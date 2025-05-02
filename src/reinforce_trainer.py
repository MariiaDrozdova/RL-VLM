# src/reinforce_trainer.py

import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.trainer import BaseTrainer, reward_function_vlm

class REINFORCETrainer(BaseTrainer):
    """
    Simple REINFORCE trainer (no critic).
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

        # how many tokens to sample per example
        self.max_new_tokens = config.get("max_new_tokens", 10)
        # optimizer for the actor only
        lr = config.get("actor_lr", 1e-5)
        self.actor_optimizer = AdamW(self.model.parameters(), lr=lr)

        # wrap in accelerator if requested
        if use_accelerator and self.accelerator is not None:
            self.model, self.actor_optimizer = self.accelerator.prepare(
                self.model, self.actor_optimizer
            )

        self.global_step = 0

    @torch.no_grad()
    def generate_one_pass(self, input_ids, pixel_values):
        """
        Sample a full response of length self.max_new_tokens, returning:
          - seqs   : [B, T] token IDs (excluding prompt)
          - logps  : [B, T] log-probabilities under current policy
        """
        B = input_ids.size(0)

        # use HF generate under no_grad + use_cache for speed
        gen_out = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        # strip off the prompt tokens
        T_prompt = input_ids.size(1)
        seqs = gen_out.sequences[:, T_prompt:]         # [B, T]
        scores = gen_out.scores                        # list of T tensors [B, V]

        # compute log-probs per time‐step
        logps = []
        for t, logits in enumerate(scores):
            dist = torch.distributions.Categorical(logits=logits)
            logps.append(dist.log_prob(seqs[:, t]))    # [B]
        logps = torch.stack(logps, dim=1)             # [B, T]

        return seqs, logps

    def train_rl(self, epochs: int = 3):
        """
        Runs basic REINFORCE: sample, compute reward, 
        policy gradient = - E[r * sum log p(a)]
        """
        self.model.train()

        for epoch in range(epochs):
            total_reward = 0.0
            total_count  = 0
            loss_sum     = 0.0

            # freeze vision tower if present
            if hasattr(self.model, "vision_tower"):
                for p in self.model.vision_tower.parameters():
                    p.requires_grad = False

            for inputs, answers in tqdm(self.train_loader, desc=f"RL Epoch {epoch+1}/{epochs}"):
                in_ids     = inputs["input_ids"].to(self.device)
                pix_vals   = inputs["pixel_values"].to(self.device)

                # 1) sample under current policy
                seqs, logps = self.generate_one_pass(in_ids, pix_vals)

                # 2) compute rewards
                texts = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                rewards = torch.tensor(
                    [reward_function_vlm(pred, ref) for pred, ref in zip(texts, answers)],
                    dtype=torch.float,
                    device=self.device
                )  # [B]

                # 3) compute REINFORCE loss
                # sum logps over T then weight by reward
                total_logp = logps.sum(dim=1)             # [B]
                rl_loss    = - (total_logp * rewards).mean()

                # 4) backward & step
                self.actor_optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(rl_loss)
                else:
                    rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.actor_optimizer.step()

                # metrics
                B = in_ids.size(0)
                total_reward += rewards.sum().item()
                total_count  += B
                loss_sum     += rl_loss.item() * B
                self.global_step += 1

                # free immedate memory
                del in_ids, pix_vals, seqs, logps, rewards
                torch.cuda.empty_cache()

            # epoch stats
            avg_r  = total_reward / total_count
            avg_l  = loss_sum     / total_count
            print(f"[RL][Epoch {epoch+1}] avg_reward={avg_r:.3f}, avg_loss={avg_l:.4f}")

            if self.tb:
                self.tb.add_scalar("RL/avg_reward", avg_r, self.global_step)
                self.tb.add_scalar("RL/avg_loss",   avg_l, self.global_step)

            # optional validation & checkpoint
            if self.val_loader:
                self._validate_rl(epoch)

        torch.cuda.empty_cache()

    def _validate_rl(self, epoch: int):
        self.model.eval()
        val_r = 0.0
        cnt  = 0
        with torch.no_grad():
            for inputs, answers in tqdm(self.val_loader, desc="RL Validation"):
                in_ids   = inputs["input_ids"].to(self.device)
                pix_vals = inputs["pixel_values"].to(self.device)
                seqs, _  = self.generate_one_pass(in_ids, pix_vals)
                texts = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                for pred, ref in zip(texts, answers):
                    val_r += reward_function_vlm(pred, ref)
                    cnt  += 1
                del in_ids, pix_vals, seqs
                torch.cuda.empty_cache()

        avg_val = val_r / cnt
        print(f"[RL][Validation] epoch={epoch+1} avg_reward={avg_val:.3f}")
        if self.tb:
            self.tb.add_scalar("RL/val_reward", avg_val, epoch)

        # save best policy
        if avg_val > self.best_val:
            self.best_val = avg_val
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            model_to_save = (self.accelerator.unwrap_model(self.model)
                             if self.accelerator else self.model)
            model_to_save.save_pretrained(self.checkpoint_dir)
            torch.save(model_to_save.state_dict(),
                       os.path.join(self.checkpoint_dir, "pytorch_model.bin"))
            self.processor.save_pretrained(self.checkpoint_dir)

