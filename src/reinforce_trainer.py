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
        checkpoint_dir: str = "reinforce_checkpoint/",
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
        self.max_new_tokens = config.get("reinforce_max_new_tokens", 17)
        # optimizer for the actor only
        lr = config.get("actor_lr", 1e-5)
        self.actor_optimizer = AdamW(self.model.parameters(), lr=lr)

        # wrap in accelerator if requested
        if use_accelerator and self.accelerator is not None:
            self.model, self.actor_optimizer = self.accelerator.prepare(
                self.model, self.actor_optimizer
            )

        self.global_step = 0
        self.best_val = -float("inf")

    def generate_one_pass(self, input_ids, pixel_values, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
    
        B = input_ids.size(0)
    
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
    
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
        return generated_ids, all_log_probs

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
                
                answers = [ans[ans.find("Counts by color"):] for ans in answers]
                rewards = torch.tensor(
                    [reward_function_vlm(pred, ref)+2 for pred, ref in zip(texts, answers)],
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
                    val_r += reward_function_vlm(pred, ref)+2
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
