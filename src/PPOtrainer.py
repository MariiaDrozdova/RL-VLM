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
    Simple critic network: takes a state embedding (from encoder) and outputs V(s).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 4)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim // 4, 1), std=1.0),
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        # state_embedding: [B, D] -> value: [B]
        return self.critic(state_embedding).squeeze(-1)

class PPOTrainer(BaseTrainer):
    """
    Extends BaseTrainer with PPO-style actor-critic training.
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
        checkpoint_dir: str = "ppo_checkpoint/",
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

        # PPO hyperparameters
        self.clip_coef      = config.get("clip_coef", 0.1)
        self.entropy_coef   = config.get("entropy_coef", 0.01)
        self.value_coef     = config.get("value_coef", 0.5)
        self.update_epochs  = config.get("update_epochs", 1)
        self.minibatch_size = config.get("minibatch_size", 1)
        self.max_new_tokens = config.get("max_new_tokens", 2)
        self.best_val       = -float("inf")
        self.global_step    = 0
        

        # Critic network
        self.hidden_size = config.get("hidden_size", 768)
        self.critic_warmup_epochs = config.get("critic_warmup_epochs", 3)
        hidden_dim = config.get("critic_hidden_dim", 128)
        self.critic = ValueNetwork(
            input_dim=self.hidden_size,
            hidden_dim=hidden_dim
        ).to(device)

        # Optimizers
        lr_actor  = config.get("actor_lr", 1e-6)
        lr_critic = config.get("critic_lr", 1e-5)
        self.actor_optimizer  = AdamW(self.model.parameters(),  lr=lr_actor)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic)

        # Wrap everything in accelerator if requested
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
        all_entropy   = []
    
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
            ent    = dist.entropy()                         # [B]

            # advance decoder inputs
            decoder_input_ids = torch.cat([decoder_input_ids, nxt.unsqueeze(-1)], dim=1)
            
            # 1) mask after EOS
            mask = ~finished
            nxt  = torch.where(mask, nxt, eos_token_id)
            lp   = lp  * mask.float()
            ent  = ent * mask.float()
        
            # 2) append
            generated_ids.append(nxt.unsqueeze(-1))
            all_log_probs.append(lp  .unsqueeze(-1))
            all_entropy  .append(ent .unsqueeze(-1))
        
            # 3) update finished and decoder inputs
            finished |= (nxt == eos_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids, nxt.unsqueeze(-1)], dim=1)
    
        # stack only the things you need to
        generated_ids = torch.cat(generated_ids, dim=1)    # [B, T]
        all_log_probs = torch.stack(all_log_probs, dim=1)  # [B, T]
        all_entropy   = torch.stack(all_entropy,   dim=1)  # [B, T]
    
        # now do the mean‐pool:
        avg_state_emb = sum_state_emb / float(num_steps)   # [B, H]
    
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
        return generated_ids, all_log_probs, avg_state_emb, all_entropy
    


    def train_rl(self):
        """
        Main PPO training loop.
        """
        self.model.train()
        self.critic.train()

        num_epochs = self.config.get("ppo_epochs", 3)
        for epoch in range(num_epochs):
            train_actor = (epoch >= self.critic_warmup_epochs)
            memory = []

            # === 1) Collect trajectories ===
            for inputs, answers in tqdm(self.train_loader, desc=f"PPO Epoch {epoch+1}/{num_epochs}"):
                input_ids    = inputs["input_ids"].to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device)

                # sample without gradients
                seqs, old_logps, emb, ents = self.generate_one_pass(input_ids, pixel_values)

                preds = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                #print(preds)
                rewards = torch.tensor(
                    [reward_function_vlm(p, g) for p, g in zip(preds, answers)],
                    dtype=torch.float,
                    device=self.device
                )
                
                values    = self.critic(emb)           # [B]
                advantages = (rewards - values).detach()

                # store on CPU
                memory.append((
                    input_ids.cpu(),
                    pixel_values.cpu(),
                    seqs.cpu(),
                    old_logps.cpu(),
                    rewards.cpu(),
                    emb.cpu(),
                    advantages.cpu(),
                    ents.cpu()
                ))

            # === 2) PPO Updates ===
            for _ in range(self.update_epochs):
                random.shuffle(memory)
                for start in range(0, len(memory), self.minibatch_size):
                    batch = memory[start:start + self.minibatch_size]
                    (b_in, b_pix, b_seqs, b_old_lp,
                     b_rew, b_emb, b_adv, b_ent) = zip(*batch)

                    # move to device
                    b_in   = torch.cat(b_in).to(self.device)
                    b_pix  = torch.cat(b_pix).to(self.device)
                    b_seqs = torch.cat(b_seqs).to(self.device)
                    b_old_lp = torch.cat(b_old_lp).to(self.device)
                    b_rew    = torch.cat(b_rew).to(self.device)
                    b_emb    = torch.cat(b_emb).to(self.device)
                    b_adv    = torch.cat(b_adv).to(self.device)
                    b_ent    = torch.cat(b_ent).to(self.device)

                    # normalize advantages
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                    # policy loss via scoring function
                    new_lp, mean_ent = self.score_sequence(b_in, b_pix, b_seqs)
                    ratio = (new_lp.sum(1) - b_old_lp.sum(1)).exp()
                    loss1 = -ratio * b_adv
                    loss2 = -torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * b_adv
                    policy_loss = torch.max(loss1, loss2).mean()

                    # value loss
                    new_vals    = self.critic(b_emb)
                    value_loss  = self.value_coef * nn.functional.mse_loss(new_vals, b_rew)

                    # entropy bonus
                    entropy_loss = -self.entropy_coef * mean_ent

                    total_loss = policy_loss + value_loss + entropy_loss

                    # backward + step (actor + critic)
                    if train_actor:
                        self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    if hasattr(self, "accelerator"):
                        self.accelerator.backward(total_loss)
                    else:
                        total_loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(),  0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    if train_actor:
                        self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # log to TensorBoard
                    if self.tb:
                        self.tb.add_scalar("PPO/policy_loss", policy_loss.item(), self.global_step)
                        self.tb.add_scalar("PPO/value_loss",   value_loss.item(),   self.global_step)
                        self.tb.add_scalar("PPO/entropy",      mean_ent.item(),     self.global_step)
                        self.global_step += 1

                    # free memory
                    del b_in, b_pix, b_seqs, b_old_lp, b_rew, b_emb, b_adv, b_ent, new_lp
                    torch.cuda.empty_cache()

            # === 3) Validation & Checkpoint ===
            if self.val_loader:
                self.model.eval()
                self.critic.eval()
                val_reward = 0.0
                n_samples  = 0

                with torch.no_grad():
                    for inputs, answers in tqdm(self.val_loader, desc="PPO Validation"):
                        inp   = inputs["input_ids"].to(self.device)
                        pix   = inputs["pixel_values"].to(self.device)
                        seqs, _, _, _ = self.generate_one_pass(inp, pix)

                        texts = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                        print(texts)
                        print(answers)
                        for t, g in zip(texts, answers):
                            val_reward += reward_function_vlm(t, g)
                            n_samples  += 1

                avg_val = val_reward / n_samples
                print(f"[Epoch {epoch+1}] Avg Validation Reward: {avg_val:.3f}")

                if self.tb:
                    self.tb.add_scalar("PPO/val_reward", avg_val, epoch)

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

        # clear GPU cache once at end
        torch.cuda.empty_cache()
