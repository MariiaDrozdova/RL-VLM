import torch
import random
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.trainer import BaseTrainer, reward_function_vlm

class GRPOTrainer(BaseTrainer):
    """
    Extends BaseTrainer with GRPO-specific training.
    """
    def __init__(self, model, processor, train_loader, val_loader, device, config, use_accelerator=True, checkpoint_dir="checkpoint/", best_val_rewards=-2**31, tb_writer=None):
        super().__init__(model, processor, train_loader, val_loader, device, config, use_accelerator, checkpoint_dir, tb_writer)
        # GRPO hyperparameters
        self.group_size = config.get("group_size", 3)
        self.clip_coef = config.get("clip_coef", 0.01)
        self.entropy_coef = config.get("entropy_coef", 0.001)
        self.update_epochs = config.get("update_epochs", 2)
        self.rollout_batch_size = config.get("rollout_batch_size", 3)
        self.minibatch_size = config.get("minibatch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 50)
        self.iterations_per_epoch = config.get("iterations_per_epoch", 20)
        self.grpo_lr = config.get("grpo_learning_rate", config.get("learning_rate", 1e-6))
        self.checkpoint_dir = checkpoint_dir
        self.tb = tb_writer
        self.global_step = 0

        self.optimizer = AdamW(self.model.parameters(), lr=self.grpo_lr)
        if use_accelerator:
            self.optimizer = self.accelerator.prepare(
                self.optimizer
            )
        else:
            self.logger.info(f"No accelerator")
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)
        self.best_val_rewards = best_val_rewards

    # ------------------------
    # GRPO Utility Methods
    # ------------------------
    @torch.no_grad()
    def generate_one_pass(self, input_ids, pixel_values, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        batch_size = pixel_values.shape[0]
        generated_ids = []
        all_log_probs = []
        all_entropy = []
        bos_token_id = 2
        eos_token_id = 2
        decoder_input_ids = torch.tensor([[2, 0]] * batch_size, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        current_input_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids, 
                pixel_values=pixel_values, 
                decoder_input_ids=decoder_input_ids, 
                output_hidden_states=True, 
                return_dict=True
            )
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            distrib = torch.distributions.Categorical(probs)
            sampled_tokens = distrib.sample()
            entropy = distrib.entropy()
            all_entropy.append(entropy)
            log_probs = distrib.log_prob(sampled_tokens)
            generated_ids.append(sampled_tokens.unsqueeze(-1))
            all_log_probs.append(log_probs)
            decoder_input_ids = torch.cat([decoder_input_ids, sampled_tokens.unsqueeze(-1)], dim=-1)
            finished |= (sampled_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        generated_ids = torch.cat(generated_ids, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_entropy = torch.stack(all_entropy, dim=1)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return generated_ids, all_log_probs, None, all_entropy

    def batch_generate_group(self, old_model, input_ids_list, pixel_list):
        """
        Generate rollout groups using the old_model (on CPU).
        """
        B = len(input_ids_list)
        all_gen_ids = []
        all_old_log_probs = []
        rollout_device = next(old_model.parameters()).device  
        for _ in range(self.group_size):
            batch_input_ids = torch.cat(input_ids_list, dim=0).to(rollout_device)
            batch_pix = torch.cat(pixel_list, dim=0).to(rollout_device)
            with torch.no_grad():
                gen_ids, old_log_probs, _, _ = self.generate_one_pass(
                    batch_input_ids, batch_pix, max_new_tokens=self.max_new_tokens
                )
            all_gen_ids.append(gen_ids.cpu())
            all_old_log_probs.append(old_log_probs.cpu())
        return all_gen_ids, all_old_log_probs

    def train_rl(self):
        self.model.train()
        best_val = float('-inf')

        for epoch in range(self.config.get("grpo_epochs", 3)):
            # 1) Rollout with old policy
            #old_model = copy.deepcopy(self.model).cpu().eval()
            old_model = copy.deepcopy(self.model).eval()
            memory, total_r, count = [], 0.0, 0
            for inputs, answers in tqdm(self.train_loader, desc=f"Rollout Epoch {epoch+1}/{self.config.get('grpo_epochs',3)}"):
                inp, pix = inputs["input_ids"].to(self.device), inputs["pixel_values"].to(self.device)
                for i in range(inp.size(0)):
                    q_ids, q_pix, ref = inp[i:i+1], pix[i:i+1], answers[i]
                    group = []
                    for _ in range(self.group_size):
                        gen_ids, old_lp, _, _ = self.generate_one_pass(q_ids, q_pix)
                        txt = self.processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        r = reward_function_vlm(txt, ref)
                        group.append((gen_ids, old_lp, r))
                    rs = torch.tensor([g[2] for g in group], device=self.device)
                    advs = (rs - rs.mean())/(rs.std()+1e-8)
                    for (gen_ids, old_lp, r), adv in zip(group, advs):
                        memory.append({
                            "enc": q_ids.cpu(), "pix": q_pix.cpu(),
                            "tokens": gen_ids.cpu(), "old_lp": old_lp.cpu(),
                            "adv": adv.item()
                        })
                        total_r += r; count += 1
            del old_model
            torch.cuda.empty_cache()
            print(f"Avg reward rollout: {total_r/count:.3f}")
            if self.tb:
                self.tb.add_scalar("GRPO/avg_rollout_reward", total_r/count, epoch)
                    

            # 2) GRPO updates with padding
            total_loss_avg = 0
            counts = 0
            for _ in range(self.update_epochs):
                random.shuffle(memory)
                for start in range(0, len(memory), self.minibatch_size):
                    batch = memory[start:start+self.minibatch_size]
                    # gather lists
                    b_enc = [x['enc'] for x in batch]
                    b_pix = [x['pix'] for x in batch]
                    b_tok = [x['tokens'] for x in batch]
                    b_old = [x['old_lp'] for x in batch]
                    b_adv = torch.tensor([x['adv'] for x in batch], device=self.device)

                    # pad sequences to max length in batch
                    max_len = max([t.shape[1] for t in b_tok])
                    padded_tok = []
                    padded_old = []
                    for tok, old in zip(b_tok, b_old):
                        pad_size = max_len - tok.shape[1]
                        if pad_size > 0:
                            pad_tokens = torch.full((tok.shape[0], pad_size), 2, device=tok.device, dtype=tok.dtype)
                            tok = torch.cat([tok, pad_tokens], dim=1)
                            pad_lp = torch.zeros((old.shape[0], pad_size), device=old.device, dtype=old.dtype)
                            old = torch.cat([old, pad_lp], dim=1)
                        padded_tok.append(tok)
                        padded_old.append(old)

                    b_enc = torch.cat(b_enc).to(self.device)
                    b_pix = torch.cat(b_pix).to(self.device)
                    b_tok = torch.cat(padded_tok).to(self.device)
                    b_old = torch.cat(padded_old).to(self.device)

                    # compute new log-probs and loss
                    new_lp, ent = self.score_sequence(b_enc, b_pix, b_tok)
                    new_seq = new_lp[:,1:].sum(1)
                    old_seq = b_old.sum(1)
                    ratio = (new_seq - old_seq).exp()
                    loss1 = -ratio * b_adv; loss2 = -torch.clamp(ratio,1-self.clip_coef,1+self.clip_coef)*b_adv
                    policy_loss = torch.max(loss1, loss2).mean()
                    entropy_loss = -self.entropy_coef * ent
                    total_loss = policy_loss + entropy_loss
                    
                    total_loss_avg += total_loss
                    counts += 1
                    
                    self.optimizer.zero_grad(); total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5); self.optimizer.step()
                    
            if self.accelerator is None or self.accelerator.is_main_process and self.tb:
                self.tb.add_scalar("GRPO/avg_loss", total_loss_avg/counts, epoch)
                
            # 3) Validation and checkpointing
            if self.val_loader:
                self.model.eval(); vr=0; nc=0
                with torch.no_grad():
                    for inputs, answers in self.val_loader:
                        inp, pix = inputs["input_ids"].to(self.device), inputs["pixel_values"].to(self.device)
                        gen_ids,_,_,_ = self.generate_one_pass(inp,pix)
                        for gi, ans in zip(gen_ids, answers):
                            txt = self.processor.tokenizer.decode(gi, skip_special_tokens=True)
                            vr += reward_function_vlm(txt, ans); nc += 1
                avg_vr = vr/nc; print(f"Epoch {epoch+1} val reward: {avg_vr:.3f}")
                if self.tb:
                    self.tb.add_scalar("GRPO/avg_vr", vr/nc, epoch)
                if avg_vr > best_val:
                    best_val = avg_vr
                    checkpoint_dir = self.config.get("checkpoint_dir","./grpo_checkpoint")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    model_to_save.save_pretrained(checkpoint_dir)
                    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                    self.processor.save_pretrained(checkpoint_dir)
        torch.cuda.empty_cache()
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("GRPO training loop completed.")