# src/trainer.py
import torch
import random
import logging
import re
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from transformers import AdamW, get_scheduler
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator

import os

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Implements a standard training loop.
    """
    def __init__(self, model, processor, train_loader, val_loader, device, config, use_accelerator=True, checkpoint_dir="checkpoint/"):
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint_dir = checkpoint_dir

        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.isdir(checkpoint_dir) and os.path.isfile(weights_path):
            print(f"Loading model weights from {weights_path}")
            state_dict = torch.load(weights_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.processor = self.processor.from_pretrained(checkpoint_dir)
        else:
            print("No existing checkpoint weights found — training from scratch.")
            
        
        self.optimizer = AdamW(self.model.parameters(), lr=config.get("learning_rate", 1e-6))
        #self.scheduler = get_scheduler(
        #    name="linear",
        #    optimizer=self.optimizer,
        #    num_warmup_steps=0,
        #    num_training_steps=config.get("epochs", 10) * len(train_loader),
        #)
        
        self.accelerator = None
        if use_accelerator:
            self.logger.info(f"ACCELERATOR activated")
            self.accelerator = Accelerator(mixed_precision="bf16") 
            self.device = self.accelerator.device
            # Prepare model, optimizer, and dataloaders for distributed training or mixed precision
            self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        else:
            self.logger.info(f"No accelerator")
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)

        self.best_val_loss = float("inf")

    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training Epoch"):
            inputs, answers = batch
            input_ids = inputs["input_ids"].to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device)
            # Prepare labels using the processor's tokenizer.
            labels = self.processor.tokenizer(
                text=answers, 
                return_tensors="pt", 
                padding=True, 
                return_token_type_ids=False
            ).input_ids.to(self.device)
            outputs = self.model(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info(f"Average training loss: {avg_loss}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation Epoch"):
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = self.processor.tokenizer(
                    text=answers, 
                    return_tensors="pt", 
                    padding=True, 
                    return_token_type_ids=False
                ).input_ids.to(self.device)
                outputs = self.model(
                    input_ids=input_ids, 
                    pixel_values=pixel_values, 
                    labels=labels
                )
                total_loss += outputs.loss.item()
        avg_loss = total_loss / len(self.val_loader)
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info(f"Average validation loss: {avg_loss}")
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                print(f"Validation loss improved. Saving checkpoint to {self.checkpoint_dir}")
                # save model weights + config + tokenizer
                self.model.save_pretrained(self.checkpoint_dir)
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "pytorch_model.bin"))
                self.processor.save_pretrained(self.checkpoint_dir)
        return avg_loss

    def train(self, epochs):
        for epoch in range(epochs):
            # Wait for all GPUs before starting new epoch
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            # Log only on main process
            if self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch()
            self.validate()
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()


class GRPOTrainer(BaseTrainer):
    """
    Extends BaseTrainer with GRPO-specific training.
    """
    def __init__(self, model, processor, train_loader, val_loader, device, config, use_accelerator=True):
        super().__init__(model, processor, train_loader, val_loader, device, config, use_accelerator)
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

        self.optimizer = AdamW(self.model.parameters(), lr=self.grpo_lr)
        if use_accelerator:
            self.optimizer = self.accelerator.prepare(
                self.optimizer
            )
        else:
            self.logger.info(f"No accelerator")
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)

    # ------------------------
    # GRPO Utility Methods
    # ------------------------
    def generate_one_pass(self, input_ids, pixel_values, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        batch_size = pixel_values.shape[0]
        generated_ids = []
        all_log_probs = []
        all_entropy = []
        state_embeddings = []
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
            state_embedding = outputs.encoder_last_hidden_state[:, 0, :]
            next_token_logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            distrib = torch.distributions.Categorical(probs)
            sampled_tokens = distrib.sample()
            entropy = distrib.entropy()
            all_entropy.append(entropy)
            log_probs = distrib.log_prob(sampled_tokens)
            generated_ids.append(sampled_tokens.unsqueeze(-1))
            all_log_probs.append(log_probs)
            state_embeddings.append(state_embedding)
            decoder_input_ids = torch.cat([decoder_input_ids, sampled_tokens.unsqueeze(-1)], dim=-1)
            finished |= (sampled_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        generated_ids = torch.cat(generated_ids, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_entropy = torch.stack(all_entropy, dim=1)
        state_embeddings = torch.stack(state_embeddings, dim=1)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return generated_ids, all_log_probs, state_embeddings, all_entropy

    def score_sequence(self, input_ids, pixel_values, decoder_input_ids):
        batch_size = decoder_input_ids.shape[0]
        start_tokens = torch.tensor([2, 0], device=self.device).unsqueeze(0).expand(batch_size, -1)
        decoder_input_ids = torch.cat([start_tokens, decoder_input_ids], dim=1)
        dec_in = decoder_input_ids[:, :-1].contiguous()
        labels = decoder_input_ids[:, 1:].contiguous()
        #input_ids = input_ids.contiguous()
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=dec_in,
            labels=labels,
            return_dict=True
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        forced_lps = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
        probs = log_probs.exp()
        token_entropies = -(probs * log_probs).sum(dim=-1)
        mean_entropy = token_entropies.mean()
        return forced_lps, mean_entropy

    @staticmethod
    def remove_outer_parentheses(s):
        return s[1:-1] if s.startswith('(') and s.endswith(')') else s

    @staticmethod
    def extract_colors_and_parity(text):
        pattern_pairs = re.compile(r"Pairs:\s+(.*?)\.\s+Leftover:\s+(\w+|none)\s*=>\s*(\w+)")
        pattern_none = re.compile(r"Pairs:\s+none\.\s+Leftover:\s+(\w+|none)\s*=>\s*(\w+)")
        match = pattern_pairs.search(text) or pattern_none.search(text)
        if not match:
            return None
        if pattern_none.search(text):
            leftover = match.group(1)
            parity  = match.group(2)
            return {
                "individual_colors": [],
                "leftover": None if leftover == "none" else leftover,
                "parity": parity
            }
        else:
            pairs_str = match.group(1)
            leftover  = match.group(2)
            parity    = match.group(3)
            pair_regex = re.compile(r"\(\s*\w+\s*,\s*\w+\s*\)")
            raw_pairs  = pair_regex.findall(pairs_str)
            individual_colors = []
            for raw_pair in raw_pairs:
                inner = GRPOTrainer.remove_outer_parentheses(raw_pair.strip())
                color1, color2 = inner.split(',')
                individual_colors.extend([color1.strip(), color2.strip()])
            individual_colors = sorted(individual_colors)
            return {
                "individual_colors": individual_colors,
                "leftover": None if leftover == "none" else leftover,
                "parity": parity
            }

    @staticmethod
    def reward_function_odd_even(generated_text: str, gold_data: str) -> float:
        gold_dict = GRPOTrainer.extract_colors_and_parity(gold_data)
        if not gold_dict:
            return 0.0
        gold_colors = gold_dict["individual_colors"]
        if gold_dict["leftover"]:
            gold_colors.append(gold_dict["leftover"])
        gold_colors_sorted = sorted(gold_colors)
        final_label = gold_dict["parity"]
        is_gold_odd = (final_label.lower() == "odd")
        reward = 0.0
        gen_dict = GRPOTrainer.extract_colors_and_parity(generated_text)
        if not gen_dict:
            return -1.0
        gen_colors = gen_dict["individual_colors"]
        if gen_dict["leftover"]:
            gen_colors.append(gen_dict["leftover"])
        gen_colors_sorted = sorted(gen_colors)
        is_gen_odd = (gen_dict["parity"].lower() == "odd")
        if is_gen_odd == is_gold_odd:
            reward += 2.0
        else:
            reward -= 2.0
        if (final_label == "none" and gen_dict["parity"] == "none") or (final_label != "none" and gen_dict["parity"] != "none"):
            reward += 1.0
        else:
            reward -= 1.0
        if gen_colors_sorted == gold_colors_sorted:
            reward += 3.0
        else:
            reward -= 1.0
        return reward

    def batch_generate_group(self, old_model, input_ids_list, pixel_list):
        B = len(input_ids_list)
        all_gen_ids = []
        all_old_log_probs = []
        for _ in range(self.group_size):
            batch_input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
            batch_pix = torch.cat(pixel_list, dim=0).to(self.device)
            #print(batch_pix.shape)
            with torch.no_grad():
                gen_ids, old_log_probs, _, _ = self.generate_one_pass(
                    batch_input_ids, batch_pix, max_new_tokens=self.max_new_tokens
                )
            all_gen_ids.append(gen_ids.cpu())
            all_old_log_probs.append(old_log_probs.cpu())
        return all_gen_ids, all_old_log_probs

    # ------------------------
    # GRPO Training Loop
    # ------------------------
    def train_grpo(self):
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("Starting GRPO training loop.")
        self.model.train()
        train_iter = iter(self.train_loader)
        best_val_rewards = 0


        for epoch in range(self.config.get("grpo_epochs", 3)):
            total_reward = 0.0
            total_count = 0
            self.model.train()
            for iteration in tqdm(range(self.iterations_per_epoch), 
                                  desc=f"GRPO Epoch {epoch+1}/{self.config.get('grpo_epochs', 3)}"):
                # Collect a rollout batch.
                rollout_batches = []
                while len(rollout_batches) < self.rollout_batch_size:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)
                    rollout_batches.append(batch)
                
                # Flatten the rollout batch.
                inputs_list = []
                answers_list = []
                for (inputs, answers) in rollout_batches:
                    B_ = inputs["input_ids"].shape[0]
                    for i in range(B_):
                        single_inputs = {
                            "input_ids": inputs["input_ids"][i].unsqueeze(0),
                            "pixel_values": inputs["pixel_values"][i].unsqueeze(0),
                        }
                        inputs_list.append(single_inputs)
                        answers_list.append(answers[i])
                B = len(inputs_list)
                if B == 0:
                    continue

                with torch.no_grad():
                    old_model = deepcopy(self.model).eval()
                
                input_ids_list = [inp["input_ids"] for inp in inputs_list]
                pixel_values_list = [inp["pixel_values"] for inp in inputs_list]
                all_gen_ids, all_old_log_probs = self.batch_generate_group(old_model, input_ids_list, pixel_values_list)
                if iteration == 5:
                    print(torch.cuda.memory_summary())
                
                memory = []
                with torch.no_grad():
                    for b_idx in range(B):
                        group_rewards = []
                        group_tokens = []
                        group_logps = []
                        for g in range(self.group_size):
                            seq_ids = all_gen_ids[g][b_idx]
                            seq_lp = all_old_log_probs[g][b_idx]
                            seq_ids_ = seq_ids
                            if 2 in seq_ids:
                                index = (seq_ids == 2).nonzero(as_tuple=True)[0][0]
                                seq_ids_ = seq_ids[:index + 1]
                            while len(seq_ids) < self.max_new_tokens:
                                seq_ids = torch.cat((seq_ids, torch.tensor([2], dtype=seq_ids.dtype, device=seq_ids.device)))
                            while len(seq_lp) < self.max_new_tokens:
                                seq_lp = torch.cat((seq_lp, torch.tensor([2], dtype=seq_lp.dtype, device=seq_lp.device)))
                            pred_text = self.processor.tokenizer.decode(seq_ids_, skip_special_tokens=True)
                            r = GRPOTrainer.reward_function_odd_even(pred_text, answers_list[b_idx])
                            if self.accelerator is None or self.accelerator.is_main_process:
                                self.logger.debug(f"Reward: {r}, Prediction: {pred_text}, Answer: {answers_list[b_idx]}")
                            group_rewards.append(r)
                            group_tokens.append(seq_ids)
                            group_logps.append(seq_lp)
                        group_rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=self.device)
                        r_mean = group_rewards_t.mean().item()
                        r_std = group_rewards_t.std().item() + 1e-8
                        for g in range(self.group_size):
                            adv_value = (group_rewards[g] - r_mean) / r_std
                            memory.append({
                                "encoder_ids": inputs_list[b_idx]["input_ids"].cpu(),
                                "pixel_vals": inputs_list[b_idx]["pixel_values"].cpu(),
                                "tokens": group_tokens[g].unsqueeze(0),
                                "old_logprobs": group_logps[g].unsqueeze(0),
                                "advantage": adv_value,
                                "reward": group_rewards[g],
                            })
                            total_reward += group_rewards[g]
                            total_count += 1
                
                # Perform teacher-forcing updates.
                for _ in range(self.update_epochs):
                    random.shuffle(memory)
                    for start_idx in range(0, len(memory), self.minibatch_size):
                        batch_data = memory[start_idx:start_idx+self.minibatch_size]
                        if not batch_data:
                            continue
                        b_enc_ids = []
                        b_pix = []
                        b_tokens = []
                        b_old_lp = []
                        b_adv = []
                        for item in batch_data:
                            b_enc_ids.append(item["encoder_ids"])
                            b_pix.append(item["pixel_vals"])
                            b_tokens.append(item["tokens"])
                            b_old_lp.append(item["old_logprobs"])
                            b_adv.append(item["advantage"])
                        b_enc_ids = torch.cat(b_enc_ids, dim=0).to(self.device)
                        b_pix = torch.cat(b_pix, dim=0).to(self.device)
                        b_tokens = torch.cat(b_tokens, dim=0).to(self.device)
                        b_old_lp = torch.cat(b_old_lp, dim=0).to(self.device)
                        b_adv = torch.tensor(b_adv, dtype=torch.float, device=self.device)
                        #print(b_pix.shape)
                        
                        new_log_probs, mean_entropy = self.score_sequence(b_enc_ids, b_pix, b_tokens)
                        
                        old_lp_seq = b_old_lp.sum(dim=1)
                        new_lp_seq = new_log_probs[:, 1:].sum(dim=1)
                        
                        ratio = (new_lp_seq - old_lp_seq).exp()
                        pg_loss1 = -ratio * b_adv
                        pg_loss2 = -torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef) * b_adv
                        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        entropy_loss = -self.entropy_coef * mean_entropy
                        total_loss = policy_loss + entropy_loss
                        
                        self.model.zero_grad()
                                    
                        if self.accelerator is not None:
                            self.accelerator.backward(total_loss)
                        else:
                            total_loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        torch.cuda.empty_cache()

                memory.clear()
            avg_reward = total_reward / max(total_count, 1e-8)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch+1}: avg_reward={avg_reward:.3f}")
            if True:
                if self.val_loader:
                    self.model.eval()
                    val_rewards = 0
                    num_samples = 0
                    with torch.no_grad():
                        for batch in tqdm(self.val_loader, desc="Validation"):
                            inputs, answers = batch
                            inp = inputs["input_ids"].to(self.device)
                            pix = inputs["pixel_values"].to(self.device)
                            B_ = inp.size(0)
                            gen_ids, _, _, _ = self.generate_one_pass(inp, pix, max_new_tokens=50)
                            pred_texts = [self.processor.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
                            for i in range(B_):
                                val_rewards += GRPOTrainer.reward_function_odd_even(pred_texts[i], answers[i])
                            num_samples += B_
                    self.logger.info(f"[Epoch {epoch+1}] Validation Reward = {val_rewards / num_samples:.3f}")

                    # Save model checkpoint
                    if best_val_rewards > val_rewards:
                        print("Saving model")
                        output_dir = "./model_checkpoints/epoch_best"
                        os.makedirs(output_dir, exist_ok=True)
                    
                        # Ensure only the main process saves the model to avoid race conditions
                        #if torch.distributed.get_rank() == 0:
                        #    if hasattr(self.model, "module"):  # If using DDP, unwrap the model
                        #        self.model.module.save_pretrained(output_dir)
                        #    else:
                        #        self.model.save_pretrained(output_dir)
                        # 
                        #    self.processor.save_pretrained(output_dir)
                    
                        best_val_rewards = val_rewards
                        print(f"Model saved in {output_dir}.")

            
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("GRPO training loop completed.")


    def batch_generate_group(self, old_model, input_ids_list, pixel_list):
        """
        Generate rollout groups using the old_model (which is on CPU).
        """
        B = len(input_ids_list)
        all_gen_ids = []
        all_old_log_probs = []
        # Use the device of the old_model (it should be CPU)
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
    
    def train_grpo(self):
        self.logger.info("Starting GRPO training loop.")
        self.model.train()
        train_iter = iter(self.train_loader)
        total_reward = 0.0
        total_count = 0
        best_val_rewards = 0.0
    
        for epoch in range(self.config.get("grpo_epochs", 3)):
            self.model.train()
            for iteration in tqdm(range(self.iterations_per_epoch), 
                                  desc=f"GRPO Epoch {epoch+1}/{self.config.get('grpo_epochs', 3)}"):
                # Collect a rollout batch.
                rollout_batches = []
                while len(rollout_batches) < self.rollout_batch_size:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)
                    rollout_batches.append(batch)
                
                # Flatten the rollout batch.
                inputs_list = []
                answers_list = []
                for (inputs, answers) in rollout_batches:
                    B_ = inputs["input_ids"].shape[0]
                    for i in range(B_):
                        single_inputs = {
                            "input_ids": inputs["input_ids"][i].unsqueeze(0),
                            "pixel_values": inputs["pixel_values"][i].unsqueeze(0),
                        }
                        inputs_list.append(single_inputs)
                        answers_list.append(answers[i])
                B = len(inputs_list)
                if B == 0:
                    continue
    
                with torch.no_grad():
                    # Unwrap the model if it is wrapped by Accelerator
                    if hasattr(self.model, "module"):
                        old_model = deepcopy(self.model.module).cpu().eval()
                    else:
                        old_model = deepcopy(self.model).cpu().eval()
                
                input_ids_list = [inp["input_ids"] for inp in inputs_list]
                pixel_values_list = [inp["pixel_values"] for inp in inputs_list]
                all_gen_ids, all_old_log_probs = self.batch_generate_group(old_model, input_ids_list, pixel_values_list)
                
                # Free the copied model immediately.
                del old_model
                import gc
                gc.collect()
    
                memory = []
                with torch.no_grad():
                    for b_idx in range(B):
                        group_rewards = []
                        group_tokens  = []
                        group_logps   = []
                        for g in range(self.group_size):
                            seq_ids = all_gen_ids[g][b_idx]
                            seq_lp  = all_old_log_probs[g][b_idx]
                            seq_ids_ = seq_ids.clone()
                            if 2 in seq_ids:
                                index = (seq_ids == 2).nonzero(as_tuple=True)[0][0]
                                seq_ids_ = seq_ids[:index + 1]
                            while len(seq_ids) < self.max_new_tokens:
                                seq_ids = torch.cat((seq_ids, torch.tensor([2], dtype=seq_ids.dtype, device=seq_ids.device)))
                            while len(seq_lp) < self.max_new_tokens:
                                seq_lp = torch.cat((seq_lp, torch.tensor([2], dtype=seq_lp.dtype, device=seq_lp.device)))
                            pred_text = self.processor.tokenizer.decode(seq_ids_, skip_special_tokens=True)
                            r = GRPOTrainer.reward_function_odd_even(pred_text, answers_list[b_idx])
                            self.logger.debug(f"Reward: {r}, Prediction: {pred_text}, Answer: {answers_list[b_idx]}")
                            group_rewards.append(r)
                            group_tokens.append(seq_ids)
                            group_logps.append(seq_lp)
                        
                        group_rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=self.device)
                        r_mean = group_rewards_t.mean().item()
                        r_std  = group_rewards_t.std().item() + 1e-8
                        for g in range(self.group_size):
                            adv_value = (group_rewards[g] - r_mean) / r_std
                            memory.append({
                                "encoder_ids": inputs_list[b_idx]["input_ids"].cpu(),
                                "pixel_vals": inputs_list[b_idx]["pixel_values"].cpu(),
                                "tokens": group_tokens[g].unsqueeze(0),
                                "old_logprobs": group_logps[g].unsqueeze(0),
                                "advantage": adv_value,
                                "reward": group_rewards[g],
                            })
                            total_reward += group_rewards[g]
                            total_count  += 1
                
                # Perform teacher-forcing updates.
                for _ in range(self.update_epochs):
                    random.shuffle(memory)
                    for start_idx in range(0, len(memory), self.minibatch_size):
                        batch_data = memory[start_idx:start_idx+self.minibatch_size]
                        if not batch_data:
                            continue
                        b_enc_ids = []
                        b_pix = []
                        b_tokens = []
                        b_old_lp = []
                        b_adv = []
                        for item in batch_data:
                            b_enc_ids.append(item["encoder_ids"])
                            b_pix.append(item["pixel_vals"])
                            b_tokens.append(item["tokens"])
                            b_old_lp.append(item["old_logprobs"])
                            b_adv.append(item["advantage"])
                        b_enc_ids = torch.cat(b_enc_ids, dim=0).to(self.device)
                        b_pix = torch.cat(b_pix, dim=0).to(self.device)
                        b_tokens = torch.cat(b_tokens, dim=0).to(self.device)
                        b_old_lp = torch.cat(b_old_lp, dim=0).to(self.device)
                        b_adv = torch.tensor(b_adv, dtype=torch.float, device=self.device)
                        new_log_probs, mean_entropy = self.score_sequence(b_enc_ids, b_pix, b_tokens)
                        old_lp_seq = b_old_lp.sum(dim=1)
                        new_lp_seq = new_log_probs[:, 1:].sum(dim=1)
                        ratio = (new_lp_seq - old_lp_seq).exp()
                        pg_loss1 = -ratio * b_adv
                        pg_loss2 = -torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef) * b_adv
                        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                        entropy_loss = -self.entropy_coef * mean_entropy
                        total_loss = policy_loss + entropy_loss
                        self.model.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        torch.cuda.empty_cache()
                avg_reward = total_reward / max(total_count, 1e-8)
                self.logger.info(f"Epoch {epoch+1}: avg_reward={avg_reward:.3f}")
            if True:
                if self.val_loader:
                    self.model.eval()
                    val_rewards = 0
                    num_samples = 0
                    with torch.no_grad():
                        for batch in tqdm(self.val_loader, desc="Validation"):
                            inputs, answers = batch
                            inp = inputs["input_ids"].to(self.device)
                            pix = inputs["pixel_values"].to(self.device)
                            B_ = inp.size(0)
                            gen_ids, _, _, _ = self.generate_one_pass(inp, pix, max_new_tokens=self.max_new_tokens)
                            pred_texts = [self.processor.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
                            for i in range(B_):
                                val_rewards += GRPOTrainer.reward_function_odd_even(pred_texts[i], answers[i])
                            num_samples += B_
                    self.logger.info(f"[Epoch {epoch+1}] Validation Reward = {val_rewards / num_samples:.3f}")
                   # Free up memory after validation
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    #val_rewards = val_rewards.cpu()
                    # Save model checkpoint
                    if best_val_rewards < val_rewards:
                        print("Saving model")
                        output_dir = "./model_checkpoints/epoch_best"
                        os.makedirs(output_dir, exist_ok=True)
                    
                        # Ensure only the main process saves the model to avoid race conditions
                        #if torch.distributed.get_rank() == 0:
                        #    if hasattr(self.model, "module"):  # If using DDP, unwrap the model
                        #        self.model.module.save_pretrained(output_dir)
                        #    else:
                        #        self.model.save_pretrained(output_dir)
                        # 
                        #    self.processor.save_pretrained(output_dir)
                    
                        best_val_rewards = val_rewards
                        print(f"Model saved in {output_dir}.")

            
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("GRPO training loop completed.")

