# src/trainer.py

import os
import re
import copy
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator

logger = logging.getLogger(__name__)

def extract_vlm_data(text: str):
        """
        Extract reasoning (<think>…</think>) and answer (<answer>…</answer>) from VLM output.
        """
        think_m = re.search(r"<think>(.*?)</think>", text, re.S)
        ans_m   = re.search(r"<answer>(.*?)</answer>", text, re.S)
        reasoning = think_m.group(1).strip() if think_m else None
        answer    = ans_m.group(1).strip()   if ans_m   else None
        return reasoning, answer

def extract_circles_from_reasoning(reasoning: str):
        """
        Parse lines like "- Color {color}, size {size}, at ({x},{y})".
        """
        pattern = re.compile(
            r"-\s*Color\s+(?P<color>\w+),\s*size\s+(?P<size>\d+),\s*at\s*\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)"
        )
        return [m.groupdict() for m in pattern.finditer(reasoning or "")]

def reward_function_vlm(generated_text: str, gold_text: str) -> float:
        """
        Reward = +3/-3 for answer correctness, +1/-1 for reasoning tags, +2/-2 for circle count match.
        """
        gen_reason, gen_ans = extract_vlm_data(generated_text)
        gold_reason, gold_ans = extract_vlm_data(gold_text)

        if gen_reason is None or gen_ans is None:
            return -5.0

        reward = 0.0
        reward += 3.0 if gen_ans.lower() == gold_ans.lower() else -3.0
        reward += 1.0 if "<think>" in generated_text and "</think>" in generated_text else -1.0

        gen_circles  = extract_circles_from_reasoning(gen_reason)
        gold_circles = extract_circles_from_reasoning(gold_reason)
        reward += 2.0 if len(gen_circles) == len(gold_circles) else -2.0

        return reward
        
class BaseTrainer:
    """
    Implements a standard supervised fine-tuning loop.
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
        checkpoint_dir: str = "checkpoint/",
        tb_writer=None
    ):
        self.model          = model
        self.processor      = processor
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.config         = config
        self.logger         = logging.getLogger(self.__class__.__name__)
        self.checkpoint_dir = checkpoint_dir
        self.tb             = tb_writer
        self.global_step    = 0

        # load existing checkpoint if present
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.isdir(checkpoint_dir) and os.path.isfile(weights_path):
            self.logger.info(f"Loading model weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.processor = self.processor.from_pretrained(checkpoint_dir)
        else:
            self.logger.info("No existing checkpoint — training from scratch.")

        # optimizer + scheduler
        lr = config.get("learning_rate", 1e-6)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)

        # accelerator (distributed / mixed precision) setup
        self.accelerator = None
        if use_accelerator:
            self.logger.info("Accelerator activated")
            self.accelerator = Accelerator(mixed_precision="bf16")
            # postpone setting device until after prepare
            self.model, self.optimizer, self.train_loader, self.val_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader
                )
            self.device = self.accelerator.device
        else:
            self.logger.info("No accelerator")

        self.best_val_loss = float("inf")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training Epoch"):
            inputs, answers = batch
            input_ids     = inputs["input_ids"].to(self.device)
            pixel_values  = inputs["pixel_values"].to(self.device)
            # prepare labels
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
            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info(f"Average training loss: {avg_loss:.4f}")
            if self.tb:
                self.tb.add_scalar("SFT/train_loss", avg_loss, self.global_step)

        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation Epoch"):
                inputs, answers = batch
                input_ids     = inputs["input_ids"].to(self.device)
                pixel_values  = inputs["pixel_values"].to(self.device)
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
            self.logger.info(f"Average validation loss: {avg_loss:.4f}")
            if self.tb:
                self.tb.add_scalar("SFT/val_loss", avg_loss, self.global_step)

            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.logger.info(f"Validation loss improved, saving to {self.checkpoint_dir}")
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                model_to_save.save_pretrained(self.checkpoint_dir)
                torch.save(model_to_save.state_dict(), os.path.join(self.checkpoint_dir, "pytorch_model.bin"))
                self.processor.save_pretrained(self.checkpoint_dir)

        return avg_loss

    def train(self, epochs: int):
        for epoch in range(epochs):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch+1}/{epochs}")

            self.train_epoch()
            self.validate()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def score_sequence(self, input_ids, pixel_values, decoder_input_ids):
        batch_size = decoder_input_ids.shape[0]
        start_tokens = torch.tensor([2, 0], device=self.device).unsqueeze(0).expand(batch_size, -1)
        decoder_input_ids = torch.cat([start_tokens, decoder_input_ids], dim=1)
        dec_in = decoder_input_ids[:, :-1].contiguous()
        labels = decoder_input_ids[:, 1:].contiguous()

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


