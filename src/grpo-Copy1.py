import torch
import random
from tqdm import tqdm
from copy import deepcopy
import re
import logging

logger = logging.getLogger(__name__)

def generate_one_pass(model, processor, input_ids, pixel_values, max_new_tokens=5):
    """
    Generates a short output from the model.
    Returns:
       generated_ids: Tensor of generated token IDs.
       log_probs: Log probabilities per token.
       state_embeddings, entropy: Additional outputs.
    """
    batch_size = pixel_values.shape[0]
    generated_ids = []
    all_log_probs = []
    all_entropy = []
    state_embeddings = []
    
    bos_token_id = 2
    eos_token_id = 2
    decoder_input_ids = torch.tensor([[2, 0]] * batch_size, dtype=torch.long, device=input_ids.device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    current_input_ids = input_ids.clone()
    
    for step in range(max_new_tokens):
        outputs = model(
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
    return generated_ids, all_log_probs, state_embeddings, all_entropy

def score_sequence_hf(model, input_ids, pixel_values, decoder_input_ids):
    """
    Computes token-level log-probabilities for a given sequence.
    Returns:
        new_log_probs: Log probabilities.
        mean_entropy: Average entropy.
    """
    batch_size = decoder_input_ids.shape[0]
    start_tokens = torch.tensor([2, 0], device=decoder_input_ids.device).unsqueeze(0)
    start_tokens = start_tokens.expand(batch_size, -1)
    decoder_input_ids = torch.cat([start_tokens, decoder_input_ids], dim=1)
    dec_in = decoder_input_ids[:, :-1]
    labels = decoder_input_ids[:, 1:]
    #print(dec_in)
    #print(input_ids)
    #print(dec_in.shape)
    #print(labels.shape)
    #print(input_ids.shape)
    #print(pixel_values.shape)
    dec_in = dec_in.contiguous()
    labels = labels.contiguous()
    input_ids = input_ids.contiguous()
    outputs = model(
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

def remove_outer_parentheses(s):
    if s.startswith('(') and s.endswith(')'):
        return s[1:-1]
    return s

def extract_colors_and_parity(text):
    """
    Extracts color pairs and parity information from the text.
    """
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
            inner = remove_outer_parentheses(raw_pair.strip())
            color1, color2 = inner.split(',')
            individual_colors.append(color1.strip())
            individual_colors.append(color2.strip())
        individual_colors = sorted(individual_colors)
        return {
            "individual_colors": individual_colors,
            "leftover": None if leftover == "none" else leftover,
            "parity": parity
        }

def reward_function_odd_even(generated_text: str, gold_data: str) -> float:
    """
    Computes a reward score comparing generated text to gold data.
    """
    gold_data_dict = extract_colors_and_parity(gold_data)
    if not gold_data_dict:
        return 0.0

    gold_colors = gold_data_dict["individual_colors"]
    if gold_data_dict["leftover"]:
        gold_colors.append(gold_data_dict["leftover"])
    gold_colors_sorted = sorted(gold_colors)
    final_label = gold_data_dict["parity"]
    is_gold_odd = (final_label.lower() == "odd")

    reward = 0.0
    generated_text_dict = extract_colors_and_parity(generated_text)
    if not generated_text_dict:
        return -1.0
    
    generated_colors = generated_text_dict["individual_colors"]
    if generated_text_dict["leftover"]:
        generated_colors.append(generated_text_dict["leftover"])
    generated_colors_sorted = sorted(generated_colors)
    generated_final_label = generated_text_dict["parity"]
    is_generated_odd = (generated_final_label.lower() == "odd")
    
    parity_match = (is_generated_odd == is_gold_odd)
    leftover_match = ((final_label == "none") and (generated_final_label == "none")) or ((final_label != "none") and (generated_final_label != "none"))
    
    if parity_match:
        reward += 2.0
    else:
        reward -= 2.0
        
    if leftover_match:
        reward += 1.0
    else:
        reward -= 1.0
    if generated_colors_sorted == gold_colors_sorted:
        reward += 3.0
    else:
        reward -= 1.0
    
    return reward


def reward_function_odd_even(generated_text: str, gold_data: str) -> float:
    """
    A partial-credit version of your leftover/pair-based reward function:
      - ±2 for matching parity (odd/even)
      - ±1 for leftover presence match
      - partial scoring for color usage: we do +0.5 per color correct, -0.5 per color missing or extra,
        then clamp that sub-score to [-3,3] (so it doesn't exceed ±3).
    """
    gold_dict = extract_colors_and_parity(gold_data)
    if not gold_dict:
        # If gold data fails parse, no reward
        return 0.0

    gold_colors = gold_dict["individual_colors"].copy()
    if gold_dict["leftover"]:
        gold_colors.append(gold_dict["leftover"])
    gold_colors_sorted = sorted([c.lower() for c in gold_colors])

    final_label = gold_dict["parity"].lower()
    is_gold_odd = (final_label == "odd")
    gold_leftover_exists = (gold_dict["leftover"] is not None)

    # parse generated text
    gen_dict = extract_colors_and_parity(generated_text)
    if not gen_dict:
        # If generated text fails parse, negative or 0
        return -1.0
    
    gen_colors = gen_dict["individual_colors"].copy()
    if gen_dict["leftover"]:
        gen_colors.append(gen_dict["leftover"])
    gen_colors_sorted = sorted([c.lower() for c in gen_colors])

    gen_label = gen_dict["parity"].lower()
    is_gen_odd = (gen_label == "odd")
    gen_leftover_exists = (gen_dict["leftover"] is not None)

    reward = 0.0

    # 1) Check parity
    if is_gen_odd == is_gold_odd:
        reward += 2.0
    else:
        reward -= 2.0

    # 2) Check leftover presence
    if gold_leftover_exists == gen_leftover_exists:
        reward += 1.0
    else:
        reward -= 1.0

    # 3) Color usage partial scoring
    #   We'll do a set-based approach:
    #   +0.5 for each color that is in both sets,
    #   -0.5 for each color that is missing (gold but not in gen)
    #   -0.5 for each color that is extra (in gen but not in gold)
    #   Then clamp to [-3, +3].
    
    gold_set = set(gold_colors_sorted)
    gen_set  = set(gen_colors_sorted)

    intersection = gold_set.intersection(gen_set)
    missing      = gold_set - gen_set
    extra        = gen_set - gold_set

    color_score = 0.0

    # +0.5 each color in the intersection
    color_score += 0.5 * len(intersection)
    # -0.5 for each missing
    color_score -= 0.5 * len(missing)
    # -0.5 for each extra
    color_score -= 0.5 * len(extra)

    # clamp color_score to [-3,3]
    color_score = max(-3.0, min(3.0, color_score))
    reward += color_score

    return reward
    
def batch_generate_group(old_model, processor, input_ids_list, pixel_list, group_size=3, max_new_tokens=2, device="cuda"):
    """
    For a batch of questions, performs group sampling.
    Returns:
      all_gen_ids: List of generated token tensors.
      all_old_log_probs: List of log probability tensors.
    """
    B = len(input_ids_list)
    all_gen_ids = []
    all_old_log_probs = []

    for _ in range(group_size):
        batch_input_ids = torch.cat(input_ids_list, dim=0).to(device)
        batch_pix = torch.cat(pixel_list, dim=0).to(device)
        with torch.no_grad():
            gen_ids, old_log_probs, _, _ = generate_one_pass(old_model, processor, batch_input_ids, batch_pix, max_new_tokens=max_new_tokens)
        all_gen_ids.append(gen_ids.cpu())
        all_old_log_probs.append(old_log_probs.cpu())
    return all_gen_ids, all_old_log_probs

def grpo_training_loop_subset(model, processor, train_loader, optimizer, device, epochs=3, group_size=3, clip_coef=0.01, entropy_coef=0.001, update_epochs=2, rollout_batch_size=3, minibatch_size=4, max_new_tokens=50, val_loader=None, iterations_per_epoch=20):
    """
    GRPO training loop implementation.
    """
    logger.info("Starting GRPO training loop.")
    model.train()
    train_iter = iter(train_loader)
    total_reward = 0.0
    total_count  = 0

    for epoch in range(epochs):
        model.train()
        for iteration in tqdm(range(iterations_per_epoch), desc=f"GRPO Epoch {epoch+1}/{epochs}"):
            rollout_batches = []
            while len(rollout_batches) < rollout_batch_size:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                rollout_batches.append(batch)
            
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
                old_model = deepcopy(model).eval()
            
            input_ids_list = [inp["input_ids"] for inp in inputs_list]
            pixel_values_list = [inp["pixel_values"] for inp in inputs_list]
            all_gen_ids, all_old_log_probs = batch_generate_group(old_model, processor, input_ids_list, pixel_values_list, group_size=group_size, max_new_tokens=max_new_tokens, device=device)
            
            memory = []
            with torch.no_grad():
                for b_idx in range(B):
                    group_rewards = []
                    group_tokens  = []
                    group_logps   = []
                    for g in range(group_size):
                        seq_ids  = all_gen_ids[g][b_idx]
                        seq_lp   = all_old_log_probs[g][b_idx]
                        seq_ids_ = seq_ids
                        if 2 in seq_ids:
                            index = (seq_ids == 2).nonzero(as_tuple=True)[0][0]
                            seq_ids_ = seq_ids[:index + 1]
                        while len(seq_ids) < max_new_tokens:
                            seq_ids = torch.cat((seq_ids, torch.tensor([2], dtype=seq_ids.dtype, device=seq_ids.device)))
                        while len(seq_lp) < max_new_tokens:
                            seq_lp = torch.cat((seq_lp, torch.tensor([2], dtype=seq_lp.dtype, device=seq_lp.device)))
                        
                        pred_text = processor.tokenizer.decode(seq_ids_, skip_special_tokens=True)
                        r = reward_function_odd_even(pred_text, answers_list[b_idx])
                        logger.debug(f"Reward: {r}, Prediction: {pred_text}, Answer: {answers_list[b_idx]}")
                        group_rewards.append(r)
                        group_tokens.append(seq_ids)
                        group_logps.append(seq_lp)
                    
                    group_rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=device)
                    r_mean = group_rewards_t.mean().item()
                    r_std  = group_rewards_t.std().item() + 1e-8
                    for g in range(group_size):
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
            
            for _ in range(update_epochs):
                random.shuffle(memory)
                for start_idx in range(0, len(memory), minibatch_size):
                    batch_data = memory[start_idx:start_idx+minibatch_size]
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
                    b_enc_ids = torch.cat(b_enc_ids, dim=0).to(device)
                    b_pix = torch.cat(b_pix, dim=0).to(device)
                    b_tokens = torch.cat(b_tokens, dim=0).to(device)
                    b_old_lp = torch.cat(b_old_lp, dim=0).to(device)
                    b_adv = torch.tensor(b_adv, dtype=torch.float, device=device)
                    new_log_probs, mean_entropy = score_sequence_hf(model, b_enc_ids, b_pix, b_tokens)
                    old_lp_seq = b_old_lp.sum(dim=1)
                    new_lp_seq = new_log_probs[:, 1:].sum(dim=1)
                    ratio = (new_lp_seq - old_lp_seq).exp()
                    pg_loss1 = -ratio * b_adv
                    pg_loss2 = -torch.clamp(ratio, 1-clip_coef, 1+clip_coef) * b_adv
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = -entropy_coef * mean_entropy
                    total_loss = policy_loss + entropy_loss
                    model.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    torch.cuda.empty_cache()
        avg_reward = total_reward / max(total_count, 1e-8)
        logger.info(f"Epoch {epoch+1}: avg_reward={avg_reward:.3f}")
        if val_loader:
            model.eval()
            val_rewards = 0
            num_samples = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    inputs, answers = batch
                    inp = inputs["input_ids"].to(device)
                    pix = inputs["pixel_values"].to(device)
                    B_ = inp.size(0)
                    gen_ids, _, _, _ = generate_one_pass(model, processor, inp, pix, max_new_tokens=50)
                    pred_texts = [processor.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
                    for i in range(B_):
                        val_rewards += reward_function_odd_even(pred_texts[i], answers[i])
                    num_samples += B_
            logger.info(f"[Epoch {epoch+1}] Validation Reward = {val_rewards / num_samples:.3f}")
    logger.info("GRPO training loop completed.")

def grpo_training_loop_subset(model, processor, train_loader, optimizer, device, epochs=3, group_size=3, clip_coef=0.01, entropy_coef=0.001, update_epochs=2, rollout_batch_size=3, minibatch_size=4, max_new_tokens=50, val_loader=None, iterations_per_epoch=20):
    """
    GRPO training loop implementation.
    """
    logger.info("Starting GRPO training loop.")
    model.train()
    train_iter = iter(train_loader)
    total_reward = 0.0
    total_count  = 0

    for epoch in range(epochs):
        model.train()
        for iteration in tqdm(range(iterations_per_epoch), desc=f"GRPO Epoch {epoch+1}/{epochs}"):
            rollout_batches = []
            while len(rollout_batches) < rollout_batch_size:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                rollout_batches.append(batch)
            
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
                # Modification: Unwrap the model if it is wrapped by Accelerator.
                if hasattr(model, "module"):
                    old_model = deepcopy(model.module).cpu().eval()
                else:
                    old_model = deepcopy(model).cpu().eval()
            
            input_ids_list = [inp["input_ids"] for inp in inputs_list]
            pixel_values_list = [inp["pixel_values"] for inp in inputs_list]
            all_gen_ids, all_old_log_probs = batch_generate_group(old_model, processor, input_ids_list, pixel_values_list, group_size=group_size, max_new_tokens=max_new_tokens, device=device)
            
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
                    for g in range(group_size):
                        seq_ids  = all_gen_ids[g][b_idx]
                        seq_lp   = all_old_log_probs[g][b_idx]
                        seq_ids_ = seq_ids
                        if 2 in seq_ids:
                            index = (seq_ids == 2).nonzero(as_tuple=True)[0][0]
                            seq_ids_ = seq_ids[:index + 1]
                        while len(seq_ids) < max_new_tokens:
                            seq_ids = torch.cat((seq_ids, torch.tensor([2], dtype=seq_ids.dtype, device=seq_ids.device)))
                        while len(seq_lp) < max_new_tokens:
                            seq_lp = torch.cat((seq_lp, torch.tensor([2], dtype=seq_lp.dtype, device=seq_lp.device)))
                        
                        pred_text = processor.tokenizer.decode(seq_ids_, skip_special_tokens=True)
                        r = reward_function_odd_even(pred_text, answers_list[b_idx])
                        logger.debug(f"Reward: {r}, Prediction: {pred_text}, Answer: {answers_list[b_idx]}")
                        group_rewards.append(r)
                        group_tokens.append(seq_ids)
                        group_logps.append(seq_lp)
                    
                    group_rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=device)
                    r_mean = group_rewards_t.mean().item()
                    r_std  = group_rewards_t.std().item() + 1e-8
                    for g in range(group_size):
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
            
            for _ in range(update_epochs):
                random.shuffle(memory)
                for start_idx in range(0, len(memory), minibatch_size):
                    batch_data = memory[start_idx:start_idx+minibatch_size]
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
                    b_enc_ids = torch.cat(b_enc_ids, dim=0).to(device)
                    b_pix = torch.cat(b_pix, dim=0).to(device)
                    b_tokens = torch.cat(b_tokens, dim=0).to(device)
                    b_old_lp = torch.cat(b_old_lp, dim=0).to(device)
                    b_adv = torch.tensor(b_adv, dtype=torch.float, device=device)
                    
                    new_log_probs, mean_entropy = score_sequence_hf(model, b_enc_ids, b_pix, b_tokens)
                    old_lp_seq = b_old_lp.sum(dim=1)
                    new_lp_seq = new_log_probs[:, 1:].sum(dim=1)
                    ratio = (new_lp_seq - old_lp_seq).exp()
                    pg_loss1 = -ratio * b_adv
                    pg_loss2 = -torch.clamp(ratio, 1-clip_coef, 1+clip_coef) * b_adv
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = -entropy_coef * mean_entropy
                    total_loss = policy_loss + entropy_loss
                    model.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    torch.cuda.empty_cache()
            memory.clear()
        avg_reward = total_reward / max(total_count, 1e-8)
        logger.info(f"Epoch {epoch+1}: avg_reward={avg_reward:.3f}")
        if False and val_loader:
            model.eval()
            val_rewards = 0
            num_samples = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    inputs, answers = batch
                    inp = inputs["input_ids"].to(device)
                    pix = inputs["pixel_values"].to(device)
                    B_ = inp.size(0)
                    gen_ids, _, _, _ = generate_one_pass(model, processor, inp, pix, max_new_tokens=50)
                    pred_texts = [processor.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
                    for i in range(B_):
                        val_rewards += reward_function_odd_even(pred_texts[i], answers[i])
                    num_samples += B_
            logger.info(f"[Epoch {epoch+1}] Validation Reward = {val_rewards / num_samples:.3f}")
    logger.info("GRPO training loop completed.")

