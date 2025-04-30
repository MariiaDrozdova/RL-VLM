import torch
import random
from tqdm import tqdm
from copy import deepcopy
import re
from collections import Counter
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
    Compute new_log_probs for each token in `decoder_input_ids`
    using the "labels=" trick for a single forward pass in HF models.

    Args:
        input_ids        : [B, enc_seq_len]
        pixel_values     : [B, ...] (if using vision)
        decoder_input_ids: [B, T], includes the <BOS> or start token at the beginning
    
    Returns:
        new_log_probs: [B, T-1], token-level log-probs
        mean_entropy : scalar, average entropy over [B, T-1]
    """
    batch_size = decoder_input_ids.shape[0]
    # Ensure [2, 0] is at the start of each sequence
    start_tokens = torch.tensor([2, 0], device=decoder_input_ids.device).unsqueeze(0)  # [1, 2]
    start_tokens = start_tokens.expand(batch_size, -1)  # Expand to [B, 2]

    # Concatenate the start tokens with the rest of decoder_input_ids
    decoder_input_ids = torch.cat([start_tokens, decoder_input_ids], dim=1)  # [B, T+2]

    
    dec_in = decoder_input_ids[:, :-1]  # everything except last
    labels = decoder_input_ids[:, 1:]   # everything except first

    labels = labels.contiguous()
    
    #print(labels)
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        decoder_input_ids=dec_in,
        labels=labels,       # triggers internal shift
        return_dict=True
    )
    logits = outputs.logits  # [B, seq_len, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)  # same shape

    # gather log-prob of the 'correct' token at each position
    forced_lps = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)  # [B, seq_len]

    # entropy
    probs = log_probs.exp()  # [B, seq_len, vocab_size]
    token_entropies = -(probs * log_probs).sum(dim=-1)  # [B, seq_len]
    mean_entropy = token_entropies.mean()

    return forced_lps, mean_entropy


def extract_vlm_data(text):
    """
    From a formatted VLM response, pull out:
      - the raw reasoning between <think>…</think>
      - the final answer between <answer>…</answer>
    Returns (reasoning:str, answer:str) or (None,None) if tags missing.
    """
    think_m = re.search(r"<think>(.*?)</think>", text, re.S)
    ans_m   = re.search(r"<answer>(.*?)</answer>", text, re.S)
    reasoning = think_m.group(1).strip() if think_m else None
    answer    = ans_m.group(1).strip()   if ans_m   else None
    return reasoning, answer

def extract_circles_from_reasoning(reasoning):
    """
    Parses lines of the form
      '- Color {color}, size {size}, at ({x},{y})'
    and returns a list of dicts with keys color,size,x,y.
    """
    pattern = re.compile(
        r"-\s*Color\s+(?P<color>\w+),\s*size\s+(?P<size>\d+),\s*at\s*\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)"
    )
    circles = [m.groupdict() for m in pattern.finditer(reasoning or "")]
    return circles




def reward_function_vlm(generated_text: str, gold_text: str) -> float:
    """
    Reward based on:
      1) Correct final answer (+3) or incorrect (–3)
      2) Presence of a well-formed <think> block (+1 if present, –1 if missing)
      3) Matching circle counts between generated reasoning and gold reasoning (+2 if match, –2 if mismatch)
    """
    # extract answer and reasoning
    gen_reason, gen_ans = extract_vlm_data(generated_text)
    gold_reason, gold_ans = extract_vlm_data(gold_text)

    # if we can’t even parse tags, heavy penalty
    if gen_reason is None or gen_ans is None:
        return -5.0

    reward = 0.0

    # 1) final‐answer correctness
    if gen_ans.lower() == gold_ans.lower():
        reward += 3.0
    else:
        reward -= 3.0

    # 2) presence of reasoning block
    if "<think>" in generated_text and "</think>" in generated_text:
        reward += 1.0
    else:
        reward -= 1.0

    # 3) compare how many circles the reasoning actually enumerated
    gen_circles  = extract_circles_from_reasoning(gen_reason)
    gold_circles = extract_circles_from_reasoning(gold_reason)
    if len(gen_circles) == len(gold_circles):
        reward += 2.0
    else:
        reward -= 2.0

    return reward



###################################################
#  E. The GRPO Training Loop
###################################################
def grpo_training_loop(
    model,
    processor,
    train_loader,
    optimizer,
    device,
    val_loader=None,
    epochs=3,
    group_size=3,       # number of outputs per question
    clip_coef=0.01,      # clipping epsilon
    entropy_coef=0.0001,
    update_epochs=1,
    minibatch_size=1,
    max_new_tokens=120,
):
    """
    Key Steps:
      1. For each question in train_loader, sample G outputs from old policy,
         compute group-based relative advantage => store old_log_probs, tokens, advantage, etc.
      2. PPO-like updates, but no critic => advantage from group-based reward normalization.
      3. Teacher-forcing log_probs of old tokens for ratio calculation.
    """
    model.train()

    for epoch in range(epochs):
        model.train()
        total_reward = 0.0
        total_count = 0

        memory = []

        #######################
        # 1) Collect Experiences with Old Policy
        #######################
        old_model = None
        with torch.no_grad():
            #config = deepcopy(model.config)  
            #old_model = type(model)(config).to(device)  
            #old_model.load_state_dict(model.state_dict())
            #old_model.eval()

            old_model = copy.deepcopy(model)
            old_model.eval()

        with torch.no_grad():
            for batch in tqdm(train_loader, desc=f"GRPO Epoch {epoch+1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
    
                B = input_ids.size(0)
                for i in range(B):
                    question_ids = input_ids[i].unsqueeze(0)
                    question_pix = pixel_values[i].unsqueeze(0)
                    ref_answer   = answers[i]
    
                    # 1) Sample G outputs for this question from old_policy
                    group_texts = []
                    group_tokens = []
                    group_old_lp = []
                    group_rewards = []
    
                    for _ in range(group_size):
                        # Sample one output
                        gen_ids, old_log_probs, _, _ = generate_one_pass(
                            old_model, processor,
                            question_ids, question_pix,
                            max_new_tokens=max_new_tokens
                        )
                        pred_text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        r = reward_function_vlm(pred_text, ref_answer)
                        
                        group_texts.append(pred_text)
                        group_tokens.append(gen_ids)
                        group_old_lp.append(old_log_probs)
                        group_rewards.append(r)
    
                    # 2) Compute group-based advantage
                    #    adv_i = (r_i - mean(r)) / std(r)
                    rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=device)
                    r_mean = rewards_t.mean()
                    r_std = rewards_t.std() + 1e-8

                    #print(group_tokens)
    
                    for g in range(group_size):
                        adv_value = (rewards_t[g] - r_mean) / r_std
                        memory.append({
                            "encoder_ids": question_ids.cpu(),
                            "pixel_vals" : question_pix.cpu(),
                            "tokens"     : group_tokens[g].cpu(),     # shape [1, T]
                            "old_logprobs": group_old_lp[g].cpu(),    # shape [1, T]
                            "advantage"  : adv_value.item(),          # scalar 
                            "reward"     : group_rewards[g],          # scalar
                        })
                    total_reward += rewards_t.sum().item()
                    total_count  += group_size

        avg_reward = total_reward / max(total_count, 1e-8)
        print(f"[Epoch {epoch+1}] Collected experiences. AvgReward = {avg_reward:.3f}")

        #######################
        # 2) GRPO Updates
        #######################
        # We'll do multiple epochs over the memory
        for _ in range(update_epochs):
            random.shuffle(memory)
            for start in tqdm(range(0, len(memory), minibatch_size)):
                end = start + minibatch_size
                batch_data = memory[start:end]

                b_enc_ids = []
                b_pix     = []
                b_tokens  = []
                b_old_lp  = []
                b_adv     = []
                for item in batch_data:
                    b_enc_ids.append(item["encoder_ids"])
                    b_pix.append(item["pixel_vals"])
                    b_tokens.append(item["tokens"])
                    b_old_lp.append(item["old_logprobs"])
                    b_adv.append(torch.tensor(item["advantage"]).unsqueeze(0))

                b_enc_ids = torch.cat(b_enc_ids).to(device)
                b_pix     = torch.cat(b_pix).to(device)
                b_tokens  = torch.cat(b_tokens).to(device)       # shape [B, T]
                b_old_lp  = torch.cat(b_old_lp).to(device)       # shape [B, T]
                b_adv     = torch.cat(b_adv).to(device)          # shape [B,]
                #print(b_tokens)
                # Teacher-forcing to get new log_probs
                new_log_probs, mean_entropy = score_sequence_hf(
                    model, b_enc_ids, b_pix, b_tokens
                )

                T_old = b_old_lp.size(1)
                T_new = new_log_probs.size(1)
                if T_old + 1 != T_new: #we add two tokens in score sequence hf, and we output one token less 
                    print(T_old)
                    print(T_new)
                    print(new_log_probs)
                    print(b_tokens)
                    print(b_old_lp)
                #min_len = min(T_old, T_new)
                new_lp_seq = new_log_probs[:, 1:].sum(dim=1) #[:, :min_len].sum(dim=1) 
                old_lp_seq = b_old_lp.sum(dim=1) #[:, :min_len].sum(dim=1)

                ratio = (new_lp_seq - old_lp_seq).exp()    # shape [B]
                pg_loss1 = -ratio * b_adv
                pg_loss2 = -torch.clamp(ratio, 1-clip_coef, 1+clip_coef) * b_adv
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # entropy bonus
                entropy_loss = -entropy_coef * mean_entropy

                # final GRPO loss
                total_loss = policy_loss + entropy_loss

                # optimize
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
                optimizer.step()

                # free memory references
                del b_enc_ids, b_pix, b_tokens, b_old_lp, b_adv, new_log_probs
                torch.cuda.empty_cache()

        # 3) Validation
        if val_loader:
            model.eval()
            val_rewards = 0
            num_samples = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    inputs, answers = batch
                    inp = inputs["input_ids"].to(device)
                    pix = inputs["pixel_values"].to(device)
                    B = inp.size(0)
                    # sample once
                    gen_ids, _, _, _ = generate_one_pass(model, processor, inp, pix, max_new_tokens=2)
                    # decode
                    pred_texts = [processor.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
                    for i in range(B):
                        val_rewards += reward_function_vlm(pred_texts[i], answers[i])
                    num_samples += B
            print(f"[Epoch {epoch+1}] Validation Reward = {val_rewards / num_samples:.3f}")

