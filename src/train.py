import torch
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_example(model, processor, device, task_prompt, text_input, image):
    """
    Runs the model on a single example and returns the processed answer.
    """
    if task_prompt == "CAPTION":
        prompt = task_prompt
    else:
        prompt = task_prompt +  text_input

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #print(task_prompt)
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer

def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

def train_model(train_loader, val_loader, model, processor, device, epochs=10, lr=1e-6):
    """
    Standard training loop.
    """
    logger.info("Starting training loop.")
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    best_val_loss = np.inf

    for epoch in range(epochs):
        model.train()
        # Freeze vision tower parameters if desired
        for param in model.vision_tower.parameters():
            param.is_trainable = False

        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Avg Training Loss: {avg_train_loss}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1} - Avg Validation Loss: {avg_val_loss}")

        # Save checkpoint if validation improves (checkpoint saving code commented)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("Validation improved. Saving checkpoint...")
            # Uncomment to save:
            # output_dir = "./model_checkpoints/epoch_best"
            # os.makedirs(output_dir, exist_ok=True)
            # model.save_pretrained(output_dir)
            # processor.save_pretrained(output_dir)
    torch.cuda.empty_cache()
    logger.info("Training complete.")

def evaluate_samples(model, processor, device, data, dataset_partition, prefix, N=20, Q="Is the number of circles odd or even?", show_images=False, print_outputs=False):
    """
    Evaluates the model on a subset of samples.
    """
    #from IPython.display import display  # for notebooks
    correct_answers = 0
    for idx in range(N):
        result = run_example(model, processor, device, prefix, Q, data[dataset_partition][idx]['image'])
        #print(result)
        words = result[prefix].split()
        last_word = words[-1].strip(".'\"").lower()
    
        if data[dataset_partition][idx]['answers'][8] == result[prefix]:
            correct_answers += 1

        if print_outputs:
            print(result[prefix], " ||| ", last_word, " ||| ", data[dataset_partition][idx]['answers'][8])
        
        #if show_images:
        #    display(data[dataset_partition][idx]['image'].resize([350, 350]))
    accuracy = correct_answers / N
    logger.info(f"Accuracy: {accuracy}")
    return accuracy
