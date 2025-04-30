# main.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from src.utils import setup_logging
from src.config import (
    BATCH_SIZE, NUM_WORKERS, EPOCHS, LEARNING_RATE, DEVICE,
    GROUP_SIZE, CLIP_COEF, ENTROPY_COEF, UPDATE_EPOCHS,
    ROLLOUT_BATCH_SIZE, MINIBATCH_SIZE, MAX_NEW_TOKENS, ITERATIONS_PER_EPOCH,
    GRPO_ITERATIONS,
)
from src.data_loader import load_and_merge_datasets, CirclesQADataset
from src.model_manager import initialize_model_and_processor
from src.train import train_model, collate_fn, evaluate_samples
from src.grpo import grpo_training_loop_subset

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting the project.")

    # Load and merge datasets from disk
    dataset = load_and_merge_datasets()
    logger.info("Dataset loaded successfully.")

    # Create custom dataset objects for training and validation
    train_data = CirclesQADataset(dataset["train"], prefix="<CirclesQA>")
    val_data = CirclesQADataset(dataset["validation"], prefix="<CirclesQA>")

    # Initialize model and processor
    model, processor = initialize_model_and_processor(DEVICE)
    
    # Create DataLoader objects.
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
        num_workers=NUM_WORKERS
    )

    # ------------------
    # SFT Training Loop
    # ------------------
    logger.info("Starting SFT training loop.")
    train_model(train_loader, val_loader, model, processor, DEVICE, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Evaluate on test split after SFT training
    logger.info("Evaluating on test split after SFT training.")
    eval_acc = evaluate_samples(
        model, processor, DEVICE, dataset, "test",
        prefix="<CirclesQA>",
        N=20, 
        Q="Is the number of circles odd or even?",
        print_outputs=True
    )
    logger.info(f"SFT Evaluation Accuracy: {eval_acc}")

    # ------------------
    # GRPO Training Loop
    # ------------------
    # Create a new optimizer for GRPO (or reuse the previous one if desired)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    logger.info("Starting GRPO training loop.")
    
    # For example, we run 30 GRPO iterations. Adjust as needed.
    for i in range(GRPO_ITERATIONS):
        logger.info(f"GRPO iteration {i+1}/30")
        grpo_training_loop_subset(
            model, processor, train_loader, optimizer, DEVICE,
            epochs=1,
            group_size=GROUP_SIZE,
            clip_coef=CLIP_COEF,
            entropy_coef=ENTROPY_COEF,
            update_epochs=UPDATE_EPOCHS,
            rollout_batch_size=ROLLOUT_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            val_loader=val_loader,
            iterations_per_epoch=ITERATIONS_PER_EPOCH
        )
        # Optionally evaluate periodically
        logger.info("Evaluating after current GRPO iteration...")
        evaluate_samples(
            model, processor, DEVICE, dataset, "test",
            prefix="<CirclesQA>",
            N=20,
            Q="Is the number of circles odd or even?",
            print_outputs=True
        )
    
    # Final evaluation after GRPO training
    logger.info("Final evaluation after GRPO training:")
    final_acc = evaluate_samples(
        model, processor, DEVICE, dataset, "test",
        prefix="<CirclesQA>",
        N=20,
        Q="Is the number of circles odd or even?",
        print_outputs=True
    )
    logger.info(f"Final Evaluation Accuracy: {final_acc}")

if __name__ == "__main__":
    main()
