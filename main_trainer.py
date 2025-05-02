# main.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from src.utils import setup_logging, collate_fn, evaluate_samples
from src.config import (
    BATCH_SIZE, NUM_WORKERS, EPOCHS, LEARNING_RATE, DEVICE,
    GROUP_SIZE, CLIP_COEF, ENTROPY_COEF, UPDATE_EPOCHS,
    ROLLOUT_BATCH_SIZE, MINIBATCH_SIZE, MAX_NEW_TOKENS, ITERATIONS_PER_EPOCH,
    GRPO_ITERATIONS, MODEL_NAME, MODEL_REVISION, TRUST_REMOTE_CODE, 
    TRAIN_SUBSET_SIZE, GRPO_EPOCHS, USE_ACCELERATOR, GRPO_LEARNING_RATE,
    TENSORBOARD_LOG, POLICY_UPDATE
)
from src.data_loader import load_and_merge_datasets, CirclesQADataset
from src.model_manager import initialize_model_and_processor
from src.trainer import *
from src.GRPOtrainer import GRPOTrainer
from src.PPOtrainer import PPOTrainer
from src.ACtrainer import ACTrainer
from src.reinforce_trainer import REINFORCETrainer
from datetime import datetime
if TENSORBOARD_LOG:
    from torch.utils.tensorboard import SummaryWriter

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting the project.")
    if TENSORBOARD_LOG:
        
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tb_writer = SummaryWriter(log_dir=f"runs/circles_experiment_{current_time}")
    else:
        tb_writer = None

    # Load and merge datasets from disk
    dataset = load_and_merge_datasets()
    logger.info("Dataset loaded successfully.")

    # Create custom dataset objects for training and validation
    answers_options="full"
    if POLICY_UPDATE=="REINFORCE":
        answers_options="shortened"
    train_data = CirclesQADataset(dataset["train"], prefix="<CirclesQA>", answers_options=answers_options)
    val_data = CirclesQADataset(dataset["validation"], prefix="<CirclesQA>", answers_options=answers_options)

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

    config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,

        "train_subset_size": TRAIN_SUBSET_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,

        "device": DEVICE,

        "grpo_epochs": GRPO_EPOCHS,
        "group_size": GROUP_SIZE,
        "clip_coef": CLIP_COEF,
        "entropy_coef": ENTROPY_COEF,
        "update_epochs": UPDATE_EPOCHS,
        "rollout_batch_size": ROLLOUT_BATCH_SIZE,
        "minibatch_size": MINIBATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "iterations_per_epoch": ITERATIONS_PER_EPOCH,
        "grpo_iterations": GRPO_ITERATIONS,
        "grpo_learning_rate": GRPO_LEARNING_RATE,

        "tensorboard_log": TENSORBOARD_LOG,
    }
    
    device = DEVICE

    # ------------------
    # SFT Training Loop
    # ------------------
    checkpoint_dir = "checkpoint/"
    if POLICY_UPDATE=="REINFORCE":
        checkpoint_dir = "checkpoint_sft_for_reinforce/"
    standard_trainer = BaseTrainer(model, processor, train_loader, val_loader, device, config, use_accelerator=USE_ACCELERATOR, tb_writer=tb_writer, checkpoint_dir=checkpoint_dir)
    if standard_trainer.accelerator is None or standard_trainer.accelerator.is_main_process:
        logger.info("Starting SFT training loop.")
    #standard_trainer.train(epochs=config["epochs"])
    # Evaluate on test split after SFT training

    for j in range(config["epochs"]):
        standard_trainer.train(epochs=1)
        if standard_trainer.accelerator is None or standard_trainer.accelerator.is_main_process:
            logger.info("Evaluating on test split after SFT training.")
            model_to_test = standard_trainer.model.module if hasattr(standard_trainer.model, "module") else standard_trainer.model
            if j % 20 == 19:
                evaluate_samples(
                    model_to_test, standard_trainer.processor, DEVICE, dataset, "test",
                    prefix="<CirclesQA>",
                    N=20, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=True,
                    logger=logger,
                )
                evaluate_samples(
                    model_to_test, standard_trainer.processor, DEVICE, dataset, "test_hard_number",
                    prefix="<CirclesQA>",
                    N=20, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=True,
                    logger=logger,
                )
                test_acc = evaluate_samples(
                    model_to_test, standard_trainer.processor, DEVICE, dataset, "test",
                    prefix="<CirclesQA>",
                    N=1000, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
                logger.info(f"SFT Evaluation Accuracy on test: {test_acc}")
                if tb_writer:
                    tb_writer.add_scalar("SIFT/test_acc", test_acc, j)
                test_acc_hard = evaluate_samples(
                    model_to_test, standard_trainer.processor, DEVICE, dataset, "test_hard_number",
                    prefix="<CirclesQA>",
                    N=1000, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
                if tb_writer:
                    tb_writer.add_scalar("SIFT/test_acc_hard", test_acc_hard, j)
                logger.info(f"SFT Evaluation Accuracy on test (hard): {test_acc_hard}")


    # ------------------
    # GRPO Training Loop
    # ------------------ 
    if standard_trainer.accelerator is not None:
        # get back the underlying HF model
        model_to_test = standard_trainer.accelerator.unwrap_model(standard_trainer.model)
    else:
        model_to_test = standard_trainer.model
    evaluate_samples(
        model_to_test, standard_trainer.processor, DEVICE, dataset, "test",
        prefix="<CirclesQA>",
        N=20, 
        Q="Is the number of circles odd or even?",
        print_outputs=True,
        logger=logger,
    )
    if POLICY_UPDATE=="GRPO":
        rl_trainer = GRPOTrainer(model_to_test, standard_trainer.processor, train_loader, val_loader, device, config, use_accelerator=USE_ACCELERATOR, tb_writer=tb_writer)
    elif POLICY_UPDATE=="PPO":
        rl_trainer = PPOTrainer(model_to_test, standard_trainer.processor, train_loader, val_loader, device, config, use_accelerator=USE_ACCELERATOR, tb_writer=tb_writer)
    elif POLICY_UPDATE=="AC":
        rl_trainer = ACTrainer(model_to_test, standard_trainer.processor, train_loader, val_loader, device, config, use_accelerator=USE_ACCELERATOR, tb_writer=tb_writer)
    elif POLICY_UPDATE=="REINFORCE":
        rl_trainer = REINFORCETrainer(model_to_test, standard_trainer.processor, train_loader, val_loader, device, config, use_accelerator=USE_ACCELERATOR, tb_writer=tb_writer)
    else:
        print(f"Unknown policy passed {POLICY_UPDATE}. Terminating.")
        print(f"Available policies GRPO, PPO, AC and REINFORCE.")
        return
    evaluate_samples(
        model_to_test, rl_trainer.processor, DEVICE, dataset, "test",
        prefix="<CirclesQA>",
        N=20, 
        Q="Is the number of circles odd or even?",
        print_outputs=True,
        logger=logger,
    )

    for i in range(1000):
        rl_trainer.train_rl()
        if rl_trainer.accelerator is None or rl_trainer.accelerator.is_main_process:
            # Final evaluation after GRPO training
            #model_to_test = rl_trainer.model.module if hasattr(standard_trainer.model, "module") else standard_trainer.model
            if rl_trainer.accelerator is not None:
                # get back the underlying HF model
                model_to_test = rl_trainer.accelerator.unwrap_model(rl_trainer.model)
            else:
                model_to_test = rl_trainer.model
            if i % 5 == 4:
                logger.info(f"Final evaluation after {POLICY_UPDATE} training:")
                logger.info("Test:")
                evaluate_samples(
                    model_to_test, rl_trainer.processor, DEVICE, dataset, "test",
                    prefix="<CirclesQA>",
                    N=20, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=True,
                    logger=logger,
                )
                logger.info("Test (hard):")
                evaluate_samples(
                    model_to_test, rl_trainer.processor, DEVICE, dataset, "test_hard_number",
                    prefix="<CirclesQA>",
                    N=20, 
                    Q="Is the number of circles odd or even?",
                    print_outputs=True,
                    logger=logger,
                )
            
                final_acc = evaluate_samples(
                    model_to_test, rl_trainer.processor, DEVICE, dataset, "test",
                    prefix="<CirclesQA>",
                    N=100,
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
                logger.info(f"Final Evaluation Accuracy on test: {final_acc}")
                if tb_writer:
                    tb_writer.add_scalar(f"{POLICY_UPDATE}/test_acc", final_acc, i+j)
                final_acc = evaluate_samples(
                    model_to_test, rl_trainer.processor, DEVICE, dataset, "test_hard_number",
                    prefix="<CirclesQA>",
                    N=100,
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
                logger.info(f"Final Evaluation Accuracy on test (hard): {final_acc}")
                if tb_writer:
                    tb_writer.add_scalar(f"{POLICY_UPDATE}/test_acc_hard", final_acc, i+j)

if __name__ == "__main__":
    main()
