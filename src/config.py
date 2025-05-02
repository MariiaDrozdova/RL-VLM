import torch

# Data paths
DATA_PATH = "/home/drozdova/data/data_circles/"
TRAIN_DATASET_PATH = f"{DATA_PATH}/circle_dataset_train"
VAL_DATASET_PATH = f"{DATA_PATH}/circle_dataset_val"
TEST_DATASET_PATH = f"{DATA_PATH}/circle_dataset_test"
TEST_HARD_NUMBER_DATASET_PATH = f"{DATA_PATH}/circle_dataset_test_hard_number"

# JSON data files (if used)
TRAIN_JSON = "train.json"
VAL_JSON = "validation.json"
TEST_JSON = "test.json"

# Model settings
MODEL_NAME = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "refs/pr/6"
MODEL_VISION_TOWER_TRAIN = False
TRUST_REMOTE_CODE = True

# Training parameters
TRAIN_SUBSET_SIZE = 200
TEST_SUBSET_SIZE = 200
TEST_HARD_SUBSET_SIZE = 200
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 1
LEARNING_RATE = 1e-5


# Device settings
USE_ACCELERATOR = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)

# GRPO training parameters
POLICY_UPDATE="PPO"
GRPO_EPOCHS = 1
GROUP_SIZE = 3
CLIP_COEF = 0.1
ENTROPY_COEF = 0.001
UPDATE_EPOCHS = 2
ROLLOUT_BATCH_SIZE = 1
MINIBATCH_SIZE = 4
MAX_NEW_TOKENS = 120
ITERATIONS_PER_EPOCH = 50
GRPO_ITERATIONS = 20
GRPO_LEARNING_RATE = 4e-8

# Logs
TENSORBOARD_LOG = True 