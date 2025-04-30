import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import logging
from .config import MODEL_NAME, MODEL_REVISION, MODEL_VISION_TOWER_TRAIN

logger = logging.getLogger(__name__)

def initialize_model_and_processor(device):
    """
    Initializes and returns the model and processor.
    """
    logger.info("Initializing model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        revision=MODEL_REVISION
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        revision=MODEL_REVISION
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if not MODEL_VISION_TOWER_TRAIN:
        logger.info("Vision tower is frozen.")
        for param in model.vision_tower.parameters():
            #param.is_trainable = False
            param.requires_grad = False
        
    logger.info("Model and processor initialized.")
    return model, processor
