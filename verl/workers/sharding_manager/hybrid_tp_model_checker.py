from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class HybridTPModelTypeDetector:
    """Hybrid tp model supports detector"""
    
    SUPPORTED_MODELS = {
        "llama", "llama2", "llama3", "qwen", "qwen2", "qwen2.5", "qwen3",
    }
    
    DEFAULT_LAYER_MAPPINGS = {
        "attention_output": ["self_attn.o_proj"],
        "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "lm_head": ["lm_head"]
    }
    
    @classmethod
    def detect_model_type(cls, model_config) -> str:
        """Detect model type"""
        model_type = getattr(model_config, "model_type", "").lower()
        return model_type
    
    @classmethod
    def is_supported_model(cls, model_type: str) -> bool:
        """Check if is supported model"""
        return model_type in cls.SUPPORTED_MODELS
    
    @classmethod
    def get_default_layer_mappings(cls, model_type: str) -> Dict[str, List[str]]:
        """Get default layer mappings"""
        if cls.is_supported_model(model_type):
            return cls.DEFAULT_LAYER_MAPPINGS.copy()
        else:
            logger.warning(f"Model type {model_type} not in supported list, using generic mappings")
            return cls.DEFAULT_LAYER_MAPPINGS.copy()