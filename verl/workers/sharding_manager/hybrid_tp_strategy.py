from typing import Dict, Optional

class HybridTPStrategy:
    """Hybrid TP strategy"""
    
    UNIFIED_STRATEGY = {
        "attention_output": {"dim": 1, "gather": False},
        "mlp": {
            "gate_proj": {"dim": 0, "gather": False},
            "up_proj": {"dim": 0, "gather": False},
            "down_proj": {"dim": 1, "gather": False},
            # generic mlp strategy
            "c_fc": {"dim": 0, "gather": False},      # GPT-2
            "c_proj": {"dim": 1, "gather": False},    # GPT-2
            "intermediate.dense": {"dim": 0, "gather": False},  # BERT
            "output.dense": {"dim": 1, "gather": False},        # BERT
        },
        "lm_head": {"dim": 0, "gather": True}
    }
    
    @classmethod
    def get_unified_strategy(cls) -> Dict:
        """Get unified strategy"""
        return cls.UNIFIED_STRATEGY.copy()
    
    @classmethod
    def get_layer_strategy(cls, layer_type: str) -> Dict:
        """Get specific layer strategy"""
        return cls.UNIFIED_STRATEGY.get(layer_type, {})
    
    @classmethod
    def get_mlp_layer_strategy(cls, weight_name: str) -> Optional[Dict]:
        """Get mlp layer strategy"""
        mlp_strategy = cls.UNIFIED_STRATEGY.get("mlp", {})
        for layer_name, strategy in mlp_strategy.items():
            if layer_name in weight_name:
                return strategy
        return None