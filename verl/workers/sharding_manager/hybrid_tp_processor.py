import torch
import torch.distributed
from typing import Dict, List, Optional
from verl.third_party.vllm import parallel_state as vllm_ps
from .hybrid_tp_config import HybridTPConfig
from .hybrid_tp_strategy import HybridTPStrategy
from .hybrid_tp_model_checker import HybridTPModelTypeDetector
import logging

logger = logging.getLogger(__name__)

class HybridTPProcessor:
    """Hybrid TP Processor"""
    
    def __init__(self, tp_config: HybridTPConfig, model_config):
        self.tp_config = tp_config
        self.model_config = model_config
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        
        self.model_type = HybridTPModelTypeDetector.detect_model_type(model_config)
        self.is_supported = HybridTPModelTypeDetector.is_supported_model(self.model_type)
        
        self.layer_mappings = tp_config.get_layer_mappings(model_config)
        self.tp_strategy = HybridTPStrategy.get_unified_strategy()
        
        logger.info(f"Initialized HybridTPProcessor for {self.model_type} "
                   f"(supported: {self.is_supported})")
    
    def apply_hybrid_tp(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply hybrid tp"""
        if not self.tp_config.is_hybrid_enabled():
            return weights
        
        logger.info(f"Applying hybrid TP for {self.model_type}")
        
        result = {}
        
        # process each layer with type
        for layer_type, patterns in self.layer_mappings.items():
            layer_weights = self._extract_weights_by_patterns(weights, patterns)
            if layer_weights:
                processed_weights = self._process_layer_weights(layer_weights, layer_type)
                result.update(processed_weights)
        
        # process other weights
        other_weights = self._extract_other_weights(weights)
        result.update(other_weights)
        
        logger.info(f"Hybrid TP processing completed for {self.model_type}")
        return result
    
    def _extract_weights_by_patterns(self, weights: Dict[str, torch.Tensor], 
                                   patterns: List[str]) -> Dict[str, torch.Tensor]:
        """Extract weights with patterns"""
        extracted = {}
        for name, weight in weights.items():
            for pattern in patterns:
                if pattern in name:
                    extracted[name] = weight
                    break
        return extracted
    
    def _extract_other_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract other weights"""
        all_patterns = []
        for patterns in self.layer_mappings.values():
            all_patterns.extend(patterns)
        
        other_weights = {}
        for name, weight in weights.items():
            if not any(pattern in name for pattern in all_patterns):
                other_weights[name] = weight
        
        return other_weights
    
    def _process_layer_weights(self, weights: Dict[str, torch.Tensor], 
                             layer_type: str) -> Dict[str, torch.Tensor]:
        """Process layer weights by layer type"""
        tp_size = self.tp_config.get_tp_size_for_layer_type(layer_type)
        if tp_size == 1:
            return weights
        
        strategy = self.tp_strategy.get(layer_type, {})
        
        if layer_type == "attention_output":
            return self._process_attention_output(weights, tp_size, strategy)
        elif layer_type == "mlp":
            return self._process_mlp(weights, tp_size)
        elif layer_type == "lm_head":
            return self._process_lm_head(weights, tp_size, strategy)
        else:
            return weights
    
    def _process_attention_output(self, weights: Dict[str, torch.Tensor], 
                                tp_size: int, strategy: Dict) -> Dict[str, torch.Tensor]:
        """Process attention output layer"""
        dim = strategy.get("dim", 1)
        gather = strategy.get("gather", False)
        
        result = {}
        for name, weight in weights.items():
            processed_weight = self._apply_tp_sharding(weight, tp_size, dim, gather)
            result[name] = processed_weight
        
        return result
    
    def _process_mlp(self, weights: Dict[str, torch.Tensor], 
                    tp_size: int) -> Dict[str, torch.Tensor]:
        """Process mlp layers"""
        result = {}
        for name, weight in weights.items():
            layer_strategy = HybridTPStrategy.get_mlp_layer_strategy(name)
            if layer_strategy:
                dim = layer_strategy.get("dim", 0)
                gather = layer_strategy.get("gather", False)
                processed_weight = self._apply_tp_sharding(weight, tp_size, dim, gather)
                result[name] = processed_weight
            else:
                result[name] = weight
        
        return result
    
    def _process_lm_head(self, weights: Dict[str, torch.Tensor], 
                        tp_size: int, strategy: Dict) -> Dict[str, torch.Tensor]:
        """Process lm_head layer"""
        dim = strategy.get("dim", 0)
        gather = strategy.get("gather", True)
        
        result = {}
        for name, weight in weights.items():
            processed_weight = self._apply_tp_sharding(weight, tp_size, dim, gather)
            result[name] = processed_weight
        
        return result
    
    def _apply_tp_sharding(self, weight: torch.Tensor, tp_size: int, 
                          dim: int, gather: bool) -> torch.Tensor:
        """Apply tp sharding"""
        if tp_size == 1:
            return weight
        
        # gather weights from all tp ranks
        gathered_weights = [torch.empty_like(weight) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered_weights, weight, group=self.tp_group)
        
        # merge weights
        merged_weight = torch.cat(gathered_weights, dim=dim)
        
        # resharding weights
        chunk_size = merged_weight.size(dim) // tp_size
        start_idx = self.tp_rank * chunk_size
        end_idx = start_idx + chunk_size
        
        if dim == 0:
            result = merged_weight[start_idx:end_idx]
        else:
            slices = [slice(None)] * merged_weight.dim()
            slices[dim] = slice(start_idx, end_idx)
            result = merged_weight[slices]
        
        return result