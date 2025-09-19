# VERL FSDP D2D Implementation Plan

## Current Status Analysis

### Megatron vs FSDP Key Differences

| Aspect | Megatron | FSDP |
|--------|----------|------|
| **Parallel State** | `megatron.core.parallel_state` | `torch.distributed` + `device_mesh` |
| **Rank Management** | `mpu.get_data_parallel_rank()` | `torch.distributed.get_rank()` |
| **World Size** | `mpu.get_data_parallel_world_size()` | `device_mesh.size()` |
| **Dispatch Mode** | `make_nd_compute_dataproto_dispatch_fn` | `Dispatch.DP_COMPUTE_PROTO` |
| **Sharding Strategy** | TP/PP/DP explicit control | FSDP automatic sharding |

## Required Modifications for FSDP D2D Support

### 1. TensorCache Adaptation

#### Current Limitation
```python
# tensor_cache.py - Megatron specific
def get_cached_tensors(self, input_data: DataProto, keys_to_get: List[str]):
    src_dp_size = self.mini_bs // src_shape[0]  # Assumes Megatron DP
    dst_dp_size = mpu.get_data_parallel_world_size()  # Megatron API
```

#### Required Changes
```python
# New: FSDP-compatible version
def get_cached_tensors(self, input_data: DataProto, keys_to_get: List[str]):
    if self.parallel_mode == "megatron":
        dst_dp_size = mpu.get_data_parallel_world_size()
    elif self.parallel_mode == "fsdp":
        dst_dp_size = self.device_mesh.size() // self.ulysses_sequence_parallel_size
    else:
        raise ValueError(f"Unsupported parallel mode: {self.parallel_mode}")
```

### 2. Reshard Function Enhancement

#### Current Implementation
```python
# reshard.py - Only supports Megatron, hardcoded parameter name
def get_dp_reshard_tensor_via_alltoall(
    src_tensor: torch.Tensor,
    src_dp_size: int,
    dst_dp_size: int,
    dst_shape: List[int],
    global_megatron_dp_ranks: List[int]  # 硬编码参数名
):
    dst_dp_rank = mpu.get_data_parallel_rank()  # Megatron specific
```

#### Required Enhancement (最佳方案)
```python
def get_dp_reshard_tensor_via_alltoall(
    src_tensor: torch.Tensor,
    src_dp_size: int,
    dst_dp_size: int,
    dst_shape: List[int],
    global_dp_ranks: List[int],
    dst_dp_rank: int  # 直接传入，避免函数内部获取
):
    """Enhanced version with dst_dp_rank parameter"""
    assert src_dp_size == torch.distributed.get_world_size(), (
        "We only support src_dp_size equals world_size for now in the cached tensor resharding."
    )
    
    micro_dp_size = src_dp_size // dst_dp_size
    
    # 直接使用传入的dst_dp_rank，无需内部获取
    src_ranks = set(range(dst_dp_rank * micro_dp_size, (dst_dp_rank + 1) * micro_dp_size, 1))
    # ... existing resharding logic using global_dp_ranks instead of global_megatron_dp_ranks ...
```

### 3. FSDP Workers Integration

#### A. Initialize TensorCache in FSDP Workers
```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        # ... existing initialization ...
        
        # Initialize tensor cache for D2D optimization (FSDP version)
        self.d2d_enabled = os.getenv("D2D_DATA_TRANSFER", "false").lower() == "true"
        if self.d2d_enabled:
            from verl.tensor_cache import TensorCache
            # Pass FSDP-specific configuration
            cache_config = {
                "parallel_mode": "fsdp",
                "device_mesh": self.device_mesh,
                "ulysses_sequence_parallel_size": self.ulysses_sequence_parallel_size
            }
            self.tensor_cache = TensorCache(config=self.config, **cache_config)
            logger.info("D2D data transfer enabled via tensor cache (FSDP mode)")
```

#### B. Modify generate_sequences
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto):
    # ... existing generation logic ...
    
    # D2D optimization: cache core tensors for subsequent D2D transfer
    if self.d2d_enabled:
        # Same logic as Megatron, but with FSDP-aware caching
        keys_to_reserve = ["responses", "attention_mask"]
        keys_no_cache = ["prompts"]
        
        all_keys_before_cache = list(output.batch.keys())
        
        self.tensor_cache.cache_tensors(
            data=output,
            keys_to_reserve=keys_to_reserve,
            keys_no_cache=keys_no_cache
        )
        
        # Statistics logging
        reserved_keys = [k for k in keys_to_reserve if k in all_keys_before_cache]
        no_cache_keys = [k for k in keys_no_cache if k in all_keys_before_cache]
        cached_and_popped_keys = [k for k in all_keys_before_cache if k not in keys_to_reserve and k not in keys_no_cache]
        logger.info(f"D2D (FSDP): Reserved for CPU: {reserved_keys}, No cache: {no_cache_keys}, Cached&Popped: {cached_and_popped_keys}")
    
    return output
```

#### C. Modify compute_ref_log_prob
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_ref_log_prob(self, data: DataProto):
    # D2D optimization: retrieve cached tensors from tensor cache
    if self.d2d_enabled:
        try:
            keys_to_get = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
            cached_tensors = self.tensor_cache.get_cached_tensors(data, keys_to_get)
            
            if cached_tensors:
                data = data.union(cached_tensors)
                logger.info(f"D2D (FSDP): Retrieved {len(cached_tensors.batch)} cached tensors for ref computation")
        except Exception as e:
            logger.warning(f"D2D (FSDP): Failed to retrieve cached tensors for ref computation: {e}")
    
    # ... existing ref computation logic ...
    
    # Cache ref_log_prob for update stage
    if self.d2d_enabled:
        ref_data = DataProto.from_dict(tensors={"ref_log_prob": output.clone()})
        self.tensor_cache.cache_tensors(
            data=ref_data,
            keys_to_reserve=["ref_log_prob"],
            keys_no_cache=[]
        )
        logger.info("D2D (FSDP): Cached ref_log_prob tensor (reserved for CPU)")
    
    return DataProto.from_dict(tensors={"ref_log_prob": output})
```

#### D. Modify update_actor
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    # D2D optimization: retrieve cached tensors from tensor cache
    if self.d2d_enabled:
        try:
            keys_to_get = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask", 
                          "rollout_log_probs", "ref_log_prob"]
            cached_tensors = self.tensor_cache.get_cached_tensors(data, keys_to_get)
            
            if cached_tensors:
                data = data.union(cached_tensors)
                logger.info(f"D2D (FSDP): Retrieved {len(cached_tensors.batch)} cached tensors for actor update")
        except Exception as e:
            logger.warning(f"D2D (FSDP): Failed to retrieve cached tensors for actor update: {e}")
    
    # ... existing training logic ...
    
    output = DataProto(meta_info={"metrics": metrics})
    output = output.to("cpu")
    
    # D2D optimization: clear cache after training completion
    if self.d2d_enabled:
        self.tensor_cache.clear()
        logger.info("D2D (FSDP): Cleared tensor cache after actor training completed")
    
    return output
```

### 4. Dispatcher Enhancement

#### Current Limitation
FSDP workers使用 `Dispatch.DP_COMPUTE_PROTO`，对应的dispatch函数是 `dispatch_dp_compute_data_proto`，但当前的D2D实现只在 `dispatch_lazy_compute_data_proto` 中设置rank mapping。

#### Required Enhancement
需要在 `dispatch_dp_compute_data_proto` 中也添加D2D支持：

```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup
    import os

    assert isinstance(worker_group, WorkerGroup)
    
    # D2D optimization: automatically set global_dp_ranks for FSDP workers
    if os.getenv("D2D_DATA_TRANSFER", "false").lower() == "true":
        # For DP_COMPUTE_PROTO, we use simple rank mapping [0, 1, 2, ...]
        dp_rank_mapping = list(range(worker_group.world_size))
        
        for arg in args:
            if hasattr(arg, 'meta_info') and arg.meta_info is not None:
                arg.meta_info["global_dp_ranks"] = dp_rank_mapping
                
        for key, val in kwargs.items():
            if hasattr(val, 'meta_info') and val.meta_info is not None:
                val.meta_info["global_dp_ranks"] = dp_rank_mapping
    
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs
```

### 5. TensorCache Enhancement for Multiple Rank Key Support

#### Current Limitation
TensorCache硬编码使用 `"global_megatron_dp_ranks"` key：

```python
# tensor_cache.py line 87
tensor_dict[key] = get_dp_reshard_tensor_via_alltoall(
    cached_tensor, 
    src_dp_size, 
    dst_dp_size, 
    dst_shape, 
    input_data.meta_info["global_megatron_dp_ranks"]  # 硬编码
)
```

#### Required Enhancement
支持不同的rank key，简化parallel mode处理：

```python
class TensorCache:
    def __init__(self, config, parallel_mode="megatron", **kwargs):
        self.parallel_mode = parallel_mode
        
        if parallel_mode == "megatron":
            self.mini_bs = config.actor.ppo_mini_batch_size * config.rollout.n
            self.rank_key = "global_megatron_dp_ranks"
        elif parallel_mode == "fsdp":
            self.mini_bs = config.actor.ppo_mini_batch_size * config.rollout.n
            self.rank_key = "global_dp_ranks"
        
        self.tensor_cached = {}

    def get_cached_tensors(self, input_data: DataProto, keys_to_get: Optional[List[str]] = None):
        """Enhanced version with parallel_mode support"""
        if keys_to_get is None:
            keys_to_get = self.tensor_cached.keys()

        tensor_dict = {}
        for key in keys_to_get:
            if key not in self.tensor_cached:
                raise KeyError(f"{key} is not in cached keys {self.tensor_cached.keys()}")
            cached_tensor = self.tensor_cached[key]
            src_shape = list(cached_tensor.shape)
            src_dp_size = self.mini_bs // src_shape[0]
            
            # Get dst_dp_size based on parallel_mode
            if self.parallel_mode == "megatron":
                dst_dp_size = mpu.get_data_parallel_world_size()
            elif self.parallel_mode == "fsdp":
                # For FSDP with DP_COMPUTE_PROTO, dst_dp_size equals world_size
                # The ulysses SP is handled by sharding manager, not here
                dst_dp_size = torch.distributed.get_world_size()
            else:
                raise ValueError(f"Unsupported parallel mode: {self.parallel_mode}")
            
            dst_shape = src_shape.copy()
            dst_shape[0] = self.mini_bs // dst_dp_size
            
            # tensor transfer
            if dst_dp_size == src_dp_size:
                tensor_dict[key] = cached_tensor
            else:
                # Get dst_dp_rank based on parallel_mode
                if self.parallel_mode == "megatron":
                    dst_dp_rank = mpu.get_data_parallel_rank()
                elif self.parallel_mode == "fsdp":
                    dst_dp_rank = torch.distributed.get_rank()
                
                # Use appropriate rank key based on parallel_mode
                global_dp_ranks = input_data.meta_info[self.rank_key]
                tensor_dict[key] = get_dp_reshard_tensor_via_alltoall(
                    cached_tensor, 
                    src_dp_size, 
                    dst_dp_size, 
                    dst_shape, 
                    global_dp_ranks,
                    dst_dp_rank  # 传入dst_dp_rank参数
                )
                self.tensor_cached[key] = tensor_dict[key]
        
        output = DataProto.from_dict(tensors=tensor_dict)
        return output
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. ✅ Enhance TensorCache with parallel_mode support
2. ✅ Update reshard function for FSDP compatibility
3. ✅ Create FSDP-specific rank mapping utilities

### Phase 2: FSDP Workers Integration
1. ✅ Add D2D initialization to FSDP workers
2. ✅ Implement caching in generate_sequences
3. ✅ Implement retrieval in compute_ref_log_prob and update_actor

### Phase 3: Testing and Optimization
1. ✅ Unit tests for FSDP D2D functionality
2. ✅ Performance benchmarking vs traditional approach
3. ✅ Integration tests with different FSDP configurations

## Key Challenges and Solutions

### Challenge 1: Rank Mapping Differences
- **Problem**: FSDP uses different rank mapping than Megatron
- **Solution**: Abstract rank management through parallel_mode parameter

### Challenge 2: Device Mesh Integration
- **Problem**: FSDP uses device_mesh for parallel coordination
- **Solution**: Pass device_mesh to TensorCache for FSDP-aware operations

### Challenge 3: Ulysses Sequence Parallelism
- **Problem**: FSDP workers may use Ulysses sequence parallelism
- **Solution**: Account for sequence parallel size in DP calculations

### Challenge 4: Different Dispatch Patterns
- **Problem**: FSDP uses simpler Dispatch.DP_COMPUTE_PROTO
- **Solution**: Maintain compatibility with both dispatch modes

## Expected Benefits

### Performance Gains
- **Data Transfer Reduction**: Same 93%+ reduction as Megatron
- **Memory Efficiency**: ~20-30MB additional GPU memory usage
- **Latency Improvement**: 30-60% training step speedup for long sequences

### Compatibility
- **Seamless Integration**: Works with existing FSDP configurations
- **Backward Compatibility**: No impact when D2D is disabled
- **Multi-Modal Support**: Compatible with vision-language models

## Usage Example

```bash
# Enable D2D for FSDP training
export D2D_DATA_TRANSFER=true

# Run FSDP GRPO training with D2D optimization
python -m verl.trainer.main_ppo \
    --config-path examples/grpo_trainer \
    --config-name fsdp_config \
    trainer.use_fsdp=true
```

## Conclusion

The FSDP D2D implementation requires strategic adaptations to handle different parallel paradigms while maintaining the core optimization benefits. The modular design ensures compatibility across both Megatron and FSDP frameworks, providing significant performance improvements for long-sequence GRPO training scenarios.
