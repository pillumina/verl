# VERL离线转换脚本内存优化方案

## 问题分析

### 当前内存占用分析（修正后）
对于1T参数规模的BailingMoeV2模型，考虑bfloat16数据类型：

```bash
# 实际内存占用计算：
1T参数 × 2字节/参数(bfloat16) = 2TB基础内存

# BailingMoeV2的MoE结构可能放大参数量：
- 活跃参数: 1T
- 专家参数: 可能是活跃参数的2-4倍
- 实际参数总量: 3-5T
- 实际内存需求: 6-10TB (bfloat16)

# 加上模型结构开销和转换缓存：
单进程实际需求: 7-12TB
双进程总需求: 14-24TB  # 远超2.9TB限制！
```

### 核心问题
1. **参数量估计错误**：1T只是活跃参数，MoE实际更多
2. **重复加载问题**：每个PP rank都加载完整HF模型
3. **内存碎片化**：大tensor分配导致内存利用率低

## 简化优化方案

### 方案1：立即可用的单进程优化（推荐）

#### 1.1 使用现有脚本的内存监控
```bash
# 单进程运行并监控内存
python scripts/converter_hf_to_mcore.py \
    --hf_model_path /path/to/bailing-moe-v2 \
    --output_path /path/to/output \
    --use_cpu_initialization \
    --trust_remote_code 2>&1 | tee conversion.log
```

#### 1.2 观察内存使用阶段
观察输出中的内存标记：
- `[MEMORY] Before HF model loading` - 起始内存
- `[MEMORY] After HF model loaded` - HF模型加载后
- `[MEMORY] After HF state_dict created` - state_dict创建后
- `[MEMORY] After HF models deleted` - 清理后
- `[MEMORY] Before dist_checkpointing.save` - 保存前

#### 1.3 如果单进程失败的原因分析
如果单进程仍然爆内存（~7TB+），可能原因：
1. **参数量低估**：实际模型可能1.5-2T参数
2. **专家参数放大**：MoE的专家参数被低估
3. **系统内存不足**：2.9TB物理内存 + swap空间不足

### 方案2：系统级内存优化（立即可实施）

如果单进程失败，先实施系统级优化：

#### 2.1 环境优化
```bash
# 1. 增加swap空间
sudo fallocate -l 32G /swapfile  # 创建32G swap
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon -s  # 验证swap启用

# 2. 清理系统缓存
sudo echo 3 > /proc/sys/vm/drop_caches  # 清理page cache
sudo sync

# 3. 关闭不必要服务
sudo systemctl stop docker  # 如果不需要
sudo systemctl stop snapd  # 如果不需要

# 4. 调整内存参数
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf  # 更激进的swap使用
```

#### 2.2 Python环境优化
```bash
# 在运行脚本前设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 减少内存碎片
export OMP_NUM_THREADS=4  # 减少线程内存使用
export MALLOC_TRIM_THRESHOLD_=100000  # 更频繁的内存回收

# 运行转换
python scripts/converter_hf_to_mcore.py ...
```

### 方案3：最小化的代码优化（如果需要）

#### 3.1 减少内存复制
在转换过程中修改converter_hf_to_mcore.py，添加即时清理：

```python
# 在转换函数中添加内存清理
def convert_with_memory_management():
    for layer_idx in range(layers_to_convert):
        # 处理一层
        convert_single_layer(layer_idx)

        # 每10层清理一次内存
        if layer_idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 每20层强制清理
        if layer_idx % 20 == 0:
            aggressive_memory_cleanup()
```

#### 3.2 状态dict优化
```python
# 修改sharded_state_dict创建方式
def create_sharded_state_dict_optimized(model):
    # 避免创建完整的state_dict
    return model[0].module._efficient_sharded_state_dict()
```

## 实施建议（优先级从高到低）

### 第一优先：单进程 + 系统优化
```bash
# 1. 系统优化
sudo swapon /swapfile  # 启用swap
sudo echo 3 > /proc/sys/vm/drop_caches

# 2. 环境优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 3. 运行转换
python scripts/converter_hf_to_mcore.py ... --use_cpu_initialization
```

### 第二优先：渐进式代码优化
如果系统优化后仍然失败，才进行代码修改：
1. 添加更频繁的内存清理
2. 优化权重复制逻辑
3. 减少中间tensor创建

### 第三优先：高级优化
最后才考虑完全重构为流式处理

### 方案3：系统级优化（立即可实施）

#### 3.1 环境准备
```bash
# 1. 关闭不必要的进程
sudo systemctl stop unnecessary-services

# 2. 清理系统缓存
sudo echo 3 > /proc/sys/vm/drop_caches
sudo sync

# 3. 设置大页内存（如果支持）
echo madvise > /proc/sys/vm/overcommit_memory

# 4. 增加swap空间（如果需要）
sudo fallocate -l 100G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3.2 Python优化
```python
# 在脚本开头添加
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 减少内存碎片
os.environ['OMP_NUM_THREADS'] = '1'  # 减少线程内存使用

# 强制使用更高效的内存分配器
import torch
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')  # 平衡精度和内存
```

## 推荐实施步骤

### 第一步：立即测试单进程
```bash
# 使用已添加的内存监控运行单进程版本
python scripts/converter_hf_to_mcore.py \
    --hf_model_path /path/to/bailing-moe-v2 \
    --output_path /path/to/output \
    --use_cpu_initialization \
    --trust_remote_code 2>&1 | tee conversion.log
```

观察输出中的内存使用情况，特别关注：
- `[MEMORY] Before HF model loading`
- `[MEMORY] After HF model loaded`
- `[MEMORY] After sharded_state_dict created`
- `[MEMORY] After HF models deleted`

### 第二步：如果单进程失败，实施方案2
- 修改代码实现部分权重加载
- 重点优化内存峰值最高的阶段

### 第三步：系统优化
- 实施系统级优化
- 监控系统资源使用

## 预期效果

| 方案 | 预期内存峰值 | 成功率 | 实施复杂度 |
|------|--------------|--------|------------|
| 单进程 | 7-12TB | 中等 | 简单 |
| 部分权重加载 | 3-5TB | 较高 | 中等 |
| 系统优化 | 降低10-20% | 低 | 简单 |

## 监控命令

```bash
# 内存监控（每5秒）
watch -n 5 'free -h && echo "---" && ps aux --sort=-%mem | head -5'

# 转换过程监控
python scripts/converter_hf_to_mcore.py ... 2>&1 | tee conversion.log

# 磁盘空间监控
df -h /path/to/output
```

## 总结

**核心建议：**
1. **先试单进程**：使用现有内存监控观察实际占用
2. **需要时才优化**：避免过度优化
3. **渐进式改进**：从最简单的改动开始

这种方法更加现实和可操作，避免了过于复杂的实现。

#### 1.2 修改主函数

替换原有的HF模型加载逻辑（约626-628行）：

```python
# 原来的代码（注释掉）：
# hf_model = AutoModelForCausalLM.from_pretrained(
#     hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
# )
# hf_state_dict = hf_model.state_dict()

# 新的代码：
hf_model = load_hf_model_minimal_memory(hf_model_path, layer_start, layer_end, hf_config, trust_remote_code)
hf_state_dict = hf_model.state_dict()  # 现在只有部分权重的state_dict
```

#### 1.3 增强内存清理

替换682行后的清理逻辑：

```python
# 原来的代码：
# del hf_state_dict, hf_model

# 新的代码：
print("Starting aggressive memory cleanup...")
del hf_state_dict, hf_model
aggressive_memory_cleanup(args.use_cpu_initialization)

# 等待内存稳定（重要）
import time
time.sleep(5)  # 等待系统回收内存
```

#### 1.4 使用方式

```bash
# 单进程转换（推荐）
python scripts/converter_hf_to_mcore.py \
    --hf_model_path /path/to/bailing-moe-v2 \
    --output_path /path/to/output \
    --use_cpu_initialization \
    --trust_remote_code
```

### 方案2：修改为真正的流式处理（高级）

#### 2.1 重构转换函数

创建新的流式转换函数：

```python
def convert_checkpoint_streaming_bailing(
    hf_model_path: str,
    model,
    hf_config,
    layer_start_end: Tuple[int, int],
    use_cpu_initialization: bool = False,
) -> int:
    """
    流式转换HF BailingMoeV2 checkpoint到Megatron-Core格式
    逐层处理，立即清理，最小化内存占用
    """
    layer_start, layer_end = layer_start_end
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    numel = 0

    print(f"[PP{pp_rank}] Streaming conversion for layers {layer_start}-{layer_end}")

    # 准备safetensors文件句柄
    model_files = [f for f in os.listdir(hf_model_path) if f.endswith('.safetensors')]

    # 按层流式处理
    for global_layer_idx in range(layer_start, layer_end):
        print(f"[PP{pp_rank}] Converting layer {global_layer_idx}")

        # 1. 加载单层权重（最小内存）
        layer_weights = load_single_layer_weights(hf_model_path, global_layer_idx, model_files)
        if not layer_weights:
            continue

        numel_cur = numel

        # 2. 转换单层
        layer_idx = global_layer_idx - layer_start
        target_layer = model.decoder.layers[layer_idx]

        # 转换embedding（第一层）
        if global_layer_idx == 0 and pp_rank == 0:
            if 'model.word_embeddings.weight' in layer_weights:
                numel += safe_copy(
                    layer_weights['model.word_embeddings.weight'],
                    target_layer.embedding.word_embeddings.weight
                )

        # 转换attention
        if 'input_layernorm.weight' in layer_weights:
            numel += safe_copy(
                layer_weights['input_layernorm.weight'],
                target_layer.self_attention.linear_qkv.layer_norm_weight
            )

        # 转换QKV权重
        if 'attention.query_key_value.weight' in layer_weights:
            numel += safe_copy(
                layer_weights['attention.query_key_value.weight'],
                target_layer.self_attention.linear_qkv.weight
            )

        # 转换MLP（根据层类型）
        if global_layer_idx < getattr(hf_config, "first_k_dense_replace", 1):
            # Dense层
            if 'post_attention_layernorm.weight' in layer_weights:
                numel += safe_copy(
                    layer_weights['post_attention_layernorm.weight'],
                    target_layer.mlp.linear_fc1.layer_norm_weight
                )
            # 继续处理其他dense权重...
        else:
            # MoE层
            if 'post_attention_layernorm.weight' in layer_weights:
                numel += safe_copy(
                    layer_weights['post_attention_layernorm.weight'],
                    target_layer.pre_mlp_layernorm.weight
                )

            # 处理专家权重（简化版）
            expert_count = getattr(hf_config, 'num_experts', 8)
            for expert_idx in range(expert_count):
                expert_prefix = f'mlp.experts.{expert_idx}.'
                if f'{expert_prefix}gate_proj.weight' in layer_weights and f'{expert_prefix}up_proj.weight' in layer_weights:
                    fc1 = torch.cat([
                        layer_weights[f'{expert_prefix}gate_proj.weight'],
                        layer_weights[f'{expert_prefix}up_proj.weight']
                    ], dim=0)

                    target_expert = target_layer.mlp.experts.local_experts[expert_idx]
                    numel += safe_copy(fc1, target_expert.linear_fc1.weight)

                    if f'{expert_prefix}down_proj.weight' in layer_weights:
                        numel += safe_copy(
                            layer_weights[f'{expert_prefix}down_proj.weight'],
                            target_expert.linear_fc2.weight
                        )

                    del fc1

        # 3. 立即清理当前层内存
        del layer_weights
        aggressive_memory_cleanup(use_cpu_initialization)

        print(f"[PP{pp_rank}] Layer {global_layer_idx} converted, total numel: {numel}, layer added: {numel - numel_cur}")

    # 处理final norm和lm head
    if pp_rank == pp_size - 1:
        print(f"[PP{pp_rank}] Processing final norm and lm head")
        final_weights = load_final_weights(hf_model_path, model_files)

        if 'model.norm.weight' in final_weights:
            numel += safe_copy(
                final_weights['model.norm.weight'],
                model.decoder.final_layernorm.weight
            )

        if 'lm_head.weight' in final_weights and not getattr(hf_config, "tie_word_embeddings", False):
            numel += safe_copy(
                final_weights['lm_head.weight'],
                model.output_layer.weight
            )

        del final_weights
        aggressive_memory_cleanup(use_cpu_initialization)

    return numel

def load_single_layer_weights(hf_model_path: str, layer_idx: int, model_files: list) -> dict:
    """加载单层权重到内存"""
    layer_weights = {}

    for model_file in model_files:
        file_path = os.path.join(hf_model_path, model_file)

        with safe_open(file_path, framework="pt") as f:
            # 只加载当前层相关的权重
            layer_prefix = f'model.layers.{layer_idx}.'

            for key in f.keys():
                if key.startswith(layer_prefix):
                    layer_weights[key] = f.get_tensor(key)

    return layer_weights

def load_final_weights(hf_model_path: str, model_files: list) -> dict:
    """加载final norm和lm head权重"""
    final_weights = {}

    for model_file in model_files:
        file_path = os.path.join(hf_model_path, model_file)

        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                if 'model.norm.weight' in key or 'lm_head.weight' in key:
                    final_weights[key] = f.get_tensor(key)

    return final_weights
```

#### 2.2 修改主函数调用

替换原有的转换调用（约644-647行）：

```python
# 原来的代码：
# numel_partial: int = convert_checkpoint_from_transformers_to_megatron_bailing(
#     hf_model, model[0].module, hf_config, layer_start_end=(layer_start, layer_end)
# )

# 新的代码：
numel_partial: int = convert_checkpoint_streaming_bailing(
    hf_model_path, model[0].module, hf_config,
    layer_start_end=(layer_start, layer_end),
    use_cpu_initialization=args.use_cpu_initialization
)
```

### 方案3：分批次转换（如果以上方案仍失败）

#### 3.1 分批转换逻辑

```python
def batch_convert_hf_to_mcore(hf_model_path: str, output_path: str, hf_config, num_batches: int = 2):
    """
    分批次转换 - 将模型分成多个批次分别转换
    """
    total_layers = hf_config.num_hidden_layers
    layers_per_batch = total_layers // num_batches

    print(f"Starting batch conversion: {total_layers} layers in {num_batches} batches, {layers_per_batch} layers per batch")

    # 临时输出目录
    temp_dirs = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * layers_per_batch
        batch_end = (batch_idx + 1) * layers_per_batch if batch_idx < num_batches - 1 else total_layers

        batch_output = f"{output_path}_batch_{batch_idx}"
        temp_dirs.append(batch_output)

        print(f"Converting batch {batch_idx}: layers {batch_start}-{batch_end}")

        # 创建临时转换脚本
        create_batch_converter_script(
            hf_model_path, batch_output, hf_config,
            batch_start, batch_end, batch_idx
        )

        # 执行批量转换
        import subprocess
        result = subprocess.run([
            "python", "batch_converter_temp.py"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Batch {batch_idx} failed: {result.stderr}")
            return False

        # 清理内存
        aggressive_memory_cleanup()
        print(f"Batch {batch_idx} completed successfully")

    # 合并批次结果
    print("Merging batch results...")
    merge_batch_results(temp_dirs, output_path)

    # 清理临时目录
    for temp_dir in temp_dirs:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Batch conversion completed successfully!")
    return True
```

## 实施建议

### 阶段1：立即实施（最容易）

1. **使用单进程转换**
   ```bash
   python scripts/converter_hf_to_mcore.py \
       --hf_model_path /path/to/bailing-moe-v2 \
       --output_path /path/to/output \
       --use_cpu_initialization \
       --trust_remote_code
   ```

2. **如果单进程仍失败**，实施方案1的按需加载

### 阶段2：中等复杂度（方案1）

1. 实现按需加载函数
2. 修改主函数调用
3. 测试内存使用情况

### 阶段3：高复杂度（方案2或3）

1. 如果方案1仍失败，实施方案2的流式处理
2. 或者实施方案3的分批次转换

## 预期效果

### 内存使用对比

| 方案 | 预期内存使用 | 成功率 | 实施难度 |
|------|--------------|--------|----------|
| 当前多进程 | ~5.2T | 极低 | 已失败 |
| 单进程 | ~2.6T | 中等 | 简单 |
| 方案1（按需加载） | ~1.2T | 高 | 中等 |
| 方案2（流式处理） | ~0.8T | 很高 | 困难 |
| 方案3（分批次） | ~0.6T | 很高 | 中等 |

### 推荐实施路径

1. **立即尝试**：单进程转换
2. **如果失败**：实施方案1（按需加载）
3. **最后选择**：方案2或3

## 注意事项

1. **备份原始脚本**：修改前备份converter_hf_to_mcore.py
2. **监控内存使用**：使用`htop`或`free -h`监控转换过程
3. **磁盘空间**：确保输出路径有足够空间（~1.5T）
4. **网络稳定性**：如果使用网络存储，确保连接稳定
5. **耐心**：1T模型转换可能需要数小时完成

## 监控命令

```bash
# 监控内存使用
watch -n 5 'free -h'

# 监控进程内存
ps aux --sort=-%mem | head -10

# 监控磁盘使用
df -h /path/to/output
```