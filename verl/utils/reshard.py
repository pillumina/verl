import torch
from typing import List

from megatron.core import parallel_state as mpu



def get_dp_reshard_tensor_via_alltoall(
    src_tensor: torch.Tensor,
    src_dp_size: int,
    dst_dp_size: int,
    dst_shape: List[int],
    global_dp_ranks: List[int],
    dst_dp_rank: int
):
    """ For resharding during D2D tensor transfer between generate_sequences and compute_ref_log_prob.
    
    Args:
        src_tensor: Source tensor to be resharded
        src_dp_size: Source data parallel size
        dst_dp_size: Destination data parallel size
        dst_shape: Destination tensor shape
        global_dp_ranks: Global DP rank mapping (supports both Megatron and FSDP)
        dst_dp_rank: Destination DP rank of current process
    """
    assert src_dp_size == torch.distributed.get_world_size(), (
        "We only support src_dp_size (generate_sequences) equals world_size for now in the cached tensor resharding."
    )
    micro_dp_size = src_dp_size // dst_dp_size
    
    # this rank receives tensors from these src_ranks
    src_ranks = set(range(dst_dp_rank * micro_dp_size, (dst_dp_rank + 1) * micro_dp_size, 1))
    # output tensor buffer for AllToAllV communication
    buffer = torch.empty(dst_shape, dtype=src_tensor.dtype, device=src_tensor.device)
    src_bs = src_tensor.shape[0]
    recv_tensors = []
    send_tensors = []

    for global_rank, dp_rank in enumerate(global_dp_ranks):
        src_dp_rank = global_rank   # src_dp_size equals world_size
        if src_dp_rank in src_ranks:
            micro_dp_idx = src_dp_rank % micro_dp_size
            recv_tensors.append(buffer[micro_dp_idx * src_bs: micro_dp_idx * src_bs + src_bs, ...])
        else:
            # this rank does not recv tensor from the src_dp_rank, add empty tensor as placeholder
            recv_tensors.append(torch.empty(0, dtype=src_tensor.dtype, device=src_tensor.device))

        rank_i_src_ranks = set(range(dp_rank * micro_dp_size, (dp_rank + 1) * micro_dp_size, 1))
        if torch.distributed.get_rank() in rank_i_src_ranks:
            send_tensors.append(src_tensor)
        else:
            # this rank as src does not send tensor to the dst dp_rank, add empty tensor as placeholder
            send_tensors.append(torch.empty(0, dtype=src_tensor.dtype, device=src_tensor.device))

    torch.distributed.all_to_all(recv_tensors, send_tensors)
    return buffer