# Copyright 2024 The VERL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
"""
Partial Rollout Buffer Implementation

This module implements a buffer for storing and managing partial rollout results
in token ID format. The buffer supports step-based eviction and provides
efficient continuation request management for SGLang rollout workers.

Key Features:
- Token ID based storage (no text processing)
- Step-based eviction policy
- FIFO ordering for continuation requests
- Configurable size and step window
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional
import torch

logger = logging.getLogger(__name__)


class PartialRolloutBuffer:
    """
    Buffer for managing partial rollout results based on token IDs.

    This buffer stores incomplete generation results and provides continuation
    requests for subsequent rollout steps. It operates entirely on token IDs
    to avoid tokenization consistency issues.

    Args:
        max_size: Maximum number of partial requests to store
        max_steps: Maximum number of training steps to keep requests
    """

    def __init__(self, max_size: int = 1000, max_steps: int = 3):
        self.partial_requests: Dict[str, Dict] = {}  # request_id -> partial_request_data
        self.request_order: List[str] = []  # Maintain insertion order for FIFO
        self.current_step: int = 0
        self.max_buffer_size: int = max_size
        self.step_window_size: int = max_steps

        logger.info(f"PartialRolloutBuffer initialized: max_size={max_size}, max_steps={max_steps}")

    def store_partial_requests(self, partial_results: List[Dict]) -> None:
        """
        Store partial results in the buffer for future continuation.

        Args:
            partial_results: List of partial result dictionaries containing
                           original_input_ids and partial_response_token_ids
        """
        # Filter out invalid results first
        valid_results = []
        for result in partial_results:
            request_id = result.get('request_id', '')
            if request_id:
                valid_results.append(result)

        if not valid_results:
            return

        # Batch eviction: calculate total space needed and evict in batch
        space_needed = len(valid_results)
        current_size = len(self.partial_requests)
        total_size_after = current_size + space_needed

        if total_size_after > self.max_buffer_size:
            # Calculate how many requests need to be evicted
            eviction_needed = total_size_after - self.max_buffer_size
            logger.debug(f"Batch eviction needed: {eviction_needed} requests to make space for {space_needed} new requests")

            # Batch evict oldest requests (FIFO)
            evicted_count = 0
            while evicted_count < eviction_needed and self.request_order:
                oldest_request_id = self.request_order.pop(0)
                if oldest_request_id in self.partial_requests:
                    del self.partial_requests[oldest_request_id]
                    evicted_count += 1
                else:
                    # Inconsistent state, skip this request
                    continue

            logger.debug(f"Batch evicted {evicted_count} oldest requests from buffer")

        # Store all valid requests
        for result in valid_results:
            request_id = result.get('request_id')

            # Store token ID based data (no text processing)
            request_data = {
                'request_id': request_id,
                'original_input_ids': result.get('original_input_ids'),  # Original prompt token IDs
                'partial_response_token_ids': result.get('partial_response_token_ids', []),  # Partial response token IDs
                'completion_tokens': result.get('completion_tokens', 0),
                'remaining_max_tokens': result.get('remaining_max_tokens', 512),
                'sampling_params': result.get('sampling_params', {}),
                'created_step': self.current_step,
                'batch_index': result.get('batch_index', -1),
                'is_continuation': result.get('is_continuation', True),
                # Support multi-modal data
                'image_data': result.get('image_data'),
                'multi_modal_data': result.get('multi_modal_data')
            }

            self.partial_requests[request_id] = request_data
            # Maintain insertion order for FIFO
            self.request_order.append(request_id)

        logger.debug(f"Stored {len(valid_results)} partial results in buffer (current size: {len(self.partial_requests)})")

    def get_continuation_requests(self, needed_count: int) -> List[Dict]:
        """
        Get continuation requests from the buffer.

        Args:
            needed_count: Number of continuation requests needed

        Returns:
            List of continuation request dictionaries with concatenated token IDs
        """
        continuation_requests = []

        # Get all pending requests that still have remaining tokens
        pending_requests = [
            req for req in self.partial_requests.values()
            if req['remaining_max_tokens'] > 0
        ]

        # Sort by created_step to prioritize older requests
        pending_requests.sort(key=lambda x: x['created_step'])

        for request_data in pending_requests[:needed_count]:
            # Direct token ID concatenation (no text processing)
            original_input_ids = request_data['original_input_ids'].clone().detach()
            partial_response_token_ids = request_data['partial_response_token_ids'].clone().detach()

            if len(partial_response_token_ids) == 0:
                continued_input_ids = original_input_ids
            else:
                continued_input_ids = torch.cat([original_input_ids, partial_response_token_ids], dim=-1)

            continuation_request = {
                'request_id': request_data['request_id'],
                'input_ids': continued_input_ids,  # Concatenated token IDs for SGLang
                'sampling_params': request_data['sampling_params'],
                'is_continuation': True,
                'completion_tokens_so_far': request_data['completion_tokens'],
                'remaining_max_tokens': request_data['remaining_max_tokens'],
                # Store original token IDs for potential future continuation
                'original_input_ids': original_input_ids,
                'partial_response_token_ids': request_data['partial_response_token_ids'],
                'batch_index': request_data.get('batch_index', -1),
                # Preserve multi-modal data
                'image_data': request_data.get('image_data'),
                'multi_modal_data': request_data.get('multi_modal_data')
            }
            continuation_requests.append(continuation_request)

        logger.debug(f"Retrieved {len(continuation_requests)} continuation requests from buffer")
        return continuation_requests

    def remove_completed_requests(self, completed_request_ids: List[str]) -> None:
        """
        Remove completed requests from the buffer.

        Args:
            completed_request_ids: List of request IDs to remove
        """
        removed_count = 0
        for request_id in completed_request_ids:
            if request_id in self.partial_requests:
                del self.partial_requests[request_id]
                # Also remove from order list if present
                if request_id in self.request_order:
                    self.request_order.remove(request_id)
                removed_count += 1

        if removed_count > 0:
            logger.debug(f"Removed {removed_count} completed requests from buffer")

    def increment_step(self) -> None:
        """
        Increment current step and trigger step-based eviction.
        """
        self.current_step += 1
        self._evict_by_step()

        logger.debug(f"Buffer step incremented to {self.current_step}, "
                    f"current size: {len(self.partial_requests)}")

    def is_empty(self) -> bool:
        """
        Check if buffer is empty.

        Returns:
            True if buffer is empty, False otherwise
        """
        return len(self.partial_requests) == 0

    def get_stats(self) -> Dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary containing buffer statistics
        """
        step_counts = {}
        for req in self.partial_requests.values():
            step = req['created_step']
            step_counts[step] = step_counts.get(step, 0) + 1

        return {
            'total_requests': len(self.partial_requests),
            'current_step': self.current_step,
            'step_distribution': step_counts,
            'pending_requests': len([req for req in self.partial_requests.values()
                                   if req['remaining_max_tokens'] > 0])
        }

    def clear(self) -> None:
        """
        Clear all requests from the buffer.
        """
        self.partial_requests.clear()
        logger.info("PartialRolloutBuffer cleared")

    def _evict_oldest_by_step(self) -> None:
        """
        Evict the oldest request from the buffer using FIFO policy.

        Removes the request that was inserted first (FIFO - First In, First Out).
        This method is kept for backward compatibility, but actual eviction
        now happens in batches in store_partial_requests().
        """
        if self.partial_requests and self.request_order:
            # Get the first inserted request (FIFO)
            oldest_request_id = self.request_order[0]

            # Remove from both structures
            del self.partial_requests[oldest_request_id]
            self.request_order.remove(oldest_request_id)

            logger.debug(f"Evicted oldest request {oldest_request_id} from buffer (FIFO)")

    def _evict_by_step(self) -> None:
        """
        Evict requests that are older than the step window.
        """
        expired_steps = self.current_step - self.step_window_size
        expired_requests = [
            req_id for req_id, req in self.partial_requests.items()
            if req['created_step'] <= expired_steps
        ]

        for request_id in expired_requests:
            del self.partial_requests[request_id]
            # Also remove from order list if present
            if request_id in self.request_order:
                self.request_order.remove(request_id)

        if expired_requests:
            logger.debug(f"Evicted {len(expired_requests)} expired requests "
                        f"older than step {expired_steps}")

    def clear(self) -> None:
        """
        Clear all requests from buffer.
        """
        self.partial_requests.clear()
        self.request_order.clear()
        logger.info("PartialRolloutBuffer cleared")