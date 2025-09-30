# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Unit tests for PartialRolloutBuffer.

This test file follows VERL conventions with minimal mocking, focusing on
testing the core buffer functionality without unnecessary dependency mocking.
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
import importlib.util

# Import PartialRolloutBuffer directly from file to avoid ray dependency
spec = importlib.util.spec_from_file_location(
    'partial_rollout_buffer',
    os.path.join(os.path.dirname(__file__), '..', '..', 'verl', 'utils', 'partial_rollout_buffer.py')
)
buffer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(buffer_module)
PartialRolloutBuffer = buffer_module.PartialRolloutBuffer


class TestPartialRolloutBuffer:
    """Test suite for PartialRolloutBuffer class with minimal dependencies."""

    def test_initialization(self):
        """Test buffer initialization with default and custom parameters."""
        # Test default initialization
        buffer = PartialRolloutBuffer()
        assert buffer.max_buffer_size == 1000
        assert buffer.step_window_size == 3
        assert buffer.current_step == 0
        assert len(buffer.partial_requests) == 0
        assert buffer.is_empty()

        # Test custom initialization
        custom_buffer = PartialRolloutBuffer(max_size=500, max_steps=5)
        assert custom_buffer.max_buffer_size == 500
        assert custom_buffer.step_window_size == 5

    def test_store_partial_requests(self):
        """Test storing partial requests in buffer."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Create test partial results
        partial_results = [
            {
                'request_id': 'req_001',
                'original_input_ids': torch.tensor([1, 2, 3]),
                'partial_response_token_ids': torch.tensor([100, 101]),
                'completion_tokens': 2,
                'remaining_max_tokens': 8,
                'sampling_params': {'temperature': 0.8},
                'created_step': 0,
                'batch_index': 0
            },
            {
                'request_id': 'req_002',
                'original_input_ids': torch.tensor([4, 5, 6]),
                'partial_response_token_ids': torch.tensor([200]),
                'completion_tokens': 1,
                'remaining_max_tokens': 9,
                'sampling_params': {'temperature': 0.9},
                'created_step': 0,
                'batch_index': 1
            }
        ]

        # Store partial requests
        buffer.store_partial_requests(partial_results)

        # Verify storage
        assert len(buffer.partial_requests) == 2
        assert 'req_001' in buffer.partial_requests
        assert 'req_002' in buffer.partial_requests
        assert not buffer.is_empty()

    def test_get_continuation_requests(self):
        """Test retrieving continuation requests from buffer."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Store test data
        partial_results = [
            {
                'request_id': 'req_001',
                'original_input_ids': torch.tensor([1, 2, 3]),
                'partial_response_token_ids': torch.tensor([100, 101]),
                'completion_tokens': 2,
                'remaining_max_tokens': 8,
                'sampling_params': {'temperature': 0.8},
                'created_step': 0,
                'batch_index': 0,
                'multi_modal_data': {'image': np.random.rand(3, 32, 32)}
            }
        ]

        buffer.store_partial_requests(partial_results)

        # Get continuation requests (need to specify needed_count)
        continuation_requests = buffer.get_continuation_requests(needed_count=1)

        # Verify continuation request structure
        assert len(continuation_requests) == 1

        req = continuation_requests[0]
        assert req['request_id'] == 'req_001'
        assert req['batch_index'] == 0
        assert 'image_data' in req or 'multi_modal_data' in req

        # Verify token concatenation logic
        expected_input_ids = torch.tensor([1, 2, 3, 100, 101])
        assert torch.equal(req['input_ids'], expected_input_ids)
        assert req['remaining_max_tokens'] == 8
        assert req['completion_tokens_so_far'] == 2

    def test_buffer_size_eviction(self):
        """Test buffer eviction when size limit is exceeded."""
        buffer = PartialRolloutBuffer(max_size=2, max_steps=3)

        # Store first request
        partial_results_1 = [{
            'request_id': 'req_001',
            'original_input_ids': torch.tensor([1, 2, 3]),
            'partial_response_token_ids': torch.tensor([100]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 0
        }]

        buffer.store_partial_requests(partial_results_1)
        assert len(buffer.partial_requests) == 1

        # Store second request (still within limit)
        partial_results_2 = [{
            'request_id': 'req_002',
            'original_input_ids': torch.tensor([4, 5, 6]),
            'partial_response_token_ids': torch.tensor([200]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 1
        }]

        buffer.store_partial_requests(partial_results_2)
        assert len(buffer.partial_requests) == 2

        # Store third request (should trigger eviction)
        partial_results_3 = [{
            'request_id': 'req_003',
            'original_input_ids': torch.tensor([7, 8, 9]),
            'partial_response_token_ids': torch.tensor([300]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 2
        }]

        buffer.store_partial_requests(partial_results_3)

        # Should have evicted oldest request to maintain size limit
        assert len(buffer.partial_requests) == 2
        assert 'req_001' not in buffer.partial_requests  # Should be evicted
        assert 'req_002' in buffer.partial_requests
        assert 'req_003' in buffer.partial_requests

    def test_step_based_eviction(self):
        """Test step-based eviction of old requests."""
        buffer = PartialRolloutBuffer(max_size=10, max_steps=2)

        # Store requests at different steps
        partial_results_step_0 = [{
            'request_id': 'req_old',
            'original_input_ids': torch.tensor([1, 2, 3]),
            'partial_response_token_ids': torch.tensor([100]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 0
        }]

        buffer.store_partial_requests(partial_results_step_0)
        assert len(buffer.partial_requests) == 1

        # Increment step (should still keep requests within window)
        buffer.increment_step()
        assert buffer.current_step == 1
        assert len(buffer.partial_requests) == 1  # Still within window

        # Add new request at current step
        partial_results_step_1 = [{
            'request_id': 'req_new',
            'original_input_ids': torch.tensor([4, 5, 6]),
            'partial_response_token_ids': torch.tensor([200]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 1,
            'batch_index': 1
        }]

        buffer.store_partial_requests(partial_results_step_1)
        assert len(buffer.partial_requests) == 2

        # Increment step to current_step=2
        # Expired steps = 2 - 2 = 0, so requests with created_step <= 0 should be evicted
        buffer.increment_step()
        assert buffer.current_step == 2

        # Old request should be evicted (created_step=0 <= expired_steps=0)
        assert 'req_old' not in buffer.partial_requests
        assert 'req_new' in buffer.partial_requests  # created_step=1 > 0, should remain

    def test_remove_completed_requests(self):
        """Test removing completed requests from buffer."""
        buffer = PartialRolloutBuffer(max_size=10, max_steps=3)

        # Store multiple requests
        partial_results = [
            {
                'request_id': 'req_001',
                'original_input_ids': torch.tensor([1, 2, 3]),
                'partial_response_token_ids': torch.tensor([100]),
                'completion_tokens': 1,
                'remaining_max_tokens': 9,
                'sampling_params': {},
                'created_step': 0,
                'batch_index': 0
            },
            {
                'request_id': 'req_002',
                'original_input_ids': torch.tensor([4, 5, 6]),
                'partial_response_token_ids': torch.tensor([200]),
                'completion_tokens': 1,
                'remaining_max_tokens': 9,
                'sampling_params': {},
                'created_step': 0,
                'batch_index': 1
            }
        ]

        buffer.store_partial_requests(partial_results)
        assert len(buffer.partial_requests) == 2

        # Remove one completed request
        buffer.remove_completed_requests(['req_001'])
        assert len(buffer.partial_requests) == 1
        assert 'req_001' not in buffer.partial_requests
        assert 'req_002' in buffer.partial_requests

        # Remove non-existent request (should not fail)
        buffer.remove_completed_requests(['non_existent'])
        assert len(buffer.partial_requests) == 1

    def test_empty_partial_response_tokens(self):
        """Test handling of empty partial response tokens."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Store request with empty partial response
        partial_results = [{
            'request_id': 'req_empty',
            'original_input_ids': torch.tensor([1, 2, 3]),
            'partial_response_token_ids': torch.tensor([]),  # Empty tensor
            'completion_tokens': 0,
            'remaining_max_tokens': 10,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 0
        }]

        buffer.store_partial_requests(partial_results)

        # Get continuation request
        continuation_requests = buffer.get_continuation_requests(needed_count=1)
        assert len(continuation_requests) == 1

        # Should use original input_ids when partial response is empty
        req = continuation_requests[0]
        expected_input_ids = torch.tensor([1, 2, 3])  # No concatenation
        assert torch.equal(req['input_ids'], expected_input_ids)

    def test_multi_modal_data_preservation(self):
        """Test preservation of multi-modal data through buffer operations."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Original multi-modal data
        image_data = np.random.rand(3, 64, 64)
        multi_modal_data = {
            'image': image_data,
            'text': 'Test image caption'
        }

        # Store request with multi-modal data
        partial_results = [{
            'request_id': 'req_mm',
            'original_input_ids': torch.tensor([1, 2, 3]),
            'partial_response_token_ids': torch.tensor([100, 101]),
            'completion_tokens': 2,
            'remaining_max_tokens': 8,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 0,
            'image_data': image_data,
            'multi_modal_data': multi_modal_data
        }]

        buffer.store_partial_requests(partial_results)

        # Retrieve continuation request
        continuation_requests = buffer.get_continuation_requests(needed_count=1)
        assert len(continuation_requests) == 1

        req = continuation_requests[0]

        # Verify multi-modal data preservation
        if 'image_data' in req:
            assert np.array_equal(req['image_data'], image_data)

        if 'multi_modal_data' in req:
            assert np.array_equal(req['multi_modal_data']['image'], image_data)
            assert req['multi_modal_data']['text'] == 'Test image caption'

    def test_get_stats(self):
        """Test buffer statistics functionality."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Initial stats
        stats = buffer.get_stats()
        assert stats['total_requests'] == 0
        assert stats['current_step'] == 0
        assert stats['pending_requests'] == 0

        # Store some requests
        partial_results = [
            {
                'request_id': 'req_001',
                'original_input_ids': torch.tensor([1, 2, 3]),
                'partial_response_token_ids': torch.tensor([100]),
                'completion_tokens': 1,
                'remaining_max_tokens': 9,
                'sampling_params': {},
                'created_step': 0,
                'batch_index': 0
            },
            {
                'request_id': 'req_002',
                'original_input_ids': torch.tensor([4, 5, 6]),
                'partial_response_token_ids': torch.tensor([200, 201]),
                'completion_tokens': 2,
                'remaining_max_tokens': 8,
                'sampling_params': {},
                'created_step': 0,
                'batch_index': 1
            }
        ]

        buffer.store_partial_requests(partial_results)

        # Updated stats
        stats = buffer.get_stats()
        assert stats['total_requests'] == 2
        assert stats['current_step'] == 0
        assert stats['pending_requests'] == 2  # Both requests have remaining_max_tokens > 0

        # Increment step and check stats
        buffer.increment_step()
        stats = buffer.get_stats()
        assert stats['current_step'] == 1
        assert stats['total_requests'] == 2  # Still within window

    def test_clear(self):
        """Test clearing all requests from buffer."""
        buffer = PartialRolloutBuffer(max_size=5, max_steps=3)

        # Store some requests
        partial_results = [{
            'request_id': 'req_001',
            'original_input_ids': torch.tensor([1, 2, 3]),
            'partial_response_token_ids': torch.tensor([100]),
            'completion_tokens': 1,
            'remaining_max_tokens': 9,
            'sampling_params': {},
            'created_step': 0,
            'batch_index': 0
        }]

        buffer.store_partial_requests(partial_results)
        assert len(buffer.partial_requests) == 1
        assert not buffer.is_empty()

        # Clear buffer
        buffer.clear()
        assert len(buffer.partial_requests) == 0
        assert buffer.is_empty()
        assert buffer.current_step == 0  # Step should be preserved

    def test_concurrent_operations(self):
        """Test thread safety of buffer operations."""
        import threading

        buffer = PartialRolloutBuffer(max_size=100, max_steps=10)
        results = []
        errors = []

        def store_requests(thread_id):
            try:
                for i in range(10):
                    partial_results = [{
                        'request_id': f'req_{thread_id}_{i}',
                        'original_input_ids': torch.tensor([1, 2, 3]),
                        'partial_response_token_ids': torch.tensor([100 + i]),
                        'completion_tokens': 1,
                        'remaining_max_tokens': 9,
                        'sampling_params': {},
                        'created_step': i,
                        'batch_index': thread_id * 10 + i
                    }]

                    buffer.store_partial_requests(partial_results)
                    results.append(thread_id * 10 + i)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=store_requests, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all requests were stored
        expected_total_requests = 5 * 10  # 5 threads * 10 requests each
        actual_total_requests = len(buffer.partial_requests)
        assert actual_total_requests == expected_total_requests, f"Expected {expected_total_requests}, got {actual_total_requests}"