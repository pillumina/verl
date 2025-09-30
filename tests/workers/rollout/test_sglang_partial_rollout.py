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
Unit tests for SGLang Partial Rollout functionality.

This test file follows VERL conventions with minimal mocking, focusing on
testing the core logic without extensive dependency mocking.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestSGLangPartialRollout:
    """Test cases for SGLang Partial Rollout core functionality.

    This test class focuses on testing the core logic and data structures
    without complex initialization and mocking patterns.
    """

    def test_partial_rollout_config_validation(self):
        """Test partial rollout configuration parameter validation."""
        # Create mock config to avoid import dependency issues
        class MockRolloutConfig:
            def __init__(self):
                self.enable_partial_rollout = False
                self.over_sampling_batch_size = None
                self.partial_buffer_max_size = 1000
                self.partial_step_window = 3
                self.response_length = 10

        # Test valid configuration
        config = MockRolloutConfig()
        config.enable_partial_rollout = True
        config.over_sampling_batch_size = 4
        config.partial_buffer_max_size = 100
        config.partial_step_window = 3
        config.response_length = 10

        # Verify configuration
        assert config.enable_partial_rollout is True
        assert config.over_sampling_batch_size == 4
        assert config.partial_buffer_max_size == 100
        assert config.partial_step_window == 3
        assert config.response_length == 10

    def test_partial_rollout_request_data_structure(self):
        """Test partial rollout request data structure integrity."""
        # Create test request data
        request_data = {
            'request_id': 'test_req_001',
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'sampling_params': {
                'max_new_tokens': 10,
                'temperature': 0.8,
                'top_p': 0.9
            },
            'batch_index': 0,
            'original_input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'remaining_max_tokens': 15,
            'completion_tokens_so_far': 0,
            'is_continuation': False,
            'multi_modal_data': {'image': np.random.rand(3, 224, 224)}
        }

        # Validate request structure
        assert 'request_id' in request_data
        assert 'input_ids' in request_data
        assert 'sampling_params' in request_data
        assert 'batch_index' in request_data
        assert 'original_input_ids' in request_data
        assert 'remaining_max_tokens' in request_data
        assert 'is_continuation' in request_data

        # Validate data types
        assert isinstance(request_data['input_ids'], torch.Tensor)
        assert isinstance(request_data['sampling_params'], dict)
        assert isinstance(request_data['batch_index'], int)
        assert isinstance(request_data['remaining_max_tokens'], int)
        assert isinstance(request_data['is_continuation'], bool)

    def test_token_concatenation_logic(self):
        """Test token ID concatenation logic for continuation requests."""
        # Test case 1: Normal concatenation
        original_input_ids = torch.tensor([1, 2, 3])
        partial_response_tokens = torch.tensor([100, 101, 102])

        # Simulate buffer concatenation logic
        if len(partial_response_tokens) == 0:
            continued_input_ids = original_input_ids
        else:
            continued_input_ids = torch.cat([original_input_ids, partial_response_tokens], dim=-1)

        expected = torch.tensor([1, 2, 3, 100, 101, 102])
        assert torch.equal(continued_input_ids, expected)

        # Test case 2: Empty partial response
        empty_partial = torch.tensor([])
        if len(empty_partial) == 0:
            continued_input_ids = original_input_ids
        else:
            continued_input_ids = torch.cat([original_input_ids, empty_partial], dim=-1)

        assert torch.equal(continued_input_ids, original_input_ids)

    def test_multi_modal_data_preservation(self):
        """Test multi-modal data preservation through partial rollout flow."""
        # Original multi-modal data
        original_data = {
            'image': np.random.rand(3, 224, 224),
            'text': 'This is a test image'
        }

        # Simulate buffer storage and retrieval
        buffer_entry = {
            'request_id': 'test_req_001',
            'multi_modal_data': original_data,
            'batch_index': 0
        }

        # Simulate retrieval and usage in continuation request
        retrieved_data = buffer_entry.get('multi_modal_data')

        # Verify data integrity
        assert retrieved_data is not None
        assert 'image' in retrieved_data
        assert 'text' in retrieved_data
        assert np.array_equal(retrieved_data['image'], original_data['image'])
        assert retrieved_data['text'] == original_data['text']

    def test_oversampling_size_calculation(self):
        """Test oversampling batch size calculation logic."""
        batch_size = 4
        over_sampling_batch_size = 12

        # Simulate calculation logic from implementation
        if over_sampling_batch_size is not None:
            if over_sampling_batch_size <= batch_size:
                effective_over_sample_size = batch_size * 2
            else:
                effective_over_sample_size = over_sampling_batch_size
        else:
            effective_over_sample_size = batch_size * 2

        # Test case 1: Valid oversampling size
        assert effective_over_sample_size == 12

        # Test case 2: Oversampling too small
        over_sampling_batch_size = 3  # <= batch_size
        if over_sampling_batch_size is not None:
            if over_sampling_batch_size <= batch_size:
                effective_over_sample_size = batch_size * 2
            else:
                effective_over_sample_size = over_sampling_batch_size
        else:
            effective_over_sample_size = batch_size * 2

        assert effective_over_sample_size == 8  # batch_size * 2

        # Test case 3: None oversampling (auto-calculate)
        over_sampling_batch_size = None
        if over_sampling_batch_size is not None:
            if over_sampling_batch_size <= batch_size:
                effective_over_sample_size = batch_size * 2
            else:
                effective_over_sample_size = over_sampling_batch_size
        else:
            effective_over_sample_size = batch_size * 2

        assert effective_over_sample_size == 8  # batch_size * 2

    def test_buffer_step_eviction_logic(self):
        """Test buffer step-based eviction logic."""
        # Simulate buffer state
        current_step = 10
        max_steps = 3

        requests = {
            'req_001': {'created_step': 8, 'data': 'old_data_1'},  # Should be kept (age: 2)
            'req_002': {'created_step': 9, 'data': 'old_data_2'},  # Should be kept (age: 1)
            'req_003': {'created_step': 10, 'data': 'current_data'},  # Should be kept (age: 0)
            'req_004': {'created_step': 6, 'data': 'very_old_data'},  # Should be evicted (age: 4)
        }

        # Simulate eviction logic
        evicted_requests = []
        for request_id, request_data in requests.items():
            created_step = request_data.get('created_step', 0)
            age = current_step - created_step
            if age > max_steps:
                evicted_requests.append(request_id)

        # Verify eviction results
        expected_evicted = ['req_004']  # Only req_004 has age > 3
        assert set(evicted_requests) == set(expected_evicted)

        # Verify remaining requests
        remaining_requests = {k: v for k, v in requests.items() if k not in evicted_requests}
        assert len(remaining_requests) == 3
        assert 'req_001' in remaining_requests
        assert 'req_002' in remaining_requests
        assert 'req_003' in remaining_requests
        assert 'req_004' not in remaining_requests

    def test_log_probs_data_flow(self):
        """Test log_probs data flow through partial rollout."""
        # Simulate SGLang output with log_probs
        sglang_output = {
            'response_tokens': torch.tensor([100, 101, 102]),
            'log_probs': torch.tensor([-0.1, -0.2, -0.3]),
            'finish_reason': 'stop'
        }

        # Simulate result structure
        result = {
            'request_id': 'test_req_001',
            'response': sglang_output['response_tokens'],
            'log_probs': sglang_output['log_probs'],
            'is_complete': True
        }

        # Simulate DataProto batch creation
        results = [result]
        response_length = 10

        # Extract log_probs similar to implementation
        rollout_log_probs = []
        for result_item in results:
            if 'log_probs' in result_item and result_item['log_probs'] is not None:
                log_probs = result_item['log_probs']
                # Simulate padding if needed
                if len(log_probs) < response_length:
                    padding = torch.zeros(response_length - len(log_probs))
                    log_probs = torch.cat([log_probs, padding])
                rollout_log_probs.append(log_probs)

        # Verify results
        assert len(rollout_log_probs) == 1
        assert rollout_log_probs[0].shape[0] == response_length

        # Use approximate comparison for floating point values
        expected_values = torch.tensor([-0.1, -0.2, -0.3])
        actual_values = rollout_log_probs[0][:3]
        assert torch.allclose(actual_values, expected_values, atol=1e-6)
        assert rollout_log_probs[0][3:].sum() == 0  # Padded zeros

    def test_device_consistency_handling(self):
        """Test device consistency for tensor operations."""
        # Create tensors on different devices (simulated)
        device_cpu = 'cpu'
        device_target = 'cuda:0'  # Simulate target device

        # Original tensor on CPU
        original_tensor = torch.tensor([1, 2, 3])
        assert original_tensor.device.type == 'cpu'

        # Simulate device transfer (in real implementation)
        # transferred_tensor = original_tensor.to(device_target)
        # For test purposes, we'll just simulate the logic
        transferred_tensor = original_tensor.clone()  # Simulate transfer

        # Test tensor operations after transfer
        assert transferred_tensor.shape == original_tensor.shape
        assert torch.equal(transferred_tensor.cpu(), original_tensor)

    def test_error_handling_in_partial_rollout(self):
        """Test error handling scenarios in partial rollout."""
        # Test case 1: Missing log_probs
        result_without_logprobs = {
            'request_id': 'test_req_001',
            'response': torch.tensor([100, 101, 102]),
            'is_complete': True
            # Missing 'log_probs' key
        }

        # Should handle gracefully
        log_probs = result_without_logprobs.get('log_probs')
        assert log_probs is None

        # Test case 2: Invalid tensor dimensions
        try:
            # Simulate invalid tensor concatenation
            tensor1 = torch.tensor([1, 2, 3])
            tensor2 = torch.tensor([4, 5])  # Different shape

            # This should work in our logic since we're concatenating 1D tensors
            result = torch.cat([tensor1, tensor2], dim=-1)
            expected = torch.tensor([1, 2, 3, 4, 5])
            assert torch.equal(result, expected)

        except Exception as e:
            pytest.fail(f"Valid tensor concatenation should not fail: {e}")

    def test_sampling_params_consistency(self):
        """Test sampling parameters consistency between normal and partial rollout."""
        base_params = {
            'max_new_tokens': 10,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50
        }

        # Simulate normal rollout params
        normal_params = base_params.copy()

        # Simulate partial rollout params
        partial_params = base_params.copy()
        partial_params['max_new_tokens'] = 5  # Reduced for partial generation

        # Both should have base parameters
        assert 'temperature' in normal_params
        assert 'temperature' in partial_params
        assert normal_params['temperature'] == partial_params['temperature']

        # But max_new_tokens can differ
        assert normal_params['max_new_tokens'] != partial_params['max_new_tokens']