from __future__ import annotations

import unittest

import torch

from src.models import MaskedCNNClassifier, NativeXLSTMClassifier


class ModelTests(unittest.TestCase):
    def test_native_xlstm_forward_and_backward(self):
        torch.manual_seed(7)
        model = NativeXLSTMClassifier(
            vocab_size=50,
            num_classes=7,
            embedding_dim=32,
            dropout=0.0,
            context_length=8,
            num_blocks=1,
            num_heads=4,
            proj_factor=2.0,
            qkv_proj_blocksize=4,
            conv1d_kernel_size=4,
        )
        inputs = torch.randint(2, 50, (3, 8))
        lengths = torch.tensor([8, 5, 2])
        inputs[1, 5:] = 0
        inputs[2, 2:] = 0
        logits = model(inputs, lengths)
        self.assertEqual(tuple(logits.shape), (3, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.stack.parameters()))

    def test_cnn_prediction_is_invariant_to_batch_padding(self):
        torch.manual_seed(11)
        model = MaskedCNNClassifier(
            vocab_size=30,
            num_classes=7,
            embedding_dim=16,
            dropout=0.0,
            kernel_sizes=[3, 5, 7],
            num_filters=8,
        ).eval()
        short_alone = torch.tensor([[2, 3]])
        alone_logits = model(short_alone, torch.tensor([2]))

        mixed = torch.tensor([[2, 3, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9]])
        mixed_logits = model(mixed, torch.tensor([2, 6]))
        torch.testing.assert_close(alone_logits[0], mixed_logits[0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

