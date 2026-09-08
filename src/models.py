"""Neural classifiers for the controlled UIT-VSMEC comparison.

The native mLSTM components in this file are adapted and simplified from the
NX-AI xLSTM 2.0.5 reference implementation:
https://github.com/NX-AI/xlstm

Copyright (c) NXAI GmbH and its affiliates 2024. Licensed under Apache-2.0.

Project modifications:
- isolate an mLSTM-only path from unconditional sLSTM/CUDA-extension imports;
- use only ordinary PyTorch tensor operations;
- remove language-model and recurrent-generation interfaces;
- add sentence-classification pooling and project validation;
- keep the reference parallel stabilized mLSTM memory equations.

This native backend intentionally does not require nvcc, Conda, Triton or the
official package import. It is an mLSTM-only xLSTM classifier, not an sLSTM or
xLSTM-Large reproduction.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def last_valid_state(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, features], got {tuple(sequence.shape)}")
    if lengths.ndim != 1 or lengths.shape[0] != sequence.shape[0]:
        raise ValueError("lengths must be one-dimensional and match the batch size.")
    if torch.any(lengths <= 0) or torch.any(lengths > sequence.shape[1]):
        raise ValueError("Every length must be between 1 and the padded sequence length.")
    batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_indices, lengths.to(sequence.device) - 1]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        if self.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            features = hidden[-1]
        return self.classifier(self.dropout(features))


class MaskedCNNClassifier(nn.Module):
    """Kim-style CNN whose global max pooling excludes padded positions."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
        kernel_sizes: list[int],
        num_filters: int,
    ) -> None:
        super().__init__()
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("MaskedCNNClassifier requires positive odd kernel sizes.")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel,
                padding=kernel // 2,
            )
            for kernel in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        valid = positions < lengths.to(input_ids.device).unsqueeze(1)

        pooled = []
        for convolution in self.convolutions:
            features = F.relu(convolution(embedded))
            features = features.masked_fill(~valid.unsqueeze(1), torch.finfo(features.dtype).min)
            pooled.append(features.max(dim=-1).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=-1)))


def _small_init(parameter: torch.Tensor, dimension: int) -> None:
    nn.init.normal_(parameter, mean=0.0, std=math.sqrt(2 / (5 * dimension)))


def _wang_init(parameter: torch.Tensor, dimension: int, num_blocks: int) -> None:
    nn.init.normal_(parameter, mean=0.0, std=2 / num_blocks / math.sqrt(dimension))


class HeadwiseLinear(nn.Module):
    """Block-diagonal linear projection used by the reference mLSTM layer."""

    def __init__(self, features: int, num_projection_heads: int, bias: bool = False) -> None:
        super().__init__()
        if features % num_projection_heads != 0:
            raise ValueError("features must be divisible by num_projection_heads.")
        self.features = features
        self.num_projection_heads = num_projection_heads
        per_head = features // num_projection_heads
        self.weight = nn.Parameter(torch.empty(num_projection_heads, per_head, per_head))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape
        inputs = inputs.reshape(*shape[:-1], self.num_projection_heads, -1)
        outputs = torch.einsum("...hd,hod->...ho", inputs, self.weight)
        outputs = outputs.reshape(*shape[:-1], self.features)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, feature_dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.padding = kernel_size - 1
        self.convolution = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=feature_dim,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convolution(inputs.transpose(1, 2))
        if self.padding:
            outputs = outputs[:, :, : -self.padding]
        return outputs.transpose(1, 2)


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, heads, sequence, head_dim = inputs.shape
        if heads != self.num_heads or heads * head_dim != self.embedding_dim:
            raise ValueError("Unexpected multi-head tensor shape.")
        flattened = inputs.transpose(1, 2).reshape(batch * sequence, self.embedding_dim)
        normalized = F.group_norm(
            flattened,
            num_groups=self.num_heads,
            weight=self.weight,
            bias=None,
            eps=self.eps,
        )
        return normalized.reshape(batch, sequence, heads, head_dim).transpose(1, 2)


def parallel_stabilized_mlstm(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    input_gate_preact: torch.Tensor,
    forget_gate_preact: torch.Tensor,
    causal_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference-style stabilized parallel mLSTM memory retrieval."""

    batch, heads, sequence, head_dim = queries.shape
    dtype = queries.dtype
    device = queries.device
    causal = causal_mask[:sequence, :sequence].to(device=device)

    log_forget = F.logsigmoid(forget_gate_preact)
    cumulative = torch.cat(
        [
            torch.zeros((batch, heads, 1, 1), dtype=dtype, device=device),
            torch.cumsum(log_forget, dim=-2),
        ],
        dim=-2,
    )
    repeated = cumulative.repeat(1, 1, 1, sequence + 1)
    log_forget_matrix_full = repeated - repeated.transpose(-2, -1)
    log_forget_matrix = torch.where(
        causal,
        log_forget_matrix_full[:, :, 1:, 1:],
        torch.tensor(-float("inf"), dtype=dtype, device=device),
    )

    log_decay = log_forget_matrix + input_gate_preact.transpose(-2, -1)
    max_log_decay = log_decay.max(dim=-1, keepdim=True).values
    decay = torch.exp(log_decay - max_log_decay)

    scaled_keys = keys / math.sqrt(head_dim)
    query_key = queries @ scaled_keys.transpose(-2, -1)
    combination = query_key * decay
    normalizer = torch.maximum(
        combination.sum(dim=-1, keepdim=True).abs(),
        torch.exp(-max_log_decay),
    )
    return (combination / (normalizer + eps)) @ values


class NativeMLSTMCell(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("mLSTM embedding_dim must be divisible by num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.input_gate = nn.Linear(3 * embedding_dim, num_heads)
        self.forget_gate = nn.Linear(3 * embedding_dim, num_heads)
        self.output_norm = MultiHeadLayerNorm(embedding_dim, num_heads)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.forget_gate.weight)
        with torch.no_grad():
            self.forget_gate.bias.copy_(torch.linspace(3.0, 6.0, self.num_heads))
        nn.init.zeros_(self.input_gate.weight)
        nn.init.normal_(self.input_gate.bias, mean=0.0, std=0.1)
        nn.init.ones_(self.output_norm.weight)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = query.shape
        if sequence > self.causal_mask.shape[0]:
            raise ValueError(
                f"Sequence length {sequence} exceeds configured context {self.causal_mask.shape[0]}."
            )
        gate_input = torch.cat([query, key, value], dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, sequence, self.num_heads, -1).transpose(1, 2)

        query_heads = split_heads(query)
        key_heads = split_heads(key)
        value_heads = split_heads(value)
        input_gate = self.input_gate(gate_input).transpose(1, 2).unsqueeze(-1)
        forget_gate = self.forget_gate(gate_input).transpose(1, 2).unsqueeze(-1)
        hidden = parallel_stabilized_mlstm(
            queries=query_heads,
            keys=key_heads,
            values=value_heads,
            input_gate_preact=input_gate,
            forget_gate_preact=forget_gate,
            causal_mask=self.causal_mask,
        )
        hidden = self.output_norm(hidden)
        return hidden.transpose(1, 2).reshape(batch, sequence, self.embedding_dim)


class NativeMLSTMBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_blocks: int,
        proj_factor: float,
        qkv_proj_blocksize: int,
        conv1d_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        projected = math.ceil((proj_factor * embedding_dim) / 64) * 64
        if projected % num_heads != 0:
            raise ValueError("Projected mLSTM dimension must be divisible by num_heads.")
        if projected % qkv_proj_blocksize != 0:
            raise ValueError("Projected mLSTM dimension must be divisible by qkv_proj_blocksize.")

        self.embedding_dim = embedding_dim
        self.inner_dim = projected
        self.pre_norm = nn.LayerNorm(embedding_dim, bias=False)
        self.up_projection = nn.Linear(embedding_dim, 2 * projected, bias=False)
        self.causal_conv = CausalDepthwiseConv1d(projected, conv1d_kernel_size)
        projection_heads = projected // qkv_proj_blocksize
        self.query_projection = HeadwiseLinear(projected, projection_heads, bias=False)
        self.key_projection = HeadwiseLinear(projected, projection_heads, bias=False)
        self.value_projection = HeadwiseLinear(projected, projection_heads, bias=False)
        self.cell = NativeMLSTMCell(projected, num_heads, context_length)
        self.learnable_skip = nn.Parameter(torch.ones(projected))
        self.down_projection = nn.Linear(projected, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pre_norm.reset_parameters()
        _small_init(self.up_projection.weight, self.embedding_dim)
        _wang_init(self.down_projection.weight, self.embedding_dim, self.num_blocks)
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
        ):
            _small_init(projection.weight, self.embedding_dim)
        self.causal_conv.convolution.reset_parameters()
        self.cell.reset_parameters()
        nn.init.ones_(self.learnable_skip)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(inputs)
        projected = self.up_projection(normalized)
        memory_branch, output_gate = projected.split(self.inner_dim, dim=-1)
        convolved = F.silu(self.causal_conv(memory_branch))
        query = self.query_projection(convolved)
        key = self.key_projection(convolved)
        value = self.value_projection(memory_branch)
        memory = self.cell(query, key, value)
        memory = memory + self.learnable_skip * convolved
        block_output = memory * F.silu(output_gate)
        block_output = self.dropout(self.down_projection(block_output))
        return inputs + block_output


class NativeMLSTMStack(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_blocks: int,
        proj_factor: float,
        qkv_proj_blocksize: int,
        conv1d_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            NativeMLSTMBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                context_length=context_length,
                num_blocks=num_blocks,
                proj_factor=proj_factor,
                qkv_proj_blocksize=qkv_proj_blocksize,
                conv1d_kernel_size=conv1d_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.post_norm = nn.LayerNorm(embedding_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return self.post_norm(outputs)


class NativeXLSTMClassifier(nn.Module):
    """mLSTM-only xLSTM sentence classifier using the native PyTorch backend."""

    backend_name = "native_pytorch_mlstm"

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
        context_length: int,
        num_blocks: int,
        num_heads: int,
        proj_factor: float,
        qkv_proj_blocksize: int,
        conv1d_kernel_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.stack = NativeMLSTMStack(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            num_blocks=num_blocks,
            proj_factor=proj_factor,
            qkv_proj_blocksize=qkv_proj_blocksize,
            conv1d_kernel_size=conv1d_kernel_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        sequence = self.stack(embedded)
        features = last_valid_state(sequence, lengths)
        return self.classifier(self.dropout(features))


def build_model(config: dict[str, Any], vocab_size: int, num_classes: int) -> nn.Module:
    model_config = config["model"]
    model_type = model_config["type"]
    common = {
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "embedding_dim": int(model_config["embedding_dim"]),
        "dropout": float(model_config["dropout"]),
    }
    if model_type in {"lstm", "bilstm"}:
        return LSTMClassifier(
            **common,
            hidden_dim=int(model_config["hidden_dim"]),
            bidirectional=model_type == "bilstm",
        )
    if model_type == "cnn":
        return MaskedCNNClassifier(
            **common,
            kernel_sizes=[int(value) for value in model_config.get("kernel_sizes", [3, 5, 7])],
            num_filters=int(model_config.get("num_filters", 128)),
        )
    if model_type == "xlstm_native":
        return NativeXLSTMClassifier(
            **common,
            context_length=int(config["data"]["max_len"]),
            num_blocks=int(model_config["num_blocks"]),
            num_heads=int(model_config["num_heads"]),
            proj_factor=float(model_config["proj_factor"]),
            qkv_proj_blocksize=int(model_config["qkv_proj_blocksize"]),
            conv1d_kernel_size=int(model_config["conv1d_kernel_size"]),
        )
    raise ValueError(f"Unsupported model type: {model_type!r}")

