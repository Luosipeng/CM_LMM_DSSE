"""Causal-guided multimodal reconstruction model backed by local Qwen3.5."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .dgp import DenseGraphPropagation


def load_qwen_backbone(
    model_id_or_path: str,
    *,
    dtype: str = "bfloat16",
    use_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    local_files_only: bool = True,
) -> tuple[Any, nn.Module]:
    """Load Qwen3.5 and optionally inject LoRA without changing base weights.

    Mathematical model: each selected frozen matrix W0 is used as
    ``W = W0 + (alpha/r) B A``, with A in R^(r x k), B in R^(d x r). Only A
    and B receive gradients. With LoRA disabled, every Qwen parameter is frozen.
    """
    from modelscope import snapshot_download

    # ModelScope and Hugging Face use different cache layouts. Resolve the
    # local ModelScope snapshot first because the supplied Qwen deployment uses
    # snapshot_download in the existing environment checks.
    resolved_path = model_id_or_path
    if "/" in model_id_or_path and not torch.jit.is_scripting():
        try:
            resolved_path = snapshot_download(
                model_id_or_path, local_files_only=local_files_only
            )
        except Exception:
            if local_files_only:
                raise
    torch_dtype = getattr(torch, dtype)
    processor = AutoProcessor.from_pretrained(
        resolved_path, local_files_only=local_files_only
    )
    processor.tokenizer.padding_side = "left"
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        resolved_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if use_lora:
        # Long sensor windows create hundreds of numerical tokens. Activation
        # checkpointing trades additional compute for substantially lower
        # memory while preserving the frozen-base LoRA optimization problem.
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_config)
    return processor, backbone


class SpatioTemporalEmbedding(nn.Module):
    """Embed sensor-time tokens before graph propagation."""

    def __init__(
        self,
        feature_count: int,
        sensor_count: int,
        hidden_size: int,
        max_time_steps: int = 4096,
    ) -> None:
        super().__init__()
        self.value_projection = nn.Linear(feature_count, hidden_size)
        self.sensor_embedding = nn.Embedding(sensor_count, hidden_size)
        self.time_embedding = nn.Embedding(max_time_steps, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_steps, sensors, _ = values.shape
        sensor_ids = torch.arange(sensors, device=values.device)
        time_ids = torch.arange(time_steps, device=values.device)
        output = self.value_projection(values)
        output = output + self.sensor_embedding(sensor_ids)[None, None, :, :]
        output = output + self.time_embedding(time_ids)[None, :, None, :]
        return self.norm(output)


class CMLLM(nn.Module):
    """Paper CM-LLM spatialized to sensor-time tokens.

    If X has shape B x T x N x F, temporal and DGP causal embeddings are
    combined by element-wise addition, flattened to T*N tokens, and appended
    after prompt embeddings. Qwen hidden states for these tokens are projected
    back to B x T x N x F.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        feature_count: int,
        sensor_count: int,
        adapter_hidden_size: int = 256,
        dgp_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        text_config = getattr(backbone.config, "text_config", backbone.config)
        llm_hidden = int(text_config.hidden_size)
        self.temporal = SpatioTemporalEmbedding(
            feature_count, sensor_count, adapter_hidden_size
        )
        self.causal = DenseGraphPropagation(
            adapter_hidden_size,
            dgp_layers,
            dropout,
            feature_count=feature_count,
        )
        self.to_llm = nn.Linear(adapter_hidden_size, llm_hidden)
        self.output_projection = nn.Linear(llm_hidden, feature_count)
        self.sensor_count = sensor_count
        self.feature_count = feature_count

    def _tokenize_prompts(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    def forward(
        self,
        values: torch.Tensor,
        causal_graph: torch.Tensor | dict[str, torch.Tensor],
        prompts: list[str],
        valid_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, time_steps, sensors, _ = values.shape
        if sensors != self.sensor_count:
            raise ValueError(f"Expected {self.sensor_count} sensors, got {sensors}")
        temporal = self.temporal(values)
        causal = self.causal(temporal, causal_graph, values)
        fused = self.to_llm(temporal + causal).reshape(batch, time_steps * sensors, -1)

        prompt_tokens = self._tokenize_prompts(prompts, values.device)
        prompt_embeddings = self.backbone.get_input_embeddings()(prompt_tokens["input_ids"])
        inputs_embeds = torch.cat((prompt_embeddings, fused), dim=1)
        if valid_time is None:
            series_attention = torch.ones(
                batch, time_steps * sensors, dtype=torch.long, device=values.device
            )
        else:
            series_attention = valid_time.repeat_interleave(sensors, dim=1).long()
        attention_mask = torch.cat(
            (prompt_tokens["attention_mask"].long(), series_attention), dim=1
        )
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1][:, -time_steps * sensors :, :]
        return self.output_projection(hidden).reshape(
            batch, time_steps, sensors, self.feature_count
        )


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return an auditable trainable/frozen parameter inventory."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_fraction": trainable / max(total, 1),
    }
