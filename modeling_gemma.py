from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from modeling_siglip import SiglipVisionModel, SiglipVisionConfig


class GemmaConfig:
    """Gemma 2 (2B) defaults, matching google/gemma-2-2b / PaliGemma 2 3B."""

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=True,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        use_bidirectional_attention=None,
        **kwargs,
    ):

        super().__init__()

        self.model_type = kwargs.pop("model_type", "gemma2")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i % 2 == 0) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types


class KVCache:
    """
    Stores previous Key and Value states during autoregressive generation.
    This prevents the model from recomputing attention for old tokens on every step.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [B, Num_heads_KV, Seq_len, Head_dim]
            return self.key_cache[0].shape[-2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If cache for this layer doesn't exist, append it (prefill phase)
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Otherwise, concatenate the new token's states along the sequence dimension (decode phase)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class PaliGemmaConfig:
    """PaliGemma 2 3B-pt-224 defaults (Gemma 2 2B language backbone)."""

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257216,
        projection_dim=2304,
        hidden_size=2304,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.image_token_index = kwargs.get("image_token_id", image_token_index)
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.vocab_size = kwargs.get("_vocab_size", vocab_size)

        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = SiglipVisionConfig()
        else:
            self.vision_config = vision_config

        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        self.vision_config.projection_dim = projection_dim

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config.setdefault("pad_token_id", pad_token_id)
            self.text_config = GemmaConfig(**text_config)
        elif isinstance(text_config, GemmaConfig):
            self.text_config = text_config
        elif text_config is None:
            text_config_dict = {
                "vocab_size": self.vocab_size,
                "hidden_size": hidden_size,
                "pad_token_id": pad_token_id,
            }
            for k in [
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "hidden_activation",
                "max_position_embeddings",
                "initializer_range",
                "rms_norm_eps",
                "use_cache",
                "rope_theta",
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "eos_token_id",
                "tie_word_embeddings",
                "query_pre_attn_scalar",
                "sliding_window",
                "layer_types",
                "attn_logit_softcapping",
                "final_logit_softcapping",
                "use_bidirectional_attention",
            ]:
                if k in kwargs:
                    text_config_dict[k] = kwargs[k]
            self.text_config = GemmaConfig(**text_config_dict)
        else:
            self.text_config = text_config

        if self.text_config.use_bidirectional_attention is None:
            self.text_config.use_bidirectional_attention = True

        img_size = getattr(self.vision_config, "image_size", getattr(self.vision_config, "img_size", 224))
        patch_size = self.vision_config.patch_size
        num_patches = (img_size // patch_size) ** 2
        self.text_config.num_image_tokens = num_patches
        self.vision_config.num_img_tokens = num_patches
        self.vision_config.num_image_tokens = num_patches

    @property
    def image_token_id(self):
        return self.image_token_index

    @image_token_id.setter
    def image_token_id(self, value):
        self.image_token_index = value


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # Root Mean Square normalization stabilizes the hidden state variances
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())  # converts x to float32
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings=8192, base=10000):
        super().__init__()

        self.dim = dim
        self.base = base  # This controls the different frequencies used by RoPE
        self.max_position_embeddings = max_position_embeddings

        # Calculate the theta according to the formula theta_i = base ^ (-2i/dim) where i = 0, 1, 2, ....., dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )

        # This tells torch that inv_freq belongs to this module, but it isn't a trainable parameter
        # So it won't be updated by gradient descent
        # It's just a precomputed tensor used during the forward pass
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Broadcast shapes for vectorized matrix multiplication
        # (Dim/2) => (1, Dim/2 , 1) => (B, Dim/2, 1)
        inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )

        # (B, Seq_len) => (B, 1, Seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type

        device_type = device_type if isinstance(device_type, str) else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # Calculate rotation angles (m * theta)
            # (B, Dim/2, 1) @ (B, 1, Seq_len) => (B, Seq_len, Head_dim/2)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)

            # (B, Seq_len, Head_dim)
            emb = torch.cat((freqs, freqs), dim=-1)

            # compute the element-wise sine and cosine of the calculated angle tensor
            # (B, Seq_len, Head_dim)
            cos = emb.cos()
            # (B, Seq_len, Head_dim)
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Splits the embedding dimension in half and negates the second half to construct the 2D rotation matrix
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # Adds a dimension so cos/sin [B, Seq_len, Head_Dim] can broadcast across num_heads
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


def repeat_kv(n_rep: int, hidden_states: torch.Tensor) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        # Gemma 2 scales attention by query_pre_attn_scalar, not head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.scale = self.scaling
        self.sliding_window = config.sliding_window
        self.is_causal = True
        if config.layer_types is not None and layer_idx is not None:
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"

        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.size()

        # (B, Seq_len, Head_dim) => (B, Seq_len, num_heads * head_dim)
        query_states = self.q_proj(hidden_states)
        # (B, Seq_len, Head_dim) => (B, Seq_len, num_kv_heads * head_dim)
        key_states = self.k_proj(hidden_states)
        # (B, Seq_len, Head_dim) => (B, Seq_len, num_kv_heads * head_dim)
        value_states = self.v_proj(hidden_states)

        # (B, Num_patches, Num_heads * Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # (B, Num_patches, num_kv_heads * Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # (B, Num_patches, num_kv_heads * Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # We could have passed query_states or key_states instead of value_states
        # it makes absolutely no difference because they all share the same device and data type
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(self.num_key_value_groups, key_states)
        value_states = repeat_kv(self.num_key_value_groups, value_states)

        # Q * K^T / sqrt(d_k)
        # (B, Num_heads, Num_patches, Embed_dim) @ (B, Num_heads, Embed_dim, Num_patches) => (B, Num_heads, Num_patches, Num_patches)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )  # It comes directly from nn.Module

        # (B, Num_heads, Num_patches, Num_patches) @ (B, Num_heads, Num_patches, head_dim) => (B, Num_heads, Num_patches, head_dim)
        attn_outputs = torch.matmul(attn_weights, value_states)

        if attn_outputs.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)} but it is of size {attn_outputs.size()}"
            )

        # (B, Num_heads, Num_patches, head_dim) => (B, Num_patches, Num_heads, head_dim)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()

        # (B, Num_patches, Num_heads, head_dim) =>  (B, Num_patches, hidden_size)
        attn_outputs = attn_outputs.reshape(batch_size, seq_len, -1)

        # (B, Num_patches, embed_dim) =>  (B, Num_patches, embed_dim)
        attn_outputs = self.o_proj(attn_outputs)

        return attn_outputs, attn_weights


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states, attention_mask, position_ids, kv_cache
        )

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states + residual


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.FloatTensor:

        # (B, Seq_len, hidden_size)
        hidden_states = inputs_embeds * (self.config.hidden_size**0.5)

        for decoder_layer in self.layers:
            # (B, Seq_len, hidden_size)
            hidden_states = decoder_layer(
                hidden_states, attention_mask, position_ids, kv_cache
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        kv_cache: KVCache,
    ):

        outputs = self.model(input_embeds, attention_mask, position_ids, kv_cache)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.float()
        return_data = {"logits": logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, config.vision_config.projection_dim
        )

    def forward(self, img_feats):
        return self.linear(img_feats)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)

        # SigLIP and Gemma don't necessarily use the same embedding dimension
        # So the projector does approximately: SigLIP features -> Linear layers -> Gemma-sized features
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    # a language-model optimization where the input embedding weights and output vocabulary projection can share weights
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        # (Batch, Num_Patches, Embed_Dim)
        embed_dim = inputs_embeds.shape[-1]

        batch_size, seq_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # a scaling step so that the magnitude of the image embeddings is compatible with the model's embedding scale
        # (B, Seq_len, Hidden_size)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        final_embeddings = torch.zeros(
            batch_size, seq_len, embed_dim, dtype=dtype, device=device
        )

        text_mask = input_ids != self.config.image_token_index
        if self.pad_token_id is not None and self.pad_token_id >= 0:
            padding_mask = input_ids == self.pad_token_id
            text_mask = text_mask & (~padding_mask)
        else:
            padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_mask = input_ids == self.config.image_token_index

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        # (B,S) -> (B,S,1) -> (B,S, embed_dim)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # (B,S) -> (B,S,1) -> (B,S, embed_dim)
        pad_mask_expanded = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # (B,S) -> (B,S,1) -> (B,S, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Inject text embeddings
        final_embeddings = torch.where(
            text_mask_expanded, inputs_embeds, final_embeddings
        )

        if scaled_image_features.numel() > 0 and image_mask.any():
            final_embeddings = final_embeddings.masked_scatter(
                image_mask_expanded, scaled_image_features
            )
        # Zero out padding tokens

        final_embeddings = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embeddings), final_embeddings
        )

        min_dtype = torch.finfo(dtype).min

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: full 2D bidirectional attention for prefix tokens
            causal_mask = torch.full(
                (batch_size, 1, seq_len, seq_len),
                fill_value=0.0,
                dtype=dtype,
                device=device,
            )
            if attention_mask is not None:
                # Expand attention_mask (B, S) -> (B, 1, 1, S)
                pad_mask = (attention_mask[:, None, None, :] == 0)
                causal_mask = causal_mask.masked_fill(pad_mask, min_dtype)

            position_ids = (
                (attention_mask.cumsum(-1) - 1)
                .masked_fill_((attention_mask == 0), 0)
                .to(device)
            )
        else:
            assert seq_len == 1
            total_kv_len = kv_cache.num_items() + 1
            if attention_mask is None or attention_mask.shape[-1] < total_kv_len:
                attention_mask = torch.ones(
                    (batch_size, total_kv_len), dtype=torch.int64, device=device
                )
            kv_len = attention_mask.shape[-1]
            causal_mask = torch.full(
                (batch_size, 1, 1, kv_len), fill_value=0.0, dtype=dtype, device=device
            )
            pad_mask = (attention_mask[:, None, None, :] == 0)
            causal_mask = causal_mask.masked_fill(pad_mask, min_dtype)

            position_ids = (
                (attention_mask.cumsum(-1)[:, -1:] - 1)
                .masked_fill_((attention_mask[:, -1:] == 0), 0)
                .to(device)
            )

        return final_embeddings, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            if kv_cache is not None and kv_cache.num_items() > 0:
                attention_mask = torch.ones(
                    (batch_size, kv_cache.num_items() + 1),
                    device=input_ids.device,
                    dtype=torch.int64,
                )
            else:
                attention_mask = torch.ones_like(input_ids)

        # appending (input_ids) to the end immediately feeds the sequence of token integers into
        # that retrieved embedding layer to convert them into high-dimensional vectors.
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            selected_image_feature = self.vision_tower(
                pixel_values.to(input_embeds.dtype)
            )
            image_features = self.multi_modal_projector(selected_image_feature)
        else:
            image_features = torch.zeros(
                (batch_size, 0, self.config.vision_config.projection_dim),
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            )

        final_embeddings, causal_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, input_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            input_embeds=final_embeddings,
            kv_cache=kv_cache,
        )

        return outputs
