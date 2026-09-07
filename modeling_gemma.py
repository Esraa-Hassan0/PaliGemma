from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from modeling_siglip import SiglipVisionModel, SiglipVisionConfig


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.attention_bias = attention_bias
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout


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
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        vision_config.pop("model_type", None)
        if isinstance(vision_config, dict):
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
            self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        elif isinstance(text_config, GemmaConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = GemmaConfig(pad_token_id=pad_token_id)
        else:
            self.text_config = text_config

        self.text_config.num_image_tokens = (
            self.vision_config.img_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.num_img_tokens = self.text_config.num_image_tokens


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
    def __init__(self, dim: int, max_position_embeddings=2048, base=10000):
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
        # Ensure frequency tensor is on the correct GPU/CPU device
        self.inv_freq.to(x.device)

        # Broadcast shapes for vectorized matrix multiplication
        # (Dim/2) => (1, Dim/2 , 1) => (B, Dim/2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
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
        self.scale = self.head_dim**-0.5
        self.is_causal = True

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
        hidden_states = inputs_embeds

        # (B, Seq_len, hidden_size)
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )

        hidden_states = hidden_states * normalizer

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
        _, _, embed_dim = image_features.shape

        batch_size, seq_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # a scaling step so that the magnitude of the image embeddings is compatible with the model's embedding scale
        # (B, Seq_len, Hidden_size)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        final_embeddings = torch.zeros(
            batch_size, seq_len, embed_dim, dtype=dtype, device=device
        )

        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.config.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        padding_mask = input_ids == self.config.pad_token_id

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

        final_embeddings = final_embeddings.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        # Zero out padding tokens

        final_embeddings = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embeddings), final_embeddings
        )

        # dtype, device = inputs_embeds.dtype, inputs_embeds.device

        min_dtype = torch.finfo(dtype).min

        if kv_cache is None or kv_cache.num_items() == 0:
            # Don't mask any token cause we are in the prefill phase
            # this only works when we have no padding
            # Mask out future tokens (Causal mapping)
            causal_mask = torch.full(
                (batch_size, seq_len, seq_len),
                fill_value=0,
                dtype=dtype,
                device=device,
            )
            # causal_mask = torch.full((batch_size, seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
            # causal_mask = torch.triu(causal_mask, diagonal=1)

        else:
            assert seq_len == 1
            kv_len = kv_cache.num_items() + seq_len
            # In decode phase, attend to all previous tokens
            causal_mask = torch.full(
                (batch_size, seq_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # (B, Seq_len, kv_cache) => (B, Num_heads, seq_len, kv_cache)
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # position_ids = attention_mask.cumsum(-1)[:, -1]
            # if position_ids.dim() == 1:
            #     position_ids = position_ids.unsqueeze(0)
            # kv_cache.num_items() already equals current sequence length (0-indexed position)
            position_ids = torch.tensor([[kv_cache.num_items()]], device=device)

        else:
            position_ids = (
                (attention_mask.cumsum(-1) - 1)
                .masked_fill_((attention_mask == 0), 0)
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

        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # appending (input_ids) to the end immediately feeds the sequence of token integers into
        # that retrieved embedding layer to convert them into high-dimensional vectors.
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        batch_size, seq_len = input_ids.shape

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
