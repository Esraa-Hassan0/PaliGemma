import torch
import torch.nn as nn
from typing import Optional, Tuple


class SiglipVisionConfig:
    """
    Configuration class storing hyperparameters for the SigLIP Vision model.
    """
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        img_size=224,
        patch_size=16,
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        num_img_tokens: int = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_img_tokens = num_img_tokens
        self.img_size = img_size
        self.num_attention_heads = num_attention_heads


class SiglipVisionEmbeddings(nn.Module):
    """
    Constructs the patch embeddings and adds positional encodings
    Converts 2D image tensors into a 1D sequence of patch embeddings
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.img_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # This indicates no padding is added
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [B, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)

        # [B, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [B, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)

        # [B, Embed_Dim, Num_Patches] => [B, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)

        # (B, Num_patches, embed_dim) + (1, Num_patches, embed_dim) => (B, Num_patches, embed_dim)
        # The positional concept of "top-left patch" is identical for every image in any batch
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # (B, Num_patches, embed_dim)
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed self-attention block
    Computes scaled dot-product attention over the patch embeddings
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.size()

        # (B, Num_patches, Embed_dim)
        query_states = self.q_proj(hidden_states)

        # (B, Num_patches, Embed_dim)
        key_states = self.k_proj(hidden_states)

        # (B, Num_patches, Embed_dim)
        value_states = self.v_proj(hidden_states)

        # (B, Num_heads, Num_patches, Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (B, Num_heads, Num_patches, Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (B, Num_heads, Num_patches, Embed_dim) =>  (B, Num_patches, Num_heads, Embed_dim)  =>  (B, Num_heads, Num_patches, head_dim)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Q * K^T / sqrt(d_k)
        # (B, Num_heads, Num_patches, Embed_dim) @ (B, Num_heads, Embed_dim, Num_patches) => (B, Num_heads, Num_patches, Num_patches)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but it is of size {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # It comes directly from nn.Module

        # (B, Num_heads, Num_patches, Num_patches) @ (B, Num_heads, Num_patches, head_dim) => (B, Num_heads, Num_patches, head_dim)
        attn_outputs = torch.matmul(attn_weights, value_states)

        if attn_outputs.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)} but it is of size {attn_outputs.size()}"
            )

        # (B, Num_heads, Num_patches, head_dim) => (B, Num_patches, Num_heads, head_dim)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()

        # (B, Num_patches, Num_heads, head_dim) =>  (B, Num_patches, embed_dim)
        attn_outputs = attn_outputs.reshape(batch_size, seq_len, self.embed_dim)

        # (B, Num_patches, embed_dim) =>  (B, Num_patches, embed_dim)
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs, attn_weights


class SiglipMLP(nn.Module):
    """
    Multilayer perceptron block applied after attention in each transformer layer.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    A single ViT layer containing Layer Norms, Self-Attention, and the MLP block
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # (B, Num_patches, embed_dim) -> (B, Num_patches, embed_dim)
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    A stack of SiglipEncoderLayers constituting the main body of the Vision Transformer
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (B, Num_patches, embed_dim)
        residual = hidden_states

        # (B, Num_patches, embed_dim) => (B, Num_patches, embed_dim)
        hidden_states = self.layer_norm1(hidden_states)

        # (B, Num_patches, embed_dim) => (B, Num_patches, embed_dim)
        hidden_states, _ = self.self_attn(hidden_states)

        # (B, Num_patches, embed_dim)
        hidden_states = hidden_states + residual

        # (B, Num_patches, embed_dim)
        residual = hidden_states

        # (B, Num_patches, embed_dim) => (B, Num_patches, embed_dim)
        hidden_states = self.layer_norm2(hidden_states)

        # (B, Num_patches, embed_dim) => (B, Num_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)

        # (B, Num_patches, embed_dim)
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    The complete Vision Transformer combining patch embeddings and the encoder stack
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps= config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        # (B, C, H, W) => (B, Num_patches, embed_dim)
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):
    """
    Top-level wrapper for the SigLIP Vision architecture
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # (B, C, Height, Width) -> (B, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)
