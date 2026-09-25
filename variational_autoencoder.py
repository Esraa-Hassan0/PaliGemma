"""
PaliGemma OID VQ-VAE Decoder
==============================
Architecture is inferred directly from the saved .npy weight files in data/vae-oid/:

  decoder.0   : Conv2d(512 → 128, 1×1)         (bias: 128)
  decoder.1   : ReLU  (implicit)
  decoder.2   : ResidualBlock(128)
  decoder.3   : ResidualBlock(128)
  decoder.4   : ConvTranspose2d(128 → 128, 4, stride=2, padding=1)
  decoder.5   : ReLU
  decoder.6   : ConvTranspose2d(128 → 64,  4, stride=2, padding=1)  weight shape (128,64,4,4)
  decoder.7   : ReLU
  decoder.8   : ConvTranspose2d(64  → 32,  4, stride=2, padding=1)  weight shape (64,32,4,4)
  decoder.9   : ReLU
  decoder.10  : ConvTranspose2d(32  → 16,  4, stride=2, padding=1)  weight shape (32,16,4,4)
  decoder.11  : ReLU
  decoder.12  : Conv2d(16 → 1, 1×1)

Input:  (B, 16) int token indices in [0, 127]
Hidden: embed each to 512-d, reshape to (B, 512, 4, 4)
Output: (B, 1, 64, 64) float in [0, 1] after sigmoid
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Three-layer residual block matching the OID VAE weights."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # .net.0
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # .net.2
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),              # .net.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PaliGemmaVAEDecoder(nn.Module):
    """
    VQ-VAE decoder used by PaliGemma for segmentation.

    Takes 16 codebook indices, looks them up in a learned embedding table
    (vocab_size=128, embed_dim=512), arranges them into a 4×4 feature grid,
    then upsamples to a 64×64 binary mask via transposed convolutions.
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        # Codebook lookup: maps each token id → 512-d vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.decoder = nn.Sequential(
            # 0  – project embed_dim → 128 channels
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            # 1
            nn.ReLU(),
            # 2, 3 – residual refinement
            ResidualBlock(128),
            ResidualBlock(128),
            # 4 – 4×4 → 8×8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            # 5
            nn.ReLU(),
            # 6 – 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # 7
            nn.ReLU(),
            # 8 – 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # 9
            nn.ReLU(),
            # 10 – 32×32 → 64×64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # 11
            nn.ReLU(),
            # 12 – reduce to single-channel mask
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16) int64 tensor of codebook indices.

        Returns:
            mask: (B, 1, 64, 64) float32 tensor with values in [0, 1].
        """
        # (B, 16) → (B, 16, embed_dim)
        x = self.embedding(x)
        # Reshape 16 tokens to spatial 4×4 grid: (B, embed_dim, 4, 4)
        x = x.view(-1, 4, 4, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        s = self.decoder(x)          # (B, 1, 64, 64)
        return torch.sigmoid(s)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

def load_vae_decoder(vae_dir: str, device: str = "cpu") -> PaliGemmaVAEDecoder:
    """
    Instantiate PaliGemmaVAEDecoder and load weights from the .npy files
    stored in *vae_dir*.

    Expected files (subset that is loaded):
        _vq_vae._embedding.npy          → embedding.weight  (128, 512)
        decoder.0.weight / bias.npy     → decoder[0]
        decoder.2.net.{0,2,4}.weight/bias → decoder[2].net
        decoder.3.net.{0,2,4}.weight/bias → decoder[3].net
        decoder.4.weight / bias.npy     → decoder[4]
        decoder.6.weight / bias.npy     → decoder[6]
        decoder.8.weight / bias.npy     → decoder[8]
        decoder.10.weight / bias.npy    → decoder[10]
        decoder.12.weight / bias.npy    → decoder[12]
    """

    def t(fname: str) -> torch.Tensor:
        """Load a .npy file from vae_dir as a float32 tensor."""
        return torch.from_numpy(
            np.load(os.path.join(vae_dir, fname)).astype(np.float32)
        )

    model = PaliGemmaVAEDecoder(vocab_size=128, embed_dim=512)

    with torch.no_grad():
        # ---- codebook embedding -----------------------------------------
        model.embedding.weight.copy_(t("_vq_vae._embedding.npy"))

        # ---- decoder.0  Conv2d(512→128, 1×1) ----------------------------
        model.decoder[0].weight.copy_(t("decoder.0.weight.npy"))
        model.decoder[0].bias.copy_(t("decoder.0.bias.npy"))

        # ---- decoder.2  ResidualBlock ------------------------------------
        model.decoder[2].net[0].weight.copy_(t("decoder.2.net.0.weight.npy"))
        model.decoder[2].net[0].bias.copy_(  t("decoder.2.net.0.bias.npy"))
        model.decoder[2].net[2].weight.copy_(t("decoder.2.net.2.weight.npy"))
        model.decoder[2].net[2].bias.copy_(  t("decoder.2.net.2.bias.npy"))
        model.decoder[2].net[4].weight.copy_(t("decoder.2.net.4.weight.npy"))
        model.decoder[2].net[4].bias.copy_(  t("decoder.2.net.4.bias.npy"))

        # ---- decoder.3  ResidualBlock ------------------------------------
        model.decoder[3].net[0].weight.copy_(t("decoder.3.net.0.weight.npy"))
        model.decoder[3].net[0].bias.copy_(  t("decoder.3.net.0.bias.npy"))
        model.decoder[3].net[2].weight.copy_(t("decoder.3.net.2.weight.npy"))
        model.decoder[3].net[2].bias.copy_(  t("decoder.3.net.2.bias.npy"))
        model.decoder[3].net[4].weight.copy_(t("decoder.3.net.4.weight.npy"))
        model.decoder[3].net[4].bias.copy_(  t("decoder.3.net.4.bias.npy"))

        # ---- decoder.4  ConvTranspose2d(128→128) -------------------------
        model.decoder[4].weight.copy_(t("decoder.4.weight.npy"))
        model.decoder[4].bias.copy_(  t("decoder.4.bias.npy"))

        # ---- decoder.6  ConvTranspose2d(128→64) --------------------------
        model.decoder[6].weight.copy_(t("decoder.6.weight.npy"))
        model.decoder[6].bias.copy_(  t("decoder.6.bias.npy"))

        # ---- decoder.8  ConvTranspose2d(64→32) ---------------------------
        model.decoder[8].weight.copy_(t("decoder.8.weight.npy"))
        model.decoder[8].bias.copy_(  t("decoder.8.bias.npy"))

        # ---- decoder.10 ConvTranspose2d(32→16) ---------------------------
        model.decoder[10].weight.copy_(t("decoder.10.weight.npy"))
        model.decoder[10].bias.copy_(  t("decoder.10.bias.npy"))

        # ---- decoder.12 Conv2d(16→1, 1×1) --------------------------------
        model.decoder[12].weight.copy_(t("decoder.12.weight.npy"))
        model.decoder[12].bias.copy_(  t("decoder.12.bias.npy"))

    return model.to(device).eval()
