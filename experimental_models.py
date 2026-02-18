"""Experimental image and multimodal models for NeuroDSL Omni mode."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.block(x) + x)


class ImageAutoencoder(nn.Module):
    """Compact convolutional autoencoder for creative image-mode experiments."""

    def __init__(self, image_size: int = 32, latent_dim: int = 128, channels: int = 3):
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4 for this autoencoder.")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = channels
        hidden = 64
        reduced = image_size // 4

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(hidden),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(hidden * 2),
            nn.Flatten(),
            nn.Linear((hidden * 2) * reduced * reduced, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, (hidden * 2) * reduced * reduced)
        self.decoder = nn.Sequential(
            ResidualConvBlock(hidden * 2),
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(hidden),
            nn.ConvTranspose2d(hidden, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        reduced = self.image_size // 4
        x = self.decoder_fc(z).view(z.shape[0], 128, reduced, reduced)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class CrossGatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.img_to_vec = nn.Linear(dim, dim, bias=False)
        self.vec_to_img = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, img_feat, vec_feat):
        g1 = torch.sigmoid(self.img_to_vec(img_feat))
        g2 = torch.sigmoid(self.vec_to_img(vec_feat))
        fused = torch.cat([img_feat * g2, vec_feat * g1], dim=-1)
        return self.out(fused)


class MultiModalFusionModel(nn.Module):
    """Simple multimodal model: image encoder + metadata vector encoder + gated fusion."""

    def __init__(self, image_size: int = 32, vec_dim: int = 16, hidden_dim: int = 128, out_dim: int = 8):
        super().__init__()
        self.image_size = int(image_size)
        self.vec_dim = int(vec_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )
        self.vec_encoder = nn.Sequential(
            nn.Linear(vec_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fusion = CrossGatedFusion(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image, vector):
        img_feat = self.image_encoder(image)
        vec_feat = self.vec_encoder(vector)
        fused = self.fusion(img_feat, vec_feat)
        return self.head(fused)


class SyntheticImageDataset(Dataset):
    """Procedural dataset that creates uncanny stripe/swirl composites."""

    def __init__(self, n_samples: int = 1024, image_size: int = 32, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        size = self.image_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size),
            torch.linspace(-1.0, 1.0, size),
            indexing="ij",
        )
        phase = torch.rand(1, generator=g).item() * 2 * math.pi
        f1 = torch.rand(1, generator=g).item() * 8 + 2
        f2 = torch.rand(1, generator=g).item() * 8 + 2

        r = torch.sin(f1 * xx + phase) * 0.5 + 0.5
        gch = torch.cos(f2 * yy - phase) * 0.5 + 0.5
        swirl = torch.sin((xx * xx + yy * yy) * (f1 + f2) + phase) * 0.5 + 0.5
        img = torch.stack([r, gch, swirl], dim=0).clamp(0, 1).float()
        return img


class SyntheticMultiModalDataset(Dataset):
    def __init__(self, n_samples: int = 2048, image_size: int = 32, vec_dim: int = 16, out_dim: int = 8, seed: int = 123):
        super().__init__()
        self.image_ds = SyntheticImageDataset(n_samples=n_samples, image_size=image_size, seed=seed)
        self.vec_dim = vec_dim
        self.out_dim = out_dim
        self.seed = seed

    def __len__(self):
        return len(self.image_ds)

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx * 7)
        img = self.image_ds[idx]
        vec = torch.randn(self.vec_dim, generator=g)
        # Create a synthetic target from both modalities.
        pooled = img.mean(dim=(1, 2))
        y = torch.cat([pooled, vec[: max(0, self.out_dim - 3)]], dim=0)[: self.out_dim]
        y = torch.tanh(y)
        return img, vec.float(), y.float()


@dataclass
class ExperimentalTrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    mixup_alpha: float = 0.0
    latent_noise: float = 0.03
    ema_decay: float = 0.995
    use_ema: bool = True


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            state = model.state_dict()
            for k, v in state.items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


def _mixup(images: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return images
    b = images.shape[0]
    lam = torch.distributions.Beta(alpha, alpha).sample((1,)).item()
    idx = torch.randperm(b, device=images.device)
    return lam * images + (1.0 - lam) * images[idx]


def _chaotic_latent(z: torch.Tensor, strength: float = 0.2, steps: int = 3) -> torch.Tensor:
    if strength <= 0 or steps <= 0:
        return z
    x = z
    for _ in range(steps):
        # Logistic map style perturbation for uncanny sample diversity.
        x = 3.77 * x * (1 - x.tanh())
    return z + strength * torch.tanh(x)


def train_image_autoencoder(
    model: ImageAutoencoder,
    config: ExperimentalTrainConfig,
    device,
    dataset: Optional[Dataset] = None,
    stop_flag: Optional[List[bool]] = None,
) -> List[float]:
    model.to(device)
    model.train()
    ds = dataset or SyntheticImageDataset(n_samples=2048, image_size=model.image_size)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

    history: List[float] = []
    for _epoch in range(config.epochs):
        running = 0.0
        steps = 0
        for images in loader:
            if stop_flag and stop_flag[0]:
                break
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(device)
            images = _mixup(images, config.mixup_alpha)
            if config.latent_noise > 0:
                images = (images + torch.randn_like(images) * config.latent_noise).clamp(0, 1)
            opt.zero_grad(set_to_none=True)

            recon = model(images)
            rec_loss = F.mse_loss(recon, images)
            # Edge/texture encouraging term for visually richer reconstructions.
            lap = (
                torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]).mean()
                + torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]).mean()
            )
            loss = rec_loss + 0.05 * lap
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)

            running += float(loss.item())
            steps += 1

        if steps > 0:
            history.append(running / steps)
        if stop_flag and stop_flag[0]:
            break

    if ema is not None:
        ema.apply_to(model)
    return history


def train_multimodal(
    model: MultiModalFusionModel,
    config: ExperimentalTrainConfig,
    device,
    dataset: Optional[Dataset] = None,
    stop_flag: Optional[List[bool]] = None,
) -> List[float]:
    model.to(device)
    model.train()
    ds = dataset or SyntheticMultiModalDataset(
        image_size=getattr(model, "image_size", 32),
        vec_dim=getattr(model, "vec_dim", 16),
        out_dim=getattr(model, "out_dim", 8),
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: List[float] = []

    for _epoch in range(config.epochs):
        running = 0.0
        steps = 0
        for image, vector, target in loader:
            if stop_flag and stop_flag[0]:
                break
            image = image.to(device)
            vector = vector.to(device)
            target = target.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(image, vector)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            steps += 1

        if steps > 0:
            history.append(running / steps)
        if stop_flag and stop_flag[0]:
            break
    return history


def generate_uncanny_images(
    model: ImageAutoencoder,
    n_samples: int = 8,
    chaos_strength: float = 0.2,
    seed: int = 7,
    device="cpu",
) -> torch.Tensor:
    model.to(device)
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.rand(n_samples, model.latent_dim, generator=g, device=device)
    z = _chaotic_latent(z, strength=chaos_strength, steps=4)
    with torch.no_grad():
        out = model.decode(z).clamp(0, 1)
    return out.detach().cpu()


def generate_interpolation_images(
    model: ImageAutoencoder,
    steps: int = 8,
    seed_a: int = 7,
    seed_b: int = 19,
    device="cpu",
) -> torch.Tensor:
    model.to(device)
    model.eval()
    g1 = torch.Generator(device=device).manual_seed(seed_a)
    g2 = torch.Generator(device=device).manual_seed(seed_b)
    z1 = torch.rand(1, model.latent_dim, generator=g1, device=device)
    z2 = torch.rand(1, model.latent_dim, generator=g2, device=device)

    frames = []
    with torch.no_grad():
        for i in range(max(2, steps)):
            t = i / float(max(1, steps - 1))
            z = (1.0 - t) * z1 + t * z2
            img = model.decode(z).clamp(0, 1)
            frames.append(img)
    return torch.cat(frames, dim=0).cpu()


def text_to_feature_vector(text: str, dim: int) -> torch.Tensor:
    """Deterministic lightweight text embedding by hashed character n-grams."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be > 0")
    vec = torch.zeros(dim, dtype=torch.float32)
    if not text:
        return vec
    t = text.lower().strip()
    # 1-gram and 2-gram hashing
    for i, ch in enumerate(t):
        idx = (ord(ch) * 31 + i * 17) % dim
        vec[idx] += 1.0
        if i + 1 < len(t):
            idx2 = (ord(ch) * 13 + ord(t[i + 1]) * 19 + i * 7) % dim
            vec[idx2] += 0.7
    vec = vec / vec.norm(p=2).clamp_min(1e-6)
    return vec


def load_images_from_folder(path: str, image_size: int = 32, max_images: int = 2000) -> TensorDataset:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for folder image loading. Install with: pip install pillow") from exc

    if not os.path.isdir(path):
        raise ValueError(f"Image folder does not exist: {path}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    items = []
    for root, _, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname.lower())[1] in exts:
                items.append(os.path.join(root, fname))
                if len(items) >= max_images:
                    break
        if len(items) >= max_images:
            break

    if not items:
        raise ValueError(f"No image files found in folder: {path}")

    tensors = []
    for fp in items:
        with Image.open(fp) as im:
            im = im.convert("RGB").resize((image_size, image_size))
            arr = torch.from_numpy(__import__("numpy").array(im)).float() / 255.0
            arr = arr.permute(2, 0, 1).contiguous()
            tensors.append(arr)
    stacked = torch.stack(tensors, dim=0)
    return TensorDataset(stacked)


def save_image_grid(images: torch.Tensor, output_path: str, cols: int = 4):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for saving image grids. Install with: pip install pillow") from exc

    n, c, h, w = images.shape
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    canvas = torch.zeros(3, rows * h, cols * w)
    for i in range(n):
        r = i // cols
        cc = i % cols
        canvas[:, r * h : (r + 1) * h, cc * w : (cc + 1) * w] = images[i][:3]
    arr = (canvas.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(output_path)
