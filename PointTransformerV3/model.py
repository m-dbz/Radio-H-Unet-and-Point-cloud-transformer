from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptv3model import PointTransformerV3, Point


@dataclass
class ModelConfig:
    # Input feature composition (xyz + engineered radio features defined in RadioFeatures)
    in_channels: int = 10

    # PointTransformerV3 hierarchical configuration (encoder + internal decoder if any)
    enc_depths: Tuple[int, ...] = (2, 3, 4, 6, 3)
    enc_channels: Tuple[int, ...] = (48, 96, 192, 384, 512)
    enc_num_heads: Tuple[int, ...] = (3, 6, 12, 24, 32)
    dec_depths: Tuple[int, ...] = (2, 2, 3, 3)
    dec_channels: Tuple[int, ...] = (96, 96, 192, 384)
    dec_num_heads: Tuple[int, ...] = (6, 6, 12, 24)
    drop_path: float = 0.1

    # Fusion & latent dims
    tx_embed_dim: int = 256
    latent_dim: int = 512
    fusion_dropout: float = 0.1

    # 2D decoder target configuration
    target_size: int = 256
    base_grid: int = 16          # must divide target_size by power-of-two factor
    base_channels: int = 256

    # Radio physics params
    frequency_ghz: float = 3.5
    enable_fresnel: bool = True

    # Output activation (None|'sigmoid'|'tanh')
    output_activation: Optional[str] = None

    # Scene pooling strategy: 'mean' | 'attn'
    pooling: Literal['mean', 'attn'] = 'attn'
    pooling_dropout: float = 0.0

    # 2D decoder positional encoding injection
    posenc2d: bool = True
    posenc_num_freqs: int = 8
    posenc_learned_projection: bool = True

    # Optional feature normalization (stabilizes scale spread for radiophysics engineered features)
    normalize_engineered: bool = True


class RadioFeatures(nn.Module):
    """Engineer deterministic radio features per point with light scale normalization.

    Raw features:
        dist3, dist2, |dz|, elev_angle, azim_angle, fspl_db, fresnel_clearance
    Normalizations:
        * Distances -> log1p for dynamic range compression.
        * Angles -> sin / cos embedding (rotation continuity) replacing raw angle values.
        * FSPL dB -> / 100 approx scaling.
    Final layout (N, 7) kept constant to avoid changing PointTransformer input dim externally.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.frequency_hz = config.frequency_ghz * 1e9
        self.wavelength = 3e8 / self.frequency_hz

    def forward(self, coords: torch.Tensor, tx_pos: torch.Tensor) -> torch.Tensor:
        diff = coords - tx_pos.unsqueeze(0)              # (N,3)
        dist3 = diff.norm(dim=1, keepdim=True)
        dist2 = diff[:, :2].norm(dim=1, keepdim=True)
        dz = diff[:, 2:3].abs()
        dist2_safe = dist2.clamp(min=1e-6)
        elev = torch.atan2(dz, dist2_safe)
        azim = torch.atan2(diff[:, 1:2], diff[:, 0:1])
        fspl_linear = (4 * math.pi * dist3 / self.wavelength).clamp(min=1e-6)
        fspl_db = 20 * torch.log10(fspl_linear)
        if self.cfg.enable_fresnel:
            fresnel_r = torch.sqrt(self.wavelength * dist3 / 2).clamp(min=1e-6)
            fresnel_clear = dz / fresnel_r
        else:
            fresnel_clear = torch.zeros_like(dz)

        if self.cfg.normalize_engineered:
            dist3_n = torch.log1p(dist3)
            dist2_n = torch.log1p(dist2)
            dz_n = torch.log1p(dz)
            
            elev_sin, elev_cos = torch.sin(elev), torch.cos(elev)
            azim_sin, azim_cos = torch.sin(azim), torch.cos(azim)
            angle_mix = 0.5 * (elev_sin + azim_cos)
            fspl_n = fspl_db / 100.0
            fresnel_n = fresnel_clear.clamp(-3, 3) / 3.0
            engineered = torch.cat([
                dist3_n, dist2_n, dz_n, angle_mix, fspl_n, fresnel_n, azim_sin 
            ], dim=1)
            return engineered

        return torch.cat([dist3, dist2, dz, elev, azim, fspl_db, fresnel_clear], dim=1)


class AttnScenePooling(nn.Module):
    """Learned attention pooling over variable-length point sets (segment softmax).
    For each scene (segment defined by offsets), compute scalar scores -> softmax -> weighted sum.
    """
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feats: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(len(offsets) - 1):
            s, e = offsets[b].item(), offsets[b + 1].item()
            seg = feats[s:e]
            w = self.score(seg).squeeze(-1)
            w = torch.softmax(w, dim=0)
            pooled = (seg * w.unsqueeze(-1)).sum(dim=0, keepdim=True)
            out.append(self.drop(pooled))
        return torch.cat(out, dim=0)


class FiLMFusion(nn.Module):
    """FiLM-based fusion of global point embedding and Tx embedding."""
    def __init__(self, point_dim: int, tx_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.film = nn.Linear(latent_dim, 2 * latent_dim)  # -> gamma | beta
        self.refine = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, point_global: torch.Tensor, tx_embed: torch.Tensor) -> torch.Tensor:
        p = self.point_proj(point_global)
        t = self.tx_proj(tx_embed)
        gamma, beta = self.film(t).chunk(2, dim=-1)
        fused = (1 + gamma.tanh()) * p + beta  # multiplicative + additive modulation
        fused = fused + self.refine(fused)      # residual refinement
        return fused


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (normalizes over H,W per channel)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.point = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.point(self.depth(x)))


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, channels: int):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(channels)

    def forward(self, x):
        return x + self.fn(self.norm(x))


class PosEnc2D(nn.Module):
    """Sinusoidal 2D positional encoding with learnable projection."""
    def __init__(self, num_freqs: int, out_channels: int, learned_projection: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_channels = out_channels
        self.learned_projection = learned_projection
        pe_dim = num_freqs * 4  # sin/cos x + sin/cos y
        if learned_projection:
            self.proj = nn.Linear(pe_dim, out_channels)
        else:
            assert pe_dim == out_channels, "Without projection, pe_dim must match out_channels"

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        freqs = 2 ** torch.arange(self.num_freqs, device=device).float()
        enc = []
        for f in freqs:
            enc.extend([torch.sin(f * math.pi * xx), torch.cos(f * math.pi * xx),
                        torch.sin(f * math.pi * yy), torch.cos(f * math.pi * yy)])
        pe = torch.stack(enc, dim=-1)  # (H,W,pe_dim)
        if self.learned_projection:
            pe = self.proj(pe)
        return pe.permute(2, 0, 1)  # (C,H,W)


class Decoder2D(nn.Module):
    """Latent vector -> 2D map via progressive transpose convs + optional 
    2D positional encoding at each scale.
    """
    def __init__(self, latent_dim: int, base_channels: int, base_grid: int, target_size: int,
                 out_activation: Optional[str], posenc_cfg: Optional[dict]):
        super().__init__()
        self.target_size = target_size
        self.base_grid = base_grid
        self.out_activation = out_activation
        num_ups = int(math.log2(target_size // base_grid))
        assert 2 ** num_ups * base_grid == target_size, "target_size/base_grid must be power of 2"

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, base_channels * base_grid * base_grid),
            nn.LayerNorm(base_channels * base_grid * base_grid),
            nn.GELU(),
        )
        channels = [base_channels // (2 ** i) for i in range(num_ups + 1)]
        self.up_blocks = nn.ModuleList()
        self.pos_encoders = nn.ModuleList()
        in_ch = channels[0]
        for i in range(num_ups):
            out_ch = channels[i + 1]
            self.up_blocks.append(
                nn.ModuleDict({
                    'deconv': nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    'norm': LayerNorm2d(out_ch),
                    'act': nn.GELU(),
                    'res': Residual(DepthwiseSeparableConv(out_ch), out_ch),
                })
            )
            if posenc_cfg is not None:
                self.pos_encoders.append(PosEnc2D(
                    num_freqs=posenc_cfg['num_freqs'],
                    out_channels=out_ch,
                    learned_projection=posenc_cfg['learned_projection'],
                ))
            else:
                self.pos_encoders.append(nn.Identity())
            in_ch = out_ch
        self.head = nn.Conv2d(in_ch, 1, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        x = self.proj(latent)
        x = x.view(B, -1, self.base_grid, self.base_grid)
        for blk, pe in zip(self.up_blocks, self.pos_encoders):
            x = blk['deconv'](x)
            # inject positional encoding (additive) after deconv but before norm
            if isinstance(pe, PosEnc2D):
                pe_map = pe(x.shape[2], x.shape[3], x.device).unsqueeze(0)
                x = x + pe_map
            x = blk['norm'](x)
            x = blk['act'](x)
            x = blk['res'](x)
        x = self.head(x)
        if self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.out_activation == 'tanh':
            x = torch.tanh(x)
        return x


class TransmitterEmbedding(nn.Module):
    def __init__(self, in_dim: int = 3, embed_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, tx_pos: torch.Tensor) -> torch.Tensor:
        return self.net(tx_pos)


class RadioPointTransformerV3(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = config or ModelConfig()

        # Radio feature engineering
        self.radio_features = RadioFeatures(self.cfg)

        # Point Transformer backbone (unchanged external implementation assumed)
        self.point_transformer = PointTransformerV3(
            in_channels=self.cfg.in_channels,
            enc_depths=self.cfg.enc_depths,
            enc_channels=self.cfg.enc_channels,
            enc_num_head=self.cfg.enc_num_heads,
            dec_depths=self.cfg.dec_depths,
            dec_channels=self.cfg.dec_channels,
            dec_num_head=self.cfg.dec_num_heads,
            drop_path=self.cfg.drop_path,
            cls_mode=False,
        )

        # Project PTv3 output to latent_dim (assumes external PTv3 returns feat dim = dec_channels[0])
        pt_out_dim = self.cfg.dec_channels[0]
        self.point_proj = nn.Sequential(
            nn.Linear(pt_out_dim, self.cfg.latent_dim),
            nn.LayerNorm(self.cfg.latent_dim),
            nn.GELU(),
        )

        # Tx embedding & fusion
        self.tx_embed = TransmitterEmbedding(3, self.cfg.tx_embed_dim)
        self.fusion = FiLMFusion(
            point_dim=self.cfg.latent_dim,
            tx_dim=self.cfg.tx_embed_dim,
            latent_dim=self.cfg.latent_dim,
            dropout=self.cfg.fusion_dropout,
        )

        # Scene pooling (mean or learned attention)
        if self.cfg.pooling == 'attn':
            self.scene_pool = AttnScenePooling(pt_out_dim, dropout=self.cfg.pooling_dropout)
        else:
            self.scene_pool = None  # use mean

        # 2D decoder (with optional positional encoding)
        posenc_cfg = None
        if self.cfg.posenc2d:
            posenc_cfg = {
                'num_freqs': self.cfg.posenc_num_freqs,
                'learned_projection': self.cfg.posenc_learned_projection,
            }
        self.decoder2d = Decoder2D(
            latent_dim=self.cfg.latent_dim,
            base_channels=self.cfg.base_channels,
            base_grid=self.cfg.base_grid,
            target_size=self.cfg.target_size,
            out_activation=self.cfg.output_activation,
            posenc_cfg=posenc_cfg,
        )

        self.apply(self._init_weights)

    # ----------------- Helpers -----------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_point_batch(self, point_clouds: List[torch.Tensor], tx: torch.Tensor) -> Point:
        device = tx.device
        coords_acc = []
        feats_acc = []
        offsets = [0]
        for i, pc in enumerate(point_clouds):
            pc = pc.to(device)
            rf = self.radio_features(pc, tx[i])  # (Ni,7)
            feats = torch.cat([pc, rf], dim=1)   # (Ni,10)
            coords_acc.append(pc)
            feats_acc.append(feats)
            offsets.append(offsets[-1] + pc.size(0))
        coord = torch.cat(coords_acc, dim=0)
        feat = torch.cat(feats_acc, dim=0)
        offset_t = torch.tensor(offsets, device=device, dtype=torch.long)
        return Point({'coord': coord, 'feat': feat, 'offset': offset_t, 'grid_size': 0.1})

    def _mean_pool(self, feats: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Fallback mean pooling (deterministic, zero extra parameters)."""
        pooled = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i].item(), offsets[i + 1].item()
            pooled.append(feats[s:e].mean(dim=0, keepdim=True))
        return torch.cat(pooled, dim=0)

    # ----------------- Forward -----------------
    def forward(self, point_clouds: List[torch.Tensor], tx_positions: torch.Tensor) -> torch.Tensor:
        # 1. Pack points + engineered features
        point_struct = self._build_point_batch(point_clouds, tx_positions)
        # 2. Encode via PointTransformerV3
        pt_out = self.point_transformer(point_struct)  # expects Point output
        feats = pt_out.feat  # (TotalPoints, C)
        offsets = pt_out.offset  # (B+1)
        # 3. Scene pooling (attention or mean)
        if self.scene_pool is not None:
            pooled = self.scene_pool(feats, offsets)
        else:
            pooled = self._mean_pool(feats, offsets)
        # 4. Project to latent
        point_latent = self.point_proj(pooled)
        # 5. Tx embedding + FiLM fusion
        tx_latent = self.tx_embed(tx_positions)
        fused = self.fusion(point_latent, tx_latent)
        # 6. Decode to 2D map
        out_map = self.decoder2d(fused)  # (B,1,256,256)
        return out_map


def create_model(size: str = "default") -> RadioPointTransformerV3:
    if size == 'small':
        cfg = ModelConfig(
            enc_channels=(32, 64, 128, 256, 384),
            enc_num_heads=(2, 4, 8, 16, 24),
            dec_channels=(64, 64, 128, 256),
            dec_num_heads=(4, 4, 8, 16),
            latent_dim=384,
            tx_embed_dim=192,
            base_channels=192,
            drop_path=0.05,
            pooling='attn',
        )
    elif size == 'large':
        cfg = ModelConfig(
            enc_channels=(64, 128, 256, 512, 768),
            enc_num_heads=(4, 8, 16, 32, 48),
            dec_channels=(128, 128, 256, 512),
            dec_num_heads=(8, 8, 16, 32),
            latent_dim=768,
            tx_embed_dim=384,
            base_channels=384,
            drop_path=0.15,
            pooling='attn',
        )
    else:
        cfg = ModelConfig()
    print(f"[Model] Building '{size}' variant: latent_dim={cfg.latent_dim}, base_channels={cfg.base_channels}")
    return RadioPointTransformerV3(cfg)
