from __future__ import annotations
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from .head_modules import TransformerBlockSelfAttn, _init_weights


class SceneQueryGSHead(nn.Module):
    """
    Anchor-based Gaussian prediction head.

    Pipeline:
      1. Extract per-pixel 2D features via feat_extractor.forward_features()
      2. Voxelize pts_all into N anchor points (confidence-guided, O(N))
      3. Initialize per-anchor queries from 3D positions
      4. Project anchors into each view, sample 2D features at projected locations
      5. Aggregate multi-view features with soft attention (masked for visibility)
      6. Combine query + aggregated features
      7. Refine with self-attention among anchors
      8. Output GS params per anchor
    """
    def __init__(
        self,
        feat_extractor,           # VGGT_DPT_GS_Head instance; provides forward_features()
        feat_dim: int = 128,      # DPT intermediate feature dim (head_features_1)
        hidden_dim: int = 256,    # internal query / attention dim
        raw_gs_dim: int = 84,     # raw GS param dim from gaussian_adapter (excluding opacity)
        num_anchors: int = 8192,  # max number of scene anchors / Gaussians
        voxel_size: float = 0.01,
        num_self_attn_layers: int = 4,
        head_dim: int = 64,
        fourier_freq: int = 6,    # number of frequency bands for positional encoding
        latent_dim: int = 64,
    ):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.raw_gs_dim = raw_gs_dim
        self.num_anchors = num_anchors
        self.voxel_size = voxel_size
        self.fourier_freq = fourier_freq

        # Fourier encoding: 3 (xyz) + 6*3*2 (sin/cos for each freq) = 39 dims
        self.fourier_dim = 3 + fourier_freq * 3 * 2

        # 3D position encoder: maps (x,y,z) → hidden_dim query vector
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project sampled 2D features to hidden_dim
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)

        # Per-view attention logit: (hidden_dim) → 1
        self.view_agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Self-attention refinement among anchors
        self.self_attn_blocks = nn.ModuleList([
            TransformerBlockSelfAttn(hidden_dim, head_dim)
            for _ in range(num_self_attn_layers)
        ])

        # For downstream diffusion
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)

        # Output MLP: hidden_dim → raw_gs_dim + 1 (opacity logit + GS params)
        self.output_mlp = nn.Linear(hidden_dim, raw_gs_dim + 1)

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        pts_all: torch.Tensor,          # (B, V, H, W, 3)  world-space points
        imgs: torch.Tensor,             # (B, V, 3, H, W)
        extrinsic: torch.Tensor,        # (B, V, 3, 4)  w2c
        intrinsic: torch.Tensor,        # (B, V, 3, 3)  pixel-space [fx,0,cx; 0,fy,cy; 0,0,1]
        patch_start_idx: int,
        conf: Optional[torch.Tensor] = None,  # (B, V, H, W) confidence
    ):
        """
        Returns:
            gs_params:   (B, N, raw_gs_dim + 1)  — first channel is opacity logit
            anchor_pts:  (B, N, 3)               — world-space anchor positions
        """
        B, V, _, H, W = imgs.shape

        # 1. Extract 2D feature maps
        feat_2d = self.feat_extractor.forward_features(
            aggregated_tokens_list, imgs, patch_start_idx
        )  # (B, V, feat_dim, H, W)
        feat_2d = feat_2d.float()

        # 2. Voxelize pts_all → anchor_pts
        anchor_pts = self.voxelize_anchors(pts_all, None, None, self.num_anchors)
        # (B, N, 3)
        N = anchor_pts.shape[1]

        # Normalize anchor_pts to canonical space [-1, 1]
        scene_center = anchor_pts.mean(dim=1, keepdim=True) # (B, 1, 3)
        # Use Euclidean distance (scene radius) with quantile to be robust to outliers
        dists = (anchor_pts - scene_center).norm(dim=-1) # (B, N)
        scene_scale = torch.quantile(dists, 0.95, dim=1, keepdim=True) # (B, 1)
        scene_scale = scene_scale.clamp(min=1e-8) # avoid division by zero
        anchor_pts_normalized = (anchor_pts - scene_center) / scene_scale.unsqueeze(-1) # (B, N, 3), mostly in [-1, 1]

        # 3. Query initialization from normalized 3D positions (with Fourier encoding)
        anchor_pts_encoded = self.fourier_encode(anchor_pts_normalized)
        query = self.pos_encoder(anchor_pts_encoded)  # (B, N, hidden_dim)

        # 4. Project anchors into views and sample 2D features
        sampled, vis_mask = self._project_and_sample(
            anchor_pts, feat_2d, extrinsic, intrinsic, H, W
        )
        # sampled:  (B, N, V, feat_dim)
        # vis_mask: (B, N, V)  bool

        # 5. View aggregation with masked soft attention
        proj_feat = self.feat_proj(sampled)  # (B, N, V, hidden_dim)
        # Add query broadcast over views
        logits = self.view_agg_mlp(
            proj_feat + query.unsqueeze(2)
        ).squeeze(-1)  # (B, N, V)
        # Mask out invisible views with large negative
        logits = logits + (~vis_mask).float() * -1e9
        weights = torch.softmax(logits, dim=-1)  # (B, N, V)
        agg_feat = (weights.unsqueeze(-1) * proj_feat).sum(dim=2)  # (B, N, hidden_dim)

        # 6. Combine query and aggregated features
        query = torch.cat([query, agg_feat], dim=-1)  # (B, N, hidden_dim * 2)

        # 7. Self-attention refinement
        for block in self.self_attn_blocks:
            query = torch.utils.checkpoint.checkpoint(block, query, use_reentrant=False)

        # 8. for downstream diffusion
        query_latent = self.latent_proj(query)
        query_latent = self.layer_norm(query_latent) # (B, N, D)
        # !!! query_latent for diffusion generation

        # 9. Output GS params
        gs_params = self.output_mlp(query_latent)  # (B, N, raw_gs_dim + 1)

        return gs_params, anchor_pts, query_latent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style Fourier positional encoding optimized for [-1, 1] space.
        Args:
            x: (B, N, 3) input 3D coordinates normalized to [-1, 1]
        Returns:
            (B, N, fourier_dim) encoded features
            fourier_dim = 3 + fourier_freq * 3 * 2 (e.g., 3 + 10 * 3 * 2 = 63)
        """
        B, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        freq_bands = 2.0 ** torch.linspace(0, self.fourier_freq - 1, self.fourier_freq, device=device, dtype=dtype)

        x_in = x.unsqueeze(-1) * (freq_bands * math.pi)

        sin_enc = torch.sin(x_in).flatten(2)
        cos_enc = torch.cos(x_in).flatten(2)

        return torch.cat([x, sin_enc, cos_enc], dim=-1)

    def voxelize_anchors(
        self,
        pts_all: torch.Tensor,          # (B, V, H, W, 3)
        conf: Optional[torch.Tensor],   # (B, V, H, W) or None
        voxel_size: Optional[float],    # Optional voxel size override
        max_n: int,
    ) -> torch.Tensor:
        B = pts_all.shape[0]
        results = []
        
        for b in range(B):
            pts = pts_all[b].reshape(-1, 3).float()   # (N_points, 3)
            c = conf[b].reshape(-1).float() if conf is not None else None
            # binary search to find the optimal voxel size
            curr_voxel_size = voxel_size
            if curr_voxel_size is None:
                curr_voxel_size = self._estimate_adaptive_voxel_size(pts, max_n)

            vox_idx = (pts / curr_voxel_size).round().long()
            unique_vox, inv = torch.unique(vox_idx, dim=0, return_inverse=True)
            if c is not None:
                c_pos = c.clamp(min=0)
                vox_sum_c = scatter_add(c_pos, inv, dim=0)
                vox_pts = scatter_add(pts * c_pos.unsqueeze(-1), inv, dim=0)
                vox_pts = vox_pts / (vox_sum_c.unsqueeze(-1) + 1e-8)
            else:
                counts = scatter_add(torch.ones(pts.shape[0], device=pts.device), inv, dim=0)
                vox_pts = scatter_add(pts, inv, dim=0) / counts.unsqueeze(-1)

            if vox_pts.shape[0] > max_n:
                if c is not None:
                    top_idx = vox_sum_c.topk(max_n).indices
                else:
                    top_idx = torch.randperm(vox_pts.shape[0], device=pts.device)[:max_n]
                vox_pts = vox_pts[top_idx]
            results.append(vox_pts)
        
        max_voxels = max(r.shape[0] for r in results)
        return self.pad_tensor_list(results, (max_voxels,), value=-1e4)
        
    def _estimate_adaptive_voxel_size(self, pts: torch.Tensor, max_n: int, iterations: int = 10) -> float:
        """
        Binary search to find the optimal voxel size that results in max_n anchors.
        """
        pt_min = pts.min(dim=0).values
        pt_max = pts.max(dim=0).values
        diag = torch.norm(pt_max - pt_min).item()
        
        low = 0.0001
        high = diag
        best_size = high
        
        target_high = int(max_n * 1.1) 
        for _ in range(iterations):
            mid = (low + high) / 2
            v_idx = (pts / mid).round().long()
            num_voxels = torch.unique(v_idx, dim=0).shape[0]
            
            if num_voxels > target_high:
                low = mid
                best_size = mid
            elif num_voxels < max_n:
                high = mid
            else:
                return mid
                
        return best_size

    def pad_tensor_list(
        self,
        tensor_list: list,
        pad_shape: tuple,
        value: float = 0.0,
    ) -> torch.Tensor:
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def _project_and_sample(
        self,
        anchor_pts: torch.Tensor,   # (B, N, 3)  world coords
        feat_2d: torch.Tensor,      # (B, V, C, H, W)
        extrinsic: torch.Tensor,    # (B, V, 3, 4)  w2c
        intrinsic: torch.Tensor,    # (B, V, 3, 3)  pixel-space
        H: int,
        W: int,
    ):
        """
        Returns:
            sampled:  (B, N, V, feat_dim)  — zero for invisible anchors
            vis_mask: (B, N, V)            — True where anchor is visible in view
        """
        B, N, _ = anchor_pts.shape
        V = feat_2d.shape[1]
        C = feat_2d.shape[2]
        device = anchor_pts.device

        # Homogeneous world coords (B, N, 4)
        ones = torch.ones(B, N, 1, device=device, dtype=anchor_pts.dtype)
        pts_h = torch.cat([anchor_pts, ones], dim=-1)  # (B, N, 4)

        # World → camera:  (B, V, N, 3)
        # extrinsic: (B, V, 3, 4)
        P_cam = torch.einsum("bvij,bnj->bvni", extrinsic.float(), pts_h)  # (B, V, N, 3)

        # Project to pixel coords: (B, V, N, 3)
        P_proj = torch.einsum("bvij,bvnj->bvni", intrinsic.float(), P_cam)  # (B, V, N, 3)
        depth_cam = P_cam[..., 2]  # (B, V, N)

        uv_pixel = P_proj[..., :2] / (P_proj[..., 2:3] + 1e-8)  # (B, V, N, 2)

        # Normalize to [0, 1]
        hw_tensor = torch.tensor([W, H], dtype=uv_pixel.dtype, device=device)
        uv_norm = uv_pixel / hw_tensor  # (B, V, N, 2)

        # Visibility: depth > 0 and uv in [0, 1]
        vis_mask = (
            (depth_cam > 0)
            & (uv_norm[..., 0] > 0) & (uv_norm[..., 0] < 1)
            & (uv_norm[..., 1] > 0) & (uv_norm[..., 1] < 1)
        )  # (B, V, N)

        # grid_sample expects grid in [-1, 1]
        uv_grid = uv_norm * 2 - 1  # (B, V, N, 2)

        # Flatten B*V for grid_sample
        feat_flat = feat_2d.flatten(0, 1)                        # (B*V, C, H, W)
        grid = uv_grid.flatten(0, 1).unsqueeze(2)                # (B*V, N, 1, 2)
        sampled_flat = F.grid_sample(
            feat_flat.float(), grid.float(),
            align_corners=True, padding_mode="zeros", mode="bilinear"
        )  # (B*V, C, N, 1)
        sampled_flat = sampled_flat.squeeze(-1).permute(0, 2, 1)  # (B*V, N, C)
        sampled = sampled_flat.view(B, V, N, C).permute(0, 2, 1, 3)  # (B, N, V, C)

        # Zero out invisible anchors
        sampled = sampled * vis_mask.permute(0, 2, 1).unsqueeze(-1).float()  # (B, N, V, C)

        return sampled, vis_mask.permute(0, 2, 1)  # (B, N, V, C), (B, N, V)