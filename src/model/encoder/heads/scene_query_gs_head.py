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
        num_self_attn_layers: int = 4,
        head_dim: int = 64,
        fourier_freq: int = 6,    # number of frequency bands for positional encoding
        latent_dim: int = 64,
        n_offsets: int = 4,       # number of Gaussians per anchor
    ):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.raw_gs_dim = raw_gs_dim
        self.num_anchors = num_anchors
        self.fourier_freq = fourier_freq
        self.n_offsets = n_offsets

        # Fourier encoding: 3 (xyz) + 6*3*2 (sin/cos for each freq) = 39 dims
        self.fourier_dim = 3 + fourier_freq * 3 * 2

        # 3D position encoder: maps (x,y,z) → hidden_dim query vector
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.fourier_dim, hidden_dim),
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
            TransformerBlockSelfAttn(hidden_dim * 2, head_dim)
            for _ in range(num_self_attn_layers)
        ])

        # For downstream diffusion
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)

        # Output MLP: [query_latent, slot_token] → GS params
        # Input: (B, N, K, latent_dim * 2), Output: (B, N, K, raw_gs_dim + 1)
        self.output_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, raw_gs_dim + 1),
        )

        # Predict per-slot 3D offset; takes slot-conditioned feature → 3
        self.offset_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3),
        )

        # K learnable slot tokens to diversify K Gaussian decodings (prevent collapse)
        self.slot_tokens = nn.Parameter(torch.randn(n_offsets, latent_dim))

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
            gs_params:   (B, N*K, raw_gs_dim + 1)  — first channel is opacity logit
            anchor_pts:  (B, N*K, 3)               — world-space Gaussian positions (anchor + offset)
            query_latent: (B, N, latent_dim)        — per-anchor latent for downstream diffusion
        """
        B, V, _, H, W = imgs.shape

        # 1. Extract 2D feature maps
        feat_2d = self.feat_extractor.forward_features(
            aggregated_tokens_list, imgs, patch_start_idx
        )  # (B, V, feat_dim, H, W)
        feat_2d = feat_2d.float()

        # 2. Voxelize pts_all → anchor_pts
        anchor_pts, valid_counts, voxel_sizes = self.voxelize_anchors(pts_all, None, self.num_anchors)
        # anchor_pts: (B, N, 3),  valid_counts: (B,),  voxel_sizes: (B,)
        B_cur, N, _ = anchor_pts.shape

        # Build a boolean mask for valid (non-padded) anchors
        idx = torch.arange(N, device=anchor_pts.device).unsqueeze(0)  # (1, N)
        valid_mask_anchor = idx < valid_counts.unsqueeze(1)             # (B, N)

        # Normalize anchor_pts to canonical space [-1, 1] — use only valid anchors
        valid_counts_f = valid_counts.float().clamp(min=1).view(B_cur, 1, 1)
        masked_pts = anchor_pts * valid_mask_anchor.unsqueeze(-1).float()
        scene_center = masked_pts.sum(dim=1, keepdim=True) / valid_counts_f  # (B, 1, 3)

        dists = (anchor_pts - scene_center).norm(dim=-1)  # (B, N)
        # Quantile over valid anchors only (loop per sample to avoid padding interference)
        scene_scale_list = []
        for i in range(B_cur):
            valid_dists = dists[i, valid_mask_anchor[i]]
            q = torch.quantile(valid_dists, 0.95) if valid_dists.numel() > 0 else dists.new_tensor(1.0)
            scene_scale_list.append(q)
        scene_scale = torch.stack(scene_scale_list).view(B_cur, 1).clamp(min=1e-8)  # (B, 1)

        anchor_pts_normalized = (anchor_pts - scene_center) / scene_scale.unsqueeze(-1)  # (B, N, 3)

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

        # Zero out padded anchors before self-attention to prevent them from
        # corrupting valid anchor representations as spurious keys
        query = query * valid_mask_anchor.unsqueeze(-1).float()

        # 7. Self-attention refinement
        for block in self.self_attn_blocks:
            query = torch.utils.checkpoint.checkpoint(block, query, use_reentrant=False)

        # 8. for downstream diffusion
        query_latent = self.latent_proj(query)
        query_latent = self.layer_norm(query_latent) # (B, N, D)
        # !!! query_latent for diffusion generation

        # 9. Build slot-conditioned features, then predict K offsets and K GS params
        K = self.n_offsets

        # Concat query_latent with each slot token → shared slot-conditioned representation
        query_exp = query_latent.unsqueeze(2).expand(-1, -1, K, -1)          # (B, N, K, latent_dim)
        slot_exp = self.slot_tokens[None, None].expand(B_cur, N, -1, -1)     # (B, N, K, latent_dim)
        slot_feat = torch.cat([query_exp, slot_exp], dim=-1)                 # (B, N, K, latent_dim*2)

        # Predict K 3D offsets; tanh constrains each offset within the adaptive voxel radius
        # voxel_sizes: (B,) → (B, 1, 1, 1) for broadcasting over (B, N, K, 3)
        offset_scale = voxel_sizes.view(B_cur, 1, 1, 1)
        offsets = torch.tanh(self.offset_mlp(slot_feat)) * offset_scale      # (B, N, K, 3)

        # Predict K sets of GS params
        gs_params = self.output_mlp(slot_feat)                               # (B, N, K, raw_gs_dim+1)

        # Compute K Gaussian world positions
        gs_positions = anchor_pts.unsqueeze(2) + offsets                     # (B, N, K, 3)

        # Flatten to (B, N*K, ...)
        gs_positions_flat = gs_positions.reshape(B_cur, N * K, 3)
        gs_params_flat = gs_params.reshape(B_cur, N * K, self.raw_gs_dim + 1)

        # Set opacity logit of padded anchor's K Gaussians to -20 (sigmoid ≈ 0)
        valid_mask_flat = valid_mask_anchor.unsqueeze(2).expand(-1, -1, K).reshape(B_cur, N * K)
        gs_params_flat = gs_params_flat.clone()
        gs_params_flat[~valid_mask_flat, 0] = -20.0

        return gs_params_flat, gs_positions_flat, query_latent

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
        max_n: int,
    ):
        """
        Returns:
            anchor_pts:   (B, max_voxels, 3)  — padded with 0 for invalid entries
            valid_counts: (B,)  — number of valid anchors per sample
            voxel_sizes:  (B,)  — adaptive voxel size per sample (for offset radius scaling)
        """
        B = pts_all.shape[0]
        results = []
        used_voxel_sizes = []

        for b in range(B):
            pts = pts_all[b].reshape(-1, 3).float()   # (N_points, 3)
            c = conf[b].reshape(-1).float() if conf is not None else None
            # binary search to find the optimal voxel size
            curr_voxel_size = self._estimate_adaptive_voxel_size(pts, max_n)
            used_voxel_sizes.append(curr_voxel_size)

            vox_idx = (pts / curr_voxel_size).round().long()
            _, inv = torch.unique(vox_idx, dim=0, return_inverse=True)
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

        valid_counts = torch.tensor(
            [r.shape[0] for r in results], device=pts_all.device, dtype=torch.long
        )
        voxel_sizes = torch.tensor(
            used_voxel_sizes, device=pts_all.device, dtype=torch.float32
        )  # (B,)
        max_voxels = int(valid_counts.max().item())
        # Pad with 0 (neutral value) instead of -1e4, valid_counts tracks real boundaries
        anchor_pts = self.pad_tensor_list(results, (max_voxels,), value=0.0)
        return anchor_pts, valid_counts, voxel_sizes
        
    def _estimate_adaptive_voxel_size(self, pts: torch.Tensor, max_n: int, iterations: int = 10) -> float:
        """
        Binary search to find the optimal voxel size that results in max_n anchors.
        """
        # Use 5th/95th percentile to exclude outlier points (sky, far background)
        # that would otherwise inflate diag and degrade binary search convergence
        pt_min = torch.quantile(pts, 0.05, dim=0)
        pt_max = torch.quantile(pts, 0.95, dim=0)
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