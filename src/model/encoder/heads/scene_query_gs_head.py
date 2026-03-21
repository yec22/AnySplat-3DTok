from __future__ import annotations
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from .head_modules import TransformerBlockSelfAttn, TransformerBlockCrossAttn, _init_weights


class SceneQueryGSHead(nn.Module):
    """
    Hierarchical two-level anchor-based Gaussian prediction head.

    Pipeline:
      Coarse level (N0 anchors): uniform voxelization → global scene coverage
      Fine level   (N1 anchors): confidence-guided voxelization → complex regions
      Cross-attention fine→coarse injects global context into fine queries
      Joint self-attention refinement over all N0+N1 anchors
      Slot tokens decode K Gaussians per anchor
    """
    def __init__(
        self,
        feat_extractor,              # VGGT_DPT_GS_Head; provides forward_features()
        feat_dim: int = 128,         # DPT intermediate feature dim
        hidden_dim: int = 256,       # internal query / attention dim
        raw_gs_dim: int = 84,        # raw GS param dim from gaussian_adapter (excl. opacity)
        num_anchors: int = 32768,    # total anchors = coarse + fine
        num_coarse_anchors: int = 8192,  # coarse anchors (uniform global coverage)
        num_self_attn_layers: int = 4,   # joint self-attn layers after concat
        num_cross_attn_layers: int = 2,  # fine→coarse cross-attn layers
        head_dim: int = 64,
        fourier_freq: int = 6,
        latent_dim: int = 64,
        n_offsets: int = 4,          # Gaussians per anchor
    ):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.raw_gs_dim = raw_gs_dim
        assert num_coarse_anchors < num_anchors, (
            f"num_coarse_anchors ({num_coarse_anchors}) must be < num_anchors ({num_anchors}), "
            f"otherwise num_fine=0 and the fine branch collapses"
        )
        self.num_anchors = num_anchors
        self.num_coarse = num_coarse_anchors
        self.num_fine = num_anchors - num_coarse_anchors
        self.fourier_freq = fourier_freq
        self.n_offsets = n_offsets

        # Fourier encoding: 3 (xyz) + fourier_freq*3*2 (sin/cos per freq)
        self.fourier_dim = 3 + fourier_freq * 3 * 2

        # 3D position encoder: Fourier → hidden_dim query vector
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project sampled 2D features to hidden_dim
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)

        # Per-view attention logit: hidden_dim → 1
        self.view_agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Coarse branch: 2 self-attention layers before cross-attn
        self.coarse_self_attn_blocks = nn.ModuleList([
            TransformerBlockSelfAttn(hidden_dim * 2, head_dim)
            for _ in range(2)
        ])

        # Cross-attention: fine queries attend to coarse key/values
        self.cross_attn_blocks = nn.ModuleList([
            TransformerBlockCrossAttn(hidden_dim * 2, head_dim)
            for _ in range(num_cross_attn_layers)
        ])

        # Joint self-attention refinement over all anchors
        self.self_attn_blocks = nn.ModuleList([
            TransformerBlockSelfAttn(hidden_dim * 2, head_dim)
            for _ in range(num_self_attn_layers)
        ])

        # Latent projection for downstream diffusion
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)

        # Output MLP: [query_latent, slot_token] → GS params
        self.output_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, raw_gs_dim + 1),
        )

        # Offset MLP: slot-conditioned feature → 3D offset
        self.offset_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3),
        )

        # K learnable slot tokens to diversify K Gaussian decodings
        self.slot_tokens = nn.Parameter(torch.randn(n_offsets, latent_dim))

        # Per-slot learnable UV offsets: each slot k samples 2D features from a
        # different location around the anchor's projection (window-based sampling).
        # Initialized on a unit circle so all K slots start at distinct directions.
        angles = torch.linspace(0, 2 * math.pi * (n_offsets - 1) / n_offsets, n_offsets)
        self.slot_uv_offsets = nn.Parameter(
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (K, 2)
        )
        # Project raw 2D features (feat_dim) to latent_dim for additive injection
        self.slot_feat_proj = nn.Linear(feat_dim, latent_dim)

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
        intrinsic: torch.Tensor,        # (B, V, 3, 3)  pixel-space
        patch_start_idx: int,
        conf: Optional[torch.Tensor] = None,  # (B, V, H, W) confidence
    ):
        """
        Returns:
            gs_params:    (B, (N0+N1)*K, raw_gs_dim + 1)  — first channel is opacity logit
            gs_positions: (B, (N0+N1)*K, 3)               — world-space Gaussian positions
            query_latent: (B, N0+N1, latent_dim)           — per-anchor latent for downstream
        """
        B, V, _, H, W = imgs.shape

        # 1. Extract 2D feature maps
        feat_2d = self.feat_extractor.forward_features(
            aggregated_tokens_list, imgs, patch_start_idx
        )  # (B, V, feat_dim, H, W)
        feat_2d = feat_2d.float()

        # 2. Coarse voxelization first — its valid anchors serve as scene stat representatives.
        #    Using voxelized coarse anchors (not raw pts_all) avoids center/scale being
        #    inflated by sky/background pixels, which would compress the Fourier encoding
        #    resolution for the actual scene of interest.
        coarse_pts, coarse_valid, coarse_voxel_sizes = self.voxelize_anchors(
            pts_all, conf=None, max_n=self.num_coarse
        )  # (B, num_coarse, 3), (B,), (B,)

        coarse_mask = (
            torch.arange(self.num_coarse, device=coarse_pts.device).unsqueeze(0)
            < coarse_valid.unsqueeze(1)
        )  # (B, num_coarse)

        # 3. Shared scene normalization — computed from valid coarse anchors only.
        #    Both coarse and fine branches share the same coordinate system.
        valid_counts_f = coarse_valid.float().clamp(min=1).view(B, 1, 1)
        masked_coarse = coarse_pts * coarse_mask.unsqueeze(-1).float()
        scene_center = masked_coarse.sum(dim=1, keepdim=True) / valid_counts_f  # (B, 1, 3)

        dists = (coarse_pts - scene_center).norm(dim=-1)  # (B, num_coarse)
        scene_scale_list = []
        for i in range(B):
            valid_dists = dists[i, coarse_mask[i]]
            q = torch.quantile(valid_dists, 0.99) if valid_dists.numel() > 0 else dists.new_tensor(1.0)
            scene_scale_list.append(q)
        scene_scale = torch.stack(scene_scale_list).view(B, 1).clamp(min=1e-8)  # (B, 1)

        # 4. Fine voxelization — confidence-guided, focuses on high-complexity regions
        #    Uses the same pts_all; conf selects denser anchors in uncertain/detailed areas.
        fine_pts, fine_valid, fine_voxel_sizes = self.voxelize_anchors(
            pts_all, conf=conf, max_n=self.num_fine
        )  # (B, num_fine, 3), (B,), (B,)

        fine_mask = (
            torch.arange(self.num_fine, device=fine_pts.device).unsqueeze(0)
            < fine_valid.unsqueeze(1)
        )  # (B, num_fine)

        # 5. Process coarse branch: pos_encode + 2D feat sampling + view aggregation
        coarse_query = self._process_branch(
            coarse_pts, coarse_mask, feat_2d, extrinsic, intrinsic,
            scene_center, scene_scale, H, W
        )  # (B, num_coarse, hidden_dim * 2)

        # 2 coarse self-attention layers (with gradient checkpointing)
        for block in self.coarse_self_attn_blocks:
            coarse_query = torch.utils.checkpoint.checkpoint(
                block, coarse_query, use_reentrant=False
            )

        # 6. Process fine branch
        fine_query = self._process_branch(
            fine_pts, fine_mask, feat_2d, extrinsic, intrinsic,
            scene_center, scene_scale, H, W
        )  # (B, num_fine, hidden_dim * 2)

        # 7. Cross-attention: fine queries attend to coarse (direct call — no checkpoint,
        #    since TransformerBlockCrossAttn.forward takes a list argument)
        for block in self.cross_attn_blocks:
            fine_query = block([fine_query, coarse_query])

        # 8. Concatenate coarse + fine and apply joint self-attention
        full_query = torch.cat([coarse_query, fine_query], dim=1)  # (B, N, hidden_dim*2)
        for block in self.self_attn_blocks:
            full_query = torch.utils.checkpoint.checkpoint(
                block, full_query, use_reentrant=False
            )

        # 9. Latent projection
        query_latent = self.latent_proj(full_query)
        query_latent = self.layer_norm(query_latent)  # (B, N, latent_dim)

        # 10. Slot tokens → K offsets and K GS params per anchor
        K = self.n_offsets
        N = self.num_anchors

        # Window-based per-slot 2D feature sampling:
        # each slot k samples features from a slightly different UV location around
        # the anchor projection, giving each slot unique visual context.
        coarse_slot_2d = self._sample_slot_features(
            coarse_pts, feat_2d, extrinsic, intrinsic, H, W
        )  # (B, num_coarse, K, latent_dim)
        fine_slot_2d = self._sample_slot_features(
            fine_pts, feat_2d, extrinsic, intrinsic, H, W
        )  # (B, num_fine, K, latent_dim)
        all_slot_2d = torch.cat([coarse_slot_2d, fine_slot_2d], dim=1)  # (B, N, K, latent_dim)

        # Zero out padded (invalid) anchors so their spurious 2D samples don't contribute
        all_valid_mask = torch.cat([coarse_mask, fine_mask], dim=1)  # (B, N)
        all_slot_2d = all_slot_2d * all_valid_mask.unsqueeze(-1).unsqueeze(-1).float()

        query_exp = query_latent.unsqueeze(2).expand(-1, -1, K, -1)     # (B, N, K, latent_dim)
        slot_exp = self.slot_tokens[None, None].expand(B, N, -1, -1)    # (B, N, K, latent_dim)
        # Inject per-slot 2D context additively into slot tokens
        slot_feat = torch.cat([query_exp, slot_exp + all_slot_2d], dim=-1)  # (B, N, K, latent_dim*2)

        # Per-anchor offset scale: coarse anchors use coarse voxel size, fine use fine
        anchor_voxel_sizes = torch.cat([
            coarse_voxel_sizes.unsqueeze(1).expand(-1, self.num_coarse),
            fine_voxel_sizes.unsqueeze(1).expand(-1, self.num_fine),
        ], dim=1)  # (B, N)
        offset_scale = anchor_voxel_sizes.unsqueeze(-1).unsqueeze(-1)   # (B, N, 1, 1)

        offsets = torch.tanh(self.offset_mlp(slot_feat)) * offset_scale  # (B, N, K, 3)
        gs_params = self.output_mlp(slot_feat)                           # (B, N, K, raw_gs_dim+1)

        # 11. Compute GS world-space positions
        all_anchor_pts = torch.cat([coarse_pts, fine_pts], dim=1)        # (B, N, 3)
        gs_positions = all_anchor_pts.unsqueeze(2) + offsets             # (B, N, K, 3)

        # 12. Flatten to (B, N*K, ...)
        gs_params_flat = gs_params.reshape(B, N * K, self.raw_gs_dim + 1)
        gs_positions_flat = gs_positions.reshape(B, N * K, 3)

        # 13. Mask padded anchors (opacity → -20 ≈ 0) and scale Gaussian scales by scene_scale
        # (all_valid_mask already computed in step 10)
        valid_mask_flat = all_valid_mask.unsqueeze(2).expand(-1, -1, K).reshape(B, -1)
        gs_params_flat = gs_params_flat.clone()
        gs_params_flat[..., 0][~valid_mask_flat] = -20.0
        gs_params_flat[..., 1:4] = gs_params_flat[..., 1:4] + torch.log(scene_scale).unsqueeze(1)

        return gs_params_flat, gs_positions_flat, query_latent

    # ------------------------------------------------------------------
    # Branch helper
    # ------------------------------------------------------------------
    def _process_branch(
        self,
        anchor_pts: torch.Tensor,    # (B, N, 3)  padded with 0 for invalid entries
        valid_mask: torch.Tensor,    # (B, N)     bool
        feat_2d: torch.Tensor,       # (B, V, feat_dim, H, W)
        extrinsic: torch.Tensor,     # (B, V, 3, 4)
        intrinsic: torch.Tensor,     # (B, V, 3, 3)
        scene_center: torch.Tensor,  # (B, 1, 3)
        scene_scale: torch.Tensor,   # (B, 1)
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Encode one level of anchors into (B, N, hidden_dim * 2) query vectors.
        Steps: normalize → Fourier encode → pos_encoder → project+sample → view agg → cat
        """
        # Normalize to canonical space using shared scene stats
        # scene_scale: (B, 1) → (B, 1, 1) for broadcasting over N, 3
        anchor_pts_normalized = (anchor_pts - scene_center) / scene_scale.unsqueeze(-1)

        # Fourier positional encoding + MLP
        anchor_pts_encoded = self.fourier_encode(anchor_pts_normalized)
        query = self.pos_encoder(anchor_pts_encoded)  # (B, N, hidden_dim)

        # Project anchors into each view and sample 2D features
        sampled, vis_mask = self._project_and_sample(
            anchor_pts, feat_2d, extrinsic, intrinsic, H, W
        )
        # sampled:  (B, N, V, feat_dim)
        # vis_mask: (B, N, V)  bool

        # View aggregation with masked soft attention
        proj_feat = self.feat_proj(sampled)  # (B, N, V, hidden_dim)
        logits = self.view_agg_mlp(
            proj_feat + query.unsqueeze(2)
        ).squeeze(-1)  # (B, N, V)
        logits = logits + (~vis_mask).float() * -1e9
        weights = torch.softmax(logits, dim=-1)          # (B, N, V)
        agg_feat = (weights.unsqueeze(-1) * proj_feat).sum(dim=2)  # (B, N, hidden_dim)

        # Combine query + aggregated 2D features
        query = torch.cat([query, agg_feat], dim=-1)     # (B, N, hidden_dim * 2)

        # Zero out padded anchors to prevent them from acting as spurious keys
        query = query * valid_mask.unsqueeze(-1).float()
        return query

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style Fourier positional encoding for [-1, 1] space.
        Args:
            x: (B, N, 3) normalized 3D coordinates
        Returns:
            (B, N, fourier_dim)  where fourier_dim = 3 + fourier_freq * 3 * 2
        """
        device = x.device
        dtype = x.dtype
        freq_bands = 2.0 ** torch.linspace(
            0, self.fourier_freq - 1, self.fourier_freq, device=device, dtype=dtype
        )
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
        Voxelize pts_all into at most max_n anchor points per batch element.
        Always pads output to exactly max_n (zero-padded), so shapes are deterministic
        and suitable for direct concatenation of coarse and fine branches.

        Returns:
            anchor_pts:   (B, max_n, 3)  — zero-padded for invalid entries
            valid_counts: (B,)           — actual number of valid anchors per sample
            voxel_sizes:  (B,)           — adaptive voxel size per sample
        """
        B = pts_all.shape[0]
        results = []
        used_voxel_sizes = []

        for b in range(B):
            pts = pts_all[b].reshape(-1, 3).float()  # (N_points, 3)
            c = conf[b].reshape(-1).float() if conf is not None else None

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
        )
        # Always pad to max_n (fixed shape for coarse/fine concatenation)
        anchor_pts = self.pad_tensor_list(results, (max_n,), value=0.0)
        return anchor_pts, valid_counts, voxel_sizes

    def _estimate_adaptive_voxel_size(self, pts: torch.Tensor, max_n: int, iterations: int = 10) -> float:
        """
        Binary search for the voxel size that yields approximately max_n anchors.
        """
        pt_min = torch.quantile(pts, 0.01, dim=0)
        pt_max = torch.quantile(pts, 0.99, dim=0)
        diag = torch.norm(pt_max - pt_min).item()

        low = 0.00001
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
        pts_h = torch.cat([anchor_pts, ones], dim=-1)

        # World → camera: (B, V, N, 3)
        P_cam = torch.einsum("bvij,bnj->bvni", extrinsic.float(), pts_h)

        # Project to pixel coords: (B, V, N, 3)
        P_proj = torch.einsum("bvij,bvnj->bvni", intrinsic.float(), P_cam)
        depth_cam = P_cam[..., 2]  # (B, V, N)

        uv_pixel = P_proj[..., :2] / (P_proj[..., 2:3] + 1e-8)  # (B, V, N, 2)

        # Normalize to [0, 1]
        hw_tensor = torch.tensor([W, H], dtype=uv_pixel.dtype, device=device)
        uv_norm = uv_pixel / hw_tensor  # (B, V, N, 2)

        # Visibility: depth > 0 and uv in (0, 1)
        vis_mask = (
            (depth_cam > 0)
            & (uv_norm[..., 0] > 0) & (uv_norm[..., 0] < 1)
            & (uv_norm[..., 1] > 0) & (uv_norm[..., 1] < 1)
        )  # (B, V, N)

        # grid_sample expects grid in [-1, 1]
        uv_grid = uv_norm * 2 - 1  # (B, V, N, 2)

        # Flatten B*V for grid_sample
        feat_flat = feat_2d.flatten(0, 1)              # (B*V, C, H, W)
        grid = uv_grid.flatten(0, 1).unsqueeze(2)      # (B*V, N, 1, 2)
        sampled_flat = F.grid_sample(
            feat_flat.float(), grid.float(),
            align_corners=True, padding_mode="zeros", mode="bilinear"
        )  # (B*V, C, N, 1)
        sampled_flat = sampled_flat.squeeze(-1).permute(0, 2, 1)  # (B*V, N, C)
        sampled = sampled_flat.view(B, V, N, C).permute(0, 2, 1, 3)  # (B, N, V, C)

        # Zero out invisible anchors
        sampled = sampled * vis_mask.permute(0, 2, 1).unsqueeze(-1).float()

        return sampled, vis_mask.permute(0, 2, 1)  # (B, N, V, C), (B, N, V)

    def _sample_slot_features(
        self,
        anchor_pts: torch.Tensor,   # (B, N, 3)  world coords (zero-padded for invalid)
        feat_2d: torch.Tensor,      # (B, V, C, H, W)
        extrinsic: torch.Tensor,    # (B, V, 3, 4)
        intrinsic: torch.Tensor,    # (B, V, 3, 3)
        H: int,
        W: int,
    ) -> torch.Tensor:              # (B, N, K, latent_dim)
        """
        Window-based per-slot 2D feature sampling.

        For each slot k, samples 2D features from a slightly different UV location
        (offset by slot_uv_offsets[k], scaled to ~4px window) around each anchor's
        projection.  Features are aggregated across visible views with a masked mean,
        then projected to latent_dim for additive injection into slot tokens.

        This gives each of the K slots unique visual context, mitigating the risk that
        all K decoded Gaussians collapse to the same appearance.
        """
        B, N, _ = anchor_pts.shape
        V = feat_2d.shape[1]
        C = feat_2d.shape[2]
        K = self.n_offsets
        device = anchor_pts.device

        # Project anchor_pts to each view → base UV coords in [-1, 1]
        ones = torch.ones(B, N, 1, device=device, dtype=anchor_pts.dtype)
        pts_h = torch.cat([anchor_pts, ones], dim=-1)                          # (B, N, 4)
        P_cam = torch.einsum("bvij,bnj->bvni", extrinsic.float(), pts_h)       # (B, V, N, 3)
        P_proj = torch.einsum("bvij,bvnj->bvni", intrinsic.float(), P_cam)     # (B, V, N, 3)
        depth_cam = P_cam[..., 2]                                               # (B, V, N)
        uv_pixel = P_proj[..., :2] / (P_proj[..., 2:3] + 1e-8)
        hw_tensor = torch.tensor([W, H], dtype=uv_pixel.dtype, device=device)
        uv_norm = uv_pixel / hw_tensor                                          # (B, V, N, 2) in [0,1]
        base_uv = uv_norm * 2 - 1                                               # (B, V, N, 2) in [-1,1]

        # Visibility: depth > 0 and projection inside image bounds
        vis_mask = (
            (depth_cam > 0)
            & (uv_norm[..., 0] > 0) & (uv_norm[..., 0] < 1)
            & (uv_norm[..., 1] > 0) & (uv_norm[..., 1] < 1)
        )  # (B, V, N) — same mask for all slots (all offsets near anchor projection)

        vis_float = vis_mask.permute(0, 2, 1).float()                           # (B, N, V)
        vis_count = vis_float.sum(dim=2, keepdim=True).clamp(min=1)            # (B, N, 1)

        # Per-slot UV offsets: tanh → [-1,1], scaled to ~8px window in [-1,1] grid space
        window_scale = 8.0 / max(H, W)
        offsets = torch.tanh(self.slot_uv_offsets) * window_scale              # (K, 2)

        feat_flat = feat_2d.flatten(0, 1)  # (B*V, C, H, W)

        slot_feats = []
        for k in range(K):
            uv_k = base_uv + offsets[k].view(1, 1, 1, 2)                       # (B, V, N, 2)
            grid_k = uv_k.flatten(0, 1).unsqueeze(2)                           # (B*V, N, 1, 2)
            sampled_k = F.grid_sample(
                feat_flat.float(), grid_k.float(),
                align_corners=True, padding_mode="zeros", mode="bilinear"
            )  # (B*V, C, N, 1)
            sampled_k = sampled_k.squeeze(-1).permute(0, 2, 1)                 # (B*V, N, C)
            sampled_k = sampled_k.view(B, V, N, C).permute(0, 2, 1, 3)        # (B, N, V, C)
            sampled_k = sampled_k * vis_float.unsqueeze(-1)                    # zero invisible

            # Masked mean across views
            agg_k = sampled_k.sum(dim=2) / vis_count                           # (B, N, C)
            agg_k = self.slot_feat_proj(agg_k)                                 # (B, N, latent_dim)
            slot_feats.append(agg_k)

        return torch.stack(slot_feats, dim=2)  # (B, N, K, latent_dim)
