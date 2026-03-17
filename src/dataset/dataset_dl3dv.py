import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal
import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


def load_cam_npz(cam_path):
    cam_file = np.load(cam_path)
    intrinsics = cam_file["intrinsic"].astype(np.float32)
    pose = cam_file["pose"].astype(np.float32)
    return intrinsics, pose


@dataclass
class DatasetDl3dvCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetDl3dvCfg


class DatasetDL3DV(Dataset):
    cfg: DatasetDl3dvCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetDl3dvCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        self.data_root = cfg.roots[0]
        index_path = os.path.join(self.data_root, f"{self.data_stage}_index.json")
        with open(index_path, "r") as file:
            data_index = json.load(file)
        
        self.data_list = [os.path.join(self.data_root, item) for item in data_index]
        print(f"DL3DV: {self.stage}: Loaded {len(self.data_list)} scenes from index")
        
    def normalize_intrinsics(self, intrinsics, image_h, image_w):
        intr_norm = intrinsics.copy()
        intr_norm[0, 0] /= image_w
        intr_norm[1, 1] /= image_h
        intr_norm[0, 2] /= image_w
        intr_norm[1, 2] /= image_h
        return intr_norm

    def get_scene_metadata(self, scene_path):
        dense_dir = os.path.join(scene_path, "dense")
        rgb_dir = os.path.join(dense_dir, "rgb")
        cam_dir = os.path.join(dense_dir, "cam")

        if not os.path.exists(rgb_dir):
            return []

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        scene_frames = []
        for rgb_file in rgb_files:
            basename = rgb_file[:-4]
            cam_path = os.path.join(cam_dir, basename + ".npz")
            if not os.path.exists(cam_path):
                continue

            intrinsics, pose = load_cam_npz(cam_path)
            img_w = 2.0 * intrinsics[0, 2]
            img_h = 2.0 * intrinsics[1, 2]

            scene_frames.append({
                "file_path": os.path.join(rgb_dir, rgb_file),
                "intrinsics": self.normalize_intrinsics(intrinsics, img_h, img_w).tolist(),
                "extrinsics": pose.tolist(),
            })
        return scene_frames

    def load_frames(self, frames_info):
        torch_images = []
        for f in frames_info:
            img = Image.open(f["file_path"]).convert("RGB")
            torch_images.append(self.to_tensor(img))
        return torch.stack(torch_images)
        
    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene_path = self.data_list[index]
        scene_id = os.path.basename(scene_path)
        
        example_frames = self.get_scene_metadata(scene_path)
        if not example_frames:
            raise RuntimeError(f"Scene {scene_id} has no valid frames.")

        extrinsics_all = torch.tensor([f["extrinsics"] for f in example_frames], dtype=torch.float32)
        intrinsics_all = torch.tensor([f["intrinsics"] for f in example_frames], dtype=torch.float32)
        
        try:
            context_indices, target_indices, _ = self.view_sampler.sample(
                scene_id, num_context_views, extrinsics_all, intrinsics_all
            )
        except ValueError:
            raise RuntimeError(f"Not enough frames for sampling in {scene_id}")
        
        if (get_fov(intrinsics_all).rad2deg() > self.cfg.max_fov).any():
            raise RuntimeError(f"FoV too wide in scene {scene_id}")
        
        context_images = self.load_frames([example_frames[i] for i in context_indices])
        target_images = self.load_frames([example_frames[i] for i in target_indices])

        context_depth = torch.ones_like(context_images[:, 0])
        target_depth = torch.ones_like(target_images[:, 0])

        extrinsics = extrinsics_all.clone()
        scale = 1.0
        if self.cfg.make_baseline_1:
            a, b = extrinsics[context_indices[0], :3, 3], extrinsics[context_indices[-1], :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                raise RuntimeError(f"Baseline {scale:.6f} out of range in {scene_id}")
            extrinsics[:, :3, 3] /= scale
        
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3]))
            extrinsics[:, :3, 3] /= (scene_scale + 1e-8)

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics_all[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": torch.tensor(context_indices),
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics_all[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": torch.tensor(target_indices),
            },
            "scene": "dl3dv_" + scene_id,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), 
                                 intr_aug=(self.stage == "train" and self.cfg.intr_augment))

        example["context"]["pts3d"] = torch.ones_like(example["context"]["image"]).permute(0, 2, 3, 1)
        example["target"]["pts3d"] = torch.ones_like(example["target"]["image"]).permute(0, 2, 3, 1)
        example["context"]["valid_mask"] = torch.ones_like(example["context"]["depth"]).bool() * -1
        example["target"]["valid_mask"] = torch.ones_like(example["target"]["depth"]).bool() * -1

        return example
        
    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        
        for _ in range(20):
            try:
                return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
            except Exception as e:
                index = np.random.randint(len(self.data_list))
        
        raise RuntimeError("Failed to load any valid scene after 20 attempts.")

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        return "test" if self.stage == "val" else self.stage

    def __len__(self) -> int:
        return len(self.data_list)