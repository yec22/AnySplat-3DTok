from pathlib import Path
import torch
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from src.model.ply_export import export_ply

# Load the model from Hugging Face
ckpt_path = "/personal/AnySplat_query_v2/output/exp_multidataset_ckpt"
NUM_QUERIES = 65536

# Change the query numbers
config_path = Path(ckpt_path) / "config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
original_num_queries = config['encoder_cfg']['num_queries']
print(f"Original num_queries: {original_num_queries}")
config['encoder_cfg']['num_queries'] = NUM_QUERIES
print(f"Modified num_queries: {NUM_QUERIES}")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

model = AnySplat.from_pretrained(ckpt_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Test Scenes: Mip360 & TNT
root_path = "/nativemm/share/cpfs/MingWorldDatas/AnySplat/test_set/"
test_scene_list = ["TNT_Barn", "TNT_Barn1", "TNT_Church", "TNT_Family", "TNT_Lighthouse", "TNT_Meetingroom", "TNT_Temple"]
for test_scene in test_scene_list:
    image_names = [os.path.join(root_path, test_scene, f) for f in sorted(os.listdir(os.path.join(root_path, test_scene)))]
    print(image_names)
    images = [process_image(image_name) for image_name in image_names]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    gaussians, pred_context_pose = model.inference((images+1)*0.5)

    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    image_folder = f"./outputs/{test_scene}"
    os.makedirs(image_folder, exist_ok=True)
    save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")

# Test Scenes: DL3DV
root_path = "/nativemm/share/cpfs/MingWorldDatas/AnySplat/test_set/DL3DV"
test_scene_list = sorted(os.listdir(root_path))
for test_scene in test_scene_list:
    image_path = os.path.join(root_path, test_scene, test_scene, "nerfstudio", "images_4")
    image_names = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path))]
    image_names = image_names[0:100:10]
    print(image_names)
    images = [process_image(image_name) for image_name in image_names]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    gaussians, pred_context_pose = model.inference((images+1)*0.5)

    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    image_folder = f"./outputs/{test_scene}"
    os.makedirs(image_folder, exist_ok=True)
    save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")