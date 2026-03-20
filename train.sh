export HF_HUB_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=0

torchrun \
  --standalone \
  --nproc_per_node=8 \
  src/main.py \
  +experiment=multi-dataset \
  trainer.num_nodes=1 \
  model.encoder.use_scene_query=true \
  model.encoder.num_scene_queries=32768 \
  model.encoder.num_coarse_anchors=8192 \
  model.encoder.scene_token_latent_dim=64 \
  model.encoder.n_anchor_offset=4 \
  2>&1 | tee train.log