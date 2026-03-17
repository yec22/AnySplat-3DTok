export HF_HUB_OFFLINE=1
# 单机多卡
# export CUDA_VISIBLE_DEVICES=0
# python src/main.py +experiment=scannetpp trainer.num_nodes=1 2>&1 | tee train.log

torchrun \
  --standalone \
  --nproc_per_node=8 \
  src/main.py \
  +experiment=multi-dataset \
  trainer.num_nodes=1 \
  model.encoder.use_scene_query=true \
  model.encoder.num_scene_queries=16384 \
  model.encoder.scene_token_latent_dim=64 \
  model.encoder.n_anchor_offset=4