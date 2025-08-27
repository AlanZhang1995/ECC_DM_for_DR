#!/bin/bash
# vim /root/.cache/huggingface/accelerate/default_config.yaml
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#export MODEL_NAME="Nihirc/Prompt2MedImage"
export TRAIN_DIR='/home/haochen/DataComplex/water/JRC_OCTA_2D' #useless

accelerate launch Fundus_train_text_to_image_lora3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512  --center_crop --random_flip \
  --train_batch_size 3 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs=100 --checkpointing_steps=500 \
  --learning_rate=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-retina-model-lora-fundus" \
  --rank 8 \
  --sync_loss_weight 0.001 \
  --prior_loss_weight 0.1 \
  --sync_loss_start_step 10 \
  --sync_loss_interval 10 \
  --stage 4 \
  --validation_prompt="A color fundus image from DDR dataset in normal stage." --report_to="wandb"

#  --sync_loss_weight 0.02 \
# --train_text_encoder \
#  --pretrain_ckpt_path='./sd-retina-model-lora-AMD-NoECC-Ava/checkpoint-6000' \