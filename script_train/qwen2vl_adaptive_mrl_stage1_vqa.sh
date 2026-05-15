# Recommended FastVLM Adaptive MRL Stage-1 config (keeps batch size and epochs unchanged).
# Key changes vs baseline: lower LR + cosine decay, longer warmup, tuned projection weights,
# and slightly stronger spectrum regularization for more stable nested-dimension quality.
#
# Orthogonal map options for Stage-1 projections:
#   ORTHO_MAP=cayley      -> strict Cayley map
#   ORTHO_MAP=matrix_exp  -> matrix exponential map (common non-Cayley alternative)
#   ORTHO_MAP=cayley_safe -> internally routes to matrix_exp
#   ORTHO_MAP=""          -> disable re-parameterization (use explicit orthogonal_weight regularizer only)

ORTHO_MAP=""
ORTHO_WEIGHT=0.001

# Compatibility mode for the previous non-Cayley orthogonal-loss setup.
# Usage: USE_PREVIOUS_ORTHO_LOSS=1 bash script_train/fastvlm_adaptive_mrl_stage1_vqa_best.sh
# This maps to: projection_orthogonal_map="" and orthogonal_weight=0.01
if [[ "${USE_PREVIOUS_ORTHO_LOSS:-0}" == "1" ]]; then
  ORTHO_MAP=""
  ORTHO_WEIGHT=0.01
fi

# Backward-compat alias (deprecated): USE_PREVIOUS_ORTHO=1 behaves the same as USE_PREVIOUS_ORTHO_LOSS=1.
if [[ "${USE_PREVIOUS_ORTHO:-0}" == "1" ]]; then
  ORTHO_MAP=""
  ORTHO_WEIGHT=0.01
fi

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --master_port=29512 \
    train_ddp_one_model.py \
    --lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --model_name Qwen/Qwen2-VL-2B \
    --model_backbone qwen2_vl \
    --bf16 \
    --gradient_checkpointing \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W \
    --dataset_split original \
    --image_dir "/home/gdi-user/enguyen/research_vllm/test/VLM_Embed/vlm2vec_train/MMEB-train" \
    --output_dir training/qwen2vl_stage1_best \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy epoch \
    --seed 42 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup_ratio 0.08 \
    --optimizer_name adamw \
    --image_resolution mid \
    --kd_loss_type adaptive_mrl_stage1 \
    --nested_dims 64 128 256 512 768 1024 1536 \
    --stage1_phase all \
    --stage1_projection_spec "1536->1024,1024->768,768->512,512->256,256->128,128->64" \
    --stage1_projection_weights "1536->1024:1.0,1024->768:1.0,768->512:1.0,512->256:1.2,256->128:1.1,128->64:0.8" \
    --align_l1_weight 1.0 \
    --full_dim_l1_weight 0.0 \
    --align_l1_weights "64:0.6,128:0.8,256:0.8,512:0.9,768:1.0,1024:1.0" \
    --orthogonal_weight "${ORTHO_WEIGHT}" \
    --projection_orthogonal_map "${ORTHO_MAP}" \
    --spectrum_kl_weight 0.35 \
    --spectrum_loss_type laplacian_kl \
    --laplacian_tau 0.07 \
    --laplacian_k_eig 16 \
    --laplacian_top_k -1 \
    --spectrum_kl_pair_weights "896->768:0.8,768->512:1.0,512->256:1.2,256->128:1.0,128->64:0.8" \
    --laplacian_pair_weights "1536->1024:1.0,1024->768:1.0,768->512:1.0,512->256:1.0,256->128:1.0,128->64:0.8"
