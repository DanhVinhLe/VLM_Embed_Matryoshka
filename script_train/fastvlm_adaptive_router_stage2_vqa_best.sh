# Recommended FastVLM Adaptive Router Stage-2 config (keeps batch size and epochs unchanged).
# Key changes vs baseline: lower LR + cosine decay and a slightly stronger router to improve
# routing fidelity while preserving aggressive dimension savings.

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --master_port=29513 \
    train_ddp_one_model.py \
    --lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --model_name apple/FastVLM-0.5B \
    --model_backbone llava_qwen2 \
    --bf16 \
    --gradient_checkpointing \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W \
    --dataset_split original \
    --image_dir /workspace/ComfyUI/models/gligen/VLM_Embed/vlm2vec_train/MMEB-train \
    --output_dir training/AdaptiveMRL_fastVLM_stage2_router_best \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy epoch \
    --seed 42 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --image_resolution high \
    --kd_loss_type adaptive_router \
    --nested_dims 64 128 256 512 768 \
    --router_alpha 0.02 \
    --router_hidden_dim 384 \
    --router_accuracy_threshold 0.92
