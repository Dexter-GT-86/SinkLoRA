

# SinkLoRA


NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" torchrun --nproc_per_node=4 supervised-fine-tune.py \
    --model_name_or_path /data1/llm_models/Llama-2-7b-chat-hf \
    --bf16 True \
    --output_dir /data1/zhy/output \
    --model_max_length 2048 \
    --use_flash_attn True \
    --data_path /home/dexter/zhy/LongAlpaca-12k/shortAlpaca.json \
    --low_rank_training True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --deepspeed "ds_configs/stage2.json" \
    --tf32 True


