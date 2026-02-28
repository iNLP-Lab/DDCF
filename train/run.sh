base_model="Qwen/Qwen2.5-Math-7B"
lamda=0.2
lr=2e-5
epochs=4
weight_decay=1e-4 
micro_batch_size=1
gradient_accumulation_steps=4 
max_steps=-1
push_to_hub=false


seed=42
outdir="outdir/ddcf/1000/${base_model//\//__}_${lamda}_${seed}" 
accelerate launch --config_file "config/ds_zero3_8.yaml" \
    train/sft.py \
    --block_size=16384 \
    --max_length=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="selected_data/ddcf/1000/${base_model//\//__}_${lamda}" \
    --model_name=${base_model} \
    --warmup_ratio=0.03 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${outdir} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --use_liger_kernel=True \
    --gradient_checkpointing=True 

sleep 60

python eval/evaluate.py --lamda ${lamda} --model_name ${base_model} --seed ${seed} 
python eval/validate.py --lamda ${lamda} --model_name ${base_model} --seed ${seed}