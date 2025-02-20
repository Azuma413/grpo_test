from unsloth import is_bfloat16_supported

def get_training_config(lora_rank: int):
    from trl import GRPOConfig
    
    training_args = GRPOConfig(
        use_vllm = True,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations = 6,
        max_prompt_length = 256,
        max_completion_length = 200,
        max_steps = 100,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb",  # Enable Weights & Biases
        output_dir = "outputs",
    )
    return training_args

def get_model_config():
    return {
        "max_seq_length": 192,
        "lora_rank": 8,
        "model_name": "unsloth/Qwen2.5-7B",
        "load_in_4bit": True,
        "fast_inference": True,
        "gpu_memory_utilization": 0.6, # 0.6だとOOM
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
    }

def get_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
