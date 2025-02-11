from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import wandb
from config import get_model_config, get_training_config, get_sampling_params
from data_utils import get_gsm8k_questions, SYSTEM_PROMPT
from reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func
)

# Get configurations
model_config = get_model_config()
max_seq_length = model_config["max_seq_length"]
lora_rank = model_config["lora_rank"]

# Initialize wandb
wandb.init(
    project="grpo-training",
    config={
        **model_config,
        "max_steps": 100,
        "learning_rate": 5e-6,
    }
)

# Initialize model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_config["model_name"],
    max_seq_length = max_seq_length,
    load_in_4bit = model_config["load_in_4bit"],
    fast_inference = model_config["fast_inference"],
    max_lora_rank = lora_rank,
    gpu_memory_utilization = model_config["gpu_memory_utilization"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = model_config["target_modules"],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Load dataset
dataset = get_gsm8k_questions()

# Initialize trainer
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = get_training_config(lora_rank),
    train_dataset = dataset,
)

# Training
try:
    trainer.train()
finally:
    wandb.finish()

model.save_lora("grpo_saved_lora")

# Test examples
def test_model(question: str, use_lora: bool = False):
    if use_lora:
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        lora_request = model.load_lora("grpo_saved_lora")
    else:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        lora_request = None

    sampling_params = get_sampling_params()
    output = model.fast_generate(
        [text] if not use_lora else text,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text

    print(f"{'With LoRA:' if use_lora else 'Without LoRA:'}")
    print(output)
    print("-" * 50)

# Test the model
test_question = "Which is bigger? 9.11 or 9.9?"
test_model(test_question, use_lora=False)
test_model(test_question, use_lora=True)
