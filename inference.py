from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

def generate_text(input, lora=None, system_prompt=None):
    if system_prompt is None:
        text = tokenizer.apply_chat_template([
            {"role" : "user", "content" : input},
        ], tokenize = False, add_generation_prompt = True)
    else:
        text = tokenizer.apply_chat_template([
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : input},
        ], tokenize = False, add_generation_prompt = True) 

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = lora,
    )[0].outputs[0].text
    return output

def main():
    # Test example
    test_board = """貴方は後手のプレイヤーです．次の局面で指すべき手を考えてください。

| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| 香 |　|　|　|　| 玉 |　|　| 香 |
|　| 飛 |　| 銀 | 金 |　| 金 | 銀 |　|
| 歩 |　| 桂 |　|　| 歩 | 桂 | 歩 | 歩 |
|　|　| 歩 |　|　| 角 |　|　|　|
|　|　|　| 歩 |　|　|　|　|　|
|　|　|　| 歩 | 銀 |　| 歩 |　| 歩 |
| 歩 | 歩 | 歩 |　|　| 歩 | 銀 |　|　|
|　|　| 金 |　|　|　| 金 | 飛 |　|
| 香 | 桂 | 角 |　| 玉 |　|　| 桂 | 香 |

持ち駒：
先手：歩 歩 歩
後手：歩 歩

"""
    # Generate moves with and without LoRA
    print("\nTesting model outputs:")
    print("-" * 50)
    print("Without LoRA:")
    output = generate_text(test_board)
    print(output)
    print("-" * 50)
    # With LoRA
    output = generate_text(test_board, lora="grpo_saved_lora")
    print("With LoRA:")
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    main()
