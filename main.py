from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import wandb
from config import get_model_config, get_training_config, get_sampling_params
from data_utils import SYSTEM_PROMPT
from shogi_engine import YaneuraOuEngine
from shogi_utils import markdown_to_sfen, move_to_usi
from reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    evaluation_reward_func,
    soft_shogi_format_reward_func,
    strict_shogi_format_reward_func
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

# Initialize Shogi engine and utilities
engine = YaneuraOuEngine()
if not engine.start():
    raise RuntimeError("Failed to start YaneuraOu engine")

def process_shogi_board(markdown_board: str, move: str) -> tuple[str, str]:
    """将棋盤のマークダウンとLLMの指し手をSFEN/USI形式に変換"""
    sfen = markdown_to_sfen(markdown_board)
    usi = move_to_usi(move)
    return sfen, usi

# Training dataset with position evaluation
dataset = [
    {
        "prompt": """次の局面で指すべき手を考えてください。

| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |
| 　 | 飛 | 　 | 　 | 　 | 　 | 　 | 角 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 角 | 　 | 　 | 　 | 　 | 　 | 飛 | 　 |
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |

持ち駒：なし
""",
        "response": "<reasoning>\n中央の支配力を高め、後の展開を容易にする手です。\n</reasoning>\n<answer>\n７六歩\n</answer>"
    }
]

# Initialize trainer
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,              # XMLの構造チェック
        soft_format_reward_func,           # 基本的なフォーマットチェック
        strict_format_reward_func,         # 厳密なフォーマットチェック
        evaluation_reward_func,            # やねうら王による評価値報酬
        soft_shogi_format_reward_func,     # 将棋の指し手形式チェック（緩い）
        strict_shogi_format_reward_func,   # 将棋の指し手形式チェック（厳密）
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
test_question = """
| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |
| 　 | 飛 | 　 | 　 | 　 | 　 | 　 | 角 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 角 | 　 | 　 | 　 | 　 | 　 | 飛 | 　 |
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |

持ち駒：なし
"""
test_model(test_question, use_lora=False)
test_model(test_question, use_lora=True)

# Clean up
engine.close()
