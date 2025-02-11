# grpo_test
## 動作環境
OS|Ubuntu24.04(wsl)
---|---
GPU|RTX3090
CUDA|12.6

## 動かし方 on wsl
```bash
pip install uv
uv sync
uv run main.py
```

## 結果
- Without LoRA
```
9.11 is bigger than 9.9. When comparing decimal numbers, you look at the digits from left to right. Both numbers have a "9" in the ones place, but 9.11 has a "1" in the tenths place, whereas 9.9 has a "9" in the tenths place. Since 1 is less than 9, you need to consider the hundredths place. In 9.11, the hundredths place is "1", and in 9.9, it can be thought of as "0" (since 9.9 is equivalent to 9.90). Therefore, 9.11 is greater than 9.9.
```
- With LoRA
```
<reasoning>
To determine which number is larger between 9.11 and 9.9, we can compare them digit by digit starting from the leftmost digit.

1. **Compare the whole number part:** Both numbers have the same whole number part, which is 9. Therefore, we proceed to the next digit.

2. **Compare the tenths place:** 
   - For 9.11, the tenths digit is 1.
   - For 9.9, the tenths digit is 9.

   Since 1 is less than 9, 9.11 is smaller than 9.9 based on the tenths place comparison.

Therefore, without needing to look at further digits, 9.9 is greater than 9.11.
</reasoning>

<answer>
9.9 is bigger.
</answer>
```
## 参考
[WSL2でunslothのGPROトレーニングを試してみる](https://note.com/ngc_shj/n/nadc7d41aa970?sub_rt=share_sb)
[phi-4のGRPOやるipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb)
