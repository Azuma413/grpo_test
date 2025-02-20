import re
import math

def extract_xml_answer(text: str) -> str:
    """
    Extracts the answer from a text containing an XML answer.
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    LLMの回答が正しければ2.0、間違っていれば0.0を返す報酬関数。
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """
    LLMの回答が整数であれば0.5、そうでなければ0.0を返す報酬関数。
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    LLMの回答が厳密なフォーマットに従っていれば0.5、そうでなければ0.0を返す報酬関数。
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    LLMの回答が緩いフォーマットに従っていれば0.5、そうでなければ0.0を返す報酬関数。
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    LLMの回答がXMLのフォーマットに従っていれば0.5、そうでなければ0.0を返す報酬関数。
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def evaluation_reward_func(completions, current_position, engine, **kwargs) -> list[float]:
    """
    やねうら王の評価値に基づいて報酬を計算する関数
    Args:
        completions: LLMの出力
        current_position: 現在の局面情報
        engine: やねうら王エンジンのインスタンス
    Returns:
        float: 評価値を-1.0から1.0の範囲に正規化した報酬
    """
    responses = [extract_xml_answer(completion[0]['content']) for completion in completions]
    rewards = []
    
    for response in responses:
        try:
            # やねうら王で指し手後の評価値を取得
            evaluation = engine.get_position_evaluation(current_position, response)
            # 評価値を-1.0から1.0の範囲に正規化
            normalized_reward = math.tanh(evaluation / 1000)  # 1000は正規化の係数
            rewards.append(normalized_reward)
        except:
            rewards.append(-1.0)  # 不正な指し手の場合
    
    return rewards

def soft_shogi_format_reward_func(completions, **kwargs) -> list[float]:
    """
    将棋の指し手が基本的な形式（数字→漢数字→駒名の順序）に従っているかチェック
    Example: 
        正: ７六歩、１一角
        誤: 歩７六、7六歩
    """
    pattern = r"^[１-９7-9][一二三四五六七八九123456789][歩香桂銀金角飛玉と馬龍]$"
    responses = [extract_xml_answer(completion[0]['content']) for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def strict_shogi_format_reward_func(completions, current_position, engine, **kwargs) -> list[float]:
    """
    将棋の指し手が以下の条件を全て満たしているかチェック:
    1. 数字が1-9の範囲（漢数字表記）
    2. 位置が一-九の範囲（漢数字表記）
    3. 駒の名前が正しい
    4. 指し手が将棋のルールに従っている（やねうら王でチェック）
    """
    responses = [extract_xml_answer(completion[0]['content']) for completion in completions]
    rewards = []
    
    number_pattern = r"^[１-９]"  # 1-9の漢数字
    position_pattern = r"[一二三四五六七八九]"  # 位置の漢数字
    piece_pattern = r"[歩香桂銀金角飛玉と馬龍]$"  # 駒の種類
    
    for response in responses:
        reward = 0.0
        # 基本的なパターンチェック
        if (re.match(number_pattern, response) and
            re.search(position_pattern, response) and
            re.search(piece_pattern, response)):
            reward = 0.5
            # やねうら王による合法手チェック
            if engine.is_legal_move(current_position, response):
                reward = 1.0
        rewards.append(reward)
    
    return rewards
