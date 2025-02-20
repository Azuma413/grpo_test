from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .grpo_trainer import GRPOTrainer
from game.engine_interface import EngineInterface
from utils.reward_functions import RewardFunction

@dataclass
class GameState:
    """Represents the current state of a shogi game."""
    position: str      # markdown format
    sfen: str         # SFEN format
    moves: List[str]  # move history
    repetition_count: Dict[str, int]  # position repetition counter

class OnlineGRPOTrainer(GRPOTrainer):
    """Online GRPO trainer for shogi, combining GRPO with real-time game interaction."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        engine: EngineInterface,
        reward_funcs: List[RewardFunction],
        system_prompt: str,
        max_moves: int = 100,
        mate_time_limit: int = 3,
        cpu_think_time: int = 5000,
        **kwargs
    ):
        super().__init__(model, tokenizer, reward_funcs, **kwargs)
        self.engine = engine
        self.system_prompt = system_prompt
        self.max_moves = max_moves
        self.mate_time_limit = mate_time_limit
        self.cpu_think_time = cpu_think_time

    def _process_game_state(self, state: GameState, last_n_moves: int = 1) -> str:
        """Convert game state to a prompt for the model."""
        prompt = f"次の局面で指すべき手を考えてください。\n\n{state.position}"
        if state.moves and last_n_moves > 0:
            recent_moves = state.moves[-last_n_moves:]
            moves_text = "、".join(recent_moves)
            prompt += f"\n\n直前の手：{moves_text}"
        return prompt

    def _check_mate(self, sfen: str, time_limit: int = 3) -> bool:
        """Check if the current position is mate."""
        self.engine.set_position(sfen)
        self.engine.go_mate(time_limit * 1000)  # Convert to milliseconds
        return self.engine.get_mate_result()

    def _get_cpu_move(self, sfen: str) -> Optional[str]:
        """Get the next move from the CPU engine."""
        self.engine.set_position(sfen)
        self.engine.go_bestmove(self.cpu_think_time)
        return self.engine.get_bestmove()

    def _is_game_over(self, state: GameState) -> Tuple[bool, float, str]:
        """Check if the game is over and return (is_over, reward, reason)."""
        # Check for repetition
        if state.repetition_count[state.position] >= 4:
            return True, 0.0, "千日手により引き分け"

        # Check for maximum moves
        if len(state.moves) > self.max_moves:
            return True, 0.0, "長手数による引き分け"

        # Check for mate
        if self._check_mate(state.sfen, self.mate_time_limit):
            return True, 1.0, "詰みを発見"

        return False, 0.0, ""

    def train_game(self) -> Tuple[float, List[str]]:
        """Train on a single game.
        
        Returns:
            Tuple[float, List[str]]: Total game reward and list of moves
        """
        from game.shogi_game import create_initial_board, process_shogi_board
        
        # Initialize game state
        state = GameState(
            position=create_initial_board(),
            sfen=create_initial_board(),
            moves=[],
            repetition_count={create_initial_board(): 1}
        )
        
        game_reward = 0.0
        
        while True:
            # LLM's turn
            prompt = self._process_game_state(state)
            text = self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ], tokenize=False, add_generation_prompt=True)

            # Generate move
            output = self.model.fast_generate(
                [text],
                max_tokens=64,
                temperature=0.7,
                num_return_sequences=self.num_generations,
            )
            completions = [out.text for out in output[0].outputs]

            try:
                # Process moves and compute rewards
                rewards = self.compute_rewards(
                    [prompt] * len(completions),
                    completions,
                    current_position=state.sfen,
                    engine=self.engine
                )

                # Train on the generated moves
                for completion, reward in zip(completions, rewards):
                    loss = self.train_step(text, completion, reward)
                    game_reward += reward.item()

                # Use the best completion
                best_idx = rewards.argmax().item()
                from utils.reward_functions import extract_xml_answer
                move = extract_xml_answer(completions[best_idx])
                
                # Update game state
                sfen, usi = process_shogi_board(state.position, move)
                state.moves.append(move)
                state.sfen = f"{state.sfen} {usi}"
                state.position = sfen  # This should be markdown format in real impl
                state.repetition_count[state.position] = state.repetition_count.get(state.position, 0) + 1

                # Check if game is over
                is_over, reward, reason = self._is_game_over(state)
                if is_over:
                    game_reward += reward
                    break

                # CPU's turn
                cpu_move = self._get_cpu_move(state.sfen)
                if not cpu_move or "resign" in cpu_move:
                    game_reward += 1.0  # Win by resignation
                    break

                # Update game state with CPU's move
                state.sfen = f"{state.sfen} {cpu_move}"
                state.position = self.engine.sfen_to_markdown(state.sfen)
                state.moves.append(cpu_move)
                state.repetition_count[state.position] = state.repetition_count.get(state.position, 0) + 1

                # Check if game is over after CPU's move
                is_over, reward, reason = self._is_game_over(state)
                if is_over:
                    game_reward += -reward  # Negate reward since it's CPU's win
                    break

            except Exception as e:
                print(f"Error during game: {e}")
                break

        return game_reward, state.moves
