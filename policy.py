import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

from storage import Storage


@dataclass
class QConfig:
    alpha: float = 0.25
    gamma: float = 0.9
    epsilon: float = 0.1
    actions: Tuple[str, ...] = ("conservative", "balanced", "aggressive")


class QLearningPolicy:
    def __init__(self, storage: Storage, function: str, config: QConfig | None = None):
        self.storage = storage
        self.function = function
        self.config = config or QConfig()
        self.q_table: Dict[tuple, float] = storage.load_policy(function)
        self.prev_state: str | None = None
        self.prev_action: str | None = None

    def _state_from_metrics(self, coverage_ratio: float, edge_gain: int) -> str:
        bucket = min(int(coverage_ratio * 10), 10)
        edge_bucket = min(edge_gain // 10, 10)
        return f"{bucket}:{edge_bucket}"

    def select_action(self, coverage_ratio: float, edge_gain: int) -> str:
        state = self._state_from_metrics(coverage_ratio, edge_gain)
        if random.random() < self.config.epsilon:
            action = random.choice(self.config.actions)
        else:
            action = self._best_action(state)
        self.prev_state = state
        self.prev_action = action
        return action

    def _best_action(self, state: str) -> str:
        best_value = -math.inf
        best_action = self.config.actions[0]
        for action in self.config.actions:
            value = self.q_table.get((state, action), 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def update(self, coverage_ratio: float, edge_gain: int, reward: float) -> None:
        if self.prev_state is None or self.prev_action is None:
            return
        state = self.prev_state
        action = self.prev_action
        current_value = self.q_table.get((state, action), 0.0)
        future = self._best_action_value(state)
        updated = current_value + self.config.alpha * (reward + self.config.gamma * future - current_value)
        self.q_table[(state, action)] = updated
        self.storage.save_policy(self.function, self.q_table)

    def _best_action_value(self, state: str) -> float:
        return max(self.q_table.get((state, action), 0.0) for action in self.config.actions)

    def mutation_scale(self, action: str) -> float:
        if action == "conservative":
            return 0.5
        if action == "aggressive":
            return 1.5
        return 1.0

