"""Tabular Q-learning implemented from scratch using NumPy."""
import numpy as np

class QLearningAgent:
    def __init__(self, n_states=500, n_actions=6, alpha=0.1, gamma=0.99, seed=42):
        self.q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        # Random tie-breaking during learning; deterministic greedy evaluation.
        if epsilon > 0:
            best = np.flatnonzero(self.q[state] == self.q[state].max())
            return int(self.rng.choice(best))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, terminated):
        target = reward if terminated else reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def save(self, path):
        np.save(path, self.q)
