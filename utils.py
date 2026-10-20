"""Training, independent evaluation, plots and policy animation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from agent import QLearningAgent

def evaluate(agent, episodes=300, seed=10000, random_policy=False):
    env = gym.make('Taxi-v3')
    rng = np.random.default_rng(seed)
    returns, lengths, successes = [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total, steps, done, success = 0, 0, False, False
        while not done:
            action = int(rng.integers(6)) if random_policy else agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            success = terminated and reward == 20
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        successes.append(success)
    env.close()
    return {'mean_return': float(np.mean(returns)), 'success_rate': float(np.mean(successes)),
            'mean_steps': float(np.mean(lengths)), 'episodes': episodes}

def train(alpha=0.1, seed=42, episodes=10000, verbose=True):
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(alpha=alpha, seed=seed)
    rewards, checkpoints = [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        epsilon = max(0.05, 0.9995 ** episode)
        total = 0
        while True:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Bootstrap across time limits, but not true terminal transitions.
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        if (episode + 1) % 1000 == 0:
            score = evaluate(agent, episodes=100, seed=50000)
            checkpoints.append({'episode': episode + 1, **score})
            if verbose:
                print(f'Episode {episode+1:5d} | exploratory return (last 100): {np.mean(rewards[-100:]):7.2f} | greedy success: {score["success_rate"]:.0%}')
    env.close()
    return agent, rewards, checkpoints

def plot_results(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = ['#247a98', '#c17b28', '#6262ac']
    for color, alpha in zip(colors, sorted(set(r['alpha'] for r in runs))):
        selected = [r for r in runs if r['alpha'] == alpha]
        smooth = np.array([np.convolve(r['rewards'], np.ones(200)/200, 'valid') for r in selected])
        axes[0].plot(np.arange(199, 199+smooth.shape[1])+1, smooth.mean(0), color=color, label=f'alpha={alpha}')
        axes[0].fill_between(np.arange(200, 200+smooth.shape[1]), smooth.min(0), smooth.max(0), color=color, alpha=.12)
        checkpoints = np.array([[c['success_rate'] for c in r['checkpoints']] for r in selected])
        axes[1].plot(np.arange(1000, 10001, 1000), checkpoints.mean(0)*100, color=color, label=f'alpha={alpha}')
    axes[0].set(title='Training: 200-episode moving average', xlabel='Episode', ylabel='Return')
    axes[1].set(title='Greedy validation: mean of 3 seeds', xlabel='Training episode', ylabel='Success (%)', ylim=(-3, 103))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

def save_gif(agent, path, seed=2026):
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    state, _ = env.reset(seed=seed)
    frames = [Image.fromarray(env.render())]
    total = 0
    while True:
        state, reward, terminated, truncated, _ = env.step(agent.act(state))
        frames.append(Image.fromarray(env.render()))
        total += reward
        if terminated or truncated:
            break
    env.close()
    if not (terminated and reward == 20):
        raise RuntimeError('Demonstration did not succeed.')
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]]*5, duration=300, loop=0)
    return {'seed': seed, 'return': total, 'steps': len(frames)-1, 'success': True}
