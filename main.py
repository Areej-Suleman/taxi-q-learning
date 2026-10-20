"""Run: python main.py (full experiment); python main.py --evaluate."""
from pathlib import Path
import argparse
import json
import numpy as np
from agent import QLearningAgent
from utils import train, evaluate, plot_results, save_gif

ROOT = Path(__file__).resolve().parent

def run_experiments():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    runs = []
    candidates = {}
    for alpha in [0.01, 0.1, 0.5]:
        for seed in [42, 43, 44]:
            print(f'\nLearning rate={alpha}, training seed={seed}', flush=True)
            agent, rewards, checkpoints = train(alpha=alpha, seed=seed)
            runs.append({'alpha': alpha, 'seed': seed, 'rewards': rewards, 'checkpoints': checkpoints})
            if seed == 42:
                candidates[alpha] = agent
    # Choose fastest early validation learning, before looking at final test results.
    scores = {alpha: float(np.mean([c['mean_return'] for r in runs if r['alpha'] == alpha for c in r['checkpoints'][:3]])) for alpha in candidates}
    best_alpha = max(scores, key=scores.get)
    agent = candidates[best_alpha]
    agent.save(output / 'q_table.npy')
    summary = {'selected_alpha': best_alpha, 'selection_scores': scores,
               'test_trained': evaluate(agent), 'test_random': evaluate(agent, random_policy=True),
               'demo': save_gif(agent, output / 'trained_agent.gif'),
               'validation': [{'alpha': r['alpha'], 'seed': r['seed'], 'early_validation_return': float(np.mean([c['mean_return'] for c in r['checkpoints'][:3]])), 'final_validation': r['checkpoints'][-1]} for r in runs]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'training_history.json').write_text(json.dumps(runs))
    plot_results(runs, output / 'learning_curves.png')
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='Load saved Q-table without training')
    args = parser.parse_args()
    if args.evaluate:
        agent = QLearningAgent()
        agent.q = np.load(ROOT / 'results' / 'q_table.npy')
        print(json.dumps(evaluate(agent), indent=2))
    else:
        run_experiments()
