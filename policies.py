import math
import torch
import numpy as np


def risk_seeking_policy(trees, epsilon, inference_mode):
    rewards = trees.rewards
    # if inference_mode:
    #     rewards = rewards.detach().numpy()
    indices = np.argsort(rewards)[::-1][:int(epsilon * len(trees.rewards))]
    rewards = np.array(rewards)
    trees.reduce(indices)
    rewards = rewards[indices]
    baseline_reward = np.min(rewards)
    best_reward = np.max(rewards)
    median = np.median(rewards)
    return trees, baseline_reward, (best_reward, median, baseline_reward)

