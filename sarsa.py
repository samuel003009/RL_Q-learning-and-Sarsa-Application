# -*- coding: utf-8 -*-
"""
sarsa.py

SARSA control for Vacuum_GridWorld.

這版已改成可讀取 Vacuum_GridWorld.py 的環境：
- state 不再假設是整數 index。
- Q-table 改用 dictionary：q[state] = np.array([Q_E, Q_W, Q_N, Q_S])。
- 加入每個 episode 的 Length / Total Reward / Avg Total Reward 統計與繪圖。
"""

from random import Random
import numpy as np


class EpsilonGreedyPolicy:
    """epsilon-greedy 動作選擇策略。"""

    def __init__(self, epsilon, actions, prng):
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.actions = list(actions)
        self.num_actions = len(actions)
        self.prng = prng
        self.update_probs()

    def update_probs(self):
        self.greedy_prob = 1.0 - self.epsilon + self.epsilon / self.num_actions
        self.rand_prob = self.epsilon / self.num_actions

    def decay_epsilon(self, episode, num_episodes):
        self.epsilon = self.epsilon0 * (num_episodes - episode) / num_episodes
        self.update_probs()

    def choose_action(self, q_values):
        """根據目前 state 的 q_values 選擇 action。"""
        policy = [self.rand_prob] * self.num_actions
        greedy_action = int(np.argmax(q_values))
        policy[greedy_action] = self.greedy_prob
        return self.prng.choices(self.actions, weights=policy)[0]


def get_q_values(q, state, num_actions):
    """取得某 state 的 action Q 值；若 state 第一次出現就初始化。"""
    if state not in q:
        q[state] = np.zeros(num_actions, dtype=np.float32)
    return q[state]


def make_greedy_policy_function(q, num_actions):
    """將訓練完成的 Q-table 轉成 greedy policy function。"""
    def policy_fn(state):
        q_values = get_q_values(q, state, num_actions)
        return int(np.argmax(q_values))

    return policy_fn


def sarsa(simenv,
          init_state=None,
          gamma=0.95,
          epsilon=1.0,
          alpha=0.3,
          num_episodes=8000,
          max_episode_len=500,
          decayEpsilon=True,
          showPlots=False,
          prng_seed=789):
    
    """
    SARSA 主訓練函式。

    SARSA 是 on-policy TD control，更新公式為：
        Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]

    和 Q-learning 最大差異：
    - Q-learning 用 next_state 裡最大的 max Q(s',a') 更新。
    - SARSA 用下一步實際依 epsilon-greedy 選到的 next_action 更新。

    Returns:
        policy_fn: 訓練完成後的 greedy policy function。
        q: dictionary 型 Q-table。
    """

    prng = Random()
    prng.seed(prng_seed)
    np.random.seed(prng_seed)

    num_actions = simenv.numActions
    actions = list(range(num_actions))
    q = {}
    egp = EpsilonGreedyPolicy(epsilon, actions, prng)

    episodeLengths = []
    episodeTotalRewards = []

    for episode in range(num_episodes):
        if init_state is None:          # Using environment default starting point
            simenv.reset()      
        else:
            simenv.reset(init_state)
        if decayEpsilon:                # Updated according to the current episode: epsilon
            egp.decay_epsilon(episode, num_episodes)
        state = simenv.currentState()
        q_values = get_q_values(q, state, num_actions)          # Obtain the Q value corresponding to the current state.
        action = egp.choose_action(q_values)                    # Select the action to be performed in this step.
        episode_length = 0
        episode_total_reward = 0.0
        while episode_length < max_episode_len:
            next_state, reward, done = simenv.step(action)          # The environment will return next_state, reward, done
            next_q_values = get_q_values(q, next_state, num_actions)        # Obtain the Q value corresponding to next_state.
            next_action = egp.choose_action(next_q_values)              # Select the action to be performed in this step.
            # Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
            q_values[action] = q_values[action] + alpha * (reward + gamma * next_q_values[next_action] - q_values[action])
            episode_length += 1
            episode_total_reward += reward
            if done:                # If the environment returns "done=True", it means the task has been completed.
                break
            state = next_state          # If the task has not yet been completed, update the current state to next_state.
            q_values = next_q_values    # If the task has not yet been completed, update the current q_values to next_q-values.
            action = next_action        # If the task has not yet been completed, update the current action to next_action.
        episodeLengths.append(episode_length)               # Record the total number of steps in an episode.
        episodeTotalRewards.append(episode_total_reward)        # Record the total reward for the episode.

    if showPlots:
        import matplotlib.pyplot as plt

        reward_array = np.array(episodeTotalRewards, dtype=np.float64)
        avgTotalRewards = np.cumsum(reward_array) / np.arange(1, len(reward_array) + 1)

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(episodeLengths)
        axes[0].set_ylabel("Length")

        axes[1].plot(episodeTotalRewards)
        axes[1].set_ylabel("Total Reward")

        axes[2].plot(avgTotalRewards)
        axes[2].set_ylabel("Avg Total Reward")
        axes[2].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig("sarsa_training_progress.png", dpi=150)
        plt.show()

        plt.cla()
        plt.close("all")

    policy_fn = make_greedy_policy_function(q, num_actions)
    return policy_fn, q
