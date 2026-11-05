def qlearn(simenv,
           init_state=None,
           gamma=0.95,
           epsilon=1.0,
           alpha=0.3,
           num_episodes=8000,
           max_episode_len=100,
           decayEpsilon=True,
           showPlots=False,
           prng_seed=789):
    """

    Parameters:
        simenv:
            掃地機器人的 GridWorld 環境物件。
            這個環境需要提供以下屬性或方法：
            - simenv.numActions：可用 action 數量。
            - simenv.reset()：重置環境。
            - simenv.currentState()：取得目前 state。
            - simenv.step(action)：執行 action 並回傳 next_state, reward, done。

        init_state:
            指定每個 episode 開始時的初始位置。
            - None 表示使用環境預設的 robotInitLoc。
            - 也可以給 [row, col]，讓環境從指定格子開始。

        gamma:
            折扣因子 discount factor。
            用來決定未來 reward 的重要程度。
            - 越接近 1，代表越重視長期報酬。
            - 越接近 0，代表越重視眼前 reward。

        epsilon:
            epsilon-greedy 的初始探索率。
            訓練初期通常設大一點，讓 agent 多探索不同走法。

        alpha:
            學習率 learning rate。
            控制每次 Q 值更新時，新經驗影響舊 Q 值的程度。
            - alpha 越大，更新越快，但可能較不穩定。
            - alpha 越小，更新較慢，但通常較平滑。

        num_episodes:
            訓練總回合數。
            每個 episode 代表從初始狀態開始，直到任務完成或超過最大步數為止。

        max_episode_len:
            每個 episode 允許的最大步數。
            這可以避免 agent 一直走不完任務，導致訓練卡在無限迴圈。

        decayEpsilon:
            是否讓 epsilon 隨 episode 逐漸下降。
            - True：訓練初期多探索，後期多利用學到的策略。
            - False：整個訓練過程使用固定 epsilon。

        showPlots:
            是否顯示訓練過程中的 episode length 曲線。
            episode length 可以粗略觀察 agent 是否逐漸學會用較少步數完成任務。
            如果只是要產生掃地機器人的路徑圖或動畫，可以設為 False。

        prng_seed:
            隨機種子 random seed。
            固定 seed 可以讓隨機探索過程比較容易重現，
            方便小組成員在不同電腦上比較結果。

    Returns:
        policy_fn:
            由訓練完成 Q-table 產生的 greedy policy function。
            後續可以用 policy_fn(state) 取得該 state 下建議採取的 action。

        q:
            dictionary 型 Q-table。
            key 是 state，value 是該 state 下所有 action 的 Q 值陣列。
    """

    prng = Random()
    prng.seed(prng_seed)
    np.random.seed(prng_seed)

    num_actions = simenv.numActions     # action quantity

    actions = list(range(num_actions))          # Create action list, ex : 0, 1, 2, 3

    q = {}      # Create Q-table

    egp = EpsilonGreedyPolicy(epsilon, actions, prng)       # Create an epsilon-greedy policy object

    episodeLengths = []         # Record how many steps were actually taken in each episode.

    episodeTotalRewards = []        # Record the total reward for each episode.

    for episode in range(num_episodes):

        if init_state is None:
            simenv.reset()          # Using environment default starting point
        else:
            simenv.reset(init_state)

        state = simenv.currentState()

        if decayEpsilon:                # Updated according to the current episode: epsilon
            egp.decay_epsilon(episode, num_episodes)

        episode_length = 0

        episode_total_reward = 0.0


        while episode_length < max_episode_len:

            q_values = get_q_values(q, state, num_actions)      # Obtain the Q value corresponding to the current state.

            action = egp.choose_action(q_values)                # Select the action to be performed in this step.

            next_state, reward, done = simenv.step(action)          # The environment will return next_state, reward, done

            next_q_values = get_q_values(q, next_state, num_actions)        # Obtain the Q value corresponding to next_state.

            # Q(s,a) ← Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
            q_values[action] = q_values[action] + alpha * (reward + gamma * np.max(next_q_values) - q_values[action])

            episode_length += 1
            episode_total_reward += reward

            if done:            # If the environment returns "done=True", it means the task has been completed.
                break
  
            state = next_state      # If the task has not yet been completed, update the current state to next_state.

        episodeLengths.append(episode_length)       # Record the total number of steps in an episode.
        episodeTotalRewards.append(episode_total_reward)        # Record the total reward for the episode.

    # 如果 showPlots=True，就畫出訓練過程圖。
    # 這張圖包含：
    #   1. Length：每個 episode 實際走了幾步
    #   2. Total Reward：每個 episode 的總 reward
    #   3. Avg Total Reward：截至目前 episode 為止的平均總 reward
    if showPlots:
        import matplotlib.pyplot as plt
        reward_array = np.array(episodeTotalRewards, dtype=np.float64)          # Convert the total reward into a NumPy array.

        # Avg Total Reward：從第 0 回合到目前回合的累積平均。
        # 例如 avgTotalRewards[i] = 前 i+1 回合 total reward 的平均。
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

        plt.savefig("qlearning_training_progress.png", dpi=150)
        plt.show()

        plt.cla()
        plt.close("all")

    # 訓練完成後，將 Q-table 轉成 greedy policy function。
    # 後續展示路徑時會直接選每個 state 的最大 Q 值 action。
    policy_fn = make_greedy_policy_function(q, num_actions)

    return policy_fn, q