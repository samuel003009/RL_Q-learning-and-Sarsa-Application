from random import Random
import numpy as np


class EpsilonGreedyPolicy:
    """
    epsilon-greedy（ε-greedy）動作選擇策略類別。

    這個類別負責「決定目前狀態下要選哪個 action」。
    在 Q-learning 訓練過程中，agent 不能一開始就永遠選目前看起來最好的動作，
    因為早期 Q 值還不準，如果完全不探索，就可能卡在局部不好的路徑。

    因此 epsilon-greedy 的核心概念是：
    - 大部分時間選擇目前 Q 值最大的動作，也就是利用 exploitation。
    - 少部分時間隨機選擇動作，也就是探索 exploration。

    在本專題中：
    - action 代表掃地機器人在格子世界中的移動方向。
    - 通常 Vacuum_GridWorld 的 action 編號為：
        0: East  往右
        1: West  往左
        2: North 往上
        3: South 往下
    """

    def __init__(self, epsilon, actions, prng):
        """
        初始化 epsilon-greedy policy 需要的參數。

        Parameters:
            epsilon:
                探索率，通常用 ε 表示。
                epsilon 越大，代表越常隨機嘗試不同 action。
                epsilon 越小，代表越常選擇目前 Q 值最大的 action。

                例如：
                - epsilon = 1.0 時，幾乎完全隨機探索。
                - epsilon = 0.1 時，大部分時間會選目前最好的 action。
                - epsilon = 0.0 時，完全不探索，只選 Q 值最大的 action。

            actions:
                所有可用 action 的 list。
                在本掃地機器人環境中通常是 [0, 1, 2, 3]，
                分別對應東、西、北、南四個方向。

            prng:
                Python 的 Random 物件，用來產生隨機選擇結果。
                這裡不用直接呼叫 random 模組，而是使用外部傳入的 prng，
                方便透過固定 random seed 讓訓練結果比較容易重現。
        """

        # 保存最初設定的 epsilon。
        # 後面如果使用 epsilon decay，會需要用 epsilon0 當作初始值來逐步遞減。
        self.epsilon0 = epsilon

        # 目前實際使用中的 epsilon。
        # 如果 decayEpsilon=True，這個值會隨著 episode 增加而慢慢變小。
        self.epsilon = epsilon

        # 將可用 action 轉成 list 保存。
        # 例如 actions = [0, 1, 2, 3]。
        self.actions = list(actions)

        # 紀錄 action 總數。
        # 在四方向移動的環境中，num_actions = 4。
        self.num_actions = len(actions)

        # 保存隨機數產生器。
        # 後面 choose_action() 會用它來依照機率抽選 action。
        self.prng = prng

        # 根據目前 epsilon 計算 greedy action 與 random action 的選擇機率。
        self.update_probs()

    def update_probs(self):
        """
        根據目前 epsilon 更新 epsilon-greedy 的機率分配。

        epsilon-greedy 的機率邏輯如下：
        - 每個 action 先平均分到 epsilon / action數 的探索機率。
        - Q 值最大的 greedy action 額外得到 1 - epsilon 的利用機率。

        因此 greedy action 的總機率為：
            1 - epsilon + epsilon / num_actions

        其他非 greedy action 的機率為：
            epsilon / num_actions

        這樣做的意思是：
        即使正在探索，greedy action 也仍然有機會被隨機抽到。
        """

        # greedy_prob 是目前 Q 值最大 action 的總選擇機率。
        # 其中 1.0 - epsilon 是「利用」部分，epsilon / num_actions 是探索時平均分到的部分。
        self.greedy_prob = 1.0 - self.epsilon + self.epsilon / self.num_actions

        # rand_prob 是每個 action 在探索部分平均分配到的機率。
        # 非 greedy action 只會拿到這個機率。
        self.rand_prob = self.epsilon / self.num_actions

    def decay_epsilon(self, episode, num_episodes):
        """
        讓 epsilon 隨著訓練回合數逐漸下降。

        訓練初期：
        - Q-table 還沒有學到足夠資訊。
        - 需要較高 epsilon，讓 agent 多嘗試不同路線。

        訓練後期：
        - Q-table 已經累積較多經驗。
        - 可以降低 epsilon，讓 agent 更常使用目前學到的最佳策略。

        這裡採用線性遞減：
            epsilon = epsilon0 * (剩餘回合數 / 總回合數)

        因此 episode 越接近 num_episodes，epsilon 會越接近 0。
        """

        # 依照目前 episode 進度，線性降低 epsilon。
        self.epsilon = self.epsilon0 * (num_episodes - episode) / num_episodes

        # epsilon 改變後，greedy_prob 與 rand_prob 也必須重新計算。
        self.update_probs()

    def choose_action(self, q_values):

        policy = [self.rand_prob] * self.num_actions        # Establish the exploration probability for each action.

        greedy_action = int(np.argmax(q_values))        # Find the action index with the largest value in q_values.

        policy[greedy_action] = self.greedy_prob        # Change the probability of the greedy action to greedy_prob, keeping the probability of other actions as rand_prob.

        action = self.prng.choices(self.actions, weights=policy)[0]         # Based on the weighted probabilities in the policy, extract an action from self.actions.
       
        return action       # Return the selected action, ex : E, W, N, S


def get_q_values(q, state, num_actions):
    """
    取得某個 state 對應的所有 action Q 值。

    在一般簡單範例中，state 可能只是整數，例如 0、1、2、3，
    但本專題的 Vacuum_GridWorld 狀態比較複雜，不只包含機器人位置，
    還包含目前哪些髒污格已經被清理、哪些還沒清理。

    因此 Vacuum_GridWorld.currentState() 回傳的 state 通常不是單一整數，
    而是類似以下結構：
        ((robot_row, robot_col), dirtyMap_tuple)

    其中：
    - (robot_row, robot_col) 表示掃地機器人目前所在位置。
    - dirtyMap_tuple 表示整個環境中髒污格的狀態。

    因為 state 不是連續整數，所以這裡不用二維 numpy array 當 Q-table，
    而是使用 dictionary：
        q[state] = np.array([Q(s,0), Q(s,1), Q(s,2), Q(s,3)])

    如果某個 state 第一次出現，代表 Q-table 還沒有它的紀錄，
    就建立一組初始 Q 值，全部設為 0。
    """

    # 如果這個 state 尚未出現在 Q-table 中，就新增一筆資料。
    if state not in q:
        # 建立長度為 num_actions 的 Q 值陣列。
        # 每個 action 的初始 Q 值都設為 0。
        # dtype=np.float32 可以減少記憶體使用量，對大型 Q-table 比較有幫助。
        q[state] = np.zeros(num_actions, dtype=np.float32)

    # 回傳這個 state 對應的 Q 值陣列。
    return q[state]


def make_greedy_policy_function(q, num_actions):
    """
    將訓練完成的 Q-table 轉成 greedy policy function。

    Q-learning 訓練完成後，q 裡面已經存有各 state-action 的估計價值。
    實際展示路徑或產生動畫時，通常不再需要 epsilon-greedy 的隨機探索，
    而是直接選擇每個 state 中 Q 值最大的 action。

    這裡建立並回傳一個 policy_fn：
        policy_fn(state) -> action

    也就是：
    - 輸入目前 state。
    - 查詢該 state 的 Q 值。
    - 回傳 Q 值最大的 action。

    這個 policy_fn 可以直接丟給環境中的路徑展示或動畫函式，
    例如 env.policy2gif()、env.policy2png() 或類似函式。
    """

    def policy_fn(state):
        # 從 Q-table 取得目前 state 的所有 action Q 值。
        q_values = get_q_values(q, state, num_actions)

        # 選擇 Q 值最大的 action。
        # 這裡不再使用 epsilon 隨機探索，而是完全採用 greedy 策略。
        action = int(np.argmax(q_values))

        # 回傳選到的最佳 action。
        return action

    # 回傳內部定義好的 policy function。
    return policy_fn


def qlearn(simenv,
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
    Q-learning 主訓練函式。

    這個函式會讓掃地機器人在 Vacuum_GridWorld 中反覆訓練多個 episode，
    透過環境給的 reward 更新 Q-table，最後學出一個 greedy policy。

    Q-learning 是一種 off-policy temporal-difference control 方法。
    它的更新公式為：

        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

    對應到本程式：
    - s  是目前 state。
    - a  是目前選擇的 action。
    - r  是執行 action 後環境回傳的 reward。
    - s' 是下一個 state。
    - max_a' Q(s',a') 是下一個 state 中所有 action 的最大 Q 值。
    - α alpha 是學習率。
    - γ gamma 是折扣因子。

    在本掃地機器人專題中，Q-learning 的目標通常是學會：
    - 避開障礙物。
    - 盡量用較短路徑清完所有髒污格。
    - 清完後回到起點或指定終點。
    - 讓累積 reward 最大。

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

    # 建立 Python Random 物件。
    # 這個 prng 主要給 epsilon-greedy policy 抽 action 使用。
    prng = Random()

    # 設定 Python Random 的隨機種子，讓 action 探索過程可重現。
    prng.seed(prng_seed)

    # 同時設定 numpy 的隨機種子。
    # 雖然本檔案主要隨機抽樣使用 prng，但設定 numpy seed 可避免其他 numpy 隨機操作造成不可重現。
    np.random.seed(prng_seed)

    # 從環境中取得 action 數量。
    # 例如四方向移動時，num_actions = 4。
    num_actions = simenv.numActions

    # 建立 action 編號 list。
    # 若 num_actions = 4，則 actions = [0, 1, 2, 3]。
    actions = list(range(num_actions))

    # 建立 Q-table。
    # 這裡使用 dict 是因為本環境的 state 不是單純連續整數。
    q = {}

    # 建立 epsilon-greedy policy 物件，訓練過程用它決定 action。
    egp = EpsilonGreedyPolicy(epsilon, actions, prng)

    # 用來記錄每個 episode 實際走了幾步。
    # 如果訓練有效，後期 episode length 通常會有下降趨勢。
    episodeLengths = []

    # 用來記錄每個 episode 的 total reward。
    # 這可以觀察 agent 是否從亂走、撞牆、超過步數限制，
    # 慢慢變成能完成任務並拿到較高總分。
    episodeTotalRewards = []

    for episode in range(num_episodes):
        if init_state is None:      # Using environment default starting point
            simenv.reset()          
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

        # 將 total reward 轉成 numpy array，方便計算累積平均。
        reward_array = np.array(episodeTotalRewards, dtype=np.float64)

        # Avg Total Reward：從第 0 回合到目前回合的累積平均。
        # 例如 avgTotalRewards[i] = 前 i+1 回合 total reward 的平均。
        avgTotalRewards = np.cumsum(reward_array) / np.arange(1, len(reward_array) + 1)

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        # 第一張：每回合步數，也就是訓練中的 Length。
        axes[0].plot(episodeLengths)
        axes[0].set_ylabel("Length")

        # 第二張：每回合 total reward。
        axes[1].plot(episodeTotalRewards)
        axes[1].set_ylabel("Total Reward")

        # 第三張：累積平均 total reward。
        axes[2].plot(avgTotalRewards)
        axes[2].set_ylabel("Avg Total Reward")
        axes[2].set_xlabel("Episode")

        plt.tight_layout()

        # 同時輸出成 PNG，方便放進報告或簡報。
        plt.savefig("qlearning_training_progress.png", dpi=150)
        plt.show()

        # 清除目前圖表，並關閉所有 matplotlib 視窗。
        # 這可以避免後續畫其他圖時受到上一張圖影響。
        plt.cla()
        plt.close("all")

    # 訓練完成後，將 Q-table 轉成 greedy policy function。
    # 後續展示路徑時會直接選每個 state 的最大 Q 值 action。
    policy_fn = make_greedy_policy_function(q, num_actions)

    # 回傳訓練後的 policy function 以及完整 Q-table。
    return policy_fn, q
