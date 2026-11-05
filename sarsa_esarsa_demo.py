# -*- coding: utf-8 -*-
"""
sarsa_esarsa_demo.py

這份 demo 已改成讀取 Vacuum_GridWorld.py，而不是 Cat_and_Mouse。
功能和掃地機器人 Q-learning demo 對齊：
1. 使用同一張掃地機器人地圖。
2. 分別訓練 SARSA 與 Expected SARSA。
3. 記錄並繪製訓練回合統計圖：Length / Total Reward / Avg Total Reward。
4. 輸出 Q-table txt / csv。
5. 用 greedy policy 跑一次任務，列印每一步與 reward 統計。
6. 輸出路徑圖與 GIF。
"""

import numpy as np

from Vacuum_GridWorld import Vacuum_GridWorld
from sarsa import sarsa


# ------------------------------------------------------------
# action 編號轉方向文字
# ------------------------------------------------------------
def action_name(env, action):
    return env.actions[action]


# ------------------------------------------------------------
# 建立與 qlearn_demo.py 相同的掃地機器人地圖
# ------------------------------------------------------------
def build_vacuum_env():
    rows = 12
    columns = 12
    robotInitLoc = [2, 0]

    obstacleLocs = [
        # 小桌
        [0, 0], [0, 1],
        [1, 0], [1, 1],

        # 雙人床
        [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
        [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
        [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
        [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
        [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],

        # 書桌
        [10, 3], [10, 4], [10, 5], [10, 6],
        [11, 3], [11, 4], [11, 5], [11, 6],

        # 櫃子1
        [0, 10], [0, 11],
        [1, 10], [1, 11],
        [2, 10], [2, 11],
        [3, 10], [3, 11],

        # 櫃子2
        [6, 10], [6, 11],
        [7, 10], [7, 11],
        [8, 10], [8, 11],
        [9, 10], [9, 11],
        [10, 10], [10, 11],
        [11, 10], [11, 11],

        # 櫃子3
        [0, 7], [0, 8], [0, 9],
        [1, 7], [1, 8], [1, 9],

        # 家電
        [7, 7], [7, 8],
        [8, 7], [8, 8],
    ]

    obstacleSet = set(tuple(loc) for loc in obstacleLocs)

    # 與 qlearn_demo.py 相同：所有可走格子，除了 Home，都設成 target。
    targetLocs = [
        [r, c]
        for r in range(rows)
        for c in range(columns)
        if (r, c) not in obstacleSet and [r, c] != robotInitLoc
    ]

    return Vacuum_GridWorld(
        rows=rows,
        columns=columns,
        robotInitLoc=robotInitLoc,
        targetLocs=targetLocs,
        obstacleLocs=obstacleLocs
    )


# ------------------------------------------------------------
# 印出環境資訊
# ------------------------------------------------------------
def print_environment_info(env):
    print("========== Vacuum_GridWorld 環境資訊 ==========")
    print("rows =", env.rows)
    print("columns =", env.columns)
    print("Home =", env.robotInitLoc)
    print("targetLocs =", env.targetLocs)
    print("obstacleLocs =", env.obstacleLocs)
    print("actions =", env.actions)
    print("numActions =", env.numActions)
    print("numStates 理論值 =", env.numStates)
    print("currentStateDetail =", env.currentStateDetail())


# ------------------------------------------------------------
# 解碼 Q-table state
# ------------------------------------------------------------
def decode_q_state(env, state):
    robot_pos, dirty_flat = state
    robot_loc = [robot_pos[0], robot_pos[1]]
    remaining_dirty = int(np.sum(dirty_flat))
    cleaned_count = env.numTargets - remaining_dirty
    return robot_loc, remaining_dirty, cleaned_count


# ------------------------------------------------------------
# Q-table console 預覽
# ------------------------------------------------------------
def print_q_table(env, q, title="Q-table", max_rows=30):
    print("\n========== {} 預覽 ==========".format(title))
    print("Q-table 總 state 數 =", len(q))
    print("只預覽前 {} 筆，完整內容請看輸出的 txt / csv".format(max_rows))

    header = (
        "{:<6s} {:<10s} {:<12s} {:<10s} "
        "{:>10s} {:>10s} {:>10s} {:>10s} {:<12s}"
    ).format(
        "index", "robot", "remain_dirty", "cleaned",
        "Q(E)", "Q(W)", "Q(N)", "Q(S)", "best_action"
    )

    print(header)
    print("-" * len(header))

    sorted_items = sorted(
        q.items(),
        key=lambda item: (
            decode_q_state(env, item[0])[2],
            decode_q_state(env, item[0])[0][0],
            decode_q_state(env, item[0])[0][1]
        )
    )

    for idx, (state, q_values) in enumerate(sorted_items[:max_rows]):
        robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)
        best_action = int(np.argmax(q_values))
        best_action_text = "{}({})".format(best_action, action_name(env, best_action))

        print(
            "{:<6d} {:<10s} {:<12d} {:<10d} "
            "{:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:<12s}".format(
                idx,
                str(robot_loc),
                remaining_dirty,
                cleaned_count,
                q_values[0], q_values[1], q_values[2], q_values[3],
                best_action_text
            )
        )


# ------------------------------------------------------------
# Q-table 輸出 txt / csv
# ------------------------------------------------------------
def save_q_table(env, q, txt_outfile, csv_outfile):
    sorted_items = sorted(
        q.items(),
        key=lambda item: (
            decode_q_state(env, item[0])[2],
            decode_q_state(env, item[0])[0][0],
            decode_q_state(env, item[0])[0][1]
        )
    )

    with open(txt_outfile, "w", encoding="utf-8") as f:
        f.write("========== Q-table 完整輸出 ==========\n")
        f.write("Q-table 總 state 數 = {}\n\n".format(len(q)))

        f.write(
            "{:<6s} {:<10s} {:<12s} {:<10s} "
            "{:>10s} {:>10s} {:>10s} {:>10s} {:<12s}\n".format(
                "index", "robot", "remain_dirty", "cleaned",
                "Q(E)", "Q(W)", "Q(N)", "Q(S)", "best_action"
            )
        )

        for idx, (state, q_values) in enumerate(sorted_items):
            robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)
            best_action = int(np.argmax(q_values))
            best_action_text = "{}({})".format(best_action, action_name(env, best_action))

            f.write(
                "{:<6d} {:<10s} {:<12d} {:<10d} "
                "{:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:<12s}\n".format(
                    idx,
                    str(robot_loc),
                    remaining_dirty,
                    cleaned_count,
                    q_values[0], q_values[1], q_values[2], q_values[3],
                    best_action_text
                )
            )

    with open(csv_outfile, "w", encoding="utf-8-sig") as f:
        f.write("index,robot_row,robot_col,remain_dirty,cleaned,Q_E,Q_W,Q_N,Q_S,best_action,best_action_name\n")

        for idx, (state, q_values) in enumerate(sorted_items):
            robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)
            best_action = int(np.argmax(q_values))

            f.write(
                "{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n".format(
                    idx,
                    robot_loc[0],
                    robot_loc[1],
                    remaining_dirty,
                    cleaned_count,
                    q_values[0], q_values[1], q_values[2], q_values[3],
                    best_action,
                    action_name(env, best_action)
                )
            )

    print("\n已輸出完整 Q-table：")
    print("文字檔：", txt_outfile)
    print("CSV 檔：", csv_outfile)


# ------------------------------------------------------------
# 使用 greedy policy 實際跑一次並統計 reward
# ------------------------------------------------------------
def run_greedy_episode(env, policy_fn, max_steps=500, title="Greedy policy", path_outfile="gridworld_path.png"):
    env.reset()

    print("\n========== {} 文字測試 ==========".format(title))
    print("初始位置 =", env.currentStateXY())
    print("初始 dirtyMap:")
    print(env.dirtyMap)

    path = [list(env.currentStateXY())]
    total_reward = 0.0

    wall_hit_count = 0
    obstacle_hit_count = 0
    moved_count = 0
    cleaned_count = 0

    wall_reward_total = 0.0
    obstacle_reward_total = 0.0
    move_penalty_total = 0.0
    clean_reward_total = 0.0
    finish_reward_total = 0.0

    executed_steps = 0

    for step in range(1, max_steps + 1):
        state = env.currentState()
        action = policy_fn(state)
        next_state, reward, done = env.step(action)
        info = env.lastStepInfo

        path.append(list(env.currentStateXY()))
        total_reward += reward
        executed_steps = step

        if info["hit_wall"]:
            wall_hit_count += 1
            wall_reward_total += info["wall_penalty"]

        if info["hit_obstacle"]:
            obstacle_hit_count += 1
            obstacle_reward_total += info["obstacle_penalty"]

        if info["moved"]:
            moved_count += 1
            move_penalty_total += info["step_penalty"]

        if info["cleaned_new_target"]:
            cleaned_count += 1
            clean_reward_total += info["clean_reward"]

        if info["finished_home"]:
            finish_reward_total += info["finish_reward"]

        event_text = []
        if info["hit_wall"]:
            event_text.append("碰牆")
        if info["hit_obstacle"]:
            event_text.append("碰障礙物")
        if info["moved"]:
            event_text.append("成功移動")
        if info["cleaned_new_target"]:
            event_text.append("清到新髒污")
        if info["finished_home"]:
            event_text.append("完成回家")
        if len(event_text) == 0:
            event_text.append("無事件")

        print(
            "Step {:02d}: action={}({}), reward={:6.1f}, loc={}, event={}, done={}".format(
                step,
                action,
                action_name(env, action),
                reward,
                env.currentStateXY(),
                "+".join(event_text),
                done
            )
        )

        if done:
            print("\n任務完成：所有髒污清完，且機器人已回到 Home。")
            break

    print("\n========== Reward 統計 ==========")
    print("實際執行步數 =", executed_steps)
    print("成功移動步數 = {}, 移動扣分合計 = {:.1f}".format(moved_count, move_penalty_total))
    print("碰牆次數 = {}, 碰牆 reward 合計 = {:.1f}".format(wall_hit_count, wall_reward_total))
    print("碰障礙物次數 = {}, 碰障礙物 reward 合計 = {:.1f}".format(obstacle_hit_count, obstacle_reward_total))
    print("清掃新髒污格數 = {}, 清掃 reward 合計 = {:.1f}".format(cleaned_count, clean_reward_total))
    print("完成任務回家 reward 合計 = {:.1f}".format(finish_reward_total))

    recomputed_total = (
        move_penalty_total
        + wall_reward_total
        + obstacle_reward_total
        + clean_reward_total
        + finish_reward_total
    )

    print("\nReward 重新加總 = {:.1f}".format(recomputed_total))
    print("Total reward = {:.1f}".format(total_reward))

    if abs(recomputed_total - total_reward) < 1e-9:
        print("檢查結果：Reward 分項加總與 Total reward 一致。")
    else:
        print("警告：Reward 分項加總與 Total reward 不一致，請檢查統計邏輯。")

    print("\n最後 dirtyMap:")
    print(env.dirtyMap)

    print("\n========== {} 行走路徑 ==========".format(title))
    for idx, loc in enumerate(path):
        print("Path {:03d}: {}".format(idx, loc))

    env.render_path(path, outfile=path_outfile)
    print("\n已輸出路徑圖：", path_outfile)

    return path


# ------------------------------------------------------------
# 單一演算法完整流程
# ------------------------------------------------------------
def train_and_demo(algorithm_name, train_func, env, alpha, progress_note):
    print("\n\n============================================================")
    print("開始訓練：{}".format(algorithm_name))
    print(progress_note)
    print("============================================================")

    policy_fn, q = train_func(
        simenv=env,
        init_state=None,
        gamma=0.95,
        epsilon=1.0,
        alpha=alpha,
        num_episodes=20000,
        max_episode_len=500,
        decayEpsilon=True,
        showPlots=True,
        prng_seed=789
    )

    print("\n========== {} 訓練完成 ==========".format(algorithm_name))
    print("Q-table 實際記錄 state 數 =", len(q))

    prefix = algorithm_name.lower().replace(" ", "_").replace("-", "_")

    print_q_table(env, q, title="{} Q-table".format(algorithm_name), max_rows=30)
    save_q_table(
        env,
        q,
        txt_outfile="{}_qtable_output.txt".format(prefix),
        csv_outfile="{}_qtable_output.csv".format(prefix)
    )

    run_greedy_episode(
        env,
        policy_fn,
        max_steps=500,
        title="{} greedy policy".format(algorithm_name),
        path_outfile="{}_gridworld_path.png".format(prefix)
    )

    total_steps = env.policy2gif(
        policy=policy_fn,
        robotInitLoc=env.robotInitLoc,
        filebaseName="{}_vacuum_gridworld".format(prefix),
        maxEpisodeLen=500
    )

    print("\n========== {} GIF 產生完成 ==========".format(algorithm_name))
    print("檔名：{}_vacuum_gridworld.gif".format(prefix))
    print("GIF 總步數 =", total_steps)

    return policy_fn, q


# ------------------------------------------------------------
# main 主流程
# ------------------------------------------------------------
def main():
    np.set_printoptions(precision=3, suppress=True)

    env = build_vacuum_env()
    print_environment_info(env)

    env.render(outfile="vacuum_initial.png")
    print("\n已輸出初始環境圖：vacuum_initial.png")

    # SARSA
    train_and_demo(
        algorithm_name="SARSA",
        train_func=sarsa,
        env=env,
        alpha=0.3,
        progress_note="SARSA 使用下一步實際依 epsilon-greedy 選到的 action 來更新。"
    )



if __name__ == "__main__":
    main()
