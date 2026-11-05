# 匯入 NumPy：主要用來處理陣列、計算總和、尋找最大 Q 值等數值運算
import numpy as np

# 匯入自訂的掃地機器人 GridWorld 環境類別
# Vacuum_GridWorld 負責定義房間格子、障礙物、髒污格、機器人位置、reward、畫圖等功能
from Vacuum_GridWorld import Vacuum_GridWorld

# 匯入 Q-learning 訓練函式
# qlearn() 會根據環境互動結果訓練出 Q-table，最後回傳 policy 與 q 值資料
from qlearn import qlearn


# ------------------------------------------------------------
# 函式：action_name
# 功能：把 action 的數字編號轉成人類看得懂的方向文字
# 例如：0 可能代表 E，1 可能代表 W，實際對應以 env.actions 為準
# ------------------------------------------------------------
def action_name(env, action):
    """
    將 action 編號轉成文字。
    """

    # env.actions 是環境內部定義的 action 對照表
    # 例如 env.actions[0] = 'E'，代表 action 0 是往東走
    return env.actions[action]


# ------------------------------------------------------------
# 函式：print_environment_info
# 功能：印出目前掃地機器人環境的重要設定，方便確認環境是否建立正確
# 用途：在訓練前先檢查 rows、columns、障礙物、髒污格、action 等資訊
# ------------------------------------------------------------
def print_environment_info(env):
    """
    印出 Vacuum_GridWorld 環境資訊。
    """

    # 印出環境標題，讓 console 輸出比較好閱讀
    print("========== Vacuum_GridWorld 環境資訊 ==========")

    # rows：房間的列數，也就是垂直方向有幾格
    print("rows =", env.rows)

    # columns：房間的行數，也就是水平方向有幾格
    print("columns =", env.columns)

    # Home：機器人的初始位置，也是最後完成任務時需要回到的位置
    print("Home =", env.robotInitLoc)

    # targetLocs：需要清掃的髒污格座標列表
    print("targetLocs =", env.targetLocs)

    # obstacleLocs：障礙物座標列表，機器人不能進入這些格子
    print("obstacleLocs =", env.obstacleLocs)

    # actions：動作編號與方向的對照表，例如 {0:'E', 1:'W', 2:'N', 3:'S'}
    print("actions =", env.actions)

    # numActions：可用動作總數，本專題通常是東西南北四個方向
    print("numActions =", env.numActions)

    # numStates：理論上的 state 數量，由機器人位置與髒污狀態組合而成
    print("numStates 理論值 =", env.numStates)

    # currentStateDetail：印出目前 state 的詳細內容，方便確認初始狀態
    print("currentStateDetail =", env.currentStateDetail())



# ------------------------------------------------------------
# 函式：decode_q_state
# 功能：把 Q-table 裡面較難直接閱讀的 state 拆解成直覺資訊
# Q-learning 的 state 通常包含：
#   1. 機器人目前位置
#   2. dirtyMap 攤平成 tuple 後的髒污狀態
# 這個函式會把它轉成：機器人座標、剩餘髒污數、已清掃數
# ------------------------------------------------------------
def decode_q_state(env, state):
    """
    將 Q-table 裡的 state 拆成人看得懂的資訊。

    state 形式：
        ((robot_row, robot_col), dirtyMap_flat_tuple)

    回傳：
        robot_loc:      [row, col]
        remaining_dirty: 還沒清掃的髒污格數
        cleaned_count:   已清掃的髒污格數
    """

    # state 由兩部分組成：
    # robot_pos：機器人目前座標，例如 (2, 0)
    # dirty_flat：dirtyMap 攤平成一維 tuple 後的結果，其中 1 代表尚未清掃，0 代表已清掃或不是髒污格
    robot_pos, dirty_flat = state

    # 將機器人座標從 tuple 轉成 list，方便後面輸出成 [row, col] 的形式
    robot_loc = [robot_pos[0], robot_pos[1]]

    # np.sum(dirty_flat) 會把 dirty_flat 中所有 1 加總
    # 因為 1 代表尚未清掃，所以總和就是「剩下幾格髒污還沒清」
    remaining_dirty = int(np.sum(dirty_flat))

    # env.numTargets 是一開始總共有幾個目標髒污格
    # 已清掃數 = 總髒污格數 - 剩餘髒污格數
    cleaned_count = env.numTargets - remaining_dirty

    # 回傳拆解後的人類可讀資訊
    return robot_loc, remaining_dirty, cleaned_count


# ------------------------------------------------------------
# 函式：print_q_table
# 功能：把 Q-table 以表格形式印在 console 中，方便觀察每個 state 的 Q 值
# 注意：Q-table 可能非常大，所以只預覽前 max_rows 筆
# ------------------------------------------------------------
def print_q_table(env, q, max_rows=30):
    """
    將 Q-table 印成比較好讀的表格。

    注意：
        Q-table 可能非常大，所以 console 只印前 max_rows 筆。
        完整 Q-table 會另外用 save_q_table() 存成文字檔。
    """

    # 印出 Q-table 預覽標題
    print("\n========== Q-table 預覽 ==========")

    # len(q) 是 Q-table 實際有紀錄到的 state 數
    # 實際 state 數可能小於理論 state 數，因為訓練不一定會走到所有狀態
    print("Q-table 總 state 數 =", len(q))

    # 提醒使用者 console 只顯示部分資料，完整資料已另外存檔
    print("只預覽前 {} 筆，完整內容請看 qtable_output.txt / qtable_output.csv".format(max_rows))

    # 建立表格欄位名稱
    # Q(E), Q(W), Q(N), Q(S) 分別代表往東、西、北、南的 Q 值
    # best_action 代表目前 state 下 Q 值最大的建議動作
    header = (
        "{:<6s} {:<10s} {:<12s} {:<10s} "
        "{:>10s} {:>10s} {:>10s} {:>10s} {:<12s}"
    ).format(
        "index", "robot", "remain_dirty", "cleaned",
        "Q(E)", "Q(W)", "Q(N)", "Q(S)", "best_action"
    )

    # 印出表格標題列
    print(header)

    # 印出分隔線，長度與表格標題相同
    print("-" * len(header))

    # 排序規則：先看已清掃數，再看 robot 位置
    # 這樣輸出的 Q-table 比較有順序，不會是 dictionary 原本較混亂的順序
    sorted_items = sorted(
        q.items(),
        key=lambda item: (
            decode_q_state(env, item[0])[2],
            decode_q_state(env, item[0])[0][0],
            decode_q_state(env, item[0])[0][1]
        )
    )

    # 只取前 max_rows 筆資料印出，避免 console 輸出太長
    for idx, (state, q_values) in enumerate(sorted_items[:max_rows]):
        # 將 state 解碼成機器人位置、剩餘髒污數、已清掃數
        robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)

        # np.argmax(q_values) 會找出 Q 值最大的 action index
        # 這就是 greedy policy 在此 state 會選擇的動作
        best_action = int(np.argmax(q_values))

        # 將最佳 action 同時顯示成「編號(方向)」格式，例如 0(E)
        best_action_text = "{}({})".format(best_action, action_name(env, best_action))

        # 印出單一 state 的資訊與四個方向的 Q 值
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
# 函式：save_q_table
# 功能：將完整 Q-table 輸出成 txt 與 csv 檔
# txt：適合直接用記事本閱讀
# csv：適合用 Excel 開啟、篩選、排序或做表格分析
# ------------------------------------------------------------
def save_q_table(env, q, txt_outfile="qtable_output.txt", csv_outfile="qtable_output.csv"):
    """
    將完整 Q-table 輸出成：
        1. qtable_output.txt：人比較好讀
        2. qtable_output.csv：可用 Excel 開啟
    """

    # 和 print_q_table() 使用相同排序方式
    # 讓 console 預覽與輸出檔案的順序一致
    sorted_items = sorted(
        q.items(),
        key=lambda item: (
            decode_q_state(env, item[0])[2],
            decode_q_state(env, item[0])[0][0],
            decode_q_state(env, item[0])[0][1]
        )
    )

    # 輸出文字檔，encoding="utf-8" 可以避免中文亂碼
    with open(txt_outfile, "w", encoding="utf-8") as f:
        # 寫入檔案標題與 Q-table state 數量
        f.write("========== Q-table 完整輸出 ==========\n")
        f.write("Q-table 總 state 數 = {}\n\n".format(len(q)))

        # 寫入表格欄位名稱
        f.write(
            "{:<6s} {:<10s} {:<12s} {:<10s} "
            "{:>10s} {:>10s} {:>10s} {:>10s} {:<12s}\n".format(
                "index", "robot", "remain_dirty", "cleaned",
                "Q(E)", "Q(W)", "Q(N)", "Q(S)", "best_action"
            )
        )

        # 將排序後的每一個 state 與其 Q 值逐筆寫入 txt 檔
        for idx, (state, q_values) in enumerate(sorted_items):
            # 解析 state，取得人類比較容易理解的資訊
            robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)

            # 找出此 state 下 Q 值最大的 action
            best_action = int(np.argmax(q_values))

            # 將最佳 action 轉成「編號(方向)」格式
            best_action_text = "{}({})".format(best_action, action_name(env, best_action))

            # 寫入該筆 state 的完整資料
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

    # 輸出 CSV 檔
    # utf-8-sig 會在檔案開頭加入 BOM，通常可讓 Excel 正確辨識中文編碼
    with open(csv_outfile, "w", encoding="utf-8-sig") as f:
        # CSV 第一列是欄位名稱
        f.write("index,robot_row,robot_col,remain_dirty,cleaned,Q_E,Q_W,Q_N,Q_S,best_action,best_action_name\n")

        # 將每個 state 的資訊寫成逗號分隔格式
        for idx, (state, q_values) in enumerate(sorted_items):
            # 解析 state
            robot_loc, remaining_dirty, cleaned_count = decode_q_state(env, state)

            # 找出 Q 值最大的 action
            best_action = int(np.argmax(q_values))

            # 將資料寫入 csv
            # Q 值使用小數點後 6 位，方便後續比較細微差異
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

    # 在 console 告知使用者檔案已成功輸出
    print("\n已輸出完整 Q-table：")
    print("文字檔：", txt_outfile)
    print("CSV 檔：", csv_outfile)

# ------------------------------------------------------------
# 函式：run_greedy_episode
# 功能：用訓練後得到的 policy 實際跑一次任務
# greedy policy：每一步都選擇目前 state 下 Q 值最大的 action
# 此函式會輸出：
#   1. 每一步的動作、reward、位置與事件
#   2. 各類 reward 統計
#   3. 最後 dirtyMap
#   4. 完整行走路徑
#   5. 路徑圖 gridworld_path.png
# ------------------------------------------------------------
def run_greedy_episode(env, policy_fn, max_steps=500):
    """
    用訓練好的 greedy policy 實際跑一次，
    並用文字印出每一步。

    這版額外統計：
        - 碰牆次數與碰牆 reward 加總
        - 碰障礙物次數與碰障礙物 reward 加總
        - 成功移動步數與移動扣分加總
        - 清掃新髒污格數與清掃 reward 加總
        - 回到 Home 完成任務 reward
    """

    # 將環境重設到初始狀態
    # 包含機器人回到起點、dirtyMap 回復成尚未清掃狀態
    env.reset()

    # 印出 greedy policy 測試標題
    print("\n========== Greedy policy 文字測試 ==========")

    # 印出初始位置
    print("初始位置 =", env.currentStateXY())

    # 印出初始 dirtyMap
    # dirtyMap 中通常 1 代表該格是尚未清掃的髒污格，0 代表沒有髒污或已清掃
    print("初始 dirtyMap:")
    print(env.dirtyMap)
    
    # 紀錄機器人走過的座標
    # 一開始先把初始位置加入 path，後面每走一步再追加新位置
    path = [list(env.currentStateXY())]

    # total_reward 用來累計整趟 greedy 測試的總 reward
    total_reward = 0.0

    # reward 統計
    # 以下變數分別計算各類事件出現的次數
    wall_hit_count = 0
    obstacle_hit_count = 0
    moved_count = 0
    cleaned_count = 0

    # 以下變數分別累計各類事件造成的 reward 或 penalty
    wall_reward_total = 0.0
    obstacle_reward_total = 0.0
    move_penalty_total = 0.0
    clean_reward_total = 0.0
    finish_reward_total = 0.0

    # executed_steps 用來記錄實際執行了幾步
    # 若任務提前完成，會小於 max_steps
    executed_steps = 0

    # 從第 1 步開始執行，最多執行 max_steps 步，避免 policy 不佳時無限迴圈
    for step in range(1, max_steps + 1):

        # 取得目前環境 state
        state = env.currentState()

        # 透過訓練好的 policy_fn 決定此 state 應該採取哪個 action
        action = policy_fn(state)

        # 對環境執行 action
        # next_state：執行後的新 state
        # reward：這一步得到的 reward
        # done：任務是否結束
        next_state, reward, done = env.step(action)

        # env.lastStepInfo 是環境在 step() 後紀錄的事件資訊
        # 例如是否碰牆、碰障礙物、成功移動、清到新髒污、完成回家等
        info = env.lastStepInfo
        
        # 每走一步後，把目前座標記下來
        # 注意：如果碰牆或碰障礙物，機器人位置可能不變，但仍然記錄這一步後的位置
        path.append(list(env.currentStateXY()))

        # 累加總 reward
        total_reward += reward

        # 更新目前已執行步數
        executed_steps = step

        # 如果這一步碰到牆，記錄碰牆次數與碰牆 penalty
        if info["hit_wall"]:
            wall_hit_count += 1
            wall_reward_total += info["wall_penalty"]

        # 如果這一步碰到障礙物，記錄碰障礙物次數與 obstacle penalty
        if info["hit_obstacle"]:
            obstacle_hit_count += 1
            obstacle_reward_total += info["obstacle_penalty"]

        # 如果這一步成功移動，記錄移動步數與每步移動扣分
        if info["moved"]:
            moved_count += 1
            move_penalty_total += info["step_penalty"]

        # 如果這一步清到新的髒污格，記錄清掃次數與清掃 reward
        if info["cleaned_new_target"]:
            cleaned_count += 1
            clean_reward_total += info["clean_reward"]

        # 如果這一步完成「所有髒污清完且回到 Home」，記錄完成任務 reward
        if info["finished_home"]:
            finish_reward_total += info["finish_reward"]

        # event_text 用來組合這一步發生的事件文字
        # 一步可能同時包含多個事件，例如「成功移動 + 清到新髒污」
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

        # 印出單步執行結果
        # 包含 step 編號、action 編號與方向、reward、目前位置、事件、是否完成
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

        # 如果 done=True，代表任務完成，跳出迴圈
        if done:
            print("\n任務完成：所有髒污清完，且機器人已回到 Home。")
            break

    # 印出 reward 統計結果
    print("\n========== Reward 統計 ==========")
    print("實際執行步數 =", executed_steps)
    print("成功移動步數 = {}, 移動扣分合計 = {:.1f}".format(moved_count, move_penalty_total))
    print("碰牆次數 = {}, 碰牆 reward 合計 = {:.1f}".format(wall_hit_count, wall_reward_total))
    print("碰障礙物次數 = {}, 碰障礙物 reward 合計 = {:.1f}".format(obstacle_hit_count, obstacle_reward_total))
    print("清掃新髒污格數 = {}, 清掃 reward 合計 = {:.1f}".format(cleaned_count, clean_reward_total))
    print("完成任務回家 reward 合計 = {:.1f}".format(finish_reward_total))

    # 重新把各類 reward / penalty 加總一次
    # 這是為了檢查統計邏輯是否與 total_reward 一致
    recomputed_total = (
        move_penalty_total
        + wall_reward_total
        + obstacle_reward_total
        + clean_reward_total
        + finish_reward_total
    )

    # 印出重新加總結果與直接累加得到的 total_reward
    print("\nReward 重新加總 = {:.1f}".format(recomputed_total))
    print("Total reward = {:.1f}".format(total_reward))

    # 檢查兩種總 reward 計算方式是否一致
    # abs(...) < 1e-9 是浮點數比較常用寫法，避免小數誤差造成誤判
    if abs(recomputed_total - total_reward) < 1e-9:
        print("檢查結果：Reward 分項加總與 Total reward 一致。")
    else:
        print("警告：Reward 分項加總與 Total reward 不一致，請檢查統計邏輯。")

    # 印出任務結束後的 dirtyMap
    # 若任務成功，理論上所有 target 對應位置都應該已被清成 0
    print("\n最後 dirtyMap:")
    print(env.dirtyMap)
    
    # 印出完整行走路徑
    # Path 000 是初始位置，Path 001 開始是第一步後的位置
    print("\n========== Greedy policy 行走路徑 ==========")
    for idx, loc in enumerate(path):
        print("Path {:03d}: {}".format(idx, loc))
    
    # 呼叫環境的 render_path()，將 path 畫成圖片
    env.render_path(path, outfile="gridworld_path.png")
    print("\n已輸出路徑圖：gridworld_path.png")
    
    # 回傳 path，方便後續若要做其他分析或畫圖時使用
    return path


# ------------------------------------------------------------
# 函式：main
# 功能：整個 qlearn_demo.py 的主流程
# 執行順序：
#   1. 設定輸出格式
#   2. 建立掃地機器人環境
#   3. 自動產生 targetLocs
#   4. 訓練 Q-learning
#   5. 輸出 Q-table
#   6. 使用 greedy policy 測試
#   7. 產生路徑 GIF
# ------------------------------------------------------------
def main():
    """
    qlearn_demo 主程式。

    這份 demo 的重點：
    - 不 import Cat_and_Mouse
    - 不自己畫圖
    - 不顯示 Q-learning 訓練曲線
    - 只使用 Vacuum_GridWorld.render() / policy2gif() 畫圖
    """

    # 設定 NumPy 印出陣列時的格式
    # precision=3：小數點後最多顯示 3 位
    # suppress=True：避免使用科學記號，讓輸出比較容易閱讀
    np.set_printoptions(precision=3, suppress=True)

    # ------------------------------------------------------------
    # 這裡設 True，訓練結束後會顯示 Q-learning 訓練過程圖。
    #
    # 圖中包含：
    #   1. Length：每個 episode 實際走了幾步
    #   2. Total Reward：每個 episode 的總 reward
    #   3. Avg Total Reward：截至目前 episode 為止的平均總 reward
    #
    # 同時會輸出圖片檔：qlearning_training_progress.png
    # ------------------------------------------------------------
    show_plots = True

    # ------------------------------------------------------------
    # 建立 Vacuum_GridWorld 環境
    # rows / columns 決定房間大小
    # robotInitLoc 是掃地機器人的起點，也就是 Home
    # obstacleLocs 是不可進入的障礙物座標
    # ------------------------------------------------------------
    rows = 12
    columns = 12
    robotInitLoc = [2, 0]

    # obstacleLocs 用格子座標描述房間中不能走的位置
    # 座標格式都是 [row, col]
    # row 越大代表越下面，col 越大代表越右邊
    obstacleLocs = [
        # 小桌
        # 左上角 2x2 區域視為小桌，機器人不能進入
        [0, 0], [0, 1], 
        [1, 0], [1, 1],

        # 雙人床
        # 由多個連續格子組成的大型障礙物，代表床佔據的範圍
        [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
        [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
        [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
        [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
        [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],

        # 書桌
        # 房間下方偏左的 2x4 區域
        [10, 3], [10, 4], [10, 5], [10, 6],
        [11, 3], [11, 4], [11, 5], [11, 6],

        # 櫃子1
        # 右上側直向櫃子區域
        [0, 10], [0, 11],
        [1, 10], [1, 11],
        [2, 10], [2, 11],
        [3, 10], [3, 11],
        
        # 櫃子2
        # 右下側直向櫃子區域
        [6, 10], [6, 11],
        [7, 10], [7, 11],
        [8, 10], [8, 11],
        [9, 10], [9, 11],
        [10, 10], [10, 11],
        [11, 10], [11, 11],
        
        # 櫃子3
        # 上方中間偏右的櫃子區域
        [0, 7], [0, 8], [0, 9],
        [1, 7], [1, 8], [1, 9],
        
        # 家電
        # 房間中間偏右的 2x2 區域
        [7, 7], [7, 8], 
        [8, 7], [8, 8],
        
        
        
    ]

    # 把障礙物轉成 set，方便快速判斷
    # tuple(loc) 是因為 list 不能放進 set，但 tuple 可以
    # set 查詢速度比 list 快，適合用來判斷某格是不是障礙物
    obstacleSet = set(tuple(loc) for loc in obstacleLocs)

    # targetLocs = 所有不是障礙物、也不是機器人起點的格子
    # 也就是：全部可走格子 - robotInitLoc
    # 在這份設定中，代表除了障礙物與起點之外，所有可走區域一開始都視為需要清掃
    targetLocs = [
        [r, c]
        for r in range(rows)
        for c in range(columns)
        if (r, c) not in obstacleSet and [r, c] != robotInitLoc
    ]

    # 建立 Vacuum_GridWorld 環境物件
    # env 之後會被 qlearn() 使用，作為訓練時互動的模擬環境
    env = Vacuum_GridWorld(
        rows=rows,
        columns=columns,
        robotInitLoc=robotInitLoc,
        targetLocs=targetLocs,
        obstacleLocs=obstacleLocs
    )

    # 印出環境資訊，方便確認 rows、columns、障礙物、target 是否正確
    print_environment_info(env)

    # ------------------------------------------------------------
    # 先輸出一張初始畫面 PNG
    #
    # 檔名：
    #   vacuum_initial.png
    #
    # 用這張圖確認畫面是不是 Vacuum_GridWorld。
    # ------------------------------------------------------------
    env.render(outfile="vacuum_initial.png")
    print("\n已輸出初始環境圖：vacuum_initial.png")

    # ------------------------------------------------------------
    # Q-learning 訓練
    # ------------------------------------------------------------
    # qlearn() 會讓機器人在環境中反覆試走多個 episode
    # 透過 reward 更新 Q-table，逐漸學到每個 state 下哪個 action 比較好
    policy_fn, q = qlearn(
        # simenv：傳入要訓練的掃地機器人環境
        simenv=env,

        # init_state=None：每個 episode 使用環境預設初始狀態
        init_state=None,

        # gamma：折扣因子，越接近 1 代表越重視未來 reward
        gamma=0.95,

        # epsilon：初始探索率
        # epsilon 越大，訓練初期越常隨機嘗試不同動作
        epsilon=1.0,

        # alpha：學習率，控制新的經驗對 Q 值更新的影響程度
        alpha=0.3,

        # num_episodes：訓練回合數
        # 回合數越多，通常 Q-table 越有機會收斂，但訓練時間也越長
        num_episodes=20000,

        # max_episode_len：單一 episode 最多允許走幾步
        # 避免機器人卡住或繞圈時無限執行
        max_episode_len=500,

        # decayEpsilon=True：讓 epsilon 隨訓練逐漸下降
        # 也就是前期多探索，後期較常利用已學到的好策略
        decayEpsilon=True,

        # showPlots=False：不要顯示訓練曲線圖，避免和 Vacuum_GridWorld 圖混淆
        showPlots=show_plots,

        # prng_seed：隨機種子，讓訓練過程比較容易重現
        prng_seed=789
    )

    # 訓練完成後，印出 Q-table 實際紀錄的 state 數
    print("\n========== 訓練完成 ==========")
    print("Q-table 實際記錄 state 數 =", len(q))

    # ------------------------------------------------------------
    # Q-table 輸出
    # ------------------------------------------------------------
    # 先在 console 預覽前 30 筆，再把完整 Q-table 存成 txt 與 csv
    print_q_table(env, q, max_rows=30)
    save_q_table(env, q, txt_outfile="qtable_output.txt", csv_outfile="qtable_output.csv")

    # ------------------------------------------------------------
    # 用文字測試 policy 是否合理
    # ------------------------------------------------------------
    # 使用訓練完成後的 greedy policy 實際跑一次
    # 觀察它是否能清完所有髒污並回到 Home
    run_greedy_episode(env, policy_fn, max_steps=500)

    # ------------------------------------------------------------
    # 產生 Vacuum_GridWorld GIF
    #
    # 這裡會呼叫：
    #   env.policy2gif(...)
    #
    # 而 policy2gif() 裡面會呼叫：
    #   self.render(...)
    #
    # 所以 GIF 一定是 Vacuum_GridWorld 的畫面介面。
    #
    # 檔名：
    #   vacuum_gridworld_qlearn.gif
    #
    # 我故意不用舊檔名 vacuum_ql.gif，
    # 避免你打開到之前舊的 GIF。
    # ------------------------------------------------------------
    total_steps = env.policy2gif(
        # policy：傳入訓練後得到的 policy_fn
        policy=policy_fn,

        # robotInitLoc：指定 GIF 模擬時機器人的初始位置
        robotInitLoc=env.robotInitLoc,

        # filebaseName：輸出 GIF 的基本檔名，最後會產生 vacuum_gridworld_qlearn.gif
        filebaseName="vacuum_gridworld_qlearn",

        # maxEpisodeLen：GIF 最多模擬幾步，避免策略異常時無限產生圖片
        maxEpisodeLen=500
    )

    # 印出 GIF 輸出結果
    print("\n========== GIF 產生完成 ==========")
    print("檔名：vacuum_gridworld_qlearn.gif")
    print("GIF 總步數 =", total_steps)


# Python 程式進入點
# 當這個檔案被直接執行時，才會呼叫 main()
# 如果未來被其他檔案 import，則不會自動執行 main()
if __name__ == "__main__":
    main()
