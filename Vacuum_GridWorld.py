# -*- coding: utf-8 -*-
"""
Vacuum_GridWorld.py

本檔案是「掃地機器人強化學習環境」的環境類別定義檔。

設計目的：
    讓 Q-learning / Sarsa / Expected Sarsa 等強化學習演算法，
    可以在一個格子世界 GridWorld 裡面訓練掃地機器人。

環境概念：
    1. 地圖由 rows × columns 個格子組成。
    2. 機器人一次只能往東、西、北、南其中一個方向移動一格。
    3. obstacleLocs 表示障礙物格子，機器人不能進入。
    4. targetLocs 表示髒污格子，機器人走到該格就會把它清乾淨。
    5. dirtyMap 用來記錄每個髒污格目前是否仍然髒。
    6. 所有髒污清完後，機器人還要回到起點 Home，episode 才算結束。

Reward 設計：
    - 撞牆：-1
    - 撞障礙物：-1
    - 成功走一步：-1
    - 清到新的髒污格：額外 +3
    - 清完所有髒污並回到 Home：額外 +100

重要提醒：
    這份檔案主要負責「環境本身」。
    它不負責訓練 Q-table，也不負責決定最佳動作。
    訓練演算法通常會寫在 qlearn.py、demo.py 或其他主程式裡。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import os


class Vacuum_GridWorld:
    """
    掃地機器人 GridWorld 環境模擬器。

    這個 class 可以想成「遊戲規則管理員」：
        - 紀錄地圖大小
        - 紀錄障礙物位置
        - 紀錄髒污格位置
        - 紀錄機器人目前位置
        - 根據 action 更新機器人位置
        - 回傳 reward 與 episode 是否結束

    目前版本：
        - 有 obstacle 障礙物
        - 有多個 target dirty cells 髒污目標格
        - 使用 dirtyMap 矩陣紀錄髒污狀態
        - dirtyMap[row, col] = 1 表示該格還是髒的
        - dirtyMap[row, col] = 0 表示該格已經乾淨
        - 走過 dirty target 後，該格設為 0
        - 所有 target 清理完後，必須回到出發點 Home 才結束

    注意：
        這裡的 row / column 是矩陣座標概念：
            row 往下增加
            column 往右增加

        所以：
            [0, 0] 是左上角
            [rows - 1, columns - 1] 是右下角
    """

    # ============================================================
    # 建構子 __init__
    # ------------------------------------------------------------
    # 當我們建立 env = Vacuum_GridWorld(...) 時，會自動執行這裡。
    # 主要工作：
    #   1. 儲存地圖大小 rows / columns
    #   2. 設定機器人起點 robotInitLoc
    #   3. 設定髒污格 targetLocs
    #   4. 設定障礙物 obstacleLocs
    #   5. 檢查輸入是否合法，例如不能超出地圖、不能和障礙物重疊
    #   6. 建立 dirtyMap 髒污矩陣
    #   7. 建立 action 與 move 的對應關係
    # ============================================================
    def __init__(self, rows=10, columns=10, robotInitLoc=None,
                 targetLocs=None, obstacleLocs=None):

        # env config params
        self.rows = rows
        self.columns = columns

        if obstacleLocs is None:
            obstacleLocs = []

        if robotInitLoc is None:
            self.robotInitLoc = [0, 0]
        else:
            self.robotInitLoc = list(robotInitLoc)

        if targetLocs is None:
            self.targetLocs = [[rows - 1, columns - 1]]
        else:
            self.targetLocs = [list(t) for t in targetLocs]

        self.obstacleLocs = [list(o) for o in obstacleLocs]

        # consistency checking
        if self.rows < 1 or self.columns < 1:
            raise Exception("Error: rows, columns must be >= 1")

        if self.offOfMap(self.robotInitLoc):
            raise Exception("Error: robot initial location out of bounds: {}".format(self.robotInitLoc))

        if self.robotInitLoc in self.obstacleLocs:
            raise Exception("Error: robot initial location cannot be on obstacle: {}".format(self.robotInitLoc))

        for target in self.targetLocs:
            if self.offOfMap(target):
                raise Exception("Error: target location out of bounds: {}".format(target))

            if target in self.obstacleLocs:
                raise Exception("Error: Target and obstacle locations cannot overlap: {}".format(target))

        for obstacle in self.obstacleLocs:
            if self.offOfMap(obstacle):
                raise Exception("Error: obstacle location out of bounds: {}".format(obstacle))

        # 建立髒污地圖
        # dirtyMap[row, col] = 1 表示該格目前是髒的
        # dirtyMap[row, col] = 0 表示該格目前是乾淨的
        self.dirtyMap = np.zeros((self.rows, self.columns), dtype=np.int32)

        for target in self.targetLocs:
            self.dirtyMap[target[0], target[1]] = 1

        # computed env params
        self.numPositionStates = rows * columns

        # 理論狀態數：
        # 如果每一個 target 都有「髒 / 乾淨」兩種狀態
        # 則清掃狀態數 = 2 ^ target數量
        self.numTargets = len(self.targetLocs)
        self.numCleaningStates = 2 ** self.numTargets
        self.numStates = self.numPositionStates * self.numCleaningStates

        # 動作設定：固定 2D 四方向
        self.numActions = 4

        self.actions = {
            0: 'E',
            1: 'W',
            2: 'N',
            3: 'S'
        }

        self.moves = {
            0: [0, 1],     # East
            1: [0, -1],    # West
            2: [-1, 0],    # North
            3: [1, 0]      # South
        }

        # initialize robot state
        self.robotLoc = list(self.robotInitLoc)

        # 如果起點本身就是髒污 target，起點也視為已清理
        self.cleanCurrentLoc()

        # list of rendered filenames
        self.renderedFiles = []

        # 紀錄最近一步發生了什麼事件，方便 demo 統計 reward 來源
        self.lastStepInfo = self.make_empty_step_info()

    # ============================================================
    # offOfMap(coord)
    # ------------------------------------------------------------
    # 判斷某個座標 coord = [row, col] 是否超出地圖範圍。
    # 例如：
    #   rows = 10, columns = 10
    #   合法 row 是 0~9
    #   合法 col 是 0~9
    # 若 row 或 col 超出範圍，就回傳 True。
    # ============================================================
    def offOfMap(self, coord):
        """
        Returns True if coordinate is off of grid-world map,
        otherwise False
        """

        if coord[0] >= self.rows or coord[0] < 0:
            return True

        if coord[1] >= self.columns or coord[1] < 0:
            return True

        return False

    # ============================================================
    # positionState()
    # ------------------------------------------------------------
    # 將機器人的二維位置 [row, col] 轉成一個整數編號。
    # 這是傳統 GridWorld 常見的 state index 編碼方式。
    #
    # 公式：
    #   state_index = row * columns + col
    #
    # 不過本程式後來的 currentState() 採用更完整的 tuple 狀態，
    # 因為只知道位置不夠，還要知道哪些髒污已經被清掉。
    # ============================================================
    def positionState(self):
        """
        只回傳機器人所在位置的 state index

        例如：
        rows = 10, columns = 10
        robotLoc = [2, 3]

        positionState = 2 * 10 + 3 = 23
        """

        return self.robotLoc[0] * self.columns + self.robotLoc[1]

    # ============================================================
    # currentState()
    # ------------------------------------------------------------
    # 回傳「強化學習演算法真正使用的完整狀態」。
    #
    # 掃地機器人的狀態不能只看位置，還要知道：
    #   - 哪些 target 還沒清
    #   - 哪些 target 已經清完
    #
    # 因此這裡把狀態設計成：
    #   (機器人位置, dirtyMap攤平成一維tuple)
    #
    # 使用 tuple 的原因：
    #   Python 的 list / numpy array 不能直接當 dictionary key，
    #   但 tuple 可以。
    #   這樣 Q-table 就可以用：
    #       Q[state][action]
    #   或：
    #       Q[(state, action)]
    #   的方式紀錄數值。
    # ============================================================
    def currentState(self):
        """
        回傳完整 state。

        這裡採用比較直觀的表示方式：

        state = robot 位置 + dirtyMap 髒污地圖狀態

        格式：
            (
                (robot_row, robot_col),
                (dirtyMap 攤平成一維 tuple)
            )

        例如：
            robotLoc = [1, 2]

            dirtyMap =
            [[0, 1, 0],
             [0, 0, 1]]

            currentState =
            (
                (1, 2),
                (0, 1, 0, 0, 0, 1)
            )

        這種 state 可以直接當作 dictionary key，
        適合之後用 dict 形式的 Q-table。
        """

        return (tuple(self.robotLoc), tuple(self.dirtyMap.flatten()))

    # ============================================================
    # currentStateDetail()
    # ------------------------------------------------------------
    # 給人類 debug / print 觀察用的狀態。
    # 和 currentState() 不同，這裡保留 dirtyMap 的矩陣形狀，
    # 比較容易直接看出哪一格還是髒的。
    # ============================================================
    def currentStateDetail(self):
        """
        給人看的完整 state。

        回傳：
            robot 目前位置
            dirtyMap 的 copy

        這個比較適合 debug 或 print 出來觀察。
        """

        return (tuple(self.robotLoc), self.dirtyMap.copy())

    # ============================================================
    # currentStateXY()
    # ------------------------------------------------------------
    # 只回傳目前機器人的 [row, col] 座標。
    # 如果只是要畫圖或觀察機器人位置，可以用這個函式。
    # ============================================================
    def currentStateXY(self):
        """
        Returns current robot state in [row, col] coordinate form
        """

        return self.robotLoc

    # ============================================================
    # allTargetsCleaned()
    # ------------------------------------------------------------
    # 檢查所有髒污格是否都清完。
    # dirtyMap 中只要還有任何 1，就代表至少還有一格髒污沒清。
    # 如果 np.sum(dirtyMap) == 0，代表全部清完。
    # ============================================================
    def allTargetsCleaned(self):
        """
        判斷所有 target 是否都已經清理完。

        dirtyMap 裡面如果完全沒有 1，
        代表所有髒污格都清完了。
        """

        return np.sum(self.dirtyMap) == 0

    # ============================================================
    # cleanCurrentLoc()
    # ------------------------------------------------------------
    # 檢查機器人目前所在格子是否為尚未清理的髒污格。
    # 若 dirtyMap[r, c] == 1：
    #   表示這格還髒，機器人到達後會清理它，
    #   因此改成 0，並回傳 True。
    #
    # 若 dirtyMap[r, c] == 0：
    #   表示這格本來就是乾淨的，
    #   不會得到清掃獎勵，回傳 False。
    # ============================================================
    def cleanCurrentLoc(self):
        """
        如果目前機器人所在位置是髒汙格，
        就把該格清乾淨。

        Returns:
            True: 這一步有清到新的髒汙格
            False: 這一步沒有清到新的髒汙格
        """

        r = self.robotLoc[0]
        c = self.robotLoc[1]

        if self.dirtyMap[r, c] == 1:
            self.dirtyMap[r, c] = 0
            return True

        return False

    # ============================================================
    # initState(newState)
    # ------------------------------------------------------------
    # 只重設「機器人位置」，不重設 dirtyMap。
    #
    # 這個函式比較像是局部調整位置用，
    # 如果要完整重新開始一回合 episode，
    # 通常應該使用 reset()。
    #
    # newState 可以有三種情況：
    #   1. None：回到原本起點 robotInitLoc
    #   2. int：把整數 state index 轉成 [row, col]
    #   3. list / tuple：直接指定 [row, col]
    # ============================================================
    def initState(self, newState=None):
        """
        Reset robot location only.

        注意：
        這個 function 只改 robot 位置，
        不會重設 dirtyMap。

        If newState is None:
            return to robotInitLoc.

        If newState is type int:
            assume it is only a position state value.

        Otherwise:
            assume the state is specified using [row, col].
        """

        if newState is None:
            self.robotLoc = list(self.robotInitLoc)

        elif type(newState) == int:
            tmp = [newState // self.columns, newState % self.columns]

            if self.offOfMap(tmp):
                raise Exception("Error: initState: robot location out of bounds: {}".format(tmp))

            if tmp in self.obstacleLocs:
                raise Exception("Error: initState: robot location cannot be on obstacle: {}".format(tmp))

            self.robotLoc = tmp

        else:
            if self.offOfMap(newState):
                raise Exception("Error: initState: robot location out of bounds: {}".format(newState))

            if list(newState) in self.obstacleLocs:
                raise Exception("Error: initState: robot location cannot be on obstacle: {}".format(newState))

            self.robotLoc = list(newState)

    # ============================================================
    # make_empty_step_info()
    # ------------------------------------------------------------
    # 建立一個 dictionary，用來記錄「最近一步 step 發生什麼事」。
    #
    # step() 原本只回傳：
    #   next_state, reward, done
    #
    # 但為了報告或 demo 時能統計得更清楚，
    # 這裡額外記錄：
    #   - 是否撞牆
    #   - 是否撞障礙物
    #   - 是否成功移動
    #   - 是否清到新髒污
    #   - 是否完成任務回家
    #   - 各項 reward 來源
    #
    # 這不會影響 Q-learning 的基本介面，
    # 只是讓後續分析 reward 來源更方便。
    # ============================================================
    def make_empty_step_info(self):
        """
        建立單步事件紀錄。

        這不會改變 step() 原本回傳的格式，
        仍然是 next_state, reward, done。
        demo 程式可以讀 self.lastStepInfo 來統計：
            - 是否撞牆
            - 是否撞障礙物
            - 是否成功移動
            - 是否清到新的髒污格
            - 是否拿到回家完成獎勵
        """

        return {
            "hit_wall": False,
            "hit_obstacle": False,
            "moved": False,
            "cleaned_new_target": False,
            "finished_home": False,
            "step_penalty": 0.0,
            "wall_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "clean_reward": 0.0,
            "finish_reward": 0.0,
            "total_reward": 0.0
        }

    # ============================================================
    # reset(initLoc)
    # ------------------------------------------------------------
    # 完整重置環境，通常用在每一個 episode 開始前。
    #
    # reset() 會做三件事：
    #   1. 把 dirtyMap 恢復成一開始還沒清掃的狀態
    #   2. 把 robotLoc 放回起點，或指定的 initLoc
    #   3. 清空之前 render 存下來的圖片紀錄
    #
    # 注意：
    #   如果起點剛好也是 target，reset 後會立即視為已清理。
    # ============================================================
    def reset(self, initLoc=None):
        """
        Reset entire environment:

        1. Robot 回到初始位置，除非有指定 initLoc
        2. dirtyMap 回到一開始的髒污狀態
        3. renderedFiles 清空
        """

        # 重建 dirtyMap
        self.dirtyMap = np.zeros((self.rows, self.columns), dtype=np.int32)

        for target in self.targetLocs:
            self.dirtyMap[target[0], target[1]] = 1

        # 重設 robot 位置
        if initLoc is None:
            self.robotLoc = list(self.robotInitLoc)
        else:
            self.initState(initLoc)

        # 如果 reset 後的位置本身是髒污格，直接視為已清理
        self.cleanCurrentLoc()

        # 清空 render 暫存檔案列表
        self.renderedFiles = []

        # 重設最近一步事件紀錄
        self.lastStepInfo = self.make_empty_step_info()

    # ============================================================
    # step(action)
    # ------------------------------------------------------------
    # 這是強化學習環境最核心的函式。
    #
    # 外部演算法每選一個 action，就會呼叫一次：
    #   next_state, reward, done = env.step(action)
    #
    # action 對應：
    #   0: E，往右 / 往東
    #   1: W，往左 / 往西
    #   2: N，往上 / 往北
    #   3: S，往下 / 往南
    #
    # step() 會依序處理：
    #   1. 檢查目前是否已經完成任務
    #   2. 檢查 action 是否合法
    #   3. 根據 action 計算下一格座標
    #   4. 判斷是否撞牆
    #   5. 判斷是否撞障礙物
    #   6. 若可移動，更新機器人位置
    #   7. 計算走一步的扣分
    #   8. 若清到新髒污，加清掃獎勵
    #   9. 若全部清完且回到 Home，加完成獎勵並結束 episode
    # ============================================================
    def step(self, action):
        """
        Increment robot state using specified action.

        Returns:
            next_state, reward, done

        reward 設定：
            - 撞牆：-1
            - 撞障礙物：-1
            - 每成功走一步：-1
            - 清到新的髒污格：額外 +3
            - 清完所有髒污並回到終點 Home：額外 +100，done=True

        補充：
            step() 仍然維持原本的三個回傳值，避免 qlearn.py 需要大改。
            但每一步會額外把事件細節存在 self.lastStepInfo，
            給 demo 程式統計碰牆、碰障礙物、清掃格數與各自 reward。
        """

        step_info = self.make_empty_step_info()

        # 如果所有 target 已清理，且目前已經在出發地，episode 結束
        if self.allTargetsCleaned() and self.robotLoc == self.robotInitLoc:
            step_info["finished_home"] = True
            self.lastStepInfo = step_info
            return self.currentState(), 0.0, True

        # 檢查 action 是否合理
        if action not in self.moves:
            raise Exception("Error: invalid action: {}".format(action))

        # 根據 action 計算下一個位置
        newRobotLoc = [
            self.robotLoc[0] + self.moves[action][0],
            self.robotLoc[1] + self.moves[action][1]
        ]

       # If the new location is outside the map, it means you've hit a wall. The location remains the same, and the reward = -1.
        if self.offOfMap(newRobotLoc):
            reward = -1.0
            step_info["hit_wall"] = True
            step_info["wall_penalty"] = reward
            step_info["total_reward"] = reward
            self.lastStepInfo = step_info
            return self.currentState(), reward, False

        # If the new location is obstacle. The location remains the same, and the reward = -1.
        if newRobotLoc in self.obstacleLocs:
            reward = -1.0
            step_info["hit_obstacle"] = True
            step_info["obstacle_penalty"] = reward
            step_info["total_reward"] = reward
            self.lastStepInfo = step_info
            return self.currentState(), reward, False

        self.robotLoc = newRobotLoc         # Update robot position
        step_info["moved"] = True

        reward = -1.0       # For every successful step, deduct 1 point.
        done = False
        step_info["step_penalty"] = -1.0
        
        cleanedNewTarget = self.cleanCurrentLoc()       # If the current location is a dirty compartment that has not been cleaned, clean it.

        # 3 extra points for cleaning a new dirty compartment
        if cleanedNewTarget:
            reward += 3.0
            step_info["cleaned_new_target"] = True
            step_info["clean_reward"] = 3.0

        # If all targets are cleared and the game returns to the endpoint, the episode ends.
        if self.allTargetsCleaned() and self.robotLoc == self.robotInitLoc:
            reward += 100.0
            done = True
            step_info["finished_home"] = True
            step_info["finish_reward"] = 100.0

        step_info["total_reward"] = reward
        self.lastStepInfo = step_info

        return self.currentState(), reward, done


    # ============================================================
    # render(outfile)
    # ------------------------------------------------------------
    # 畫出目前環境狀態。
    #
    # 用途：
    #   - outfile=None 時，直接顯示圖片
    #   - outfile="xxx.png" 時，存成圖片檔
    #
    # 圖中元素：
    #   - 白色：一般可走格
    #   - 金色 Dirty：尚未清理的髒污格
    #   - 淺綠色 Clean：已清理的 target 格
    #   - 灰色 Obstacle：障礙物
    #   - 藍框圓形 Robot：掃地機器人
    #   - Home：機器人起點，也是最後要回來的位置
    # ============================================================
    def render(self, outfile=None):
        """
        畫出目前 Vacuum_GridWorld 畫面。

        這裡是你要的掃地機器人環境介面：
        - 每格大小一致
        - obstacle 佔滿整格
        - dirty / clean 佔滿整格
        - robot 是接近一格大小的圓形
        - Home 會用文字標示
        """

        fig, ax = plt.subplots(figsize=(self.columns, self.rows))

        # ------------------------------------------------------------
        # 1. 畫背景格子
        # ------------------------------------------------------------
        for r in range(self.rows):
            for c in range(self.columns):

                # 預設白色空地
                facecolor = "white"
                label = ""

                # 如果是 targetLocs，要依 dirtyMap 決定髒或乾淨
                if [r, c] in self.targetLocs:
                    if self.dirtyMap[r, c] == 1:
                        facecolor = "gold"
                        label = "Dirty"
                    else:
                        facecolor = "lightgreen"
                        label = "Clean"

                # 如果是 obstacle，蓋成灰色
                if [r, c] in self.obstacleLocs:
                    facecolor = "gray"
                    label = "Obstacle"

                # 畫整格方塊
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1
                )
                ax.add_patch(rect)

                # 寫 Dirty / Clean / Obstacle
                if label != "":
                    ax.text(
                        c,
                        r,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8
                    )

        # ------------------------------------------------------------
        # 2. 畫 Home
        # ------------------------------------------------------------
        home_r = self.robotInitLoc[0]
        home_c = self.robotInitLoc[1]

        ax.text(
            home_c,
            home_r - 0.32,
            "Home",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round",
                facecolor="lightblue",
                edgecolor="black",
                alpha=0.8
            )
        )

        # ------------------------------------------------------------
        # 3. 畫 Robot
        # ------------------------------------------------------------
        robot_r = self.robotLoc[0]
        robot_c = self.robotLoc[1]

        # 半徑 0.45，直徑 0.9，接近一格大小
        robot_circle = Circle(
            (robot_c, robot_r),
            radius=0.45,
            facecolor="white",
            edgecolor="blue",
            linewidth=2
        )
        ax.add_patch(robot_circle)

        ax.text(
            robot_c,
            robot_r,
            "Robot",
            ha="center",
            va="center",
            fontsize=8
        )

        # ------------------------------------------------------------
        # 4. 設定座標與格線
        # ------------------------------------------------------------
        ax.set_xlim(-0.5, self.columns - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)

        ax.set_xticks(np.arange(0, self.columns, 1))
        ax.set_yticks(np.arange(0, self.rows, 1))

        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")

        ax.set_title("Vacuum GridWorld")

        plt.tight_layout()

        # ------------------------------------------------------------
        # 5. 顯示或存檔
        # ------------------------------------------------------------
        if outfile is None:
            plt.show()
        else:
            plt.savefig(outfile)
            self.renderedFiles.append(outfile)

        plt.cla()
        plt.close("all")
        
    # ============================================================
    # render_path(path, outfile)
    # ------------------------------------------------------------
    # 將一整段機器人路徑畫成靜態圖片。
    #
    # path 格式：
    #   [
    #       [row0, col0],
    #       [row1, col1],
    #       [row2, col2],
    #       ...
    #   ]
    #
    # 主要設計：
    #   1. 用箭頭表示每一步移動方向
    #   2. 用數字標出 step 編號
    #   3. 重複走同一條邊時，自動偏移，避免線段完全重疊
    #   4. 多次走到同一格時，step label 自動錯開
    #   5. 第一次到某格使用紅色
    #   6. 重複到達某格使用深色循環，方便辨識回訪路徑
    # ============================================================
    def render_path(self, path, outfile="gridworld_path.png"):
        """
        畫出 GridWorld，並用比較清楚的方式表示機器人的行走路徑。
    
        改良重點：
        1. 可走格 / target 格改成白色，避免綠底干擾路徑
        2. 路徑用線段 + 小箭頭表示
        3. 初始路徑顏色為紅色
        4. 如果下一步走到以前走過的格子，從那一步開始切換顏色
        5. 如果下一步沒有走到以前走過的格子，就保持目前顏色
        6. 同一條邊重複走，包含來回走，會自動偏移
        7. step 編號若出現在同一格，會自動錯開
        8. Start / End 若在同一格，會分開顯示
        """
    
        fig, ax = plt.subplots(figsize=(self.columns, self.rows))
    
        # ------------------------------------------------------------
        # 1. 畫背景格子
        # ------------------------------------------------------------
        for r in range(self.rows):
            for c in range(self.columns):
    
                # 預設可走格都是白色
                facecolor = "white"
                label = ""
    
                # target 格也改成白色，不再用綠色
                if [r, c] in self.targetLocs:
                    facecolor = "white"
                    label = ""
    
                # obstacle 保留灰色
                if [r, c] in self.obstacleLocs:
                    facecolor = "gray"
                    label = "Obs"
    
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1
                )
                ax.add_patch(rect)
    
                if label != "":
                    ax.text(
                        c,
                        r,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                        zorder=2
                    )
    
        # ------------------------------------------------------------
        # 2. 畫 Home
        # ------------------------------------------------------------
        home_r = self.robotInitLoc[0]
        home_c = self.robotInitLoc[1]
    
        ax.text(
            home_c,
            home_r - 0.36,
            "Home",
            ha="center",
            va="center",
            fontsize=8,
            zorder=20,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="lightblue",
                edgecolor="black",
                alpha=0.90
            )
        )
    
        # ------------------------------------------------------------
        # 3. 小工具：產生路徑線段的對稱偏移量
        # ------------------------------------------------------------
        def get_symmetric_offset(k, gap=0.09):
            """
            k = 0,1,2,3,4,...
            回傳：
                0 ->  0
                1 -> +gap
                2 -> -gap
                3 -> +2gap
                4 -> -2gap
    
            這樣重複走同一條邊時，不會全部疊在一起。
            """
            if k == 0:
                return 0.0
    
            level = (k + 1) // 2
    
            if k % 2 == 1:
                return level * gap
            else:
                return -level * gap
    
        # ------------------------------------------------------------
        # 4. 畫路徑線段與箭頭
        #
        # 顏色規則：
        # - 如果下一格是第 1 次走到：箭頭顏色用紅色
        # - 如果下一格是第 2 次以上走到：依「到達該格的次數」換不同深色
        # - 重複到達顏色以 5 次為循環
        # - 重複到達用色不含紅色，也不用淺色系
        #
        # 例：
        #   第 1 次到某格：紅色
        #   第 2 次到某格：深藍
        #   第 3 次到某格：紫色
        #   第 4 次到某格：暗綠
        #   第 5 次到某格：棕色
        #   第 6 次到某格：黑色
        #   第 7 次到某格：回到深藍
        # ------------------------------------------------------------
        edge_count = {}
    
        # 紀錄每個格子已經被走到幾次
        cell_visit_count = {}
    
        if len(path) > 0:
            start_cell = tuple(path[0])
            cell_visit_count[start_cell] = 1
    
        # 第一次走到某格的顏色
        first_visit_color = "red"
    
        # 第 2 次以上走到某格時使用的顏色
        # 注意：這裡刻意不要放紅色，也不要放淺色
        revisit_color_cycle = [
            "#1f77b4",   # 深藍，第 2 次
            "#6a3d9a",   # 紫色，第 3 次
            "#006400",   # 暗綠，第 4 次
            "#8b4513",   # 棕色，第 5 次
            "#000000"    # 黑色，第 6 次
        ]
    
        for i in range(len(path) - 1):
    
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
    
            # 如果原地不動，代表撞牆或撞障礙物，這裡不畫箭頭
            if r1 == r2 and c1 == c2:
                continue
    
            next_cell = (r2, c2)
    
            # --------------------------------------------------------
            # 依照「下一格目前是第幾次被走到」決定箭頭顏色
            # --------------------------------------------------------
            previous_visit_count = cell_visit_count.get(next_cell, 0)
            next_visit_count = previous_visit_count + 1
    
            if next_visit_count == 1:
                segment_color = first_visit_color
            else:
                color_idx = (next_visit_count - 2) % len(revisit_color_cycle)
                segment_color = revisit_color_cycle[color_idx]
    
            # 更新下一格的走訪次數
            cell_visit_count[next_cell] = next_visit_count
    
            # --------------------------------------------------------
            # 用「無方向邊」當 key
            # A->B 與 B->A 視為同一條邊，避免來回箭頭重疊
            # --------------------------------------------------------
            p1 = (r1, c1)
            p2 = (r2, c2)
    
            if p1 <= p2:
                edge_key = (p1, p2)
            else:
                edge_key = (p2, p1)
    
            count = edge_count.get(edge_key, 0)
            edge_count[edge_key] = count + 1
    
            offset = get_symmetric_offset(count, gap=0.09)
    
            # matplotlib 座標：
            # x 是 column
            # y 是 row
            x1 = c1
            y1 = r1
            x2 = c2
            y2 = r2
    
            dx = x2 - x1
            dy = y2 - y1
    
            # 水平移動：上下偏移
            if dy == 0:
                y1 += offset
                y2 += offset
    
            # 垂直移動：左右偏移
            if dx == 0:
                x1 += offset
                x2 += offset
    
            # 把線段稍微縮短，避免箭頭壓滿整格
            shrink = 0.18
    
            if dx > 0:       # 往右
                x1 += shrink
                x2 -= shrink
            elif dx < 0:     # 往左
                x1 -= shrink
                x2 += shrink
    
            if dy > 0:       # 往下
                y1 += shrink
                y2 -= shrink
            elif dy < 0:     # 往上
                y1 -= shrink
                y2 += shrink
    
            # 畫主線段
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=segment_color,
                linewidth=2.2,
                alpha=0.92,
                zorder=3
            )
    
            # 在中間畫小箭頭
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
    
            arrow_dx = (x2 - x1) * 0.35
            arrow_dy = (y2 - y1) * 0.35
    
            ax.arrow(
                mid_x - arrow_dx / 2.0,
                mid_y - arrow_dy / 2.0,
                arrow_dx,
                arrow_dy,
                head_width=0.08,
                head_length=0.10,
                fc=segment_color,
                ec=segment_color,
                linewidth=0,
                length_includes_head=True,
                zorder=4
            )
    
            # 每一步落點加一個小圓點
            ax.plot(
                c2,
                r2,
                marker="o",
                markersize=3,
                color=segment_color,
                zorder=5
            )
    
        # ------------------------------------------------------------
        # 5. 小工具：同一格多個 step label 時自動錯開
        # ------------------------------------------------------------
        def get_label_offset(k, base=0.18, extra=0.10):
            """
            根據同一格內第 k 個標籤，回傳 (dx, dy)。
    
            位置順序大致為：
                0: 右上
                1: 左上
                2: 右下
                3: 左下
                4: 更右上
                5: 更左上
                6: 更右下
                7: 更左下
    
            注意：
                y 軸在圖上是往下增加，所以：
                dy < 0 表示往上
                dy > 0 表示往下
            """
    
            patterns = [
                (+base, -base),
                (-base, -base),
                (+base, +base),
                (-base, +base),
                (+base + extra, -base - extra),
                (-base - extra, -base - extra),
                (+base + extra, +base + extra),
                (-base - extra, +base + extra),
            ]
    
            if k < len(patterns):
                return patterns[k]
    
            # 如果同一格真的超過 8 個標籤，就往外圈擴散
            ring = k // len(patterns)
            idx = k % len(patterns)
    
            dx, dy = patterns[idx]
            scale = 1.0 + 0.5 * ring
    
            return dx * scale, dy * scale
    
        # ------------------------------------------------------------
        # 6. 標示部分 step 編號
        # ------------------------------------------------------------
        if len(path) > 0:
    
            # 紀錄每一格已經放了幾個 step label
            cell_label_count = {}
    
            for i, loc in enumerate(path):
    
                r, c = loc
                show_label = True
    
                if show_label:
    
                    key = (r, c)
                    count_here = cell_label_count.get(key, 0)
                    dx_label, dy_label = get_label_offset(count_here)
    
                    cell_label_count[key] = count_here + 1
    
                    ax.text(
                        c + dx_label,
                        r + dy_label,
                        str(i),
                        fontsize=7,
                        color="blue",
                        ha="center",
                        va="center",
                        zorder=30,
                        clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            edgecolor="blue",
                            alpha=0.90
                        )
                    )
    
        # ------------------------------------------------------------
        # 7. 標出 Start 與 End
        # ------------------------------------------------------------
        if len(path) > 0:
    
            start_r, start_c = path[0]
            end_r, end_c = path[-1]
    
            # 起點 marker
            ax.plot(
                start_c,
                start_r,
                marker="o",
                markersize=9,
                color="blue",
                zorder=25
            )
    
            # 終點 marker
            ax.plot(
                end_c,
                end_r,
                marker="s",
                markersize=9,
                color="red",
                zorder=26
            )
    
            # 如果起點和終點是同一格，通常代表最後回到 Home
            if start_r == end_r and start_c == end_c:
    
                ax.text(
                    start_c - 0.22,
                    start_r + 0.34,
                    "Start",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue",
                    fontweight="bold",
                    zorder=35,
                    clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor="blue",
                        alpha=0.85
                    )
                )
    
                ax.text(
                    end_c + 0.22,
                    end_r + 0.34,
                    "End",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                    zorder=35,
                    clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor="red",
                        alpha=0.85
                    )
                )
    
            else:
    
                ax.text(
                    start_c - 0.12,
                    start_r + 0.32,
                    "Start",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue",
                    fontweight="bold",
                    zorder=35,
                    clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor="blue",
                        alpha=0.85
                    )
                )
    
                ax.text(
                    end_c + 0.12,
                    end_r + 0.32,
                    "End",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                    zorder=35,
                    clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor="red",
                        alpha=0.85
                    )
                )
    
        # ------------------------------------------------------------
        # 8. 座標設定
        # ------------------------------------------------------------
        ax.set_xlim(-0.5, self.columns - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
    
        ax.set_xticks(np.arange(0, self.columns, 1))
        ax.set_yticks(np.arange(0, self.rows, 1))
    
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")
    
        ax.set_title("Vacuum GridWorld Path")
    
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)
    
        plt.cla()
        plt.close("all")

    # ============================================================
    # rendered2gif(outfile, framePeriod)
    # ------------------------------------------------------------
    # 將 render() 產生的一連串 PNG 圖片合成 GIF 動畫。
    #
    # renderedFiles：
    #   每次 render(outfile=...) 存圖時，
    #   檔名會被加入 self.renderedFiles。
    #
    # framePeriod：
    #   每張圖片停留時間，單位是毫秒。
    #   例如 500 代表每一步畫面停留 0.5 秒。
    # ============================================================
    def rendered2gif(self, outfile, framePeriod=500):
        """
        Convert sequence of static rendered frames to animated gif,
        with animation period of framePeriod in milliseconds.
        """

        if len(self.renderedFiles) == 0:
            print('rendered2gif: empty filelist, nothing to do!')
            return

        imageFrames = []

        for imageFile in self.renderedFiles:
            imageFrames.append(Image.open(imageFile))

        imageFrames[0].save(
            outfile,
            format='gif',
            append_images=imageFrames[1:],
            save_all=True,
            duration=framePeriod
        )

    # ============================================================
    # policy2gif(policy, robotInitLoc, filebaseName, maxEpisodeLen)
    # ------------------------------------------------------------
    # 根據已訓練好的 policy，讓機器人從起點開始走，
    # 並把整個行走過程輸出成：
    #   1. 每一步 PNG 圖片
    #   2. 一個 GIF 動畫
    #
    # policy 可以是：
    #   1. function：
    #       action = policy(state)
    #
    #   2. dictionary：
    #       action = policy[state]
    #
    # maxEpisodeLen：
    #   用來限制最多走幾步，避免 policy 還沒學好時無限迴圈。
    #
    # 回傳值：
    #   i，也就是這次 episode 總共走了幾步。
    # ============================================================
    def policy2gif(self, policy, robotInitLoc, filebaseName, maxEpisodeLen=None):
        """
        From specified initial state, this method will walk the robot through
        the environment using the given deterministic policy.
        
        因為 currentState() 現在不是 int，
        而是：
            ((robot_row, robot_col), dirtyMap_tuple)

        所以 policy 建議用 dictionary：

            policy[state] = action

        例如：
            state = env.currentState()
            policy[state] = 0

        也可以傳入 function：

            action = policy(state)

        Returns:
            Total number of steps taken during this policy-walk episode
    
        這版會：
        1. 產生 GIF
        2. 另外建立資料夾，保留每一步 PNG
        """
    
        # reset the environment
        self.reset(robotInitLoc)
    
        # 建立專門存每一步 PNG 的資料夾
        png_dir = filebaseName + "_png_steps"
        os.makedirs(png_dir, exist_ok=True)
    
        # walk the policy and generate animated gif
        self.render(
            outfile=os.path.join(
                png_dir,
                "{}_step_{:03d}.png".format(filebaseName, 0)
            )
        )
    
        done = False
        i = 0
    
        while not done:
    
            state = self.currentState()
    
            # policy 如果是 function，就直接呼叫
            if callable(policy):
                action = policy(state)
    
            # policy 如果是 dictionary，就用 state 查 action
            elif isinstance(policy, dict):
                if state not in policy:
                    raise Exception("Error: current state not found in policy dictionary: {}".format(state))
    
                action = policy[state]
    
            else:
                raise Exception("Error: policy must be a function or dictionary.")
    
            state, reward, done = self.step(action)
    
            i += 1
    
            self.render(
                outfile=os.path.join(
                    png_dir,
                    "{}_step_{:03d}.png".format(filebaseName, i)
                )
            )
    
            if maxEpisodeLen is not None and i >= maxEpisodeLen:
                done = True
    
        # GIF 仍然輸出在目前資料夾
        self.rendered2gif(outfile=filebaseName + '.gif')
    
        # 清空 renderedFiles，避免下次 GIF 混到舊圖片
        self.renderedFiles = []
    
        # reset the environment before exit
        self.reset(robotInitLoc)
    
        return i