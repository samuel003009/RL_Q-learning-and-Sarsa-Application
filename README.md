# 掃地機器人路徑規劃：Q-learning 與 SARSA

用 Q-learning 與 SARSA 訓練掃地機器人，讓它在有家具的 12×12 房間裡清完所有地面，再回到充電座，並比較兩種演算法的路徑長度與理想步數的差距。

<p align="center">
  <img src="images/qlearning_demo.gif" width="480" alt="掃地機器人依 Q-learning 學到的策略清掃房間的動畫">
</p>

## 問題描述

掃地機器人要如何在複雜的室內環境中，找出最短的清掃路徑並回到充電座？

本專題把房間轉成格子世界（GridWorld），機器人只能透過與環境互動得到的 reward 學習，事先不知道地圖，也不知道最佳路徑。

## 環境設定

<table>
  <tr>
    <th>房間配置</th>
    <th>GridWorld 地圖</th>
  </tr>
  <tr>
    <td width="50%">
      <img width="962" height="961" alt="image" src="https://github.com/user-attachments/assets/60fd6c59-634f-4496-8756-4c178606d698" />

    <td width="50%">
      <img width="971" height="972" alt="image" src="https://github.com/user-attachments/assets/6bf8e09d-652c-48f2-bcac-0a9965deedf8" />



  </tr>
</table>

| 項目 | 數量 | 說明 |
| --- | --- | --- |
| 地圖 | 144 格 | 12 × 12，座標為 `[row, col]`，`[0, 0]` 在左上角 |
| 障礙物 | 72 格 | 小桌、雙人床、書桌、三個櫃子、家電 |
| 髒污格 | 71 格 | 除了障礙物與 Home 以外的所有格子 |
| Home | 1 格 | `[2, 0]`，機器人的起點與終點 |

### Reward

| 事件 | Reward |
| --- | --- |
| 撞牆，位置不變 | −1 |
| 撞障礙物，位置不變 | −1 |
| 成功移動一步 | −1 |
| 清到新的髒污格 | 額外 +3 |
| 清完所有髒污並回到 Home | 額外 +100 |

清完所有髒污並回到 Home 時 episode 結束；步數達到 `max_episode_len` 時也會中止。

### State 與 Action

- **State**：`((row, col), dirtyMap 攤平後的 tuple)`。除了機器人位置，還要記錄每一格是否已清掃，否則機器人無法分辨「這格清過了沒」。
- **Action**：`0` 東、`1` 西、`2` 北、`3` 南。
- **Q-table**：理論 state 數高達 144 × 2⁷¹，因此使用 dictionary，只記錄實際走過的 state：`q[state] = [Q(E), Q(W), Q(N), Q(S)]`。

## 演算法

兩者都用 ε-greedy 選擇動作，差別在更新時看的是哪一個下一步。

**Q-learning**（off-policy）：以下一個 state 中最大的 Q 值更新，不管實際會不會走那一步。

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\right]$$

**SARSA**（on-policy）：以下一步實際選到的動作更新，因此探索時的行為也會反映在 Q 值上。

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\right]$$

### 參數

| 參數 | 值 | 說明 |
| --- | --- | --- |
| `gamma` | 0.95 | 折扣因子 |
| `alpha` | 0.3 | 學習率 |
| `epsilon` | 1.0 | 初始探索率，隨 episode 線性遞減至接近 0 |
| `num_episodes` | 8000 | 訓練回合數 |
| `max_episode_len` | 500 | 每回合步數上限 |
| `prng_seed` | 789 | 隨機種子 |

## 實驗結果

以下為 8,000 回合訓練後，用 greedy policy 實際跑一次的結果。

| 演算法 | 理想步數 | 實際步數 | 多走步數 | 總 reward | 清掃格數 | 碰撞次數 |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning | 72 | **94** | 22 | 219 | 71 / 71 | 0 |
| SARSA | 72 | **96** | 24 | 217 | 71 / 71 | 0 |

地圖共有 72 個可走格，每格只走一次並回到 Home 剛好是 72 步，這就是理想步數。

### 行走路徑

路徑上的數字是步數。紅色箭頭代表第一次到達該格，其他顏色代表重複走到已到過的格子，所以紅色以外的線段越多，繞路越多。

<table>
  <tr>
    <th>Q-learning（94 步）</th>
    <th>SARSA（96 步）</th>
  </tr>
  <tr>
    <td><img src="images/qlearning_path.png" alt="Q-learning 的 94 步清掃路徑"></td>
    <td><img src="images/sarsa_path.png" alt="SARSA 的 96 步清掃路徑"></td>
  </tr>
</table>

### 訓練過程

三張圖由上到下是每回合步數、每回合總 reward，以及累積平均 reward。Q-learning 約從第 5,000 回合開始頻繁完成任務，SARSA 則要到第 6,500 回合左右。兩者都在探索率接近 0 的最後幾百回合才穩定下來。

<table>
  <tr>
    <th>Q-learning</th>
    <th>SARSA</th>
  </tr>
  <tr>
    <td><img src="images/qlearning_training_progress.png" alt="Q-learning 訓練曲線"></td>
    <td><img src="images/sarsa_training_progress.png" alt="SARSA 訓練曲線"></td>
  </tr>
</table>

### 訓練回合數的影響

`qlearn_demo.py` 與 `sarsa_esarsa_demo.py` 目前的設定是 20,000 回合，跑出來的結果和上面不同：

| 訓練回合數 | Q-learning | SARSA |
| --- | --- | --- |
| 8,000 | 94 步 | 96 步 |
| 20,000 | 86 步 | 106 步 |

這些都是單一 seed 的結果，回合數增加不一定讓路徑變短，兩者的差距也不宜過度解讀。

## 結論

- **兩種演算法都能完成任務**：清完全部 71 格、回到 Home，途中沒有撞牆或撞到障礙物。
- **都比理想步數多走約三成**：髒污格密集時，機器人容易被附近的 +3 局部獎勵吸引，很難針對全域最短路徑最佳化。
- **Q-learning 較積極，SARSA 較保守**：Q-learning 的步數略少，偏重效率；SARSA 會把探索的風險算進去，步數略多，但較重視路徑安全。

## 專案結構

```text
.
├── Vacuum_GridWorld.py      # 環境：地圖、reward、路徑圖與 GIF 繪製
├── qlearn.py                # Q-learning 訓練與 ε-greedy 策略
├── qlearn_demo.py           # Q-learning 主程式
├── sarsa.py                 # SARSA 訓練與 ε-greedy 策略
├── sarsa_esarsa_demo.py     # SARSA 主程式（目前只有 SARSA，沒有 Expected SARSA）
├── qlearn_algorithm.py      # qlearn() 的節錄，僅供閱讀，無法單獨執行
└── images/                  # README 使用的圖片
```

## 安裝與執行

已用 Python 3.13、NumPy 2.5、Matplotlib 3.11、Pillow 12.3 測試。

```bash
pip install "numpy>=2" matplotlib pillow
```

```bash
python qlearn_demo.py          # 訓練 Q-learning
python sarsa_esarsa_demo.py    # 訓練 SARSA
```

### 執行前請注意

- **重現上方結果**：把兩個 demo 檔裡的 `num_episodes=20000` 改成 `num_episodes=8000`。NumPy 1.x 的數值型別規則不同，同一個 seed 也可能得到不同路徑。
- **記憶體用量大**：每個 state 都存整張 `dirtyMap`，約佔 6 KB，8,000 回合會累積約 100 萬個 state。8 GB 記憶體不夠完整跑完，建議 8,000 回合準備 16 GB 以上，20,000 回合準備 32 GB 以上。
- **訓練曲線視窗**：訓練結束後會跳出圖表視窗，關閉後程式才會繼續。沒有圖形介面的環境可以用 `MPLBACKEND=Agg python qlearn_demo.py` 執行。
- **執行時間**：8,000 回合時，每個 demo 約需 3 到 4 分鐘。

### 輸出檔案

| 內容 | Q-learning | SARSA |
| --- | --- | --- |
| 初始環境圖 | `vacuum_initial.png` | `vacuum_initial.png` |
| 訓練曲線 | `qlearning_training_progress.png` | `sarsa_training_progress.png` |
| 完整 Q-table | `qtable_output.txt`、`.csv` | `sarsa_qtable_output.txt`、`.csv` |
| 路徑圖 | `gridworld_path.png` | `sarsa_gridworld_path.png` |
| 動畫 | `vacuum_gridworld_qlearn.gif` | `sarsa_vacuum_gridworld.gif` |
| 每一步的畫面 | `vacuum_gridworld_qlearn_png_steps/` | `sarsa_vacuum_gridworld_png_steps/` |

## 組員

楊明霆、章棋翔、朱立軒、蔣淇
