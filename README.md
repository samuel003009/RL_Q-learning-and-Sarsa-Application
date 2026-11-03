# 強化式學習路徑規劃：Q-learning 與 SARSA 比較

**Q-learning vs. SARSA for multi-target path planning in grid worlds**


在 12×12 的網格中，讓機器人自己學會「**走過所有目標點，再回到起點**」的路線，並比較 **Q-learning** 與 **SARSA** 在「目標密集」與「目標分散」兩種情境下的表現。

<p align="center">
  <img src="results/delivery_paths.png" width="92%" alt="外送機器人情境：Q-learning 與 SARSA 訓練後的路徑">
</p>

## 重點結果

| 情境 | 理想步數 | Q-learning | SARSA |
|---|---|---|---|
| 外送機器人（9 個目標，分散） | 64 | **64**（10 個種子全部達到最短路徑） | 平均 77.0 |
| 掃地機器人（71 個目標，密集） | 72 | 平均 87.8 | 平均 98.6 |

<sub>10 個隨機種子（1–10）各訓練 8000 回合後，以貪婪策略走一次的步數。</sub>

- **目標分散時兩種演算法都比較好學**：外送情境離理想步數較近（Q-learning 0%、SARSA +20%），掃地情境則差得較多（+22%、+37%）。目標密集時，每走一步幾乎都有 +3 的「局部獎勵」，容易讓策略只顧眼前、不易學到全域最短路徑。
- **Q-learning 較積極**：步數較少、學得較快（第一次完成任務比 SARSA 早，成功率也爬升得比較快）。
- **SARSA 較保守**：用「實際會採取的動作（包含探索）」更新，路徑較長，收斂也較慢。

## 問題情境

| 掃地機器人（Robotic vacuum） | 外送機器人（Delivery robot） |
|---|---|
| 掃地機器人要如何在有家具的室內，找到清掃所有髒污格子的最短路徑，再回到充電座？ | 外送機器人要如何在有建築物的街區，找到送完所有包裹的最短路線，再回到物流中心？ |
| 幾乎每個空格都是目標 → **目標密集** | 9 個送貨點分散在街區中 → **目標分散** |

<p align="center">
  <img src="results/maps.png" width="92%" alt="兩種情境的網格地圖">
</p>

## 環境設定

- **地圖**：12×12 網格，灰色為障礙物（家具／建築物），黃色為目標，`H` 為起點兼終點。
- **動作**：上、下、左、右。
- **狀態**：`(row, col, mask)`，`mask` 以位元記錄哪些目標已經走過，讓代理人知道還剩哪些目標。
- **Q 表**：以 dict 儲存，只記錄實際走過的狀態（掃地情境的 `(位置, mask)` 組合多到無法用陣列存）。

**獎勵規則**（兩種情境相同）

| 事件 | 獎勵 |
|---|---|
| 移出地圖（停在原地） | −1 |
| 撞到障礙物（停在原地） | −1 |
| 一般移動 | −1 |
| 第一次到達某個目標 | +3 |
| 所有目標都完成且回到起點（回合結束） | +100 |

## 演算法

**Q-learning（off-policy）**：以下一個狀態「最好的動作」來更新。

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)\right]$$

**SARSA（on-policy）**：以下一個狀態「實際會採取的動作」來更新。

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\right]$$

兩者都用 ε-greedy 選動作，ε 從 1.0 隨回合線性遞減到 0。

**參數設定**

| 參數 | 值 |
|---|---|
| 折扣因子 γ | 0.95 |
| 學習率 α | 0.3 |
| 初始 ε | 1.0（線性遞減） |
| 訓練回合數 | 8000 |
| 每回合最多步數 | 500 |
| 亂數種子 | 789 |

## 實驗結果

### 學習曲線

<p align="center">
  <img src="results/delivery_learning_curves.png" width="80%" alt="外送機器人學習曲線">
</p>
<p align="center">
  <img src="results/vacuum_learning_curves.png" width="80%" alt="掃地機器人學習曲線">
</p>

兩種情境中，Q-learning 的回合步數都下降得比較早，成功率也爬升得比 SARSA 快。以外送情境為例，「最近 100 回合全部完成任務」這件事，Q-learning 在第 3317 回合就做到，SARSA 要到第 6518 回合。

### 訓練後的路徑（種子 789）

<p align="center">
  <img src="results/vacuum_paths.png" width="92%" alt="掃地機器人：訓練後的路徑">
</p>

| 情境 | 演算法 | 理想步數 | 實際步數 | 第一次完成任務（回合） | 最後 500 回合成功率 |
|---|---|---|---|---|---|
| 掃地機器人 | Q-learning | 72 | 86 | 1565 | 98.8% |
| 掃地機器人 | SARSA | 72 | 100 | 1745 | 95.2% |
| 外送機器人 | Q-learning | 64 | **64** | 1071 | 100.0% |
| 外送機器人 | SARSA | 64 | 66 | 1355 | 99.6% |

### 換不同亂數種子也成立嗎？

單一種子的結果可能是運氣，所以另外用種子 1–10 各重跑一次：

<p align="center">
  <img src="results/multi_seed.png" width="88%" alt="10 個種子的路徑步數分布">
</p>

| 情境 | 演算法 | 理想 | 平均步數（標準差） | 最佳 | 最差 | 完成任務 |
|---|---|---|---|---|---|---|
| 掃地機器人 | Q-learning | 72 | 87.8（1.8） | 84 | 90 | 10/10 |
| 掃地機器人 | SARSA | 72 | 98.6（5.8） | 92 | 106 | 10/10 |
| 外送機器人 | Q-learning | 64 | 64.0（0.0） | 64 | 64 | 10/10 |
| 外送機器人 | SARSA | 64 | 77.0（3.2） | 74 | 82 | 10/10 |

## 結論

1. **適用情境**：Q-learning 與 SARSA 都比較適合目標分散的任務。目標密集時，代理人容易被沿途的局部獎勵牽著走，難以針對全域最短路徑做最佳化。
2. **演算法特性**：
   - **Q-learning（積極）**：直接朝最佳動作更新，步數較少、收斂較快，重視效率。
   - **SARSA（保守）**：把探索時可能犯的錯也算進去，步數略多、收斂較慢；一般認為這種保守特性用在實體機器人上，可以降低碰撞等意外的風險。

## 專案結構

```
.
├── main.py                 # 訓練兩種演算法、輸出所有圖表與 CSV
├── rl_gridworld/
│   ├── envs.py             # 網格環境、兩張地圖、獎勵規則、理想步數計算
│   ├── agents.py           # Q-learning、SARSA、ε-greedy、貪婪策略回放
│   └── plotting.py         # 地圖、學習曲線、路徑、多種子比較圖
├── tests/
│   └── test_env.py         # 獎勵規則、理想步數、訓練可重現性的單元測試
├── results/                # main.py 產生的圖表與 summary.csv、multi_seed.csv
└── requirements.txt
```

## 快速開始

```bash
git clone https://github.com/<your-account>/rl-gridworld-qlearning-sarsa.git
cd rl-gridworld-qlearning-sarsa
pip install -r requirements.txt

python main.py                 # 依上表參數訓練（種子 789），通常 1 分鐘內完成
python main.py --seeds 10      # 另外用種子 1–10 重跑，做穩健性比較
python -m unittest discover tests
```

其他選項：

```bash
python main.py --env delivery --algo sarsa      # 只跑單一情境／演算法
python main.py --episodes 3000 --alpha 0.2      # 調整參數
python main.py --help
```

## 關於這個專案

- 2026/5–2026/6，強化式學習課程期末專題（4 人小組）。
- 本 repository 依課堂報告的環境、獎勵與參數，整理成可直接執行、結果可重現的版本；README 中的圖表與數字都由本 repo 的程式產生。
- 技術：Python、強化式學習（Q-learning、SARSA）、NumPy、Matplotlib
