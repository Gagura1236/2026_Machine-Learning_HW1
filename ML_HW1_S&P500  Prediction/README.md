# 📈 S&P 500 指數預測系統

> **課程：114-2 Machine Learning & Deep Learning**
> 使用 XGBoost 與 Random Forest 預測 S&P 500 每日收盤指數，比較 MSE（均方誤差）。

---

## 📁 專案檔案一覽

```
RA7141236_HW_1/
├── data/                             # 輸出的 CSV 結果（由程式自動產生）
│   ├── model_comparison.csv          # 版本 1.0：4 項評估指標
│   └── model_comparison_v1.1.csv     # 版本 1.1：15 項評估指標
│
├── notebooks/                        # Jupyter Notebook 互動版
│   └── sp500_prediction.ipynb        # 版本 1.0 Notebook（含路徑自動設定）
│
├── src/                              # 核心 Python 程式碼（從根目錄執行）
│   ├── sp500_prediction.py           # 版本 1.0 主程式
│   ├── sp500_prediction_v1.1.py      # 版本 1.1（詳細圖表）
│   └── sp500_interactive_report.py   # 互動式 HTML 報告產生器
│
├── reports/                          # 輸出的圖表與 HTML（由程式自動產生）
│   ├── sp500_results.png             # 版本 1.0：三合一圖
│   ├── sp500_feature_importance.png  # 版本 1.0：特徵重要性
│   ├── sp500_results_v1.1.png        # 版本 1.1：三合一（加強版）
│   ├── sp500_report_v1.1.png         # 版本 1.1：詳細報告頁
│   ├── sp500_feature_importance_v1.1.png  # 版本 1.1：含數值標注
│   └── sp500_interactive_report.html # 互動報告（懸停說明 + 詞彙表）
│
├── .gitignore                        # Git 排除清單（data/ 和 reports/ 不上傳）
├── README.md                         # 本說明文件
├── requirements.txt                  # 套件清單
└── implementation_plan.md            # AI 代理人實作計畫書
```

---

## 🖥️ 系統需求

| 項目 | 最低需求 |
|------|---------|
| 作業系統 | Windows 10+ / macOS 12+ / Ubuntu 20.04+ |
| Python 版本 | Python 3.9 以上（建議 3.10 或 3.11） |
| 網路連線 | 需要（Yahoo Finance 資料 + HTML 報告的 Plotly CDN） |
| 磁碟空間 | 約 1 GB（含所有套件） |

> 💡 確認 Python 版本：終端機輸入 `python --version`（若找不到，試 `python3 --version`）

---

## ⚠️ macOS 使用者必讀：XGBoost 需要 OpenMP

在 macOS 上，XGBoost 必須安裝 `libomp` 才能正常載入。

```bash
# Step 1：安裝 Homebrew（若尚未安裝）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Step 2：安裝 OpenMP runtime
brew install libomp
```

安裝完後，**重新啟動終端機或 VSCode**，再繼續後續步驟。

---

## 📦 第一步（必做）：安裝所有套件

```bash
# 切換到專案資料夾（請替換為你自己的實際路徑）
cd "/Users/你的名字/Library/你的資料夾路徑"

# 一鍵安裝（約需 5～10 分鐘，需要網路）
pip install -r requirements.txt
```

> ⚠️ 若出現 `pip: command not found`，請改用 `pip3 install -r requirements.txt`

---

## 🚀 四種執行方式（依需求選擇）

---

### 🟢 方式 A：基本版 1.0（最快執行）

**輸出：** 三張 PNG 圖 + `model_comparison.csv`（4 項指標）

#### 方法一：Jupyter Notebook（瀏覽器互動）
```bash
jupyter notebook
# → 開啟 sp500_prediction.ipynb → Kernel → Restart & Run All
```

#### 方法二：Python 腳本（終端機最簡單）
```bash
# 從專案根目錄（RA7141236_HW_1/）執行
python src/sp500_prediction.py
```

#### 方法三：Google Colab（免安裝）
1. 前往 [colab.research.google.com](https://colab.research.google.com/)
2. 上傳 `sp500_prediction.ipynb` → 全部執行（`Ctrl+F9`）
3. Cell 1 會自動安裝所有套件

#### 方法四：VSCode（Notebook 互動）

> ⚠️ **若出現 `XGBoostError`** → 確認已執行 `brew install libomp` 並重啟 VSCode

1. 安裝擴充套件：**Python**（Microsoft）+ **Jupyter**（Microsoft）
2. File → Open Folder → 選 `RA7141236_HW_1`
3. 點開 `.ipynb` → Select Kernel → Run All
4. 若套件找不到：在 Cell 1 執行 `%pip install -r requirements.txt`

---

### 🟡 方式 B：詳細報告版 1.1

**輸出：** 五張 PNG（含 15 項指標總表、散點圖、CDF、滾動 RMSE）+ `model_comparison_v1.1.csv`

```bash
# 從專案根目錄執行
python src/sp500_prediction_v1.1.py
```

**新增輸出說明：**

| 輸出檔案 | 內容 |
|---------|------|
| `sp500_results_v1.1.png` | 原有三圖 + 數值框（含 ±RMSE 水平線） |
| `sp500_report_v1.1.png` | **詳細報告頁**：散點圖 / 誤差直方圖 / CDF / 20日滾動RMSE / 15項指標總表 |
| `sp500_feature_importance_v1.1.png` | 特徵重要性長條圖（每條標數值） |
| `model_comparison_v1.1.csv` | 擴充為 15 項指標（含方向準確率、Q95 誤差、R² 等） |

---

### 🔵 方式 C：互動式 HTML 報告（推薦展示使用）

**輸出：** `sp500_interactive_report.html`（152 KB，瀏覽器直接開啟）

```bash
# 從專案根目錄執行
python src/sp500_interactive_report.py
# 完成後雙擊 reports/sp500_interactive_report.html 開啟
```

**互動功能：**
- 🖱️ **滑鼠懸停說明**：將滑鼠移到藍色底線文字上 → 彈出中文定義 + 數值意義 + 邊界標準
  - 覆蓋 13 個術語：MSE、RMSE、MAE、MAPE、R²、方向準確率、XGBoost、Random Forest、VIX、大波動 Spike、Adj Close、Look-ahead Bias、TimeSeriesSplit
- 📊 **6 張互動圖表**：可縮放、可懸停查看精確日期與數值
- 📖 **詞彙表新視窗**：點擊「開啟術語詞彙表」按鈕 → 彈出新視窗完整說明（附列印功能）

> 💡 HTML 報告需要網路連線（從 Plotly CDN 載入圖表引擎）

---

## 📊 預期輸出數值（2025 年測試集）

| 指標 | XGBoost | Random Forest | 評等 |
|------|---------|---------------|------|
| **MSE** | 234,014 | 260,235 | XGB 勝 |
| **RMSE** | 483.75 pt | 510.13 pt | XGB 勝 |
| **MAPE** | 5.74% | 6.31% | XGB 勝 |
| **R²** | 0.577 | 0.719 | RF 勝 |
| **方向準確率** | 51.0% | 60.0% | RF 勝 |
| **最大單日誤差** | 923.9 pt | 945.2 pt | XGB 勝 |

> XGBoost 在 MSE/RMSE 上勝出，RF 在 R²/方向準確率上反而較佳，兩模型各有優劣。

---

## ❓ 常見問題

**Q1：macOS XGBoost 報錯（libxgboost.dylib could not be loaded）**
```bash
brew install libomp  # 安裝後重啟 VSCode 或終端機
```

**Q2：`ModuleNotFoundError: No module named 'yfinance'`**
```bash
pip install -r requirements.txt
# Notebook 中：在 Cell 1 執行 %pip install -r requirements.txt
```

**Q3：HTML 報告圖表空白**

需要網路連線。若離線使用，可改用 PNG 版本的輸出。

**Q4：圖表中文顯示為方塊（□□□）**
- 程式已自動偵測平台字體（macOS/Windows/Linux）
- 若仍有問題，在 Cell 2 改為：`plt.rcParams['font.family'] = ['DejaVu Sans']`

**Q5：執行速度很慢**

模型訓練約需 5～15 分鐘，屬正常範圍。互動報告版執行時間與 1.1 版相近。

---

## 🔬 系統設計說明

### 資料處理規範（防止學術作弊）

| 規範 | 說明 |
|------|------|
| 股利還原 | `auto_adjust=True` 下載 Adj Close |
| 嚴禁預知偏差 | 所有特徵只使用 `t-1` 及更早的資料（`shift(1)`） |
| 嚴禁隨機洗牌 | 資料依時間順序切分，2025 年不進入訓練集 |
| 防止資料洩漏 | 大波動閾值只用訓練集（2021–2024）統計量計算 |

### 特徵清單（共 18 個）

| 類別 | 特徵 |
|------|------|
| 落後價格 | `lag_1/5/10` |
| 移動均線 | `ma_5/20/60` |
| 波動與量 | `volatility_10`, `volume_change` |
| 技術指標 | `rsi_14` |
| 週期效應 | `day_of_week`, `month` |
| 日報酬 | `daily_return` |
| 🆕 VIX | `vix_close`, `vix_ma_5`, `vix_change`, `vix_regime` |
| 🆕 大波動 | `is_spike`, `spike_direction` |

---

## ⚠️ 學術倫理聲明

1. 訓練集（2021–2024）與測試集（2025）嚴格分離，不得混用
2. 所有特徵均為落後型，不含任何未來資料
3. 大波動閾值只用訓練集統計量，不洩漏測試集資訊
4. 超參數調整只在訓練集的 TimeSeriesSplit 交叉驗證中進行

---

*最後更新：2026-03-27*
