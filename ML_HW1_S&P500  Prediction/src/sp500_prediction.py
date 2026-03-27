"""
=============================================================================
  S&P 500 指數預測系統
  課程：114-2 ML & DL
  目標：比較 XGBoost 與 Random Forest 對 S&P 500 收盤指數的預測效能
  作者：AI 代理人（依據計畫書 v1.1 實作）

  ⚠️ 學術倫理聲明：
    - 所有模型訓練只使用 2021-2024 年資料
    - 測試集（2025 年）僅用於最終評估，不參與訓練或調參
    - 所有特徵均為落後型，不含未來資訊（防止 Look-ahead Bias）
    - 資料切分依時間順序進行，嚴禁隨機洗牌
=============================================================================
"""

import os
import sys
# 自動將工作目錄切換到專案根目錄 (src 的上一層)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# ===========================================================================
# 第一區塊：匯入所需套件
# ===========================================================================

import warnings
warnings.filterwarnings("ignore")  # 關閉不影響結果的警告訊息，保持輸出整潔

import sys          # 用於偵測作業系統，選擇對應的中文字體
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# 跨平台中文字體設定（自動偵測作業系統）
# macOS  → Arial Unicode MS（系統內建，支援繁體中文）
# Windows → Microsoft JhengHei（微軟正黑體，支援繁體中文）
# Linux / Colab → DejaVu Sans（無中文，但避免方塊）
if sys.platform == "darwin":
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
elif sys.platform == "win32":
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 確保負號正確顯示

# ===========================================================================
# 第二區塊：資料下載函數
# ===========================================================================

def download_data(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """
    從 Yahoo Finance 下載指定標的的歷史資料。

    參數說明：
        ticker       : 標的代碼，如 "^GSPC" 或 "^VIX"
        start        : 起始日期字串，格式 "YYYY-MM-DD"
        end          : 結束日期字串，格式 "YYYY-MM-DD"
        auto_adjust  : 是否自動使用「還原除權息收盤價（Adj Close）」
                       True → yfinance 會把 Adj Close 存入 Close 欄位（股利已回溯調整）
                       ⚠️ 重要：這是防止「股利偽造跌幅」污染模型的關鍵設定
    """
    print(f"  [資料下載] 正在下載 {ticker} ({start} ~ {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    print(f"  [資料下載] 完成！共 {len(df)} 筆交易日資料")
    return df


# ===========================================================================
# 第三區塊：資料清洗函數
# ===========================================================================

def clean_data(df: pd.DataFrame, name: str = "資料") -> pd.DataFrame:
    """
    清洗資料：處理缺失值與基本品質檢查。
    使用前向填補（ffill）再後向填補（bfill），保留時間連續性。
    """
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        df = df.ffill().bfill()
        print(f"  [資料清洗] {name} 填補了 {missing_before} 個缺失值")
    else:
        print(f"  [資料清洗] {name} 無缺失值，資料品質良好")
    return df


# ===========================================================================
# 第四區塊：特徵工程函數
# ===========================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    計算 RSI（相對強弱指數）。
    RSI = 100 - 100 / (1 + RS)，RS = 平均漲幅 / 平均跌幅
    使用 Wilder 指數平滑法（ewm）。
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_features(gspc_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    建構訓練特徵矩陣（18 個特徵 + 1 個目標）。

    所有特徵皆為落後型（Lagged）或滾動型（Rolling），
    確保特徵值 F(t) 只使用 t-1 及更早的資料，防止 Look-ahead Bias。
    """
    print("\n  [特徵工程] 開始建構特徵矩陣...")

    df = gspc_df[["Close", "High", "Low", "Volume"]].copy()
    df.columns = ["adj_close", "high", "low", "volume"]

    vix_close = vix_df["Close"].rename("vix_raw")
    df = df.join(vix_close, how="left")
    df["vix_raw"] = df["vix_raw"].ffill()

    # 落後型特徵
    df["lag_1"]  = df["adj_close"].shift(1)
    df["lag_5"]  = df["adj_close"].shift(5)
    df["lag_10"] = df["adj_close"].shift(10)

    # 滾動均線（先 shift(1) 再 rolling，確保不包含當日收盤價）
    close_lag1   = df["adj_close"].shift(1)
    df["ma_5"]   = close_lag1.rolling(window=5,  min_periods=5).mean()
    df["ma_20"]  = close_lag1.rolling(window=20, min_periods=20).mean()
    df["ma_60"]  = close_lag1.rolling(window=60, min_periods=60).mean()

    # 波動率（前 10 日收盤標準差）
    df["volatility_10"] = close_lag1.rolling(window=10, min_periods=10).std()

    # 成交量變化率
    vol_lag1 = df["volume"].shift(1)
    vol_lag2 = df["volume"].shift(2)
    df["volume_change"] = (vol_lag1 - vol_lag2) / (vol_lag2 + 1e-10)

    # RSI（使用已落後的 close_lag1 計算，不含當日資訊）
    df["rsi_14"] = compute_rsi(close_lag1, period=14)

    # 週期性特徵
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month

    # 日報酬率
    df["daily_return"] = df["adj_close"].pct_change(1)

    # VIX 特徵（全部 shift(1)）
    vix_lag1         = df["vix_raw"].shift(1)
    df["vix_close"]  = vix_lag1
    df["vix_ma_5"]   = vix_lag1.rolling(window=5, min_periods=5).mean()
    df["vix_change"] = vix_lag1.pct_change(1)
    df["vix_regime"] = (vix_lag1 > 20).astype(int)

    # 預測目標：次日還原收盤價
    df["target"] = df["adj_close"].shift(-1)

    df = df.drop(columns=["vix_raw"])
    rows_before = len(df)
    df = df.dropna()
    print(f"  [特徵工程] 移除 {rows_before - len(df)} 筆 NaN 行")
    print(f"  [特徵工程] 最終特徵矩陣：{len(df)} 筆 × {len(df.columns)} 欄（含 target）")
    return df


# ===========================================================================
# 第五區塊：大幅波動標記函數
# ===========================================================================

def mark_spikes(df: pd.DataFrame, train_mask: pd.Series, threshold_sigma: float = 2.0):
    """
    標記單日大幅波動（Spike）的交易日。

    ⚠️ 防洩漏關鍵規則：
        閾值（μ 與 σ）只用「訓練集」的 daily_return 計算。
        套用至測試集時，直接沿用訓練集的統計量，不重新計算。
    """
    train_returns = df.loc[train_mask, "daily_return"]
    mu    = train_returns.mean()
    sigma = train_returns.std()
    threshold = threshold_sigma * sigma

    print(f"\n  [大波動標記] 訓練集日報酬統計：μ={mu:.4%}, σ={sigma:.4%}")
    print(f"  [大波動標記] 大波動閾值：|日報酬| > {(abs(mu) + threshold):.4%}")

    df["is_spike"]        = (df["daily_return"].abs() > (abs(mu) + threshold)).astype(int)
    df["spike_direction"] = 0
    df.loc[df["daily_return"] >  (abs(mu) + threshold), "spike_direction"] =  1
    df.loc[df["daily_return"] < -(abs(mu) + threshold), "spike_direction"] = -1

    spike_count = df["is_spike"].sum()
    print(f"  [大波動標記] 全期間共標記 {spike_count} 個大波動交易日")
    return df, mu, sigma, threshold


# ===========================================================================
# 第六區塊：模型訓練函數
# ===========================================================================

def train_xgboost(X_train, y_train, tscv):
    """訓練 XGBoost 模型，含 TimeSeriesSplit 交叉驗證。"""
    print("\n  [XGBoost] 開始訓練...")

    xgb_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, n_jobs=-1
    )

    cv_scores = []
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        m = XGBRegressor(**xgb_params)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], verbose=False)
        mse = mean_squared_error(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]))
        cv_scores.append(mse)
        print(f"    Fold {i+1}: MSE = {mse:,.2f}")

    print(f"  [XGBoost] CV 平均 MSE = {np.mean(cv_scores):,.2f} (±{np.std(cv_scores):,.2f})")

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    print("  [XGBoost] 最終模型訓練完成 ✓")
    return model


def train_random_forest(X_train, y_train, tscv):
    """訓練 Random Forest 模型，含 TimeSeriesSplit 交叉驗證。"""
    print("\n  [Random Forest] 開始訓練...")

    rf_params = dict(
        n_estimators=300, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, max_features="sqrt", bootstrap=True,
        random_state=42, n_jobs=-1
    )

    cv_scores = []
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        m = RandomForestRegressor(**rf_params)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        mse = mean_squared_error(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]))
        cv_scores.append(mse)
        print(f"    Fold {i+1}: MSE = {mse:,.2f}")

    print(f"  [Random Forest] CV 平均 MSE = {np.mean(cv_scores):,.2f} (±{np.std(cv_scores):,.2f})")

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    print("  [Random Forest] 最終模型訓練完成 ✓")
    return model


# ===========================================================================
# 第七區塊：評估指標函數
# ===========================================================================

def evaluate_model(y_true, y_pred, model_name):
    """計算 MSE、RMSE、MAE、MAPE 四項評估指標。"""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_true.values - y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred) / (y_true.values + 1e-10))) * 100

    print(f"\n  ====== {model_name} 測試集評估結果（2025 全年）======")
    print(f"    MSE  : {mse:>12,.4f}  ← 主要評估指標（越低越好）")
    print(f"    RMSE : {rmse:>12,.4f}  ← 與 S&P 500 指數點數同單位")
    print(f"    MAE  : {mae:>12,.4f}  ← 平均每日預測偏差（點數）")
    print(f"    MAPE : {mape:>12,.4f}% ← 平均百分比誤差")

    return {"模型": model_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}


# ===========================================================================
# 第八區塊：視覺化函數
# ===========================================================================

def plot_results(test_df, xgb_pred, rf_pred, vix_series, spike_dates, metrics, save_path):
    """繪製三張對照圖：預測對照 / VIX 雙軸 / 誤差分析。"""
    print(f"\n  [視覺化] 開始繪製圖表...")

    C_ACTUAL = "#58a6ff"; C_XGB = "#f78166"; C_RF = "#3fb950"
    C_SPIKE  = "#ff7b72"; C_VIX = "#d2a8ff"; C_GRID = "#21262d"; C_TEXT = "#c9d1d9"
    BG_MAIN  = "#0d1117"; BG_PANEL = "#161b22"

    test_dates  = test_df.index
    vix_aligned = vix_series.reindex(test_dates).ffill()

    xgb_m = next(m for m in metrics if "XGBoost" in m["模型"])
    rf_m  = next(m for m in metrics if "Random Forest" in m["模型"])

    fig, axes = plt.subplots(3, 1, figsize=(18, 18))
    fig.patch.set_facecolor(BG_MAIN)

    # 圖一：實際值 vs. 預測值（含大波動垂直虛線）
    ax1 = axes[0]; ax1.set_facecolor(BG_PANEL)
    ax1.plot(test_dates, test_df["target"], color=C_ACTUAL, lw=2.0, label="實際收盤價（Adj Close）")
    ax1.plot(test_dates, xgb_pred, color=C_XGB, lw=1.5, ls="--", label="XGBoost 預測")
    ax1.plot(test_dates, rf_pred,  color=C_RF,  lw=1.5, ls=":",  label="Random Forest 預測")
    for d in spike_dates:
        ax1.axvline(d, color=C_SPIKE, alpha=0.55, lw=1.2, ls="--")
    spike_patch = mpatches.Patch(color=C_SPIKE, alpha=0.6, label=f"大波動交易日（{len(spike_dates)} 天）")
    title_txt = (f"S&P 500 指數預測（2025 年測試集）\n"
                 f"XGBoost RMSE={xgb_m['RMSE']:.2f} pt / MAPE={xgb_m['MAPE(%)']:.3f}%   │   "
                 f"RF RMSE={rf_m['RMSE']:.2f} pt / MAPE={rf_m['MAPE(%)']:.3f}%")
    ax1.set_title(title_txt, color=C_TEXT, fontsize=12, pad=12)
    ax1.set_ylabel("指數點數（還原收盤價）", color=C_TEXT)
    ax1.tick_params(colors=C_TEXT); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(color=C_GRID, lw=0.8)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h+[spike_patch], l+[spike_patch.get_label()],
               facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.9, fontsize=9)
    [s.set_edgecolor(C_GRID) for s in ax1.spines.values()]

    # 圖二：VIX × S&P 500 雙軸（高波動區間塗色）
    ax2 = axes[1]; ax2.set_facecolor(BG_PANEL)
    ax2.plot(test_dates, test_df["target"], color=C_ACTUAL, lw=1.8, label="S&P 500（左軸）")
    ax2.set_ylabel("S&P 500 指數", color=C_ACTUAL)
    ax2.tick_params(axis="y", colors=C_ACTUAL); ax2.tick_params(axis="x", colors=C_TEXT)
    ax2r = ax2.twinx()
    ax2r.plot(test_dates, vix_aligned.values, color=C_VIX, lw=1.4, alpha=0.85, label="VIX（右軸）")
    ax2r.set_ylabel("VIX 恐慌指數", color=C_VIX); ax2r.tick_params(axis="y", colors=C_VIX)
    ax2.fill_between(test_dates, test_df["target"].min()*0.97, test_df["target"].max()*1.03,
                     where=(vix_aligned.values > 20), color=C_SPIKE, alpha=0.1, label="高波動區間(VIX>20)")
    ax2.set_title("VIX 恐慌指數 × S&P 500 走勢（2025 年）", color=C_TEXT, fontsize=12, pad=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax2.grid(color=C_GRID, lw=0.8)
    ln1,lb1 = ax2.get_legend_handles_labels(); ln2,lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(ln1+ln2, lb1+lb2, facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.9, fontsize=9)
    [s.set_edgecolor(C_GRID) for s in ax2.spines.values()]

    # 圖三：預測誤差分析（大波動日三角形標記）
    ax3 = axes[2]; ax3.set_facecolor(BG_PANEL)
    xgb_err = test_df["target"].values - xgb_pred
    rf_err  = test_df["target"].values - rf_pred
    ax3.plot(test_dates, xgb_err, color=C_XGB, lw=1.2, alpha=0.85, label="XGBoost 誤差")
    ax3.plot(test_dates, rf_err,  color=C_RF,  lw=1.2, alpha=0.85, label="RF 誤差")
    ax3.axhline(0, color=C_TEXT, lw=1.0, alpha=0.5)
    for d in spike_dates:
        if d in test_dates:
            idx = test_dates.get_loc(d)
            ax3.scatter(d, xgb_err[idx], color=C_SPIKE, zorder=5, s=50, marker="^")
    ax3.set_title("預測誤差分析（▲ = 大波動事件日）", color=C_TEXT, fontsize=12, pad=12)
    ax3.set_ylabel("誤差（點數）", color=C_TEXT); ax3.tick_params(colors=C_TEXT)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax3.grid(color=C_GRID, lw=0.8)
    ax3.legend(facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.9, fontsize=9)
    [s.set_edgecolor(C_GRID) for s in ax3.spines.values()]

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  [視覺化] 圖表已儲存 → {save_path} ✓")


def plot_feature_importance(xgb_model, rf_model, feature_names, save_path):
    """繪製 XGBoost 與 Random Forest 的特徵重要性長條圖（前 12 名）。"""
    print("  [視覺化] 繪製特徵重要性圖...")

    C_VIX = "#d2a8ff"; C_SPIKE = "#ff7b72"; C_GRID = "#21262d"; C_TEXT = "#c9d1d9"
    BG_MAIN = "#0d1117"; BG_PANEL = "#161b22"
    TOP_N = 12

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG_MAIN)

    for ax, model, title, cbar in zip(
        axes,
        [xgb_model, rf_model],
        ["XGBoost 特徵重要性（前 12）", "Random Forest 特徵重要性（前 12）"],
        ["#f78166", "#3fb950"]
    ):
        ax.set_facecolor(BG_PANEL)
        imp     = model.feature_importances_
        indices = np.argsort(imp)[::-1][:TOP_N]
        names   = [feature_names[i] for i in indices]
        vals    = imp[indices]

        bars = ax.barh(range(TOP_N), vals[::-1], color=cbar, alpha=0.85)
        ax.set_yticks(range(TOP_N)); ax.set_yticklabels(names[::-1], color=C_TEXT, fontsize=10)
        ax.set_xlabel("特徵重要性分數", color=C_TEXT)
        ax.set_title(title, color=C_TEXT, fontsize=12, pad=10)
        ax.tick_params(axis="x", colors=C_TEXT); ax.grid(color=C_GRID, lw=0.8, axis="x")

        for bi, name in enumerate(names[::-1]):
            if "vix"   in name: bars[bi].set_color(C_VIX);   bars[bi].set_alpha(1.0)
            if "spike" in name: bars[bi].set_color(C_SPIKE);  bars[bi].set_alpha(1.0)
        [s.set_edgecolor(C_GRID) for s in ax.spines.values()]

    legend_h = [mpatches.Patch(color=C_VIX,   label="VIX 恐慌指數相關特徵"),
                mpatches.Patch(color=C_SPIKE, label="大波動標記特徵")]
    fig.legend(handles=legend_h, loc="lower center", ncol=2,
               facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.9,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("模型特徵重要性比較", color=C_TEXT, fontsize=14, y=1.01)
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  [視覺化] 特徵重要性圖已儲存 → {save_path} ✓")


# ===========================================================================
# 第九區塊：主程式主流程
# ===========================================================================

def main():
    import os
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("=" * 65)
    print("  S&P 500 指數預測系統  |  114-2 ML & DL")
    print("=" * 65)

    START_DATE  = "2021-01-01"
    END_DATE    = "2025-12-31"
    SPLIT_DATE  = "2025-01-01"
    RESULT_PATH = "reports/sp500_results.png"
    FEAT_PATH   = "reports/sp500_feature_importance.png"
    CSV_PATH    = "data/model_comparison.csv"

    FEATURE_COLS = [
        "lag_1", "lag_5", "lag_10",
        "ma_5", "ma_20", "ma_60",
        "volatility_10", "volume_change", "rsi_14",
        "day_of_week", "month", "daily_return",
        "vix_close", "vix_ma_5", "vix_change", "vix_regime",
        "is_spike", "spike_direction"
    ]

    print("\n【Step 1/9】下載原始資料")
    gspc_raw = download_data("^GSPC", START_DATE, END_DATE, auto_adjust=True)
    vix_raw  = download_data("^VIX",  START_DATE, END_DATE, auto_adjust=False)

    print("\n【Step 2/9】清洗資料")
    gspc_clean = clean_data(gspc_raw, "^GSPC")
    vix_clean  = clean_data(vix_raw,  "^VIX")

    print("\n【Step 3/9】建構特徵矩陣")
    df_features = build_features(gspc_clean, vix_clean)

    print("\n【Step 4/9】時間序列切分（嚴格依時間順序）")
    train_mask = df_features.index < SPLIT_DATE
    test_mask  = df_features.index >= SPLIT_DATE
    train_df   = df_features[train_mask]
    test_df    = df_features[test_mask]

    assert train_df.index.max() < pd.Timestamp(SPLIT_DATE), "❌ 訓練集含 2025 年資料！"
    assert test_df.index.min() >= pd.Timestamp(SPLIT_DATE), "❌ 測試集含 2024 年以前資料！"

    print(f"  訓練集：{train_df.index.min().date()} ～ {train_df.index.max().date()} ({len(train_df)} 天) ✓")
    print(f"  測試集：{test_df.index.min().date()} ～ {test_df.index.max().date()} ({len(test_df)} 天) ✓")

    print("\n【Step 5/9】標記大幅波動日")
    df_features, mu, sigma, threshold = mark_spikes(df_features, train_mask)
    train_df = df_features[train_mask]
    test_df  = df_features[test_mask]

    test_spike_dates = test_df[test_df["is_spike"] == 1].index
    print(f"  測試集（2025 年）大波動日期：{len(test_spike_dates)} 天")
    for d in test_spike_dates:
        print(f"    {d.date()}  日報酬 = {test_df.loc[d, 'daily_return']:.3%}")

    print("\n【Step 6/9】準備模型輸入")
    X_train = train_df[FEATURE_COLS]; y_train = train_df["target"]
    X_test  = test_df[FEATURE_COLS];  y_test  = test_df["target"]
    print(f"  訓練：X{X_train.shape}  測試：X{X_test.shape}")

    tscv = TimeSeriesSplit(n_splits=5)

    print("\n【Step 7/9】訓練模型")
    xgb_model = train_xgboost(X_train, y_train, tscv)
    rf_model  = train_random_forest(X_train, y_train, tscv)

    print("\n【Step 8/9】測試集評估")
    xgb_pred    = xgb_model.predict(X_test)
    rf_pred     = rf_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")
    rf_metrics  = evaluate_model(y_test, rf_pred,  "Random Forest")

    pd.DataFrame([xgb_metrics, rf_metrics]).to_csv(CSV_PATH, index=False,
        float_format="%.6f", encoding="utf-8-sig")
    print(f"\n  [輸出] 模型比較表已儲存 → {CSV_PATH}")

    winner = "XGBoost" if xgb_metrics["MSE"] < rf_metrics["MSE"] else "Random Forest"
    print(f"  🏆 測試集 MSE 較低的模型：{winner}")

    print("\n【Step 9/9】生成視覺化報告")
    vix_series = vix_clean["Close"]
    plot_results(test_df, xgb_pred, rf_pred, vix_series,
                 test_spike_dates, [xgb_metrics, rf_metrics], RESULT_PATH)
    plot_feature_importance(xgb_model, rf_model, FEATURE_COLS, FEAT_PATH)

    print("\n" + "=" * 65)
    print("  ✅ 預測系統執行完成！輸出檔案：")
    print(f"     📊 {RESULT_PATH}")
    print(f"     📊 {FEAT_PATH}")
    print(f"     📋 {CSV_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
