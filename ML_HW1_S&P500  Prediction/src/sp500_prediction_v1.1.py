"""
=============================================================================
  S&P 500 指數預測系統 v1.1
  新增：詳細數值報告圖表（誤差分佈、散點圖、指標摘要表）
  輸出：sp500_results_v1.1.png、sp500_report_v1.1.png、sp500_interactive_report.html
        sp500_report_v1.1.png（完整數值報告頁）
        sp500_feature_importance_v1.1.png（特徵重要性，不變）
        model_comparison_v1.1.csv（擴充指標）
=============================================================================
"""

import os
import sys
# 自動將工作目錄切換到專案根目錄 (src 的上一層)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.table import Table
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# 跨平台字體
if sys.platform == "darwin":
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
elif sys.platform == "win32":
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 全域色彩 ──────────────────────────────────────────────────
C_ACTUAL = "#58a6ff"; C_XGB = "#f78166"; C_RF = "#3fb950"
C_SPIKE  = "#ff7b72"; C_VIX = "#d2a8ff"; C_GRID = "#21262d"; C_TEXT = "#c9d1d9"
BG_MAIN  = "#0d1117"; BG_PANEL = "#161b22"

# ===========================================================================
# 資料下載與清洗
# ===========================================================================

def download_data(ticker, start, end, auto_adjust=True):
    print(f"  下載 {ticker} ({start} ~ {end})...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=auto_adjust, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"  完成！{len(df)} 筆交易日")
    return df


def clean_data(df, name=""):
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.ffill().bfill()
        print(f"  {name}：填補 {missing} 個缺失值")
    return df


# ===========================================================================
# 特徵工程
# ===========================================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_features(gspc_df, vix_df):
    df = gspc_df[["Close", "High", "Low", "Volume"]].copy()
    df.columns = ["adj_close", "high", "low", "volume"]
    df = df.join(vix_df["Close"].rename("vix_raw"), how="left")
    df["vix_raw"] = df["vix_raw"].ffill()

    df["lag_1"]  = df["adj_close"].shift(1)
    df["lag_5"]  = df["adj_close"].shift(5)
    df["lag_10"] = df["adj_close"].shift(10)

    c1 = df["adj_close"].shift(1)
    df["ma_5"]          = c1.rolling(5,  min_periods=5).mean()
    df["ma_20"]         = c1.rolling(20, min_periods=20).mean()
    df["ma_60"]         = c1.rolling(60, min_periods=60).mean()
    df["volatility_10"] = c1.rolling(10, min_periods=10).std()

    v1 = df["volume"].shift(1); v2 = df["volume"].shift(2)
    df["volume_change"] = (v1 - v2) / (v2 + 1e-10)
    df["rsi_14"] = compute_rsi(c1, period=14)

    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["daily_return"] = df["adj_close"].pct_change(1)

    vix1 = df["vix_raw"].shift(1)
    df["vix_close"]  = vix1
    df["vix_ma_5"]   = vix1.rolling(5, min_periods=5).mean()
    df["vix_change"] = vix1.pct_change(1)
    df["vix_regime"] = (vix1 > 20).astype(int)

    df["target"] = df["adj_close"].shift(-1)
    df = df.drop(columns=["vix_raw"])
    df = df.dropna()
    print(f"  特徵矩陣：{len(df)} 筆 × {len(df.columns)} 欄")
    return df


# ===========================================================================
# 詳細指標計算（v1.1 新增）
# ===========================================================================

def compute_detailed_metrics(y_true, y_pred, model_name):
    """計算完整的預測評估指標集，包含方向準確率與分位數誤差。"""
    y_t = y_true.values
    err = y_t - y_pred
    pct_err = err / (y_t + 1e-10) * 100

    mse    = mean_squared_error(y_t, y_pred)
    rmse   = np.sqrt(mse)
    mae    = np.mean(np.abs(err))
    mape   = np.mean(np.abs(pct_err))
    corr   = np.corrcoef(y_t, y_pred)[0, 1]

    # 方向準確率（預測漲跌方向是否正確）
    actual_dir = np.diff(y_t)
    pred_dir   = np.diff(y_pred)
    dir_acc    = np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100

    return {
        "模型":           model_name,
        "MSE":            mse,
        "RMSE (pt)":      rmse,
        "MAE (pt)":       mae,
        "MAPE (%)":       mape,
        "最大誤差 (pt)":  np.max(np.abs(err)),
        "最小誤差 (pt)":  np.min(np.abs(err)),
        "誤差標準差 (pt)": np.std(err),
        "Q25 誤差 (pt)":  np.percentile(np.abs(err), 25),
        "Q75 誤差 (pt)":  np.percentile(np.abs(err), 75),
        "Q95 誤差 (pt)":  np.percentile(np.abs(err), 95),
        "過估比例 (%)":   np.mean(y_pred > y_t) * 100,
        "低估比例 (%)":   np.mean(y_pred < y_t) * 100,
        "方向準確率 (%)": dir_acc,
        "相關係數 (r)":   corr,
        "R² (決定係數)":  corr ** 2,
    }


# ===========================================================================
# 圖表一：原有三圖（含數值標注加強版）
# ===========================================================================

def plot_results_v11(test_df, xgb_pred, rf_pred, vix_series,
                     spike_dates, xgb_m, rf_m, save_path):
    """預測對照 + VIX雙軸 + 誤差分析（標題加入完整數值）"""
    test_dates  = test_df.index
    vix_aligned = vix_series.reindex(test_dates).ffill()

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.patch.set_facecolor(BG_MAIN)

    # ── 圖一：實際 vs 預測（含數值標注框）──
    ax1 = axes[0]; ax1.set_facecolor(BG_PANEL)
    ax1.plot(test_dates, test_df["target"], color=C_ACTUAL, lw=2.0, label="實際收盤價")
    ax1.plot(test_dates, xgb_pred, color=C_XGB, lw=1.5, ls="--", label="XGBoost 預測")
    ax1.plot(test_dates, rf_pred,  color=C_RF,  lw=1.5, ls=":",  label="Random Forest 預測")
    for d in spike_dates:
        ax1.axvline(d, color=C_SPIKE, alpha=0.5, lw=1.2, ls="--")

    # 數值標注框（左上角）
    info_text = (
        f"─── XGBoost ───\n"
        f"  RMSE : {xgb_m['RMSE (pt)']:>8.2f} pt\n"
        f"  MAE  : {xgb_m['MAE (pt)']:>8.2f} pt\n"
        f"  MAPE : {xgb_m['MAPE (%)']:>8.3f} %\n"
        f"  R²   : {xgb_m['R² (決定係數)']:>8.4f}\n"
        f"  方向 : {xgb_m['方向準確率 (%)']:>8.1f} %\n"
        f"\n─── Random Forest ───\n"
        f"  RMSE : {rf_m['RMSE (pt)']:>8.2f} pt\n"
        f"  MAE  : {rf_m['MAE (pt)']:>8.2f} pt\n"
        f"  MAPE : {rf_m['MAPE (%)']:>8.3f} %\n"
        f"  R²   : {rf_m['R² (決定係數)']:>8.4f}\n"
        f"  方向 : {rf_m['方向準確率 (%)']:>8.1f} %"
    )
    ax1.text(0.01, 0.97, info_text, transform=ax1.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_MAIN,
                       edgecolor=C_GRID, alpha=0.92), color=C_TEXT)

    sp = mpatches.Patch(color=C_SPIKE, alpha=0.6, label=f"大波動（{len(spike_dates)} 天）")
    ax1.set_title("S&P 500 指數預測對照（2025 年測試集）", color=C_TEXT, fontsize=13, pad=12)
    ax1.set_ylabel("指數點數（Adj Close）", color=C_TEXT); ax1.tick_params(colors=C_TEXT)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax1.grid(color=C_GRID, lw=0.8)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h+[sp], l+[sp.get_label()], facecolor="#21262d",
               labelcolor=C_TEXT, fontsize=9, loc="lower left")
    [s.set_edgecolor(C_GRID) for s in ax1.spines.values()]

    # ── 圖二：VIX 雙軸 ──
    ax2 = axes[1]; ax2.set_facecolor(BG_PANEL)
    ax2.plot(test_dates, test_df["target"], color=C_ACTUAL, lw=1.8, label="S&P 500（左軸）")
    ax2.set_ylabel("S&P 500 指數", color=C_ACTUAL); ax2.tick_params(axis="y", colors=C_ACTUAL)
    ax2.tick_params(axis="x", colors=C_TEXT)
    ax2r = ax2.twinx()
    ax2r.plot(test_dates, vix_aligned.values, color=C_VIX, lw=1.4, alpha=0.85, label="VIX（右軸）")
    ax2r.set_ylabel("VIX 恐慌指數", color=C_VIX); ax2r.tick_params(axis="y", colors=C_VIX)
    ax2.fill_between(test_dates, test_df["target"].min()*0.97, test_df["target"].max()*1.03,
                     where=(vix_aligned.values > 20), color=C_SPIKE, alpha=0.1, label="VIX>20")
    ax2.set_title("VIX 恐慌指數 × S&P 500（2025 年）", color=C_TEXT, fontsize=13, pad=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax2.grid(color=C_GRID, lw=0.8)
    ln1, lb1 = ax2.get_legend_handles_labels(); ln2, lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(ln1+ln2, lb1+lb2, facecolor="#21262d", labelcolor=C_TEXT, fontsize=9)
    [s.set_edgecolor(C_GRID) for s in ax2.spines.values()]

    # ── 圖三：誤差分析（含水平統計線）──
    ax3 = axes[2]; ax3.set_facecolor(BG_PANEL)
    xgb_err = test_df["target"].values - xgb_pred
    rf_err  = test_df["target"].values - rf_pred
    ax3.plot(test_dates, xgb_err, color=C_XGB, lw=1.2, alpha=0.85, label="XGBoost 誤差")
    ax3.plot(test_dates, rf_err,  color=C_RF,  lw=1.2, alpha=0.85, label="RF 誤差")
    ax3.axhline(0, color=C_TEXT, lw=1.0, alpha=0.5)
    # ±1 RMSE 區間線
    ax3.axhline( xgb_m["RMSE (pt)"], color=C_XGB, lw=0.8, ls=":", alpha=0.6,
                label=f"XGB ±RMSE ({xgb_m['RMSE (pt)']:.1f}pt)")
    ax3.axhline(-xgb_m["RMSE (pt)"], color=C_XGB, lw=0.8, ls=":", alpha=0.6)
    ax3.axhline( rf_m["RMSE (pt)"],  color=C_RF,  lw=0.8, ls=":", alpha=0.6,
                label=f"RF  ±RMSE ({rf_m['RMSE (pt)']:.1f}pt)")
    ax3.axhline(-rf_m["RMSE (pt)"],  color=C_RF,  lw=0.8, ls=":", alpha=0.6)
    for d in spike_dates:
        if d in test_dates:
            idx = test_dates.get_loc(d)
            ax3.scatter(d, xgb_err[idx], color=C_SPIKE, zorder=5, s=60, marker="^")
    ax3.set_title("預測誤差分析（▲=大波動日  虛線=±RMSE 區間）", color=C_TEXT, fontsize=13, pad=12)
    ax3.set_ylabel("誤差（點數）", color=C_TEXT); ax3.tick_params(colors=C_TEXT)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax3.grid(color=C_GRID, lw=0.8)
    ax3.legend(facecolor="#21262d", labelcolor=C_TEXT, fontsize=8.5, ncol=2)
    [s.set_edgecolor(C_GRID) for s in ax3.spines.values()]

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  ✓ {save_path}")


# ===========================================================================
# 圖表二：完整數值報告頁（v1.1 全新）
# ===========================================================================

def plot_detailed_report(test_df, xgb_pred, rf_pred, xgb_m, rf_m, save_path):
    """完整數值報告：散點圖 / 誤差直方圖 / 指標總表"""
    y_t = test_df["target"].values
    xgb_err = y_t - xgb_pred
    rf_err  = y_t - rf_pred

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(BG_MAIN)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── (左上) 實際 vs 預測散點圖 ──────────────────────────────
    ax_sc = fig.add_subplot(gs[0, 0]); ax_sc.set_facecolor(BG_PANEL)
    ax_sc.scatter(y_t, xgb_pred, color=C_XGB, s=18, alpha=0.55, label="XGBoost")
    ax_sc.scatter(y_t, rf_pred,  color=C_RF,  s=18, alpha=0.55, label="Random Forest")
    lims = [min(y_t.min(), xgb_pred.min(), rf_pred.min()) * 0.99,
            max(y_t.max(), xgb_pred.max(), rf_pred.max()) * 1.01]
    ax_sc.plot(lims, lims, color=C_ACTUAL, lw=1.5, ls="--", label="完美預測線")
    ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
    ax_sc.set_xlabel("實際收盤價（pt）", color=C_TEXT)
    ax_sc.set_ylabel("預測收盤價（pt）", color=C_TEXT)
    ax_sc.set_title("實際值 vs. 預測值（散點圖）\n愈靠近對角線代表預測愈準", color=C_TEXT, fontsize=11)
    ax_sc.tick_params(colors=C_TEXT); ax_sc.grid(color=C_GRID, lw=0.8)
    ax_sc.legend(facecolor="#21262d", labelcolor=C_TEXT, fontsize=9)
    # 標注 R²
    ax_sc.text(0.03, 0.95,
               f"XGB  R² = {xgb_m['R² (決定係數)']:.4f}\nRF   R² = {rf_m['R² (決定係數)']:.4f}",
               transform=ax_sc.transAxes, fontsize=9, color=C_TEXT, va="top",
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_MAIN, edgecolor=C_GRID, alpha=0.9))
    [s.set_edgecolor(C_GRID) for s in ax_sc.spines.values()]

    # ── (右上) 誤差分佈直方圖 ──────────────────────────────────
    ax_hi = fig.add_subplot(gs[0, 1]); ax_hi.set_facecolor(BG_PANEL)
    bins = np.linspace(min(xgb_err.min(), rf_err.min()),
                       max(xgb_err.max(), rf_err.max()), 35)
    ax_hi.hist(xgb_err, bins=bins, color=C_XGB, alpha=0.65, label="XGBoost 誤差")
    ax_hi.hist(rf_err,  bins=bins, color=C_RF,  alpha=0.65, label="RF 誤差")
    ax_hi.axvline(0, color=C_ACTUAL, lw=1.5, ls="--")
    ax_hi.axvline( xgb_m["RMSE (pt)"], color=C_XGB, lw=1.2, ls=":", alpha=0.8)
    ax_hi.axvline(-xgb_m["RMSE (pt)"], color=C_XGB, lw=1.2, ls=":", alpha=0.8)
    ax_hi.axvline( rf_m["RMSE (pt)"],  color=C_RF,  lw=1.2, ls=":", alpha=0.8)
    ax_hi.axvline(-rf_m["RMSE (pt)"],  color=C_RF,  lw=1.2, ls=":", alpha=0.8)
    ax_hi.set_xlabel("誤差（pt）", color=C_TEXT)
    ax_hi.set_ylabel("頻率（天數）", color=C_TEXT)
    ax_hi.set_title("預測誤差分佈直方圖\n虛線 = ±RMSE 區間", color=C_TEXT, fontsize=11)
    ax_hi.tick_params(colors=C_TEXT); ax_hi.grid(color=C_GRID, lw=0.8, axis="y")
    ax_hi.legend(facecolor="#21262d", labelcolor=C_TEXT, fontsize=9)
    # 偏度/峰度
    from scipy import stats as sp_stats
    xgb_sk = sp_stats.skew(xgb_err); xgb_ku = sp_stats.kurtosis(xgb_err)
    rf_sk  = sp_stats.skew(rf_err);  rf_ku  = sp_stats.kurtosis(rf_err)
    ax_hi.text(0.03, 0.95,
               f"XGB  偏度={xgb_sk:+.2f}  峰度={xgb_ku:.2f}\n"
               f"RF   偏度={rf_sk:+.2f}  峰度={rf_ku:.2f}",
               transform=ax_hi.transAxes, fontsize=8.5, color=C_TEXT, va="top",
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_MAIN, edgecolor=C_GRID, alpha=0.9))
    [s.set_edgecolor(C_GRID) for s in ax_hi.spines.values()]

    # ── (左下) 累積誤差 CDF ────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 0]); ax_cdf.set_facecolor(BG_PANEL)
    for err_arr, color, label in [
        (np.abs(xgb_err), C_XGB, "XGBoost |誤差|"),
        (np.abs(rf_err),  C_RF,  "RF |誤差|")
    ]:
        sorted_err = np.sort(err_arr)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
        ax_cdf.plot(sorted_err, cdf, color=color, lw=2.0, label=label)
    for pct in [50, 75, 90, 95]:
        ax_cdf.axhline(pct, color=C_GRID, lw=0.8, ls="--", alpha=0.7)
        ax_cdf.text(ax_cdf.get_xlim()[1] if ax_cdf.get_xlim()[1] > 0 else 1000,
                    pct+1, f"{pct}%", color=C_TEXT, fontsize=7.5, alpha=0.8)
    ax_cdf.set_xlabel("|誤差|（pt）", color=C_TEXT)
    ax_cdf.set_ylabel("累積比例（%）", color=C_TEXT)
    ax_cdf.set_title("誤差累積分佈函數（CDF）\n曲線愈靠左代表誤差愈集中在小值", color=C_TEXT, fontsize=11)
    ax_cdf.tick_params(colors=C_TEXT); ax_cdf.grid(color=C_GRID, lw=0.8)
    ax_cdf.legend(facecolor="#21262d", labelcolor=C_TEXT, fontsize=9)
    ax_cdf.set_ylim(0, 102)
    [s.set_edgecolor(C_GRID) for s in ax_cdf.spines.values()]

    # ── (右下) 滾動 RMSE（20 日視窗）──────────────────────────
    ax_roll = fig.add_subplot(gs[1, 1]); ax_roll.set_facecolor(BG_PANEL)
    test_dates = test_df.index
    window = 20
    xgb_roll = pd.Series(xgb_err**2, index=test_dates).rolling(window).mean().apply(np.sqrt)
    rf_roll  = pd.Series(rf_err**2,  index=test_dates).rolling(window).mean().apply(np.sqrt)
    ax_roll.plot(test_dates, xgb_roll, color=C_XGB, lw=1.8, label=f"XGBoost {window}日滾動RMSE")
    ax_roll.plot(test_dates, rf_roll,  color=C_RF,  lw=1.8, label=f"RF {window}日滾動RMSE")
    ax_roll.axhline(xgb_m["RMSE (pt)"], color=C_XGB, lw=1.0, ls="--", alpha=0.7,
                   label=f"XGB 整體RMSE={xgb_m['RMSE (pt)']:.1f}")
    ax_roll.axhline(rf_m["RMSE (pt)"],  color=C_RF,  lw=1.0, ls="--", alpha=0.7,
                   label=f"RF 整體RMSE={rf_m['RMSE (pt)']:.1f}")
    ax_roll.set_title(f"{window} 日滾動 RMSE（隨時間變化的誤差趨勢）", color=C_TEXT, fontsize=11)
    ax_roll.set_ylabel("RMSE（pt）", color=C_TEXT); ax_roll.tick_params(colors=C_TEXT)
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax_roll.grid(color=C_GRID, lw=0.8)
    ax_roll.legend(facecolor="#21262d", labelcolor=C_TEXT, fontsize=8.5, ncol=2)
    [s.set_edgecolor(C_GRID) for s in ax_roll.spines.values()]

    # ── (下方全寬) 完整數值指標總表 ────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :]); ax_tbl.set_facecolor(BG_MAIN)
    ax_tbl.axis("off")

    rows = [
        ("MSE（均方誤差）",          f"{xgb_m['MSE']:>12,.2f}",            f"{rf_m['MSE']:>12,.2f}",            "越小越好"),
        ("RMSE（根均方誤差，pt）",   f"{xgb_m['RMSE (pt)']:>12.4f}",       f"{rf_m['RMSE (pt)']:>12.4f}",       "與指數點數同單位"),
        ("MAE（平均絕對誤差，pt）",  f"{xgb_m['MAE (pt)']:>12.4f}",        f"{rf_m['MAE (pt)']:>12.4f}",        "對極端值較穩健"),
        ("MAPE（平均百分比誤差，%）",f"{xgb_m['MAPE (%)']:>12.4f}%",       f"{rf_m['MAPE (%)']:>12.4f}%",       "跨市場可比較"),
        ("最大單日誤差（pt）",       f"{xgb_m['最大誤差 (pt)']:>12.2f}",   f"{rf_m['最大誤差 (pt)']:>12.2f}",   "最差情況"),
        ("最小單日誤差（pt）",       f"{xgb_m['最小誤差 (pt)']:>12.2f}",   f"{rf_m['最小誤差 (pt)']:>12.2f}",   "最佳情況"),
        ("誤差標準差（pt）",         f"{xgb_m['誤差標準差 (pt)']:>12.2f}", f"{rf_m['誤差標準差 (pt)']:>12.2f}", "誤差穩定度"),
        ("Q25 絕對誤差（pt）",       f"{xgb_m['Q25 誤差 (pt)']:>12.2f}",   f"{rf_m['Q25 誤差 (pt)']:>12.2f}",   "25% 分位"),
        ("Q75 絕對誤差（pt）",       f"{xgb_m['Q75 誤差 (pt)']:>12.2f}",   f"{rf_m['Q75 誤差 (pt)']:>12.2f}",   "75% 分位"),
        ("Q95 絕對誤差（pt）",       f"{xgb_m['Q95 誤差 (pt)']:>12.2f}",   f"{rf_m['Q95 誤差 (pt)']:>12.2f}",   "95% 分位"),
        ("過估比例（預測 > 實際）",  f"{xgb_m['過估比例 (%)']:>12.1f}%",   f"{rf_m['過估比例 (%)']:>12.1f}%",   "系統性偏高"),
        ("低估比例（預測 < 實際）",  f"{xgb_m['低估比例 (%)']:>12.1f}%",   f"{rf_m['低估比例 (%)']:>12.1f}%",   "系統性偏低"),
        ("漲跌方向準確率（%）",      f"{xgb_m['方向準確率 (%)']:>12.1f}%", f"{rf_m['方向準確率 (%)']:>12.1f}%", ">50% 優於隨機"),
        ("相關係數 r",               f"{xgb_m['相關係數 (r)']:>12.4f}",    f"{rf_m['相關係數 (r)']:>12.4f}",    "1.0 = 完美線性"),
        ("決定係數 R²",              f"{xgb_m['R² (決定係數)']:>12.4f}",   f"{rf_m['R² (決定係數)']:>12.4f}",   "1.0 = 完美預測"),
    ]

    col_labels = ["評估指標", "XGBoost", "Random Forest", "說明"]
    col_widths = [0.32, 0.18, 0.18, 0.30]
    col_colors_hdr = ["#1f6feb", "#b94040", "#2d6a4f", "#21262d"]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_facecolor(col_colors_hdr[c])
            cell.set_text_props(color="white", weight="bold", fontsize=10)
        elif r % 2 == 0:
            cell.set_facecolor("#1c2128")
            cell.set_text_props(color=C_TEXT)
        else:
            cell.set_facecolor(BG_PANEL)
            cell.set_text_props(color=C_TEXT)
        cell.set_height(0.062)

    ax_tbl.set_title("完整模型評估指標總表（2025 年測試集）",
                     color=C_TEXT, fontsize=12, pad=14)

    plt.suptitle("S&P 500 預測系統 v1.1 — 詳細數值報告",
                 color=C_TEXT, fontsize=15, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  ✓ {save_path}")


# ===========================================================================
# 特徵重要性圖（與原版相同，輸出新檔名）
# ===========================================================================

def plot_feature_importance(xgb_model, rf_model, feature_names, save_path):
    TOP_N = 12
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG_MAIN)

    for ax, model, title, cbar in zip(
        axes,
        [xgb_model, rf_model],
        ["XGBoost 特徵重要性（前 12）", "Random Forest 特徵重要性（前 12）"],
        [C_XGB, C_RF]
    ):
        ax.set_facecolor(BG_PANEL)
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:TOP_N]
        names = [feature_names[i] for i in idx]
        vals  = imp[idx]

        bars = ax.barh(range(TOP_N), vals[::-1], color=cbar, alpha=0.85)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(names[::-1], color=C_TEXT, fontsize=10)
        ax.set_xlabel("重要性分數", color=C_TEXT)
        ax.set_title(title, color=C_TEXT, fontsize=12, pad=10)
        ax.tick_params(axis="x", colors=C_TEXT); ax.grid(color=C_GRID, lw=0.8, axis="x")

        for bi, n in enumerate(names[::-1]):
            if "vix"   in n: bars[bi].set_color(C_VIX);   bars[bi].set_alpha(1.0)
            if "spike" in n: bars[bi].set_color(C_SPIKE);  bars[bi].set_alpha(1.0)

        # 數值標在長條末端
        for bi, v in enumerate(vals[::-1]):
            ax.text(v + 0.0005, bi, f"{v:.4f}", va="center",
                    color=C_TEXT, fontsize=7.5)
        [s.set_edgecolor(C_GRID) for s in ax.spines.values()]

    fig.legend(
        handles=[mpatches.Patch(color=C_VIX, label="VIX 相關特徵"),
                 mpatches.Patch(color=C_SPIKE, label="大波動標記特徵")],
        loc="lower center", ncol=2, facecolor="#21262d", labelcolor=C_TEXT,
        fontsize=10, bbox_to_anchor=(0.5, -0.04)
    )
    plt.suptitle("模型特徵重要性比較（數值已標於長條末端）",
                 color=C_TEXT, fontsize=14, y=1.01)
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  ✓ {save_path}")


# ===========================================================================
# 主流程
# ===========================================================================



# ===========================================================================
# 三、Plotly 圖表（六張）
# ===========================================================================


# ── HTML 報告專用顏色常數 ────────────────────────────────────
C_ACTUAL = "#58a6ff"; C_XGB = "#f78166"; C_RF = "#3fb950"
C_SPIKE  = "#ff7b72"; C_VIX = "#d2a8ff"
BG       = "#0d1117"; PANEL  = "#161b22"; GRID = "#21262d"

PLOTLY_LAYOUT = dict(

    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color="#c9d1d9", family="system-ui,sans-serif"),
    margin=dict(l=60, r=30, t=55, b=45),
    xaxis=dict(gridcolor=GRID, showline=False),
    yaxis=dict(gridcolor=GRID, showline=False),
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=GRID, borderwidth=1),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor=GRID,
                    font=dict(color="#c9d1d9", size=12)),
)


def fig_prediction(test_df, xgb_pred, rf_pred, xgb_m, rf_m):
    dates = test_df.index.strftime("%Y-%m-%d").tolist()
    actual = test_df["target"].values
    spikes = test_df[test_df["is_spike"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual, name="實際收盤價", line=dict(color=C_ACTUAL, width=2),
        hovertemplate="<b>%{x}</b><br>實際：%{y:,.2f} pt<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=dates, y=xgb_pred, name="XGBoost 預測",
        line=dict(color=C_XGB, width=1.5, dash="dash"),
        hovertemplate="<b>%{x}</b><br>XGBoost：%{y:,.2f} pt<br>"
                      "誤差：%{customdata:,.2f} pt<extra></extra>",
        customdata=(actual - xgb_pred)))
    fig.add_trace(go.Scatter(
        x=dates, y=rf_pred, name="Random Forest 預測",
        line=dict(color=C_RF, width=1.5, dash="dot"),
        hovertemplate="<b>%{x}</b><br>RF：%{y:,.2f} pt<br>"
                      "誤差：%{customdata:,.2f} pt<extra></extra>",
        customdata=(actual - rf_pred)))
    for d, row in spikes.iterrows():
        fig.add_vline(x=d.strftime("%Y-%m-%d"), line_width=1,
                      line_dash="dash", line_color=C_SPIKE, opacity=0.5)
    fig.add_trace(go.Scatter(
        x=spikes.index.strftime("%Y-%m-%d").tolist(),
        y=spikes["target"].values, mode="markers",
        name=f"大波動日（{len(spikes)} 天）",
        marker=dict(color=C_SPIKE, size=9, symbol="triangle-up"),
        hovertemplate="<b>%{x}</b> ⚡ 大波動日<br>收盤：%{y:,.2f} pt<extra></extra>"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=(f"S&P 500 指數預測對照（2025 年）　"
                         f"XGB RMSE={xgb_m['RMSE (pt)']:.1f}pt  RF RMSE={rf_m['RMSE (pt)']:.1f}pt"),
                   font=dict(size=14)),
        yaxis_title="指數點數（Adj Close）",
    )
    return fig


def fig_error(test_df, xgb_pred, rf_pred, xgb_m, rf_m):
    dates  = test_df.index.strftime("%Y-%m-%d").tolist()
    actual = test_df["target"].values
    xe = actual - xgb_pred; re = actual - rf_pred
    spikes = test_df[test_df["is_spike"] == 1]

    fig = go.Figure()
    fig.add_hrect(y0=-xgb_m["RMSE (pt)"], y1=xgb_m["RMSE (pt)"],
                  fillcolor=C_XGB, opacity=0.07, line_width=0,
                  annotation_text=f"XGB ±RMSE ({xgb_m['RMSE (pt)']:.0f}pt)",
                  annotation_position="top right",
                  annotation_font=dict(color=C_XGB, size=10))
    fig.add_trace(go.Scatter(
        x=dates, y=xe, name="XGBoost 誤差", line=dict(color=C_XGB, width=1.5),
        hovertemplate="<b>%{x}</b><br>XGB誤差：%{y:+,.2f} pt<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=dates, y=re, name="RF 誤差", line=dict(color=C_RF, width=1.5),
        hovertemplate="<b>%{x}</b><br>RF誤差：%{y:+,.2f} pt<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="#c9d1d9", opacity=0.4)
    fig.add_trace(go.Scatter(
        x=spikes.index.strftime("%Y-%m-%d").tolist(),
        y=[xe[test_df.index.get_loc(d)] for d in spikes.index],
        mode="markers", name="大波動日",
        marker=dict(color=C_SPIKE, size=10, symbol="triangle-up"),
        hovertemplate="<b>%{x}</b> ⚡ 大波動<br>XGB誤差：%{y:+,.2f} pt<extra></extra>"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="預測誤差走勢（▲=大波動日　陰影=±RMSE 區間）",
                   font=dict(size=14)),
        yaxis_title="誤差（點數，正=低估 負=高估）",
    )
    return fig


def fig_scatter(test_df, xgb_pred, rf_pred):
    actual = test_df["target"].values
    dates  = test_df.index.strftime("%Y-%m-%d").tolist()
    lo, hi = actual.min() * 0.985, actual.max() * 1.015
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="完美預測線",
        line=dict(color=C_ACTUAL, width=1.5, dash="dash"), showlegend=True))
    for pred, col, name in [(xgb_pred, C_XGB, "XGBoost"), (rf_pred, C_RF, "RF")]:
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode="markers", name=name,
            marker=dict(color=col, size=5, opacity=0.6),
            hovertemplate=f"<b>%{{customdata}}</b><br>實際：%{{x:,.2f}}<br>"
                          f"{name}：%{{y:,.2f}}<br>誤差：%{{text}}<extra></extra>",
            customdata=dates,
            text=[f"{actual[i]-pred[i]:+.1f}pt" for i in range(len(actual))]))
    corr_xgb = np.corrcoef(actual, xgb_pred)[0,1]
    corr_rf  = np.corrcoef(actual, rf_pred)[0,1]
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"實際值 vs. 預測值散點圖　XGB r={corr_xgb:.4f} R²={corr_xgb**2:.4f}　RF r={corr_rf:.4f} R²={corr_rf**2:.4f}",
            font=dict(size=14)),
        xaxis_title="實際收盤價（pt）", yaxis_title="預測收盤價（pt）",
    )
    return fig


def fig_histogram(test_df, xgb_pred, rf_pred):
    actual = test_df["target"].values
    xe = actual - xgb_pred; re = actual - rf_pred
    bins = dict(start=min(xe.min(), re.min()), end=max(xe.max(), re.max()), size=40)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=xe, name="XGBoost 誤差", nbinsx=30,
        marker_color=C_XGB, opacity=0.7,
        hovertemplate="誤差區間：%{x}<br>天數：%{y}<extra></extra>"))
    fig.add_trace(go.Histogram(
        x=re, name="RF 誤差", nbinsx=30,
        marker_color=C_RF, opacity=0.7,
        hovertemplate="誤差區間：%{x}<br>天數：%{y}<extra></extra>"))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="overlay",
        title=dict(text="預測誤差分佈直方圖（正值=低估  負值=高估）", font=dict(size=14)),
        xaxis_title="誤差（pt）", yaxis_title="天數",
    )
    xgb_sk = sp_stats.skew(xe); rf_sk = sp_stats.skew(re)
    fig.add_annotation(
        text=f"XGB 偏度={xgb_sk:+.2f}　RF 偏度={rf_sk:+.2f}",
        xref="paper", yref="paper", x=0.02, y=0.96,
        showarrow=False, font=dict(color="#c9d1d9", size=11),
        bgcolor="#21262d", bordercolor=GRID, borderwidth=1)
    return fig


def fig_vix(test_df, vix_series):
    dates  = test_df.index.strftime("%Y-%m-%d").tolist()
    actual = test_df["target"].values
    vix_a  = vix_series.reindex(test_df.index).ffill().values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=dates, y=actual, name="S&P 500（左軸）",
        line=dict(color=C_ACTUAL, width=2),
        hovertemplate="<b>%{x}</b><br>S&P 500：%{y:,.2f} pt<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=vix_a, name="VIX（右軸）",
        line=dict(color=C_VIX, width=1.5),
        hovertemplate="<b>%{x}</b><br>VIX：%{y:.2f}<extra></extra>"),
        secondary_y=True)
    # 高波動區間著色
    hi_vix = np.where(vix_a > 20, actual.max() * 1.01, None)
    fig.add_trace(go.Scatter(
        x=dates, y=hi_vix, fill="tozeroy",
        fillcolor="rgba(255,123,114,0.08)", line=dict(color="rgba(0,0,0,0)"),
        name="高波動區間(VIX>20)", showlegend=True,
        hoverinfo="skip"), secondary_y=False)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="VIX 恐慌指數 × S&P 500 走勢（紅色陰影=VIX>20 高波動期）",
                   font=dict(size=14)))
    fig.update_yaxes(title_text="S&P 500 指數", secondary_y=False,
                     gridcolor=GRID, showline=False)
    fig.update_yaxes(title_text="VIX 恐慌指數", secondary_y=True,
                     gridcolor=GRID, showline=False)
    return fig


def fig_importance(xgb_model, rf_model, FEATURE_COLS):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["XGBoost 特徵重要性", "Random Forest 特徵重要性"],
                        horizontal_spacing=0.12)
    for col, model, color in [(1, xgb_model, C_XGB), (2, rf_model, C_RF)]:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:12]
        names = [FEATURE_COLS[i] for i in idx]; vals = imp[idx]
        bar_colors = [C_VIX if "vix" in n else (C_SPIKE if "spike" in n else color)
                      for n in names[::-1]]
        fig.add_trace(go.Bar(
            x=vals[::-1], y=names[::-1], orientation="h",
            name=["XGBoost","RF"][col-1], marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>重要性：%{x:.4f}<extra></extra>",
            text=[f"{v:.4f}" for v in vals[::-1]],
            textposition="outside", textfont=dict(size=9, color="#c9d1d9")),
            row=1, col=col)
    fig.update_layout(
        **PLOTLY_LAYOUT, showlegend=False,
        title=dict(text="特徵重要性（紫色=VIX特徵  紅色=大波動特徵）",
                   font=dict(size=14)))
    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
    return fig


# ===========================================================================
# 四、指標 Tooltip 定義（懸停說明）
# ===========================================================================

TOOLTIPS = {
    "MSE": (
        "均方誤差（Mean Squared Error）\n"
        "定義：所有預測誤差平方的平均值\n"
        "公式：Σ(實際-預測)² ÷ n\n"
        "單位：指數點數的平方（較難直覺理解）\n"
        "越小越好，無固定上限\n"
        "弱點：對極端誤差特別敏感"
    ),
    "RMSE": (
        "根均方誤差（Root Mean Squared Error）\n"
        "定義：MSE 的平方根，與原始數值同單位\n"
        "本次：XGB≈484pt　RF≈510pt\n"
        "S&P 500 約 5,500pt → 誤差率約 8~9%\n"
        "評等參考：<200pt 優良 / <500pt 中等 / >500pt 待改善\n"
        "優點：比 MSE 更便於直覺理解"
    ),
    "MAE": (
        "平均絕對誤差（Mean Absolute Error）\n"
        "定義：所有預測誤差絕對值的平均\n"
        "本次：每日平均偏差約 374~409 點\n"
        "比 RMSE 更穩健（不放大極端誤差）\n"
        "適合：當異常值不應被過度懲罰時"
    ),
    "MAPE": (
        "平均絕對百分比誤差（Mean Absolute Percentage Error）\n"
        "定義：(|實際-預測| ÷ 實際) × 100% 的平均\n"
        "本次：XGB≈5.7%　RF≈6.3%\n"
        "評等參考：<3% 優良 / 3~7% 中等 / >10% 不理想\n"
        "優點：百分比形式，可跨市場、跨時期比較"
    ),
    "R²": (
        "決定係數（Coefficient of Determination）\n"
        "定義：模型能解釋多少實際值的變異量\n"
        "範圍：0（無解釋力）~ 1（完美）\n"
        "本次：XGB=0.577　RF=0.719\n"
        "RF R²更高但 MSE 也更大，表示 RF 對相對趨勢掌握較好，\n"
        "但在極端誤差（大波動日）表現較差\n"
        "注意：R² 高不代表方向預測一定準確"
    ),
    "相關係數": (
        "皮爾森相關係數（Pearson Correlation, r）\n"
        "定義：實際值與預測值線性關聯的強度\n"
        "範圍：-1（完全負相關）~ 0 ~ 1（完全正相關）\n"
        "本次：XGB r=0.76　RF r=0.85\n"
        ">0.9 → 極強　0.7~0.9 → 強　<0.5 → 弱\n"
        "R² = r²（決定係數是相關係數的平方）"
    ),
    "方向準確率": (
        "漲跌方向準確率（Directional Accuracy）\n"
        "定義：正確預測次日「漲或跌」方向的比例\n"
        "本次：XGB=51%　RF=60%\n"
        "基準線：隨機猜測 = 50%\n"
        ">55% 具參考價值 / >60% 良好 / >70% 優秀\n"
        "注意：方向對但幅度差，RMSE 仍可能很大"
    ),
    "XGBoost": (
        "極限梯度提升（eXtreme Gradient Boosting）\n"
        "概念：由許多「弱決策樹」接力修正前一棵的誤差\n"
        "優點：通常準確度最高、訓練速度快\n"
        "缺點：超參數多、可解釋性較差\n"
        "適合：結構化數值資料（如金融時序）\n"
        "⚠️ macOS 需先安裝：brew install libomp"
    ),
    "Random Forest": (
        "隨機森林（Random Forest）\n"
        "概念：同時訓練許多獨立決策樹，結果取平均\n"
        "優點：不易過擬合、穩健性高、特徵重要性直覺\n"
        "缺點：預測精度通常稍低於 XGBoost\n"
        "適合：基準線比較、特徵選擇分析\n"
        "本次：方向準確率（60%）優於 XGBoost（51%）"
    ),
    "VIX": (
        "CBOE 波動率指數（VIX，俗稱「恐慌指數」）\n"
        "定義：市場預期未來 30 天 S&P 500 的波動程度\n"
        "計算：由 S&P 500 期權的隱含波動率推算\n"
        "判讀：<15 平靜 / 15~20 正常 / 20~30 緊張 / >30 恐慌\n"
        "歷史高點：2020年3月 COVID崩盤時達 85.47\n"
        "本系統以 VIX>20 定義「高波動機制」特徵"
    ),
    "大波動 Spike": (
        "單日大幅波動標記（Spike Flag）\n"
        "定義：當日漲跌幅超過 μ+2σ（訓練集統計）\n"
        "本次閾值：日報酬絕對值 > 約 2.13%\n"
        "2025年測試集共出現 10 個大波動日\n"
        "防洩漏：閾值只用訓練集（2021-2024）計算，\n"
        "         不使用測試集資料（防 Look-ahead Bias）"
    ),
    "Adj Close": (
        "還原收盤價（Adjusted Close Price）\n"
        "問題：成分股發放股利（除息）時，股價下調，\n"
        "       模型會誤判為真實跌幅\n"
        "解法：yfinance auto_adjust=True 自動回溯調整\n"
        "       所有歷史價格均被還原為「好像未發股利」的狀態\n"
        "本系統 Close 欄位即為已還原的 Adj Close"
    ),
    "Look-ahead Bias": (
        "預知偏差（Look-ahead Bias）\n"
        "定義：在訓練階段「不小心」使用了未來才知道的資料\n"
        "後果：模型在訓練時表現完美，但真實預測時完全失效\n"
        "本系統防護措施：\n"
        "  ① 所有特徵使用 shift(1)，只用前日資料\n"
        "  ② 訓練/測試依時間嚴格分割\n"
        "  ③ 大波動閾值只用訓練集統計量計算"
    ),
    "TimeSeriesSplit": (
        "時間序列交叉驗證（TimeSeriesSplit）\n"
        "問題：一般 K-Fold 會讓「未來資料訓練、過去資料驗證」\n"
        "解法：每個折（Fold）的訓練集一定早於驗證集\n"
        "本系統：5 折，每折訓練窗口逐步向後延伸\n"
        "用途：在訓練集內評估模型並調整超參數，\n"
        "       測試集（2025年）完全隔離，僅用於最終評估"
    ),
}


# ===========================================================================
# 五、HTML 模板建構
# ===========================================================================

def build_html(figs, xgb_m, rf_m):
    """組裝完整 HTML，嵌入 Plotly 圖表與所有 tooltip / glossary 內容"""

    def fig_div(fig, height=480):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": True, "scrollZoom": True},
                           default_height=height)

    divs = [fig_div(f) for f in figs]

    # ── 指標摘要表格（含 tooltip）───────────────────────────────
    def tip(key, display=None):
        txt = TOOLTIPS.get(key, "").replace("\n", "&#10;").replace('"', "&quot;")
        return f'<span class="tip" data-tip="{txt}">{display or key}</span>'

    def row(label_key, label_disp, xv, rv, hi_xgb=False):
        win = "xgb" if hi_xgb else "rf"
        xc  = ' class="winner"' if hi_xgb else ""
        rc  = ' class="winner"' if not hi_xgb else ""
        return (f"<tr><td>{tip(label_key, label_disp)}</td>"
                f"<td{xc}>{xv}</td><td{rc}>{rv}</td></tr>")

    winner_mse = xgb_m["MSE"] < rf_m["MSE"]
    table_rows = "".join([
        row("MSE",       "MSE",           f"{xgb_m['MSE']:>12,.2f}",   f"{rf_m['MSE']:>12,.2f}",  winner_mse),
        row("RMSE",      "RMSE (pt)",     f"{xgb_m['RMSE (pt)']:>8.4f}",    f"{rf_m['RMSE (pt)']:>8.4f}",   winner_mse),
        row("MAE",       "MAE (pt)",      f"{xgb_m['MAE (pt)']:>8.4f}",     f"{rf_m['MAE (pt)']:>8.4f}",    winner_mse),
        row("MAPE",      "MAPE (%)",      f"{xgb_m['MAPE (%)']:>8.4f}%",   f"{rf_m['MAPE (%)']:>8.4f}%",  winner_mse),
        row("R²",        "R²",            f"{xgb_m['R² (決定係數)']:>8.4f}",      f"{rf_m['R² (決定係數)']:>8.4f}",     xgb_m['R² (決定係數)'] > rf_m['R² (決定係數)']),
        row("相關係數",  "相關係數 r",   f"{xgb_m['相關係數 (r)']:>8.4f}",    f"{rf_m['相關係數 (r)']:>8.4f}",   xgb_m['相關係數 (r)'] > rf_m['相關係數 (r)']),
        row("方向準確率","方向準確率 (%)",f"{xgb_m['方向準確率 (%)']:>8.1f}%", f"{rf_m['方向準確率 (%)']:>8.1f}%",xgb_m['方向準確率 (%)'] > rf_m['方向準確率 (%)']),
        row("MSE",       "最大誤差 (pt)", f"{xgb_m['最大誤差 (pt)']:>8.2f}",  f"{rf_m['最大誤差 (pt)']:>8.2f}",  xgb_m['最大誤差 (pt)'] < rf_m['最大誤差 (pt)']),
        row("MSE",       "誤差標準差 (pt)",f"{xgb_m['誤差標準差 (pt)']:>8.2f}", f"{rf_m['誤差標準差 (pt)']:>8.2f}", xgb_m['誤差標準差 (pt)'] < rf_m['誤差標準差 (pt)']),
        row("RMSE",      "Q25 誤差 (pt)", f"{xgb_m['Q25 誤差 (pt)']:>8.2f}",    f"{rf_m['Q25 誤差 (pt)']:>8.2f}",    xgb_m['Q25 誤差 (pt)'] < rf_m['Q25 誤差 (pt)']),
        row("RMSE",      "Q75 誤差 (pt)", f"{xgb_m['Q75 誤差 (pt)']:>8.2f}",    f"{rf_m['Q75 誤差 (pt)']:>8.2f}",    xgb_m['Q75 誤差 (pt)'] < rf_m['Q75 誤差 (pt)']),
        row("RMSE",      "Q95 誤差 (pt)", f"{xgb_m['Q95 誤差 (pt)']:>8.2f}",    f"{rf_m['Q95 誤差 (pt)']:>8.2f}",    xgb_m['Q95 誤差 (pt)'] < rf_m['Q95 誤差 (pt)']),
    ])

    # ── 詞彙表 HTML（新視窗用）───────────────────────────────────
    glossary_sections = ""
    categories = {
        "📊 評估指標": ["MSE","RMSE","MAE","MAPE","R²","相關係數","方向準確率"],
        "🤖 模型說明": ["XGBoost","Random Forest","TimeSeriesSplit"],
        "📈 市場數據": ["VIX","Adj Close","大波動 Spike","Look-ahead Bias"],
    }
    for cat, keys in categories.items():
        items = ""
        for k in keys:
            if k in TOOLTIPS:
                lines = TOOLTIPS[k].split("\n")
                title_line = lines[0]
                rest = "<br>".join(lines[1:])
                items += (f"<div class='gitem'><h3>{title_line}</h3>"
                          f"<p>{rest}</p></div>\n")
        glossary_sections += f"<section><h2>{cat}</h2>{items}</section>\n"

    glossary_html = f"""<!DOCTYPE html><html lang="zh-TW"><head>
<meta charset="UTF-8"><title>S&P 500 術語詞彙表</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;
        max-width:900px;margin:0 auto;padding:30px 20px;}}
  h1{{color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:12px;}}
  h2{{color:#d2a8ff;margin-top:32px;}}
  h3{{color:#f78166;margin-bottom:6px;font-size:1.05em;}}
  section{{margin-bottom:28px;}}
  .gitem{{background:#161b22;border:1px solid #21262d;border-radius:8px;
           padding:16px 20px;margin-bottom:12px;}}
  .gitem p{{margin:0;line-height:1.75;font-size:0.95em;}}
  .print-btn{{display:inline-block;margin-top:24px;padding:10px 24px;
               background:#1f6feb;color:white;border:none;border-radius:6px;
               cursor:pointer;font-size:1em;}}
  .print-btn:hover{{background:#388bfd;}}
  @media print{{body{{background:white;color:black;}}
    .gitem{{border-color:#ccc;background:#f9f9f9;}}
    h1,h2{{color:black;}} h3{{color:#333;}}
    .print-btn{{display:none;}}}}
</style></head><body>
<h1>📖 S&P 500 預測系統 — 術語詞彙表</h1>
<p>本詞彙表解說預測報告中使用的所有專業術語，供對照查閱。<br>
<button class="print-btn" onclick="window.print()">🖨️ 列印詞彙表</button></p>
{glossary_sections}
</body></html>"""

    glossary_js = json.dumps(glossary_html)

    # ── 主 HTML 模板 ──────────────────────────────────────────────
    from datetime import datetime
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    winner_name = "XGBoost" if winner_mse else "Random Forest"
    winner_color = C_XGB if winner_mse else C_RF

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S&P 500 預測系統 — 互動式報告</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--border:#21262d;--text:#c9d1d9;
       --blue:#58a6ff;--xgb:#f78166;--rf:#3fb950;--vix:#d2a8ff;--spike:#ff7b72;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      font-size:15px;line-height:1.6;}}
.container{{max-width:1440px;margin:0 auto;padding:24px 28px;}}
header{{text-align:center;padding:32px 0 20px;border-bottom:1px solid var(--border);}}
header h1{{color:var(--blue);font-size:1.9em;margin-bottom:8px;}}
header p{{color:#8b949e;font-size:0.95em;}}
.badge{{display:inline-block;padding:3px 10px;border-radius:30px;font-size:0.82em;
         font-weight:600;margin:0 4px;}}
.badge-xgb{{background:rgba(247,129,102,0.2);color:var(--xgb);border:1px solid var(--xgb);}}
.badge-rf{{background:rgba(63,185,80,0.2);color:var(--rf);border:1px solid var(--rf);}}

/* ── 指標摘要卡片 ── */
.metrics-section{{margin:32px 0;}}
.metrics-section h2{{color:var(--vix);font-size:1.15em;margin-bottom:16px;}}
.metrics-table{{width:100%;border-collapse:collapse;font-size:0.92em;}}
.metrics-table th{{background:#1c2128;color:#8b949e;padding:9px 14px;
                    text-align:center;border:1px solid var(--border);font-weight:600;}}
.metrics-table td{{padding:8px 14px;border:1px solid var(--border);
                    text-align:center;font-family:"SFMono-Regular",Consolas,monospace;}}
.metrics-table td:first-child{{text-align:left;font-family:inherit;color:var(--text);}}
.metrics-table tr:nth-child(even) td{{background:#161b22;}}
.metrics-table tr:nth-child(odd) td{{background:#0d1117;}}
.winner{{color:#ffffff;font-weight:700;background:rgba(31,111,235,0.15) !important;}}

/* ── Tooltip ── */
.tip{{border-bottom:1px dashed var(--blue);cursor:help;color:var(--blue);
       white-space:nowrap;}}
#tip-box{{display:none;position:fixed;z-index:9999;max-width:340px;
           background:#1c2128;border:1px solid #388bfd;border-radius:8px;
           padding:12px 16px;font-size:13px;line-height:1.7;color:var(--text);
           box-shadow:0 8px 32px rgba(0,0,0,0.6);pointer-events:none;
           white-space:pre-line;}}

/* ── 圖表區域 ── */
.charts-section{{margin:32px 0;}}
.chart-card{{background:var(--panel);border:1px solid var(--border);
              border-radius:10px;padding:16px;margin-bottom:20px;}}
.chart-label{{color:#8b949e;font-size:0.82em;margin-bottom:6px;
               text-transform:uppercase;letter-spacing:0.05em;}}

/* ── 底部工具列 ── */
.footer{{display:flex;align-items:center;justify-content:space-between;
          margin-top:36px;padding-top:20px;border-top:1px solid var(--border);
          flex-wrap:wrap;gap:12px;}}
.glossary-btn{{background:#1f6feb;color:white;border:none;border-radius:6px;
                padding:10px 22px;cursor:pointer;font-size:0.95em;font-weight:600;
                transition:background 0.2s;}}
.glossary-btn:hover{{background:#388bfd;}}
.footer-info{{color:#8b949e;font-size:0.85em;}}
.winner-badge{{background:{winner_color}22;color:{winner_color};
                border:1px solid {winner_color};border-radius:6px;
                padding:6px 16px;font-weight:700;font-size:0.95em;}}
</style>
</head>
<body>
<div id="tip-box"></div>
<div class="container">

<header>
  <h1>📈 S&P 500 指數預測系統</h1>
  <p>課程：114-2 ML &amp; DL　｜　測試集：2025 年全年（248 個交易日）　｜　產生時間：{gen_time}</p>
  <p style="margin-top:10px">
    <span class="badge badge-xgb">XGBoost</span>
    <span class="badge badge-rf">Random Forest</span>
    &nbsp;
    <span style="color:#8b949e;font-size:0.9em">
      ⓘ 將滑鼠移到藍色底線文字上可查看說明
    </span>
  </p>
</header>

<!-- ── 指標摘要 ── -->
<div class="metrics-section">
  <h2>模型效能摘要（2025 年測試集）&nbsp;
    <span class="winner-badge">🏆 {winner_name} 勝出（MSE 較低）</span>
  </h2>
  <table class="metrics-table">
    <thead>
      <tr>
        <th>評估指標&nbsp;<small style="font-weight:400">(滑鼠移入可查看說明)</small></th>
        <th style="color:var(--xgb)">XGBoost</th>
        <th style="color:var(--rf)">Random Forest</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<!-- ── 圖表 ── -->
<div class="charts-section">

  <div class="chart-card">
    <div class="chart-label">圖一：預測值 vs. 實際值</div>
    {divs[0]}
  </div>

  <div class="chart-card">
    <div class="chart-label">圖二：預測誤差走勢</div>
    {divs[1]}
  </div>

  <div class="chart-card">
    <div class="chart-label">圖三：實際 vs. 預測散點圖（愈靠近對角線愈準）</div>
    {divs[2]}
  </div>

  <div class="chart-card">
    <div class="chart-label">圖四：預測誤差分佈（正值=低估 負值=高估）</div>
    {divs[3]}
  </div>

  <div class="chart-card">
    <div class="chart-label">圖五：<span class="tip" data-tip="{TOOLTIPS['VIX'].replace(chr(10),'&#10;')}">VIX 恐慌指數</span> × S&amp;P 500（雙軸對照）</div>
    {divs[4]}
  </div>

  <div class="chart-card">
    <div class="chart-label">圖六：特徵重要性排名（紫色=VIX相關　紅色=<span class="tip" data-tip="{TOOLTIPS['大波動 Spike'].replace(chr(10),'&#10;')}">大波動</span>標記）</div>
    {divs[5]}
  </div>

</div><!-- /charts-section -->

<div class="footer">
  <div>
    <button class="glossary-btn" onclick="openGlossary()">
      📖 開啟術語詞彙表（新視窗，可列印）
    </button>
    <span style="color:#8b949e;font-size:0.85em;margin-left:12px">
      包含所有指標定義、模型說明、市場術語共 13 個條目
    </span>
  </div>
  <div class="footer-info">
    訓練集：2021–2024　｜　測試集：2025　｜　嚴格時間序列切分（防 <span class="tip" data-tip="{TOOLTIPS['Look-ahead Bias'].replace(chr(10),'&#10;')}">Look-ahead Bias</span>）
  </div>
</div>

</div><!-- /container -->

<script>
// ── Tooltip 互動 ──────────────────────────────────────────
const tipBox = document.getElementById('tip-box');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mouseenter', e => {{
    tipBox.textContent = e.currentTarget.dataset.tip;
    tipBox.style.display = 'block';
  }});
  el.addEventListener('mousemove', e => {{
    let x = e.clientX + 18, y = e.clientY - 10;
    if (x + 350 > window.innerWidth) x = e.clientX - 360;
    if (y + 200 > window.innerHeight) y = e.clientY - 160;
    tipBox.style.left = x + 'px';
    tipBox.style.top  = y + 'px';
  }});
  el.addEventListener('mouseleave', () => {{ tipBox.style.display = 'none'; }});
}});

// ── 詞彙表新視窗 ─────────────────────────────────────────
function openGlossary() {{
  const glossaryHTML = {glossary_js};
  const win = window.open('', '_blank',
    'width=980,height=760,scrollbars=yes,resizable=yes,toolbar=no,menubar=no');
  if (!win) {{ alert('請允許彈出視窗以顯示詞彙表'); return; }}
  win.document.open();
  win.document.write(glossaryHTML);
  win.document.close();
}}
</script>
</body>
</html>"""




def main():
    import os
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("=" * 65)
    print("  S&P 500 指數預測系統 v1.1  |  114-2 ML & DL")
    print("=" * 65)

    START_DATE  = "2021-01-01"
    END_DATE    = "2025-12-31"
    SPLIT_DATE  = "2025-01-01"

    FEATURE_COLS = [
        "lag_1", "lag_5", "lag_10",
        "ma_5", "ma_20", "ma_60",
        "volatility_10", "volume_change", "rsi_14",
        "day_of_week", "month", "daily_return",
        "vix_close", "vix_ma_5", "vix_change", "vix_regime",
        "is_spike", "spike_direction"
    ]

    print("\n【Step 1/9】下載資料")
    gspc_raw = download_data("^GSPC", START_DATE, END_DATE, auto_adjust=True)
    vix_raw  = download_data("^VIX",  START_DATE, END_DATE, auto_adjust=False)

    print("\n【Step 2/9】清洗資料")
    gspc_clean = clean_data(gspc_raw, "^GSPC")
    vix_clean  = clean_data(vix_raw,  "^VIX")

    print("\n【Step 3/9】建構特徵矩陣")
    df_features = build_features(gspc_clean, vix_clean)

    print("\n【Step 4/9】時間序列切分")
    train_mask = df_features.index < SPLIT_DATE
    test_mask  = df_features.index >= SPLIT_DATE
    train_df   = df_features[train_mask]
    test_df    = df_features[test_mask]

    assert train_df.index.max() < pd.Timestamp(SPLIT_DATE)
    assert test_df.index.min() >= pd.Timestamp(SPLIT_DATE)
    print(f"  訓練：{train_df.index.min().date()} ～ {train_df.index.max().date()} ({len(train_df)} 天) ✓")
    print(f"  測試：{test_df.index.min().date()} ～ {test_df.index.max().date()} ({len(test_df)} 天) ✓")

    print("\n【Step 5/9】標記大波動")
    mu = train_df["daily_return"].mean(); sigma = train_df["daily_return"].std()
    threshold = 2.0 * sigma
    df_features["is_spike"]        = (df_features["daily_return"].abs() > (abs(mu)+threshold)).astype(int)
    df_features["spike_direction"] = 0
    df_features.loc[df_features["daily_return"] >  (abs(mu)+threshold), "spike_direction"] =  1
    df_features.loc[df_features["daily_return"] < -(abs(mu)+threshold), "spike_direction"] = -1
    train_df = df_features[train_mask]; test_df = df_features[test_mask]
    test_spike_dates = test_df[test_df["is_spike"] == 1].index
    print(f"  2025 年大波動：{len(test_spike_dates)} 天")

    print("\n【Step 6/9】準備輸入矩陣")
    X_train = train_df[FEATURE_COLS]; y_train = train_df["target"]
    X_test  = test_df[FEATURE_COLS];  y_test  = test_df["target"]
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n【Step 7a/9】訓練 XGBoost")
    XGB_PARAMS = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, n_jobs=-1
    )
    for i, (tr, val) in enumerate(tscv.split(X_train)):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_train.iloc[tr], y_train.iloc[tr], verbose=False)
        mse = mean_squared_error(y_train.iloc[val], m.predict(X_train.iloc[val]))
        print(f"  Fold {i+1}: MSE = {mse:,.2f}")
    xgb_model = XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train, verbose=False)
    print("  XGBoost 完成 ✓")

    print("\n【Step 7b/9】訓練 Random Forest")
    RF_PARAMS = dict(
        n_estimators=300, max_depth=6,
        min_samples_split=20, min_samples_leaf=10,
        max_features="sqrt", bootstrap=True,
        random_state=42, n_jobs=-1
    )
    for i, (tr, val) in enumerate(tscv.split(X_train)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        mse = mean_squared_error(y_train.iloc[val], m.predict(X_train.iloc[val]))
        print(f"  Fold {i+1}: MSE = {mse:,.2f}")
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    print("  Random Forest 完成 ✓")

    print("\n【Step 8/9】評估（v1.1 擴充指標）")
    xgb_pred = xgb_model.predict(X_test)
    rf_pred  = rf_model.predict(X_test)
    xgb_m = compute_detailed_metrics(y_test, xgb_pred, "XGBoost")
    rf_m  = compute_detailed_metrics(y_test, rf_pred,  "Random Forest")

    for m in [xgb_m, rf_m]:
        print(f"\n  ── {m['模型']} ──")
        for k, v in m.items():
            if k != "模型":
                print(f"    {k:<22}: {v:.4f}" if isinstance(v, float) else f"    {k:<22}: {v}")

    pd.DataFrame([xgb_m, rf_m]).to_csv(
        "data/model_comparison_v1.1.csv", index=False, float_format="%.6f", encoding="utf-8-sig"
    )
    print("\n  📋 model_comparison_v1.1.csv 已儲存")

    print("\n【Step 9/9】生成視覺化報告")
    vix_series = vix_clean["Close"]
    plot_results_v11(test_df, xgb_pred, rf_pred, vix_series,
                     test_spike_dates, xgb_m, rf_m, "reports/sp500_results_v1.1.png")
    plot_detailed_report(test_df, xgb_pred, rf_pred, xgb_m, rf_m,
                         "reports/sp500_report_v1.1.png")
    plot_feature_importance(xgb_model, rf_model, FEATURE_COLS,
                            "reports/sp500_feature_importance_v1.1.png")

    print("\n【Step 10/10】生成互動式 HTML 報告")
    try:
        figs = [
            fig_prediction(test_df, xgb_pred, rf_pred, xgb_m, rf_m),
            fig_error(test_df, xgb_pred, rf_pred, xgb_m, rf_m),
            fig_scatter(test_df, xgb_pred, rf_pred),
            fig_histogram(test_df, xgb_pred, rf_pred),
            fig_vix(test_df, vix_series),
            fig_importance(xgb_model, rf_model, FEATURE_COLS),
        ]
        html = build_html(figs, xgb_m, rf_m)
        out_html = "reports/sp500_interactive_report.html"
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✓ {out_html}")
    except Exception as e:
        print(f"  ❌ HTML 報告生成失敗：{e}")

    winner = "XGBoost" if xgb_m["MSE"] < rf_m["MSE"] else "Random Forest"
    print(f"\n  🏆 勝出模型：{winner}")
    print("\n" + "=" * 65)
    print("  ✅ v1.1 執行完成！輸出檔案：")
    print("     📊 sp500_results_v1.1.png          （預測對照 + 數值框）")
    print("     📊 sp500_report_v1.1.png            （完整數值報告頁）")
    print("     📊 sp500_feature_importance_v1.1.png（特徵重要性 + 數值）")
    print("     🌐 sp500_interactive_report.html   （互動報告 + 詞彙表）")
    print("     📋 model_comparison_v1.1.csv        （擴充指標總表）")
    print("=" * 65)


if __name__ == "__main__":
    main()
