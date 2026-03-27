"""
=============================================================================
  S&P 500 指數預測系統 v1.1
  新增：詳細數值報告圖表（誤差分佈、散點圖、指標摘要表）
  輸出：sp500_results_v1.1.png（原有三圖 + 新增兩圖）
        sp500_report_v1.1.png（完整數值報告頁）
        sp500_feature_importance_v1.1.png（特徵重要性，不變）
        model_comparison_v1.1.csv（擴充指標）
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
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

    winner = "XGBoost" if xgb_m["MSE"] < rf_m["MSE"] else "Random Forest"
    print(f"\n  🏆 勝出模型：{winner}")
    print("\n" + "=" * 65)
    print("  ✅ v1.1 執行完成！輸出檔案：")
    print("     📊 sp500_results_v1.1.png          （預測對照 + 數值框）")
    print("     📊 sp500_report_v1.1.png            （完整數值報告頁）")
    print("     📊 sp500_feature_importance_v1.1.png（特徵重要性 + 數值）")
    print("     📋 model_comparison_v1.1.csv        （擴充指標總表）")
    print("=" * 65)


if __name__ == "__main__":
    main()
