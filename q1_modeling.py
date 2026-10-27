# -*- coding: utf-8 -*-
"""
问题1：数据缺失条件下的感官评分 —— 模型建立与模型求解（一体化可复现脚本）
================================================================================
任务：以 1.csv 的 231 个化合物特征为基础，对 3.xlsx 定义的 Aroma / Flavour 等
      50 个感官维度给出 0-6 分制的分级评分结果。

两段式建模框架（对应论文"模型建立"一节）：
    第一段  连续评分回归选型（M1-M5）：
            M1 多输出 Ridge / M2 PLSR / M3 逐属性 GBR（原论文方法复现参照）
            / M4 随机森林 / M5 HistGBR（原生容忍缺失）
    第二段  0-6 分级标定（M6 序数模型）：
            路线一 回归 + 分位数锚定分箱；路线二 逐维梯度提升分类器直接分级
            两条路线用二次加权 Kappa(QWK) 在训练集内部交叉验证择优。
配套实验：缺失鲁棒性（训练/测试同步遮掩 0-30% 特征、含缺失训练；
          GBR+列均值重填补 vs HistGBR原生缺失处理 vs 随机森林+分位标定
          三条路径公平比较）
          与风格一致性校验（Tripel/Kriek/Stout/Pilsner 等画像合理性）。

参考：shujuji/Machine learning models transcript.py 中作者的
      MultiOutputRegressor(GradientBoostingRegressor) 与网格搜索思路；
      但全部代码为本队独立实现，且按赛题增加了分级标定与缺失鲁棒性环节。

数据依赖（预处理阶段已生成，见 src/processing_v3.py）：
    ../processed/1_X_log_normalized.csv        250 x 231 标准化化学特征
    ../processed/4_Y_sensory_standardized.csv  250 x 50  感官连续标签
    ../processed/split_train_test.csv          175/75 风格分层切分
输出：本脚本同目录 results/ 下全部 CSV、PNG 与最终模型文件。
运行：python q1_modeling.py   （仅依赖 numpy/pandas/scipy/sklearn/matplotlib）
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 全局配置
SEED = 0
N_JOBS = -1                      # 逐属性模型按维度并行
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent               # 项目根目录 A/
DATA = ROOT / "processed"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

ID_COLS = ["beer", "beer_id", "tasting_category_fine"]
MASK_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]   # 0.0 = 无缺失基线
MASK_SEEDS = [0, 1, 2]
N_LEVELS = 7                     # 0-6 共 7 级

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print(f"[Q1] {msg}", flush=True)


# ---------------------------------------------------------------- 工具函数
def fit_per_target(factory, X, Y):
    """逐感官维度拟合一个回归/分类器，返回按维度排列的估计器列表。"""
    return Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one)(factory, X, Y[:, j]) for j in range(Y.shape[1]))


def _fit_one(factory, X, y):
    est = factory()
    est.fit(X, y)
    return est


def predict_per_target(estimators, X):
    """逐维度预测并拼成 (n_samples, n_dims) 矩阵。"""
    return np.column_stack([est.predict(X) for est in estimators])


def per_dim_metrics(Y_true, Y_pred):
    """返回逐维度 R2、RMSE、Spearman（三个长度=n_dims 的数组）。"""
    n_dims = Y_true.shape[1]
    r2 = np.array([r2_score(Y_true[:, j], Y_pred[:, j]) for j in range(n_dims)])
    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
    rho = np.array([spearmanr(Y_true[:, j], Y_pred[:, j]).statistic
                    for j in range(n_dims)])
    return r2, rmse, rho


def make_quantile_boundaries(Y_train):
    """第二段核心：用训练集经验分布做分位数锚定，得到每维 6 条分箱边界。"""
    qs = np.arange(1, N_LEVELS) / N_LEVELS          # 1/7 ... 6/7
    return np.quantile(Y_train, qs, axis=0)          # shape (6, n_dims)


def to_levels(Y_cont, boundaries):
    """连续值 -> 0-6 整数等级（保序）。"""
    return np.column_stack([
        np.digitize(Y_cont[:, j], boundaries[:, j]) for j in range(Y_cont.shape[1])])


def grade_metrics(L_true, L_pred):
    """分级质量：逐维 QWK、准确率、±1 级容差准确率。"""
    n_dims = L_true.shape[1]
    labels = list(range(N_LEVELS))
    qwk = np.array([cohen_kappa_score(L_true[:, j], L_pred[:, j],
                                      weights="quadratic", labels=labels)
                    for j in range(n_dims)])
    acc = np.mean(L_true == L_pred, axis=0)
    acc1 = np.mean(np.abs(L_true - L_pred) <= 1, axis=0)
    return qwk, acc, acc1


# ---------------------------------------------------------------- Stage 0
def fit_preprocessor(train_raw):
    """仅由当前训练样本估计填补均值与标准化参数，返回 (标准化矩阵, 填补均值, 均值, 标准差)。

    将参数显式返回，供最终模型保存时随模型一并持久化，使新样本可直接复用同一套预处理。
    """
    col_mean = np.nanmean(train_raw, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    Xtr = np.where(np.isnan(train_raw), col_mean, train_raw)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    return Xtr_s, col_mean, mu, sd


def impute_and_scale(train_raw, other_raw=None):
    """填补与标准化：均值与标准差全部只由当前训练样本估计，再应用到待预测样本。

    交叉验证时传入折内训练/验证矩阵，即可得到折内重估；测试集评价时传入完整训练集。
    """
    Xtr_s, col_mean, mu, sd = fit_preprocessor(train_raw)
    if other_raw is None:
        return Xtr_s
    Xo = np.where(np.isnan(other_raw), col_mean, other_raw)
    return Xtr_s, (Xo - mu) / sd


def load_data():
    # 保留缺失标记的矩阵：原始缺失与训练集中识别的疑似异常为空值，
    # 由建模过程在训练集或各训练折内重新估计填补均值，避免提前使用验证/测试信息。
    X_df = pd.read_csv(DATA / "1_X_log_missing.csv")
    Y_df = pd.read_csv(DATA / "4_Y_sensory_standardized.csv")
    split = pd.read_csv(DATA / "split_train_test.csv")

    # 以 beer 编号为键对齐三张表，防止行序不一致造成静默错位
    assert X_df["beer"].is_unique and Y_df["beer"].is_unique and split["beer"].is_unique
    X_df = X_df.sort_values("beer").reset_index(drop=True)
    Y_df = Y_df.sort_values("beer").reset_index(drop=True)
    split = split.sort_values("beer").reset_index(drop=True)
    assert (X_df["beer"].values == Y_df["beer"].values).all()
    assert (X_df["beer"].values == split["beer"].values).all()

    feature_names = [c for c in X_df.columns if c not in ID_COLS]
    dims = [c for c in Y_df.columns if c not in ID_COLS]
    X = X_df[feature_names].to_numpy(dtype=float)
    Y = Y_df[dims].to_numpy(dtype=float)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    assert not np.isnan(Y).any(), "感官标签不应含缺失"

    is_train = (split["split"].values == "train")
    meta = X_df[ID_COLS].copy()
    log(f"数据装配完成：X{X.shape} Y{Y.shape}；train={is_train.sum()} test={(~is_train).sum()}")
    return X, Y, is_train, meta, feature_names, dims


# ---------------------------------------------------------------- Stage 1
def cv_evaluate(model_kind, factory, X, Y, n_splits=5):
    """训练集 K 折 CV：收集袋外预测，返回逐维 CV R2 与均值。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full_like(Y, np.nan, dtype=float)
    for tr_idx, va_idx in kf.split(X):
        # 折内重新估计填补均值与标准化参数，再作用到该折的验证部分
        Xtr, Xva = impute_and_scale(X[tr_idx], X[va_idx])
        Ytr = Y[tr_idx]
        if model_kind == "multi":
            est = factory()
            est.fit(Xtr, Ytr)
            oof[va_idx] = est.predict(Xva)
        else:
            ests = fit_per_target(factory, Xtr, Ytr)
            oof[va_idx] = predict_per_target(ests, Xva)
    r2, _, _ = per_dim_metrics(Y, oof)
    return r2, oof


def screen_gbr_config(X, Y, dims):
    """GBR 超参预筛：在 4 个代表性维度上做 3 折 CV，避免 50 维全网格的高成本。
    候选区间参照原论文网格（n_estimators/learning_rate/max_depth/subsample）。"""
    rep_names = [d for d in ["overall", "dms", "A_hops_all", "acidity"] if d in dims]
    rep_idx = [dims.index(d) for d in rep_names] or [0, 1, 2, 3]
    candidates = [
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=1.0),
        dict(n_estimators=500, learning_rate=0.02, max_depth=3, subsample=1.0),
        dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8),
        dict(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8),
        dict(n_estimators=300, learning_rate=0.10, max_depth=3, subsample=0.8),
        dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0),
    ]
    best, best_score = None, -np.inf
    for cfg in candidates:
        factory = lambda cfg=cfg: GradientBoostingRegressor(
            loss="squared_error", random_state=SEED, **cfg)
        scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        for tr_idx, va_idx in kf.split(X):
            A, B = impute_and_scale(X[tr_idx], X[va_idx])   # 折内重新填补与标准化
            for j in rep_idx:
                est = factory()
                est.fit(A, Y[tr_idx, j])
                scores.append(r2_score(Y[va_idx, j], est.predict(B)))
        score = float(np.mean(scores))
        log(f"  GBR 候选 {cfg} -> 代表维度CV R2={score:.4f}")
        if score > best_score:
            best, best_score = cfg, score
    log(f"  GBR 选定超参：{best}")
    return best


def tune_multi_cv(model_factory, Xraw, Y, param_grid, n_splits=5):
    """多输出模型的网格调参：每个训练折内重新填补与标准化，折外打分取平均R²。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    best_params, best_score = None, -np.inf
    for params in param_grid:
        scores = []
        for tr_idx, va_idx in kf.split(Xraw):
            A, B = impute_and_scale(Xraw[tr_idx], Xraw[va_idx])
            est = model_factory(**params)
            est.fit(A, Y[tr_idx])
            scores.append(float(np.mean(r2_score(Y[va_idx], est.predict(B)))))
        score = float(np.mean(scores))
        log(f"    {params} -> CV R2={score:.4f}")
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score


def stage1_regression(X, Y, is_train, dims):
    Xtr_raw, Ytr = X[is_train], Y[is_train]
    Xte_raw, Yte = X[~is_train], Y[~is_train]
    # 训练集填补均值与标准化参数只由训练样本估计，再应用到测试集
    Xtr, Xte = impute_and_scale(Xtr_raw, Xte_raw)

    # ---- 超参选择：调参交叉验证内部按折重新填补与标准化 ----
    log("M1 Ridge：网格选 alpha（折内预处理）")
    alpha, ridge_cv = tune_multi_cv(
        lambda alpha=1.0: Ridge(alpha=alpha), Xtr_raw, Ytr,
        [{"alpha": a} for a in [0.1, 1, 10, 100, 1000]])
    log(f"  alpha={alpha['alpha']} (CV R2={ridge_cv:.4f})")

    log("M2 PLSR：网格选成分数（折内预处理）")
    ncomp, pls_cv = tune_multi_cv(
        lambda n_components=5: PLSRegression(
            n_components=n_components, scale=False, max_iter=1000),
        Xtr_raw, Ytr, [{"n_components": c} for c in [2, 5, 10, 15, 20, 30]])
    log(f"  n_components={ncomp['n_components']} (CV R2={pls_cv:.4f})")

    log("M3 GBR：代表性维度超参预筛（折内预处理）")
    gbr_cfg = screen_gbr_config(Xtr_raw, Ytr, dims)

    models = {
        "M1_Ridge": ("multi", lambda: Ridge(alpha=alpha["alpha"])),
        "M2_PLSR": ("multi", lambda: PLSRegression(
            n_components=ncomp["n_components"], scale=False, max_iter=1000)),
        "M3_GBR": ("target", lambda: GradientBoostingRegressor(
            loss="squared_error", random_state=SEED, **gbr_cfg)),
        "M4_RF": ("target", lambda: RandomForestRegressor(
            n_estimators=300, max_features=0.3, min_samples_leaf=2,
            random_state=SEED, n_jobs=1)),
        "M5_HistGBR": ("target", lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=SEED)),
    }

    cv_r2, test_r2, summary, fitted = {}, {}, [], {}
    for name, (kind, factory) in models.items():
        log(f"{name}：5折CV + 测试集终评")
        r2_cv, _ = cv_evaluate(kind, factory, Xtr_raw, Ytr)
        if kind == "multi":
            est = factory()
            est.fit(Xtr, Ytr)
            pred_te = est.predict(Xte)
            fitted[name] = (kind, est)
        else:
            ests = fit_per_target(factory, Xtr, Ytr)
            pred_te = predict_per_target(ests, Xte)
            fitted[name] = (kind, ests)
        r2_te, rmse_te, rho_te = per_dim_metrics(Yte, pred_te)
        cv_r2[name], test_r2[name] = r2_cv, r2_te
        summary.append(dict(
            model=name, cv_R2_mean=r2_cv.mean(), cv_R2_median=np.median(r2_cv),
            test_R2_mean=r2_te.mean(), test_R2_median=np.median(r2_te),
            test_R2_min=r2_te.min(), test_R2_max=r2_te.max(),
            test_dims_R2_positive=int((r2_te > 0).sum()),
            test_RMSE_mean=rmse_te.mean(), test_Spearman_mean=np.nanmean(rho_te)))
        log(f"  CV R2均值={r2_cv.mean():.4f} | 测试 R2均值={r2_te.mean():.4f} "
            f"中位={np.median(r2_te):.4f} | R2>0维度数={(r2_te > 0).sum()}/50")

    summary_df = pd.DataFrame(summary).sort_values("cv_R2_mean", ascending=False)
    best_name = summary_df.iloc[0]["model"]
    summary_df["is_selected"] = summary_df["model"] == best_name
    summary_df.to_csv(OUT / "q1_model_summary.csv", index=False, encoding="utf-8-sig")

    perdim = pd.DataFrame({"sensory_dim": dims})
    for name in models:
        perdim[f"{name}_cv_R2"] = cv_r2[name]
        perdim[f"{name}_test_R2"] = test_r2[name]
    perdim.to_csv(OUT / "q1_perdim_r2.csv", index=False, encoding="utf-8-sig")
    log(f"第一段最优模型（按CV均值R2选型）：{best_name}")

    # 图1（合并版）：左=五模型逐维 R2 分布箱线图；右=最优模型逐维 R2 条形图
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    data = [test_r2[n] for n in models]
    ax1.boxplot(data, showmeans=True)
    ax1.set_xticklabels(list(models))
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("测试集逐维度 $R^2$")
    ax1.set_title("(a) 五种回归模型 $R^2$ 分布")
    ax1.tick_params(axis="x", rotation=15)
    ax2 = fig.add_subplot(gs[0, 1])
    order = np.argsort(-test_r2[best_name])
    vals = test_r2[best_name][order]
    ax2.bar(range(50), vals, color=["#C0504D" if v > 0 else "#4F81BD" for v in vals])
    ax2.axhline(vals.mean(), color="black", ls="--", lw=1,
                label=f"面板平均 $R^2$={vals.mean():.3f}")
    ax2.set_xticks(range(50))
    ax2.set_xticklabels([dims[i] for i in order], rotation=90, fontsize=6.5)
    ax2.set_ylabel("测试集 $R^2$")
    ax2.set_title(f"(b) 最优模型 {best_name} 的 50 维逐维 $R^2$（红=正，蓝=负）")
    ax2.legend()
    fig.suptitle("图1 回归模型选型结果", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "q1_regression_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return models, fitted, best_name, (Xtr, Ytr, Xte, Yte)


# ---------------------------------------------------------------- Stage 2
def scale_keep_nan(train, other):
    """仅用训练集估计均值与标准差做标准化，保留缺失值不变（供原生缺失路径使用）。"""
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd


def stage2_missing_robustness(models, Xtr_raw, Ytr, Xte_raw, Yte):
    """公平缺失实验：输入为保留缺失标记的原始矩阵。
    先同步遮掩，路径A再按遮掩后的训练集重新估计均值并填补、标准化后训练GBR；
    路径B保留缺失值，仅用训练集均值/标准差标准化后直接训练HistGBR；
    路径C为最终采用的方案（随机森林 + 分位标定），沿用与路径A相同的遮掩位置
    与重填补标准化，在报告 R² 的同时补报分级 Kappa 与等级准确率。"""
    gbr_factory = models["M3_GBR"][1]
    hgb_factory = models["M5_HistGBR"][1]
    rf_factory = models["M4_RF"][1]
    # 分级边界仅由完整训练集标签估计（缺失实验只遮掩特征，标签保持完整）
    boundaries = make_quantile_boundaries(Ytr)
    records = []
    for rate in MASK_RATES:
        seeds = [0] if rate == 0 else MASK_SEEDS
        for seed in seeds:
            rng = np.random.default_rng(seed)
            Xtr_m = Xtr_raw.copy(); Xtr_m[rng.random(Xtr_m.shape) < rate] = np.nan
            Xte_m = Xte_raw.copy(); Xte_m[rng.random(Xte_m.shape) < rate] = np.nan
            # 路径A：遮掩后按训练集重新估计均值填补，并标准化
            Atr, Ate = impute_and_scale(Xtr_m, Xte_m)
            ests_a = fit_per_target(gbr_factory, Atr, Ytr)
            pred_a = predict_per_target(ests_a, Ate)
            r2a, _, rhoa = per_dim_metrics(Yte, pred_a)
            records.append(dict(model="M3_GBR_reimpute", mask_rate=rate, seed=seed,
                                R2_mean=r2a.mean(), R2_median=np.median(r2a),
                                Spearman_mean=np.nanmean(rhoa),
                                dims_R2_positive=int((r2a > 0).sum())))
            # 路径B：保留缺失，仅标准化后直接训练 HistGBR
            Btr, Bte = scale_keep_nan(Xtr_m, Xte_m)
            ests_b = fit_per_target(hgb_factory, Btr, Ytr)
            pred_b = predict_per_target(ests_b, Bte)
            r2b, _, rhob = per_dim_metrics(Yte, pred_b)
            records.append(dict(model="M5_HistGBR_native", mask_rate=rate, seed=seed,
                                R2_mean=r2b.mean(), R2_median=np.median(r2b),
                                Spearman_mean=np.nanmean(rhob),
                                dims_R2_positive=int((r2b > 0).sum())))
            # 路径C：最终方案（随机森林 + 分位标定），复用路径A的重填补与标准化
            ests_c = fit_per_target(rf_factory, Atr, Ytr)
            pred_c = predict_per_target(ests_c, Ate)
            r2c, _, rhoc = per_dim_metrics(Yte, pred_c)
            Lte = to_levels(Yte, boundaries)
            Lpred_c = to_levels(pred_c, boundaries)
            qwk_c, acc_c, tol_c = grade_metrics(Lte, Lpred_c)
            records.append(dict(model="M4_RF_grading", mask_rate=rate, seed=seed,
                                R2_mean=r2c.mean(), R2_median=np.median(r2c),
                                Spearman_mean=np.nanmean(rhoc),
                                dims_R2_positive=int((r2c > 0).sum()),
                                Kappa_mean=np.nanmean(qwk_c),
                                grade_accuracy=acc_c.mean(),
                                grade_within1=tol_c.mean()))
        log(f"缺失实验 rate={rate:.0%} 完成（训练/测试同步遮掩，含缺失训练）")
    df = pd.DataFrame(records)
    df.to_csv(OUT / "q1_missing_robustness.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model_name, color, label in [("M3_GBR_reimpute", "#C0504D", "GBR + 列均值重填补（含缺失训练）"),
                                     ("M5_HistGBR_native", "#4F81BD", "HistGBR 原生缺失处理（含缺失训练）"),
                                     ("M4_RF_grading", "#2E8B57", "随机森林 + 分位标定（最终方案）")]:
        sub = df[df.model == model_name]
        agg = sub.groupby("mask_rate")["R2_mean"].agg(["mean", "std"])
        ax.errorbar(agg.index * 100, agg["mean"], yerr=agg["std"],
                    marker="o", capsize=4, color=color, label=label)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("化合物特征遮掩比例 / %")
    ax.set_ylabel("测试集 50 维平均 $R^2$")
    ax.set_title("图2 数据缺失条件下的模型性能衰减（训练/测试同步遮掩，3 个随机种子）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q1_missing_robustness.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- Stage 3
def route_cv_selection(Xtr_raw, Ytr, best_name, models):
    """在训练集内部用五折交叉验证比较两条分级路线，避免用测试集选择路线。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kind, factory = models[best_name]
    clf_factory = lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
    q1s, q2s = [], []
    for tr_idx, va_idx in kf.split(Xtr_raw):
        A, B = impute_and_scale(Xtr_raw[tr_idx], Xtr_raw[va_idx])
        b = make_quantile_boundaries(Ytr[tr_idx])
        if kind == "multi":
            est = factory(); est.fit(A, Ytr[tr_idx]); pred = est.predict(B)
        else:
            ests = fit_per_target(factory, A, Ytr[tr_idx]); pred = predict_per_target(ests, B)
        L_true = to_levels(Ytr[va_idx], b)
        q, _, _ = grade_metrics(L_true, to_levels(pred, b))
        q1s.append(float(np.nanmean(q)))
        Ltr = to_levels(Ytr[tr_idx], b)
        clfs = fit_per_target(clf_factory, A, Ltr)
        q2, _, _ = grade_metrics(L_true, predict_per_target(clfs, B))
        q2s.append(float(np.nanmean(q2)))
    return float(np.mean(q1s)), float(np.mean(q2s))


def stage3_grading(fitted, best_name, split_data, dims, Xtr_raw, models):
    Xtr, Ytr, Xte, Yte = split_data
    boundaries = make_quantile_boundaries(Ytr)       # 仅用训练集拟合边界
    Ltr = to_levels(Ytr, boundaries)
    Lte = to_levels(Yte, boundaries)

    # 路线一：最优回归模型 + 分位数锚定分箱
    kind, est = fitted[best_name]
    pred_te_cont = est.predict(Xte) if kind == "multi" else predict_per_target(est, Xte)
    L_pred_r1 = to_levels(pred_te_cont, boundaries)
    qwk1, acc1, tol1 = grade_metrics(Lte, L_pred_r1)
    log(f"路线一（{best_name}+分位标定）：QWK均值={np.nanmean(qwk1):.4f} "
        f"准确率={acc1.mean():.4f} ±1容差={tol1.mean():.4f}")

    # 路线二：M6 逐维梯度提升分类器，直接输出 0-6 等级
    factory = lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
    clfs = fit_per_target(factory, Xtr, Ltr)
    L_pred_r2 = predict_per_target(clfs, Xte)
    qwk2, acc2, tol2 = grade_metrics(Lte, L_pred_r2)
    log(f"路线二（M6 序数分类器）：QWK均值={np.nanmean(qwk2):.4f} "
        f"准确率={acc2.mean():.4f} ±1容差={tol2.mean():.4f}")

    cmp_df = pd.DataFrame({
        "sensory_dim": dims,
        "route1_regress_then_bin_QWK": qwk1, "route2_ordinal_clf_QWK": qwk2,
        "route1_accuracy": acc1, "route2_accuracy": acc2,
        "route1_within1": tol1, "route2_within1": tol2})
    cmp_df.to_csv(OUT / "q1_grading_compare.csv", index=False, encoding="utf-8-sig")

    # 路线择优在训练集内部完成：以五折交叉验证的二次加权Kappa为依据
    cv1, cv2 = route_cv_selection(Xtr_raw, Ytr, best_name, models)
    best_route = "route1" if cv1 >= cv2 else "route2"
    log(f"训练集折内QWK：路线一={cv1:.6f}，路线二={cv2:.6f} -> 采用 {best_route}")
    log(f"测试集（仅报告，不参与选择）：路线一={np.nanmean(qwk1):.4f}，路线二={np.nanmean(qwk2):.4f}")
    return best_route, (qwk1, qwk2, acc1, acc2, tol1, tol2), (cv1, cv2)


def stage3_final_scores(X, Y, meta, dims, feature_names, models, best_name, best_route):
    """评估完成后，用全部 250 款数据重训最终模型并生成 0-6 评分终表。"""
    # 最终模型在全部样本上重新估计填补均值与标准化参数后训练
    X, impute_mean, scale_mean, scale_std = fit_preprocessor(X)
    boundaries = make_quantile_boundaries(Y)
    if best_route == "route1":
        kind, factory = models[best_name]
        if kind == "multi":
            est = factory()
            est.fit(X, Y)
            cont = est.predict(X)
            final_estimators = est
        else:
            ests = fit_per_target(factory, X, Y)
            cont = predict_per_target(ests, X)
            final_estimators = ests
    else:
        L = to_levels(Y, boundaries)
        factory = lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
        final_estimators = fit_per_target(factory, X, L)
        cont = None

    if best_route == "route1":
        levels = to_levels(cont, boundaries)
    else:
        levels = predict_per_target(final_estimators, X)

    scores = meta.copy()
    for j, d in enumerate(dims):
        scores[d] = levels[:, j].astype(int)
    scores.to_csv(OUT / "q1_scores_0_6.csv", index=False, encoding="utf-8-sig")
    log(f"0-6 评分终表已生成：{scores.shape} -> results/q1_scores_0_6.csv")

    joblib.dump(dict(route=best_route, regression_model=best_name,
                     estimators=final_estimators, boundaries=boundaries,
                     feature_names=feature_names, dims=dims,
                     impute_mean=impute_mean, scale_mean=scale_mean,
                     scale_std=scale_std),
                OUT / "q1_final_model.joblib")

    # 等级分布图
    fig, ax = plt.subplots(figsize=(7, 4.2))
    counts = np.bincount(levels.ravel(), minlength=N_LEVELS) / levels.size * 100
    ax.bar(range(N_LEVELS), counts, color="#4F81BD")
    ax.set_xlabel("感官评分等级")
    ax.set_ylabel("占比 / %")
    ax.set_title("图3 全部 250 款啤酒 × 50 维感官评分的等级分布")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.4, f"{c:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "q1_grade_distribution.png", dpi=200)
    plt.close(fig)
    return scores


# ---------------------------------------------------------------- Stage 4
def stage4_style_check(scores, dims):
    """风格一致性校验：按风格统计预测等级均值，检验是否符合风格常识。"""
    style_col = "tasting_category_fine"
    grouped = scores.groupby(style_col)[dims].mean()
    counts = scores.groupby(style_col).size().rename("n_beers")
    grouped.insert(0, "n_beers", counts)
    grouped.to_csv(OUT / "q1_style_check.csv", encoding="utf-8-sig")

    show_dims = [d for d in ["A_esters_all", "A_hops_citrus", "A_hops_tropical",
                             "A_malt_burn", "A_malt_cara", "alcohol", "acidity",
                             "barnyard", "bitternes", "sweetness", "body", "overall"]
                 if d in dims]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(grouped[show_dims].values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=6)
    ax.set_xticks(range(len(show_dims)))
    ax.set_xticklabels(show_dims, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels([f"{s} (n={int(n)})" for s, n in
                        zip(grouped.index, grouped["n_beers"])], fontsize=8)
    ax.set_title("图4 各啤酒风格的 0-6 感官画像（等级均值，用于风格一致性校验）")
    fig.colorbar(im, ax=ax, label="等级均值")
    fig.tight_layout()
    fig.savefig(OUT / "q1_style_heatmap.png", dpi=200)
    plt.close(fig)
    log("风格校验完成 -> q1_style_check.csv / q1_style_heatmap.png")


# ---------------------------------------------------------------- main
def main():
    log("=" * 70)
    log("问题1 建模开始：两段式框架（回归选型 + 0-6 分级标定）")
    X, Y, is_train, meta, feature_names, dims = load_data()

    log("---- Stage 1：回归模型选型 M1-M5 ----")
    models, fitted, best_name, split_data = stage1_regression(X, Y, is_train, dims)

    log("---- Stage 2：缺失条件鲁棒性实验 ----")
    stage2_missing_robustness(models, X[is_train], Y[is_train], X[~is_train], Y[~is_train])

    log("---- Stage 3：0-6 分级标定（训练集内部择优）----")
    best_route, _, _ = stage3_grading(fitted, best_name, split_data, dims,
                                      X[is_train], models)
    scores = stage3_final_scores(X, Y, meta, dims, feature_names,
                                 models, best_name, best_route)

    log("---- Stage 4：风格一致性校验 ----")
    stage4_style_check(scores, dims)
    log("问题1 全流程完成。结果见 problem_1/results/")


if __name__ == "__main__":
    main()
