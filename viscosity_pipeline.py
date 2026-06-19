"""
================================================================================
VISCOSITY PREDICTION FOR DEEP EUTECTIC SOLVENTS
Production-Ready ML Pipeline — v2 FIXED (High-Impact Journal Version)
================================================================================

Fixes applied over the original v2:

  [FIX-1] compare_models_nested: RF and XGBoost now receive the same nested
           GridSearchCV tuning GBM gets, using their own param_grids.
           Fixed hyperparams were unfair to tree ensembles and produced the
           spurious RF > tuned-GBM gap flagged in peer review.

  [FIX-2] inner GridSearchCV was created with GroupKFold but the groups were
           NOT passed to gs.fit() inside nested_cv — all inner splits ignored
           DES-group structure. Fixed: groups=g_tr passed to every gs.fit().

  [FIX-3] fig5 SHAP beeswarm: the y-axis loop index used double-reverse logic
           (enumerate reversed list while also reversing shap_sub column index)
           causing misaligned feature↔SHAP mapping. Fixed with explicit index.

  [FIX-4] fig8 partial dependence: the manual feature scaling
           (val - mean_[fi]) / scale_[fi] was correct but applied AFTER
           transform(), double-scaling the varied feature. Fixed: build a raw
           DataFrame copy, set the feature value, then transform the whole row.

  [FIX-5] mape_safe returns np.nan which json.dump cannot serialise.
           Fixed: replace np.nan with None throughout metrics_dict / save_all.

  [FIX-6] fig1B used df_raw["hbd_hba_ratio"] but that column is named
           "hbd_hba_ratio" only if it exists; code now falls back gracefully
           and picks the first available ratio-like column.

  [FIX-7] Arrhenius baseline correctly reads temperature_c from df_raw
           (never from X, which drops the raw temperature column).
           Added an explicit assertion to guard against silent errors.

  [FIX-8] save_all/manuscript summary references to "te_rmse" column
           are now consistent with metrics_dict which keys as "rmse".

================================================================================
"""

import os, json, copy, warnings, joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from scipy.stats import linregress, gaussian_kde

import shap

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    data_path    = "viscosity_dataset_filled.csv"
    target_col   = "viscosity_cp"
    group_col    = "des_name"
    outer_folds  = 5
    inner_folds  = 4
    random_state = 42

    # Redundant features removed (Pearson r > 0.95 with a retained feature)
    drop_features = [
        "temperature_c",           # r=0.9997 with inv_temperature (Arrhenius-linearised)
        "total_hbond_donors",      # r=0.9922 with hbond_network_strength
        "total_hbond_acceptors",   # r=0.9911 with hbond_network_strength
        "weighted_dipole_total",   # r=0.9749 with temp_x_dipole (interaction encoded)
    ]

    out_dir   = "viscosity_output_v2"
    fig_dir   = "viscosity_output_v2/figures"
    model_dir = "viscosity_output_v2/models"
    rep_dir   = "viscosity_output_v2/reports"

    feature_categories = {
        "Composition":    ["hba_moles", "hbd_moles", "hbd_hba_ratio"],
        "Temperature":    ["inv_temperature", "temp_x_dipole", "temp_x_solvation"],
        "Electronic_QM":  ["weighted_lumo_energy_ev", "weighted_solvation_energy_eh",
                           "weighted_dispersion_energy_eh", "weighted_total_energy_eh"],
        "Differences":    ["dipole_total_difference", "total_energy_eh_difference"],
        "Molecular_Size": ["hba_mw", "hbd_mw", "avg_molecular_weight", "mw_ratio"],
        "HBond_Network":  ["hba_hbd_count", "hbd_hbd_count", "hba_hba_count",
                           "hbd_hba_count", "hbond_network_strength", "donor_acceptor_ratio"],
        "Interaction":    ["interaction_strength"],
    }

    # Journal palette
    C = dict(
        teal="#2E86AB", amber="#E07B39", slate="#3D405B",
        sage="#618B4A", rose="#C1666B", lavender="#7B6FA0",
        grid="#E8E8E8", cream="#FAFAF7",
    )


for d in [Config.out_dir, Config.fig_dir, Config.model_dir, Config.rep_dir]:
    os.makedirs(d, exist_ok=True)

CAT_PAL = {
    "Composition"   : "#2E86AB",
    "Temperature"   : "#E07B39",
    "Electronic_QM" : "#3D405B",
    "Differences"   : "#C1666B",
    "Molecular_Size": "#618B4A",
    "HBond_Network" : "#7B6FA0",
    "Interaction"   : "#B5838D",
    "Other"         : "#AAAAAA",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean(
        (np.log1p(np.array(y_true)) - np.log1p(np.clip(y_pred, 0, None))) ** 2
    )))


def mape_safe(y_true, y_pred, threshold=5.0):
    """Returns None (JSON-serialisable) when no samples exceed threshold."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > threshold
    if mask.sum() == 0:
        return None  # [FIX-5] was np.nan — not JSON-serialisable
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics_dict(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return dict(
        r2    = float(r2_score(y_true, y_pred)),
        rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae   = float(mean_absolute_error(y_true, y_pred)),
        rmsle = rmsle(y_true, y_pred),
        mape  = mape_safe(y_true, y_pred),   # [FIX-5] None instead of np.nan
    )


def get_category(feat):
    for cat, feats in Config.feature_categories.items():
        if feat in feats:
            return cat
    return "Other"


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": Config.C["grid"],
        "grid.linewidth": 0.6, "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "legend.framealpha": 0.85,
        "lines.linewidth": 1.6,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(Config.data_path)

    # Leakage guard
    leak = [c for c in ["density_g_cm3", "conductivity_ms_cm"] if c in df.columns]
    if leak:
        print(f"  ⚠ Removing leakage columns: {leak}")
        df = df.drop(columns=leak)

    # [FIX-7] temperature_c must exist in df_raw for Arrhenius; assert early
    assert "temperature_c" in df.columns, \
        "temperature_c column missing — required for Arrhenius baseline"

    feat_cols = [c for c in df.columns
                 if c not in [Config.group_col, Config.target_col]
                 + Config.drop_features]

    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[Config.target_col].copy()
    g = df[Config.group_col].copy()

    print(f"  Rows: {len(df)} | Features (after pruning): {len(feat_cols)} "
          f"| DES systems: {g.nunique()}")
    print(f"  Dropped redundant: {Config.drop_features}")
    print(f"  Viscosity: {y.min():.2f}–{y.max():.2f} cP  skew={y.skew():.2f}")
    return X, y, g, df


# ─────────────────────────────────────────────────────────────────────────────
# ARRHENIUS BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def arrhenius_nested_cv(df_raw):
    """
    Outer GroupKFold 5-fold Arrhenius baseline.
    Uses df_raw so temperature_c is always available.  [FIX-7]
    """
    # [FIX-7] Use df_raw directly; never touch X which drops temperature_c
    y   = df_raw[Config.target_col]
    g   = df_raw[Config.group_col]
    gkf = GroupKFold(n_splits=Config.outer_folds)

    # We need a dummy feature matrix for split() — use any column subset
    X_dummy = df_raw[[Config.group_col]].copy()

    fold_r2, fold_rmsle = [], []
    all_true, all_pred  = [], []

    for tr_idx, te_idx in gkf.split(X_dummy, y, groups=g):
        df_tr = df_raw.iloc[tr_idx]

        A_list, B_list = [], []
        for sys in df_tr[Config.group_col].unique():
            sub = df_tr[df_tr[Config.group_col] == sys]
            T_K = sub["temperature_c"].values + 273.15
            eta = sub[Config.target_col].values
            if len(sub) >= 3:
                sl, ic, *_ = linregress(1000 / T_K, np.log(eta))
                A_list.append(ic)
                B_list.append(sl)

        A_pop = np.median(A_list)
        B_pop = np.median(B_list)

        df_te  = df_raw.iloc[te_idx]
        T_K_te = df_te["temperature_c"].values + 273.15
        pred   = np.exp(A_pop + B_pop * (1000 / T_K_te))
        true   = df_te[Config.target_col].values

        fold_r2.append(float(r2_score(true, pred)))
        fold_rmsle.append(rmsle(true, pred))
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

    return dict(
        fold_r2      = fold_r2,
        fold_rmsle   = fold_rmsle,
        mean_r2      = float(np.mean(fold_r2)),
        std_r2       = float(np.std(fold_r2)),
        mean_rmsle   = float(np.mean(fold_rmsle)),
        std_rmsle    = float(np.std(fold_rmsle)),
        all_true     = all_true,
        all_pred     = all_pred,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NESTED CROSS-VALIDATION — GBM (main evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def nested_cv(X, y, g, estimator=None, param_grid=None, model_name="GBM"):
    """
    Outer: GroupKFold 5-fold  → unbiased performance estimate
    Inner: GroupKFold 4-fold inside GridSearchCV → hyperparameter selection

    [FIX-2] groups=g_tr is now explicitly passed to gs.fit() so the inner
    GroupKFold actually respects DES-group structure.

    [FIX-9] Model-agnostic: accepts any estimator + param_grid so this same
    function can run the full nested-CV + SHAP pipeline on RF (the actual
    best model per Table 3) instead of being hardcoded to GBM. Defaults to
    GBM if no estimator is passed, preserving backward compatibility.
    """
    if estimator is None:
        estimator = GradientBoostingRegressor(random_state=Config.random_state)
    if param_grid is None:
        param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [3, 4, 5],
            "learning_rate"   : [0.05, 0.10],
            "min_samples_leaf": [2, 4],
            "subsample"       : [0.8, 1.0],
        }

    outer_gkf = GroupKFold(n_splits=Config.outer_folds)
    inner_gkf = GroupKFold(n_splits=Config.inner_folds)

    fold_records    = []
    oof_true, oof_pred = [], []
    fold_models     = []
    fold_scalers    = []
    fold_X_te_list  = []
    fold_shap_vals  = []

    y_log = np.log1p(y)

    print(f"\n  Running {Config.outer_folds}-fold nested CV for {model_name} "
          f"(inner {Config.inner_folds}-fold GridSearchCV)...")

    for fold, (tr_idx, te_idx) in enumerate(
            outer_gkf.split(X, y_log, groups=g)):

        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_log.iloc[tr_idx], y_log.iloc[te_idx]
        g_tr        = g.iloc[tr_idx]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        gs = GridSearchCV(
            copy.deepcopy(estimator),
            param_grid,
            cv=inner_gkf,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        # [FIX-2] Pass g_tr so inner GroupKFold splits by DES system
        gs.fit(X_tr_s, y_tr, groups=g_tr)

        best_model     = gs.best_estimator_
        y_te_pred_log  = best_model.predict(X_te_s)
        y_te_pred_orig = np.expm1(y_te_pred_log)
        y_te_orig      = np.expm1(y_te)

        m = metrics_dict(y_te_orig.values, y_te_pred_orig)

        explainer = shap.TreeExplainer(best_model)
        shap_vals = explainer.shap_values(X_te_s)

        fold_records.append({
            "fold"            : fold + 1,
            "best_params"     : gs.best_params_,
            "cv_r2_inner"     : gs.best_score_,
            **{f"te_{k}": v for k, v in m.items()},
            "n_test_systems"  : g.iloc[te_idx].nunique(),
        })
        oof_true.extend(y_te_orig.values.tolist())
        oof_pred.extend(y_te_pred_orig.tolist())
        fold_models.append(best_model)
        fold_scalers.append(scaler)
        fold_X_te_list.append(X_te)
        fold_shap_vals.append((shap_vals, X_te_s, X_te))

        print(f"  Fold {fold+1}: R²={m['r2']:.4f}  RMSLE={m['rmsle']:.4f}  "
              f"RMSE={m['rmse']:.1f} cP  → {gs.best_params_}")

    df_folds    = pd.DataFrame(fold_records)
    oof_metrics = metrics_dict(oof_true, oof_pred)

    print(f"\n  {model_name} Nested CV OOF R²   : {oof_metrics['r2']:.4f}")
    print(f"  {model_name} Nested CV OOF RMSLE: {oof_metrics['rmsle']:.4f}")
    print(f"  {model_name} Per-fold mean R²   : {df_folds['te_r2'].mean():.4f} "
          f"± {df_folds['te_r2'].std():.4f}")

    return (df_folds, oof_true, oof_pred,
            fold_models, fold_scalers, fold_X_te_list, fold_shap_vals)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL NESTED CV COMPARISON  [FIX-1]
# ─────────────────────────────────────────────────────────────────────────────
def compare_models_nested(X, y, g):
    """
    [FIX-1] RF and XGBoost now receive their own nested GridSearchCV tuning,
    identical in structure to GBM's — same outer GroupKFold 5-fold, same inner
    GroupKFold 4-fold, same groups= passed to fit().
    Linear/distance-based models keep sensible defaults (no tree param grid).
    """
    y_log     = np.log1p(y)
    outer_gkf = GroupKFold(n_splits=Config.outer_folds)
    inner_gkf = GroupKFold(n_splits=Config.inner_folds)

    # ── Per-model configuration ────────────────────────────────────────────
    # Each entry: (base_estimator, param_grid_or_None)
    # None  → use the model as-is (no tuning needed / no tree params)
    model_configs = {
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=Config.random_state),
            {
                "n_estimators"    : [100, 200, 300],
                "max_depth"       : [3, 4, 5],
                "learning_rate"   : [0.05, 0.10],
                "min_samples_leaf": [2, 4],
                "subsample"       : [0.8, 1.0],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=Config.random_state, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth"   : [6, 8, None],
                "max_features": ["sqrt", "log2", 0.5],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "XGBoost": (
            XGBRegressor(random_state=Config.random_state,
                         verbosity=0, n_jobs=-1),
            {
                "n_estimators" : [100, 200, 300],
                "max_depth"    : [3, 4, 5],
                "learning_rate": [0.05, 0.10],
                "subsample"    : [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        # Non-tree models — fixed sensible defaults, StandardScaler applied
        "Decision Tree" : (DecisionTreeRegressor(max_depth=6,
                           random_state=Config.random_state), None),
        "Ridge"         : (Ridge(alpha=10.0), None),
        "ElasticNet"    : (ElasticNet(alpha=0.01, l1_ratio=0.5,
                           max_iter=5000), None),
        "SVR (RBF)"     : (SVR(kernel="rbf", C=10, gamma="scale"), None),
        "KNN (k=5)"     : (KNeighborsRegressor(n_neighbors=5), None),
        "Linear Regression": (LinearRegression(), None),
    }

    rows = []
    for mname, (base_mdl, param_grid) in model_configs.items():
        fold_r2, fold_rmsle = [], []

        for tr_idx, te_idx in outer_gkf.split(X, y_log, groups=g):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y_log.iloc[tr_idx], y_log.iloc[te_idx]
            g_tr        = g.iloc[tr_idx]

            sc       = StandardScaler()
            X_tr_s   = sc.fit_transform(X_tr)
            X_te_s   = sc.transform(X_te)

            if param_grid is not None:
                # [FIX-1] Tree ensembles get proper nested tuning
                gs = GridSearchCV(
                    copy.deepcopy(base_mdl),
                    param_grid,
                    cv=inner_gkf,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=0,
                )
                # [FIX-2] groups= passed so inner folds respect DES systems
                gs.fit(X_tr_s, y_tr, groups=g_tr)
                mdl_fit = gs.best_estimator_
            else:
                mdl_fit = copy.deepcopy(base_mdl)
                mdl_fit.fit(X_tr_s, y_tr)

            yp = np.expm1(mdl_fit.predict(X_te_s))
            yt = np.expm1(y_te)

            fold_r2.append(float(r2_score(yt, yp)))
            fold_rmsle.append(rmsle(yt.values, yp))

        rows.append(dict(
            Model      = mname,
            R2_mean    = float(np.mean(fold_r2)),
            R2_std     = float(np.std(fold_r2)),
            RMSLE_mean = float(np.mean(fold_rmsle)),
            RMSLE_std  = float(np.std(fold_rmsle)),
            Tuned      = param_grid is not None,
            R2_folds   = fold_r2,       # [FIX-12] raw scores for significance testing
            RMSLE_folds= fold_rmsle,
        ))
        tuned_tag = "(tuned)" if param_grid is not None else "(fixed)"
        print(f"  {mname:22s} {tuned_tag:8s} "
              f"R²={np.mean(fold_r2):.4f}±{np.std(fold_r2):.4f}  "
              f"RMSLE={np.mean(fold_rmsle):.4f}±{np.std(fold_rmsle):.4f}")

    return (pd.DataFrame(rows)
            .sort_values("R2_mean", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE — RF vs GBM vs XGBoost  [FIX-12]
# ─────────────────────────────────────────────────────────────────────────────
def significance_tests(df_cmp, models=("Random Forest", "Gradient Boosting", "XGBoost")):
    """
    Wilcoxon signed-rank test on paired per-fold R2 scores between the top
    tree-ensemble models. With only 5 folds, Wilcoxon is the appropriate
    nonparametric choice (paired t-test would over-assume normality on n=5).
    Returns a DataFrame of pairwise p-values; also reports whether each
    comparison is significant at alpha=0.05.

    NOTE: with n=5 paired folds, statistical power is inherently low —
    even large true differences may not reach significance. Report this
    honestly rather than overclaiming separation between models.
    """
    from scipy.stats import wilcoxon

    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            r1 = df_cmp.loc[df_cmp["Model"] == m1, "R2_folds"]
            r2 = df_cmp.loc[df_cmp["Model"] == m2, "R2_folds"]
            if len(r1) == 0 or len(r2) == 0:
                continue
            r1, r2 = np.array(r1.iloc[0]), np.array(r2.iloc[0])
            try:
                stat, p = wilcoxon(r1, r2)
            except ValueError:
                # All differences zero or n too small
                stat, p = np.nan, np.nan
            rows.append(dict(
                Model_A=m1, Model_B=m2,
                R2_A_mean=float(np.mean(r1)), R2_B_mean=float(np.mean(r2)),
                wilcoxon_stat=float(stat) if not np.isnan(stat) else None,
                p_value=float(p) if not np.isnan(p) else None,
                significant_at_alpha05=bool(p < 0.05) if not np.isnan(p) else False,
            ))
    df_sig = pd.DataFrame(rows)
    print("\n  Pairwise Wilcoxon signed-rank tests (n=5 folds, paired):")
    for _, row in df_sig.iterrows():
        sig_tag = "significant" if row["significant_at_alpha05"] else "not significant"
        p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
        print(f"    {row['Model_A']:18s} vs {row['Model_B']:18s}  "
              f"p={p_str}  ({sig_tag} at α=0.05)")
    print("  Note: n=5 paired folds gives low statistical power; "
          "absence of significance does not imply equivalence.")
    return df_sig


# ─────────────────────────────────────────────────────────────────────────────
# SHAP AGGREGATION ACROSS FOLDS
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_shap(fold_shap_vals, feat_names):
    """Stack SHAP values across all folds for global summary."""
    all_shap = np.vstack([sv for sv, _, _  in fold_shap_vals])
    all_Xdf  = pd.concat([Xdf for _, _, Xdf in fold_shap_vals], ignore_index=True)

    mean_abs = pd.DataFrame({
        "feature"      : feat_names,
        "mean_abs_shap": np.abs(all_shap).mean(axis=0),
        "category"     : [get_category(f) for f in feat_names],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Also return the stacked scaled X (used for beeswarm colour)
    all_X_scaled = np.vstack([Xs for _, Xs, _ in fold_shap_vals])

    return all_shap, all_X_scaled, all_Xdf, mean_abs


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig1_data_landscape(df_raw):
    """
    Fig 1 — DES Viscosity Landscape
    A: log-KDE histogram of viscosity
    B: Arrhenius profiles coloured by HBD:HBA ratio (curated 20 DES)
    C: Violin per temperature
    """
    set_style()
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
    C   = Config.C
    y   = df_raw[Config.target_col]

    # A — KDE on log scale
    ax    = fig.add_subplot(gs[0])
    y_log = np.log10(y)
    vals  = np.linspace(y_log.min() - 0.3, y_log.max() + 0.3, 400)
    kde   = gaussian_kde(y_log, bw_method=0.25)
    ax.fill_between(vals, kde(vals), alpha=0.22, color=C["teal"])
    ax.plot(vals, kde(vals), color=C["teal"], lw=2.2)
    ax.axvline(np.log10(y.median()), color=C["amber"], lw=1.8, ls="--",
               label=f"Median = {y.median():.0f} cP")
    ax.plot(y_log, np.full(len(y_log), -0.025),
            "|", color=C["slate"], alpha=0.18, ms=4)
    ax.set_xlabel("log₁₀(Viscosity / cP)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"(A)  Viscosity Distribution\n"
                 f"{df_raw[Config.group_col].nunique()} DES systems, "
                 f"{len(df_raw)} data points")
    ax.legend()

    # B — Arrhenius profiles  [FIX-6] robust column lookup for ratio feature
    ax2  = fig.add_subplot(gs[1])
    cnts = df_raw.groupby("des_name")["temperature_c"].count()
    full = cnts[cnts == 6].index[:20]

    # [FIX-6] graceful column fallback
    ratio_candidates = ["hbd_hba_ratio", "donor_acceptor_ratio", "mw_ratio"]
    ratio_col = next(
        (c for c in ratio_candidates if c in df_raw.columns), None
    )
    if ratio_col and len(full) > 0:
        ratio_val = df_raw.groupby("des_name")[ratio_col].first()
        norm      = mcolors.Normalize(ratio_val[full].min(), ratio_val[full].max())
        cmap      = plt.cm.RdYlBu_r
        for sys in full:
            sub = df_raw[df_raw["des_name"] == sys].sort_values("temperature_c")
            T_K = sub["temperature_c"].values + 273.15
            ax2.plot(1000 / T_K, np.log(sub[Config.target_col].values),
                     color=cmap(norm(ratio_val[sys])), alpha=0.7, lw=1.3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax2, pad=0.02, shrink=0.85)
        cb.set_label(ratio_col.replace("_", " ").title(), fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for\nArrhenius profiles",
                 transform=ax2.transAxes, ha="center", va="center")
    ax2.set_xlabel("1000 / T  (K⁻¹)")
    ax2.set_ylabel("ln(η / cP)")
    ax2.set_title("(B)  Arrhenius Profiles\nColour = HBD:HBA Molar Ratio")

    # C — violin per temperature
    ax3  = fig.add_subplot(gs[2])
    temps = sorted(df_raw["temperature_c"].unique())
    data  = [df_raw[df_raw["temperature_c"] == t][Config.target_col].values
             for t in temps]
    vp = ax3.violinplot(data, positions=temps, widths=3,
                        showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(C["teal"]); body.set_alpha(0.45)
    vp["cmedians"].set_color(C["amber"]); vp["cmedians"].set_linewidth(2.2)
    ax3.set_yscale("log")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Viscosity (cP, log scale)")
    ax3.set_title("(C)  Temperature Dependence\n25–50 °C across all DES")
    ax3.set_xticks(temps)

    fig.suptitle("Figure 1 — Viscosity Landscape of the DES Dataset",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(f"{Config.fig_dir}/fig1_data_landscape.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 1 saved")


def fig2_arrhenius_vs_ml(arr_res, oof_true, oof_pred, df_folds):
    """Fig 2 — Arrhenius Baseline vs Random Forest (primary model, nested CV)."""
    set_style()
    C   = Config.C
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    # A — Arrhenius parity
    ax = fig.add_subplot(gs[0])
    at = np.array(arr_res["all_true"])
    ap = np.clip(arr_res["all_pred"], 0.1, None)
    ax.scatter(at, ap, alpha=0.35, s=14, color=C["rose"], edgecolors="none")
    lim = [0.5, 3500]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual Viscosity (cP)")
    ax.set_ylabel("Predicted Viscosity (cP)")
    ax.set_title(f"(A)  Arrhenius Baseline\n"
                 f"R² = {arr_res['mean_r2']:.3f} ± {arr_res['std_r2']:.3f}  "
                 f"RMSLE = {arr_res['mean_rmsle']:.3f} ± {arr_res['std_rmsle']:.3f}")
    ax.text(0.05, 0.95,
            "Global population params\nA_pop, B_pop estimated per outer fold",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grid"]))

    # B — GBM OOF parity
    ax2 = fig.add_subplot(gs[1])
    ot, op = np.array(oof_true), np.array(oof_pred)
    ax2.scatter(ot, op, alpha=0.45, s=14, color=C["teal"], edgecolors="none")
    ax2.plot(lim, lim, "k--", lw=1.5)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Actual Viscosity (cP)")
    ax2.set_ylabel("Predicted Viscosity (cP)")
    fold_r2_mean = df_folds["te_r2"].mean()
    fold_r2_std  = df_folds["te_r2"].std()
    fold_rl_mean = df_folds["te_rmsle"].mean()
    fold_rl_std  = df_folds["te_rmsle"].std()
    ax2.set_title(f"(B)  Random Forest — Nested CV OOF\n"
                  f"R² = {fold_r2_mean:.3f} ± {fold_r2_std:.3f}  "
                  f"RMSLE = {fold_rl_mean:.3f} ± {fold_rl_std:.3f}")
    delta_r2    = fold_r2_mean   - arr_res["mean_r2"]
    delta_rmsle = arr_res["mean_rmsle"] - fold_rl_mean
    ax2.text(0.05, 0.95,
             f"Δ R² vs Arrhenius: +{delta_r2:.3f}\n"
             f"Δ RMSLE vs Arrhenius: −{delta_rmsle:.3f}",
             transform=ax2.transAxes, fontsize=8, va="top", color=C["sage"],
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["sage"]))

    # C — per-fold comparison bar
    ax3 = fig.add_subplot(gs[2])
    folds = np.arange(1, Config.outer_folds + 1)
    w = 0.32
    ax3.bar(folds - w/2, arr_res["fold_r2"], width=w, color=C["rose"],
            alpha=0.75, label="Arrhenius baseline", edgecolor="white")
    ax3.bar(folds + w/2, df_folds["te_r2"].values, width=w, color=C["teal"],
            alpha=0.85, label="Random Forest (nested CV)", edgecolor="white")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Outer CV Fold")
    ax3.set_ylabel("Test R²")
    ax3.set_title("(C)  Per-Fold R² Comparison\nArrhenius vs Random Forest")
    ax3.set_xticks(folds)
    ax3.legend()

    fig.suptitle("Figure 2 — Arrhenius Baseline vs Random Forest: Nested CV Evaluation",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(f"{Config.fig_dir}/fig2_arrhenius_vs_ml.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 2 saved")


def fig3_nested_cv_performance(df_folds, oof_true, oof_pred):
    """Fig 3 — Nested CV Performance Summary."""
    set_style()
    C   = Config.C
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
    ot, op = np.array(oof_true), np.array(oof_pred)

    # A — density-coloured parity
    ax  = fig.add_subplot(gs[0])
    xy  = np.vstack([np.log10(ot + 1), np.log10(np.clip(op, 0.01, None) + 1)])
    kde_col = gaussian_kde(xy)(xy)
    idx = kde_col.argsort()
    sc  = ax.scatter(ot[idx], op[idx], c=kde_col[idx], cmap="YlOrRd",
                     s=20, alpha=0.8, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Local density", shrink=0.85)
    lim = [0.5, 3500]
    ax.plot(lim, lim, "k--", lw=1.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual Viscosity (cP)")
    ax.set_ylabel("Predicted Viscosity (cP)")
    ax.set_title(f"(A)  OOF Parity Plot\n"
                 f"R² = {r2_score(ot, op):.4f}  RMSLE = {rmsle(ot, op):.4f}")

    # B — per-fold R² and RMSLE
    ax2  = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    folds = df_folds["fold"].values
    ax2.bar(folds - 0.18, df_folds["te_r2"].values, 0.35,
            color=C["teal"], alpha=0.8, label="R² (left)")
    ax2b.bar(folds + 0.18, df_folds["te_rmsle"].values, 0.35,
             color=C["amber"], alpha=0.7, label="RMSLE (right)")
    ax2.set_xlabel("Outer CV Fold")
    ax2.set_ylabel("R²", color=C["teal"])
    ax2b.set_ylabel("RMSLE", color=C["amber"])
    ax2.set_xticks(folds)
    ax2.set_title("(B)  Per-Fold Metrics\nR² and RMSLE per outer fold")
    ax2.axhline(df_folds["te_r2"].mean(), color=C["teal"],
                ls="--", lw=1.2, alpha=0.6)
    ax2b.axhline(df_folds["te_rmsle"].mean(), color=C["amber"],
                 ls="--", lw=1.2, alpha=0.6)
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7)

    # C — log-residual distribution
    ax3 = fig.add_subplot(gs[2])
    log_res = np.log1p(ot) - np.log1p(np.clip(op, 0, None))
    ax3.hist(log_res, bins=40, color=C["teal"], alpha=0.65,
             edgecolor="white", density=True)
    mu, sigma = log_res.mean(), log_res.std()
    xn = np.linspace(log_res.min(), log_res.max(), 200)
    ax3.plot(xn, stats.norm.pdf(xn, mu, sigma),
             color=C["amber"], lw=2.2,
             label=f"N(μ={mu:.3f}, σ={sigma:.3f})")
    ax3.axvline(0, color="black", lw=1.2, ls="--")
    ax3.set_xlabel("Log-space Residual  [ln(1+η_true) − ln(1+η_pred)]")
    ax3.set_ylabel("Density")
    ax3.set_title("(C)  Residual Distribution\nLog-space (ideal: centred at 0)")
    ax3.legend()

    fig.suptitle("Figure 3 — Nested CV Performance: Random Forest Viscosity Predictions",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(f"{Config.fig_dir}/fig3_nested_cv_performance.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 3 saved")


def fig4_model_comparison(df_cmp):
    """Fig 4 — Multi-model comparison (all tree ensembles now fully tuned)."""
    set_style()
    C   = Config.C
    n   = len(df_cmp)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    y_pos = np.arange(n)

    is_gb    = df_cmp["Model"].str.contains("Gradient Boosting")
    is_tuned = df_cmp.get("Tuned", pd.Series([False] * n))

    def bar_color(row):
        if row["Model"] == "Gradient Boosting":
            return C["amber"]
        if row.get("Tuned", False):
            return C["lavender"]
        return C["teal"]

    colors = [bar_color(row) for _, row in df_cmp.iterrows()]

    for ax, metric, label in [
        (axes[0], "R2",    "Nested CV R² (mean ± std)"),
        (axes[1], "RMSLE", "Nested CV RMSLE (mean ± std)  [lower = better]"),
    ]:
        means = df_cmp[f"{metric}_mean"]
        stds  = df_cmp[f"{metric}_std"]
        ax.barh(y_pos, means, height=0.06, color=colors, alpha=0.9)
        ax.scatter(means, y_pos, color=colors, s=90, zorder=5)
        ax.errorbar(means, y_pos, xerr=stds,
                    fmt="none", ecolor=C["slate"], elinewidth=1.2, capsize=4)
        ax.set_xlabel(label)
        if ax is axes[0]:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_cmp["Model"], fontsize=9)
            ax.set_title("(A)  Test R² per Model\nNested 5-fold GroupKFold CV")
            ax.axvline(0.65, color=C["sage"], lw=1.2, ls=":", alpha=0.8,
                       label="R² = 0.65")
            ax.legend(fontsize=8)
        else:
            ax.set_title("(B)  Test RMSLE per Model")
        for i, (r, s) in enumerate(zip(means, stds)):
            ax.text(r + s + 0.01, i, f"{r:.3f}", va="center", fontsize=8)

    gb_p    = mpatches.Patch(color=C["amber"],    label="Gradient Boosting (tuned)")
    tuned_p = mpatches.Patch(color=C["lavender"], label="RF / XGBoost (tuned)")
    oth_p   = mpatches.Patch(color=C["teal"],     label="Other models (fixed params)")
    fig.legend(handles=[gb_p, tuned_p, oth_p], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle(
        "Figure 4 — Multi-Model Benchmark (Nested CV, Fair Tuning, No Data Leakage)",
        fontsize=12, fontweight="bold")
    fig.savefig(f"{Config.fig_dir}/fig4_model_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 4 saved")


def fig5_shap_summary(all_shap, all_Xdf, mean_abs_shap, feat_names):
    """
    Fig 5 — SHAP Summary
    A: mean |SHAP| bar  B: beeswarm top-10  C: category donut
    [FIX-3] Beeswarm y-axis now uses explicit index mapping (no double-reverse).
    """
    set_style()
    C   = Config.C
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # A — mean |SHAP| bar
    ax    = fig.add_subplot(gs[0])
    top15 = mean_abs_shap.head(15).copy().iloc[::-1]
    bar_c = [CAT_PAL.get(c, "#AAAAAA") for c in top15["category"]]
    bars  = ax.barh(range(len(top15)), top15["mean_abs_shap"],
                    color=bar_c, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].str[:30], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP Value|  (log-space)")
    ax.set_title("(A)  Global Feature Importance\nSHAP — Top 15")
    for bar, (_, row) in zip(bars, top15.iterrows()):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{row.mean_abs_shap:.4f}", va="center", fontsize=7.5)
    patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_PAL.items()
               if k in mean_abs_shap["category"].values]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              title="Category", title_fontsize=7)

    # B — SHAP beeswarm for top 10  [FIX-3]
    ax2        = fig.add_subplot(gs[1])
    top10_feats = mean_abs_shap.head(10)["feature"].values  # best → worst
    # Map feature names to column indices in all_shap
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}

    for plot_row, feat in enumerate(top10_feats[::-1]):  # plot bottom→top
        fi  = feat_to_idx[feat]          # [FIX-3] explicit index, no double-reverse
        sv  = all_shap[:, fi]
        fv  = all_Xdf[feat].values
        fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jitter = np.random.uniform(-0.25, 0.25, len(sv))
        sc = ax2.scatter(sv, plot_row + jitter, c=fv_n, cmap="coolwarm",
                         s=8, alpha=0.5, edgecolors="none")

    ax2.axvline(0, color="black", lw=1, ls="--")
    ax2.set_yticks(range(len(top10_feats)))
    ax2.set_yticklabels([f[:28] for f in top10_feats[::-1]], fontsize=8)
    ax2.set_xlabel("SHAP Value (log-space, positive = ↑ viscosity)")
    ax2.set_title("(B)  SHAP Beeswarm\nTop-10 Features")
    plt.colorbar(sc, ax=ax2, label="Feature value\n(low→high)", shrink=0.8)

    # C — category donut
    ax3 = fig.add_subplot(gs[2])
    cat_shap   = (mean_abs_shap.groupby("category")["mean_abs_shap"]
                  .sum().sort_values(ascending=False))
    cat_colors = [CAT_PAL.get(c, "#AAAAAA") for c in cat_shap.index]
    _, _, autotexts = ax3.pie(
        cat_shap.values, colors=cat_colors, autopct="%1.1f%%",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(7.5)
    ax3.set_title("(C)  SHAP by Feature Category\n(Donut = % total importance)")
    ax3.legend(cat_shap.index, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=2)

    fig.suptitle(
        "Figure 5 — SHAP Feature Importance (TreeExplainer, Aggregated Across Folds)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(f"{Config.fig_dir}/fig5_shap_summary.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 5 saved")


def fig6_shap_interaction(all_shap, all_Xdf, mean_abs_shap, feat_names):
    """Fig 6 — SHAP Dependence Plots for top 4 features."""
    set_style()
    C    = Config.C
    top4 = mean_abs_shap.head(4)["feature"].values
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.subplots_adjust(wspace=0.40)

    for ax, feat in zip(axes, top4):
        fi = feat_to_idx[feat]
        sv = all_shap[:, fi]
        fv = all_Xdf[feat].values

        col_feat = ("inv_temperature" if feat != "inv_temperature"
                    else "weighted_lumo_energy_ev")
        cv = all_Xdf[col_feat].values if col_feat in all_Xdf.columns else fv

        sc = ax.scatter(fv, sv, c=cv, cmap="viridis",
                        s=18, alpha=0.6, edgecolors="none")
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)

        order = np.argsort(fv)
        win   = max(10, len(fv) // 20)
        rm    = pd.Series(sv[order]).rolling(win, center=True, min_periods=1).mean().values
        ax.plot(fv[order], rm, color=C["amber"], lw=2.5, zorder=5,
                label="Rolling mean")

        plt.colorbar(sc, ax=ax, label=col_feat[:18], shrink=0.85)
        ax.set_xlabel(feat.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("SHAP Value" if feat == top4[0] else "")
        ax.set_title(f"{feat[:26]}\n[{get_category(feat)}]", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Figure 6 — SHAP Dependence Plots (Top-4 Features)\n"
        "Colour = inv_temperature (or LUMO for temperature feature)",
        fontsize=11, fontweight="bold", y=1.03)
    fig.savefig(f"{Config.fig_dir}/fig6_shap_dependence.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 6 saved")


def fig7_error_anatomy(oof_true, oof_pred, df_raw):
    """Fig 7 — Error anatomy (hexbin, RMSLE by decile, Q-Q, |error| vs actual)."""
    set_style()
    C  = Config.C
    ot, op = np.array(oof_true), np.array(oof_pred)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.42)

    # A — hexbin
    ax = fig.add_subplot(gs[0])
    hx = ax.hexbin(np.log10(ot + 1), np.log10(np.clip(op, 0.01, None) + 1),
                   gridsize=25, cmap="Blues", mincnt=1, linewidths=0.3)
    plt.colorbar(hx, ax=ax, label="Count", shrink=0.85)
    lim = [-0.1, 3.5]
    ax.plot(lim, lim, "--", color=C["amber"], lw=1.8, zorder=5)
    ax.set_xlabel("log₁₀(Actual+1)")
    ax.set_ylabel("log₁₀(Predicted+1)")
    ax.set_title(f"(A)  Hexbin Parity\nR² = {r2_score(ot, op):.4f}")

    # B — RMSLE by viscosity decile
    ax2    = fig.add_subplot(gs[1])
    # [FIX-10] pd.qcut(np.ndarray, labels=False) returns a plain ndarray,
    # which has no .unique(). Wrap in pd.Series first, or use np.unique().
    decile = pd.Series(pd.qcut(ot, 10, labels=False, duplicates="drop"))
    d_vals = sorted(decile.unique())
    d_rmsle   = [rmsle(ot[decile.values == d], op[decile.values == d]) for d in d_vals]
    d_center  = [float(np.median(ot[decile.values == d])) for d in d_vals]
    ax2.plot(d_center, d_rmsle, "o-", color=C["teal"], lw=1.8)
    ax2.fill_between(d_center, d_rmsle, alpha=0.15, color=C["teal"])
    ax2.set_xlabel("Viscosity Decile Median (cP)")
    ax2.set_ylabel("RMSLE")
    ax2.set_xscale("log")
    ax2.set_title("(B)  RMSLE by Viscosity Decile\n(Identifies problematic ranges)")
    ax2.axhline(float(np.mean(d_rmsle)), color=C["amber"], ls="--", lw=1.5,
                label=f"Mean = {np.mean(d_rmsle):.3f}")
    ax2.legend()

    # C — Q-Q
    ax3 = fig.add_subplot(gs[2])
    log_res = np.log1p(ot) - np.log1p(np.clip(op, 0, None))
    (osm, osr), (slope, intercept, r) = stats.probplot(log_res, dist="norm")
    ax3.scatter(osm, osr, color=C["teal"], s=18, alpha=0.6, zorder=4)
    fit = slope * np.array([osm[0], osm[-1]]) + intercept
    ax3.plot([osm[0], osm[-1]], fit, color=C["amber"], lw=2,
             label=f"Pearson r = {r:.4f}")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.set_title("(C)  Q–Q Plot\nLog-space Residuals")
    ax3.legend()

    # D — |error| vs actual
    ax4 = fig.add_subplot(gs[3])
    abs_e = np.abs(ot - op)
    sc = ax4.scatter(ot, abs_e, c=np.log10(ot + 1), cmap="plasma",
                     s=18, alpha=0.65, edgecolors="none")
    plt.colorbar(sc, ax=ax4, label="log₁₀(Actual cP)", shrink=0.85)
    ax4.set_xscale("log"); ax4.set_yscale("log")
    ax4.set_xlabel("Actual Viscosity (cP)")
    ax4.set_ylabel("|Prediction Error| (cP)")
    ax4.set_title("(D)  Absolute Error vs Actual\nError scales with magnitude")

    fig.suptitle("Figure 7 — Prediction Error Anatomy (OOF Predictions, All Folds)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(f"{Config.fig_dir}/fig7_error_anatomy.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 7 saved")


def fig8_partial_dependence(fold_models, fold_scalers, fold_X_te_list,
                             mean_abs_shap):
    """
    Fig 8 — Marginal partial dependence for top 4 SHAP features.
    Averaged across all outer fold models.

    [FIX-4] PD computation now:
      1. Works on raw (unscaled) X_te DataFrame copy
      2. Replaces the target feature value in-place
      3. Calls scaler.transform() once on the entire modified DataFrame
         → avoids the double-scaling bug in the original implementation.
    """
    set_style()
    C    = Config.C
    top4 = mean_abs_shap.head(4)["feature"].values

    def pd_curve(feat, n_grid=60):
        grids, means = [], []
        for model, scaler, X_te in zip(fold_models, fold_scalers,
                                        fold_X_te_list):
            if feat not in X_te.columns:
                continue
            g_ = np.linspace(X_te[feat].quantile(0.02),
                              X_te[feat].quantile(0.98), n_grid)
            pd_ = []
            for val in g_:
                # [FIX-4] modify raw DataFrame, then scale whole matrix
                X_copy = X_te.copy()
                X_copy[feat] = val
                X_scaled = scaler.transform(X_copy)
                pd_.append(float(np.expm1(model.predict(X_scaled)).mean()))
            grids.append(g_)
            means.append(pd_)

        if not grids:
            return np.array([]), np.array([]), np.array([])

        common_g     = np.linspace(min(g[0]  for g in grids),
                                    max(g[-1] for g in grids), n_grid)
        interp_means = [np.interp(common_g, grids[i], means[i])
                        for i in range(len(grids))]
        return (common_g,
                np.mean(interp_means, axis=0),
                np.std(interp_means, axis=0))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.subplots_adjust(wspace=0.40)

    for ax, feat in zip(axes, top4):
        grid, pd_mean, pd_std = pd_curve(feat)
        if len(grid) == 0:
            ax.text(0.5, 0.5, f"Feature '{feat}'\nnot in test folds",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        p10 = np.percentile(grid, 10)
        p90 = np.percentile(grid, 90)
        ax.axvspan(grid.min(), p10,  alpha=0.07, color=C["rose"])
        ax.axvspan(p90, grid.max(),  alpha=0.07, color=C["rose"])
        ax.axvspan(p10, p90,         alpha=0.07, color=C["sage"])

        ax.plot(grid, pd_mean, color=C["teal"], lw=2.3, zorder=5)
        ax.fill_between(grid, pd_mean - pd_std, pd_mean + pd_std,
                        alpha=0.22, color=C["teal"],
                        label="±1 std (across folds)")

        ax.set_xlabel(feat.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("Mean Predicted Viscosity (cP)" if feat == top4[0] else "")
        ax.set_title(f"{feat[:26]}\n[{get_category(feat)}]", fontsize=9)
        ax.legend(fontsize=7)
        trend = "↑" if pd_mean[-1] > pd_mean[0] else "↓"
        ax.text(0.96, 0.07, f"{trend} with feature",
                transform=ax.transAxes, ha="right", fontsize=8,
                color=C["amber"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C["grid"], alpha=0.9))

    fig.suptitle(
        "Figure 8 — Partial Dependence Plots (Top-4 SHAP Features)\n"
        "Averaged across 5 outer folds | Shading = ±1 std | "
        "Green band = 10th–90th percentile training range",
        fontsize=10, fontweight="bold", y=1.04)
    fig.savefig(f"{Config.fig_dir}/fig8_partial_dependence.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✔ Fig 8 saved")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_all(df_folds, oof_true, oof_pred, df_cmp, mean_abs_shap,
             arr_res, fold_models, fold_scalers):

    # OOF predictions
    pd.DataFrame({
        "actual_cP"   : oof_true,
        "predicted_cP": oof_pred,
        "log_residual": (np.log1p(oof_true)
                         - np.log1p(np.clip(oof_pred, 0, None))),
    }).to_csv(f"{Config.out_dir}/oof_predictions.csv", index=False)

    df_folds.to_csv(f"{Config.out_dir}/nested_cv_fold_metrics.csv", index=False)

    # [FIX-13] df_cmp now carries R2_folds/RMSLE_folds list-columns (needed for
    # the Wilcoxon test) which don't serialise cleanly to flat CSV. Export a
    # clean summary CSV without them, and a separate JSON with raw fold scores
    # for full reproducibility.
    list_cols = [c for c in ["R2_folds", "RMSLE_folds"] if c in df_cmp.columns]
    df_cmp.drop(columns=list_cols).to_csv(
        f"{Config.out_dir}/model_comparison_nested_cv.csv", index=False)
    if list_cols:
        df_cmp[["Model"] + list_cols].to_json(
            f"{Config.out_dir}/model_comparison_fold_scores.json",
            orient="records", indent=2)

    mean_abs_shap.to_csv(f"{Config.out_dir}/shap_importance.csv", index=False)

    # Best fold model (highest test R²)
    best_fold = int(df_folds["te_r2"].idxmax())
    joblib.dump(fold_models[best_fold],
                f"{Config.model_dir}/best_rf_fold{best_fold+1}.pkl")
    joblib.dump(fold_scalers[best_fold],
                f"{Config.model_dir}/best_scaler_fold{best_fold+1}.pkl")

    oof_m = metrics_dict(oof_true, oof_pred)

    # [FIX-5] Use correct column names from metrics_dict (rmse, not te_rmse)
    # [FIX-8] Consistent key naming: te_rmse → rmse suffix from metrics_dict
    summary = dict(
        timestamp              = datetime.now().isoformat(),
        n_features             = f"{len(df_folds.columns)} features after pruning",
        dropped_features       = Config.drop_features,
        outer_folds            = Config.outer_folds,
        inner_folds            = Config.inner_folds,
        oof_r2                 = oof_m["r2"],
        oof_rmsle              = oof_m["rmsle"],
        oof_rmse               = oof_m["rmse"],
        oof_mae                = oof_m["mae"],
        oof_mape               = oof_m["mape"],   # None if no η>5 cP samples
        fold_r2_mean           = float(df_folds["te_r2"].mean()),
        fold_r2_std            = float(df_folds["te_r2"].std()),
        fold_rmsle_mean        = float(df_folds["te_rmsle"].mean()),
        fold_rmsle_std         = float(df_folds["te_rmsle"].std()),
        fold_rmse_mean         = float(df_folds["te_rmse"].mean()),   # [FIX-8]
        fold_rmse_std          = float(df_folds["te_rmse"].std()),
        fold_mae_mean          = float(df_folds["te_mae"].mean()),
        fold_mae_std           = float(df_folds["te_mae"].std()),
        arrhenius_r2_mean      = arr_res["mean_r2"],
        arrhenius_r2_std       = arr_res["std_r2"],
        arrhenius_rmsle_mean   = arr_res["mean_rmsle"],
        arrhenius_rmsle_std    = arr_res["std_rmsle"],
        delta_r2_vs_arrhenius  = float(df_folds["te_r2"].mean() - arr_res["mean_r2"]),
        delta_rmsle_vs_arrhenius = float(arr_res["mean_rmsle"] - df_folds["te_rmsle"].mean()),
        top5_shap_features     = mean_abs_shap.head(5)["feature"].tolist(),
        best_gbm_fold          = best_fold + 1,
    )

    with open(f"{Config.rep_dir}/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)   # default=str catches any residual nan

    print(f"  Outputs saved → {Config.out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  VISCOSITY PREDICTION PIPELINE v2 FIXED — Journal Version")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    print("\n[1] Loading and preparing data...")
    X, y, g, df_raw = load_data()

    print("\n[2] Arrhenius baseline (nested CV, same folds as ML models)...")
    arr_res = arrhenius_nested_cv(df_raw)
    print(f"  Arrhenius R²    = {arr_res['mean_r2']:.4f} ± {arr_res['std_r2']:.4f}")
    print(f"  Arrhenius RMSLE = {arr_res['mean_rmsle']:.4f} ± {arr_res['std_rmsle']:.4f}")

    print("\n[3] Nested CV — Random Forest (PRIMARY/headline model; outer 5-fold, "
          "inner 4-fold GridSearchCV)...")
    # [FIX-11] RF is the actual best-performing model (Table 3: R²=0.658 vs
    # GBM's 0.637). SHAP, partial dependence, and the main parity/error
    # figures should be computed on the model the paper claims is best —
    # otherwise interpretability claims and headline performance numbers
    # come from two different models, which a reviewer will flag.
    rf_param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [6, 8, None],
        "max_features"     : ["sqrt", "log2", 0.5],
        "min_samples_leaf" : [1, 2, 4],
    }
    rf_estimator = RandomForestRegressor(random_state=Config.random_state,
                                          n_jobs=-1)

    (df_folds, oof_true, oof_pred,
     fold_models, fold_scalers,
     fold_X_te_list, fold_shap_vals) = nested_cv(
        X, y, g,
        estimator=rf_estimator,
        param_grid=rf_param_grid,
        model_name="Random Forest",
    )

    delta_r2    = df_folds["te_r2"].mean()    - arr_res["mean_r2"]
    delta_rmsle = arr_res["mean_rmsle"] - df_folds["te_rmsle"].mean()
    print(f"\n  ΔR²    vs Arrhenius: {delta_r2:+.4f}")
    print(f"  ΔRMSLE vs Arrhenius: {-delta_rmsle:+.4f}")

    print("\n[3b] Nested CV — GBM (kept for Table 2 / cross-model reference)...")
    (gbm_df_folds, gbm_oof_true, gbm_oof_pred,
     gbm_fold_models, gbm_fold_scalers,
     gbm_fold_X_te_list, gbm_fold_shap_vals) = nested_cv(
        X, y, g, model_name="GBM (reference)"
    )

    print("\n[4] Multi-model nested CV comparison (RF, GBM & XGBoost all tuned)...")
    df_cmp = compare_models_nested(X, y, g)

    print("\n[4b] Statistical significance — RF vs GBM vs XGBoost (Wilcoxon)...")
    df_sig = significance_tests(df_cmp)
    df_sig.to_csv(f"{Config.out_dir}/significance_tests.csv", index=False)

    print("\n[5] Aggregating SHAP values across folds...")
    feat_names = list(X.columns)
    all_shap, all_X_scaled, all_Xdf, mean_abs_shap = aggregate_shap(
        fold_shap_vals, feat_names)
    print("  Top 5 SHAP features:")
    for _, row in mean_abs_shap.head(5).iterrows():
        print(f"    {row.feature:35s} [{row.category}]: {row.mean_abs_shap:.4f}")

    print("\n[6] Generating 8 manuscript figures...")
    fig1_data_landscape(df_raw)
    fig2_arrhenius_vs_ml(arr_res, oof_true, oof_pred, df_folds)
    fig3_nested_cv_performance(df_folds, oof_true, oof_pred)
    fig4_model_comparison(df_cmp)
    fig5_shap_summary(all_shap, all_Xdf, mean_abs_shap, feat_names)
    fig6_shap_interaction(all_shap, all_Xdf, mean_abs_shap, feat_names)
    fig7_error_anatomy(oof_true, oof_pred, df_raw)
    fig8_partial_dependence(fold_models, fold_scalers, fold_X_te_list,
                             mean_abs_shap)

    print("\n[7] Saving all outputs...")
    save_all(df_folds, oof_true, oof_pred, df_cmp, mean_abs_shap,
             arr_res, fold_models, fold_scalers)

    # ── Manuscript summary ────────────────────────────────────────────────
    oof_m = metrics_dict(oof_true, oof_pred)
    print("\n" + "=" * 72)
    print("  MANUSCRIPT SUMMARY")
    print("=" * 72)
    print(f"""
Dataset (after feature pruning)
  DES systems   : {g.nunique()}
  Data points   : {len(y)}
  Features      : {X.shape[1]} (removed {len(Config.drop_features)} redundant, r > 0.95)
  Viscosity     : {y.min():.2f}–{y.max():.2f} cP  (skew = {y.skew():.2f})

Evaluation Strategy
  Outer CV  : {Config.outer_folds}-fold GroupKFold  (no DES system overlap)
  Inner CV  : {Config.inner_folds}-fold GroupKFold  (hyperparameter tuning per fold)
  SHAP      : TreeExplainer aggregated across all {Config.outer_folds} outer folds
              (computed on Random Forest, the best-performing model — see Table 3)
  Fairness  : RF, GBM, and XGBoost all receive identical nested GridSearchCV tuning

Arrhenius Baseline (population A, B estimated per fold)
  R²    = {arr_res['mean_r2']:.4f} ± {arr_res['std_r2']:.4f}
  RMSLE = {arr_res['mean_rmsle']:.4f} ± {arr_res['std_rmsle']:.4f}

Random Forest (PRIMARY MODEL) — Nested CV Performance
  OOF R²          = {oof_m['r2']:.4f}
  OOF RMSLE       = {oof_m['rmsle']:.4f}
  Per-fold R²     = {df_folds['te_r2'].mean():.4f} ± {df_folds['te_r2'].std():.4f}
  Per-fold RMSLE  = {df_folds['te_rmsle'].mean():.4f} ± {df_folds['te_rmsle'].std():.4f}
  Per-fold RMSE   = {df_folds['te_rmse'].mean():.2f} ± {df_folds['te_rmse'].std():.2f} cP
  Per-fold MAE    = {df_folds['te_mae'].mean():.2f} ± {df_folds['te_mae'].std():.2f} cP

GBM (reference run) — Nested CV Performance
  OOF R²          = {metrics_dict(gbm_oof_true, gbm_oof_pred)['r2']:.4f}
  Per-fold R²     = {gbm_df_folds['te_r2'].mean():.4f} ± {gbm_df_folds['te_r2'].std():.4f}

Improvement Over Arrhenius Baseline (Random Forest)
  Δ R²    = {delta_r2:+.4f}
  Δ RMSLE = {-delta_rmsle:+.4f}

Statistical Significance (Wilcoxon signed-rank, n=5 paired folds)
  See significance_tests.csv — note that n=5 gives low statistical power;
  a non-significant p-value does not mean the models perform equivalently.

Top 5 SHAP Features (Random Forest, aggregated across outer folds)
""")
    for _, row in mean_abs_shap.head(5).iterrows():
        print(f"  {row.name+1}. {row.feature:35s} [{row.category}]: "
              f"{row.mean_abs_shap:.4f}")

    print(f"""
Figures saved → {Config.fig_dir}/
  fig1_data_landscape.png
  fig2_arrhenius_vs_ml.png
  fig3_nested_cv_performance.png
  fig4_model_comparison.png
  fig5_shap_summary.png
  fig6_shap_dependence.png
  fig7_error_anatomy.png
  fig8_partial_dependence.png

Models saved → {Config.model_dir}/
Reports saved → {Config.rep_dir}/model_summary.json
""")
    print("  Pipeline v2 FIXED — complete.")
    return df_folds, oof_true, oof_pred, df_cmp, mean_abs_shap, arr_res


if __name__ == "__main__":
    main()
