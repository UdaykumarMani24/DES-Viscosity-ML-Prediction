"""
================================================================================
VISCOSITY PREDICTION FOR DEEP EUTECTIC SOLVENTS — Pipeline v3
System-disjoint nested cross-validation, SHAP interpretability
================================================================================

  DATA
  [D1] Deduplication is now part of the code (was done outside the script, so
       the published results could not be reproduced from the released CSV).
       Step 1 drops exact duplicate rows; step 2 resolves rows that share
       (system, molar ratio, temperature, viscosity) but carry different
       descriptor sets. Every removed row is logged to reports/removed_rows.csv.
  [D2] Data-quality report (reports/data_qc_*.csv + figS4): flags
       non-monotonic temperature series, implausible 5 °C step ratios,
       identical viscosity series recorded under different compositions, and
       conflicting descriptor sets shared between different systems.
       It does NOT silently change data: decisions go in Config
       (drop_temperatures, value_corrections, rows_to_drop).

  METHOD
  [M1] Correlation pruning (|r| > 0.95) now happens INSIDE every training fold
       (outer and inner) via a CorrPruner transformer in the sklearn Pipeline.
       The retained feature set per outer fold is saved so stability can be
       reported (reviewer 2 comment).
  [M2] One nested-CV routine is used for ALL nine models, so every table and
       figure comes from the same folds and the same code path.
  [M3] Sample SD (ddof = 1) everywhere. v2 mixed np.std (ddof=0) and
       pandas .std() (ddof=1), so Fig. 2 panels used different conventions.
  [M4] Wilcoxon tests on both R² and RMSLE, with the n = 5 power caveat.
  [M5] Repeated system-disjoint CV over shuffled group partitions (seeded),
       with rank stability and the Nadeau–Bengio corrected resampled t-test.
       (v2's GroupKFold had no shuffle, so "changing the seed" did nothing.)
  [M6] Learning curve: R²/RMSLE vs fraction of training SYSTEMS.
  [M7] Wider Random Forest grid (max_features was always at the grid edge).

  OUTPUTS
  [O1] OOF predictions include des_name, temperature, molar ratio and fold.
  [O2] GBM per-fold metrics + hyperparameters, GBM SHAP, RF-vs-GBM rank table,
       fold-level SHAP variance (Table S1), fold membership (Table S2),
       per-fold descriptor profile (Table S3).
  [O3] model_summary.json: correct feature count, correct model labels.

  FIGURES
  [F1] No baked-in "Figure N —" titles (captions belong in the manuscript);
       set Config.embed_titles = True to restore them.
  [F2] Fig 1B shows 20 systems spread across the viscosity range (not the
       first 20 alphabetically); ratio colour bar on a log scale.
  [F3] Fig 3C axis label no longer cropped; legends no longer cover bars.
  [F4] Fig 4 value labels no longer overlap error bars.
  [F5] Fig 5C category shares moved to the legend (no overlapping labels).
  [F6] Fig 6/8 readable axis and colour-bar labels.
  [F7] Fig 8 band = 10th–90th percentile of the TRAINING data (v2 used the
       percentiles of the plotting grid); misleading trend arrow removed.
  [F8] Supplementary figures: S1 RF vs GBM SHAP, S2 learning curve,
       S3 repeated-CV model ranking, S4 data-QC diagnostic.

Usage
  python viscosity_pipeline_v3.py            # full run
  FAST=1 python viscosity_pipeline_v3.py     # quick smoke test (tiny grids)
================================================================================
"""

import os, json, copy, warnings, platform
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import linregress, gaussian_kde, wilcoxon, spearmanr

import joblib
import shap
import sklearn
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    data_path    = "viscosity_dataset_CORRECTED_ml_ready.csv"   # rebuilt dataset (see build_dataset.py)
    target_col   = "viscosity_cp"
    group_col    = "des_name"
    outer_folds  = 5
    inner_folds  = 4
    random_state = 42
    n_jobs       = -1

    # ── DATA DECISIONS (authors must set these after checking the sources) ──
    # 25 °C values are ~1 cP while the same systems are 51–2528 cP at 30 °C.
    # After verifying against the original papers, EITHER correct the values
    # in the CSV, OR exclude the temperature here, e.g. drop_temperatures = [25]
    drop_temperatures = []
    # Individual value fixes: {(des_name, temperature_c): new_value_or_None}
    # None drops the row. Example (verify first!): {("L:P400", 45): 175.3}
    value_corrections = {("L:P400", 40): None}
    # Explicit row drops: list of (des_name, hbd_hba_ratio) pairs,
    # e.g. [("BBC:L", 0.5), ("BBC:T", 0.5)] if those entries are duplicates.
    rows_to_drop = []
    # Which descriptor set to keep when one measurement has two sets:
    # "first" (v2 behaviour) or "last". Verify against xtb outputs!
    conflict_policy = "first"

    # ── Feature pruning (done inside each training fold) ────────────────────
    corr_threshold = 0.95
    # When two features are correlated above the threshold, drop the one that
    # appears in this list (in this priority), keeping the more interpretable one.
    drop_preference = ["temperature_c", "total_hbond_donors",
                       "total_hbond_acceptors", "weighted_dipole_total"]

    # Inner-loop selection criterion (on the ln(1+η) target).
    # "r2" = v2 behaviour; "neg_root_mean_squared_error" = RMSLE (primary metric)
    tuning_scoring = "r2"

    # ── Extra analyses promised in the response letter ──────────────────────
    run_repeated_cv    = True
    n_repeats          = 10          # 10 × 5-fold shuffled group partitions
    repeat_tuning      = "full"      # "full" re-tunes every fold (slow, rigorous)
                                     # "fixed" reuses the main-run modal params
    run_learning_curve = True
    lc_fractions       = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_subsamples      = 5

    embed_titles = False             # [F1] no "Figure N —" titles in images

    out_dir   = "viscosity_output_CORRECTED"
    fig_dir   = f"{out_dir}/figures"
    model_dir = f"{out_dir}/models"
    rep_dir   = f"{out_dir}/reports"

    feature_categories = {
        "Composition":    ["hba_moles", "hbd_moles", "hbd_hba_ratio"],
        "Temperature":    ["inv_temperature", "temperature_c",
                           "temp_x_dipole", "temp_x_solvation"],
        "Electronic_QM":  ["weighted_lumo_energy_ev", "weighted_solvation_energy_eh",
                           "weighted_dispersion_energy_eh", "weighted_total_energy_eh",
                           "weighted_dipole_total"],
        "Differences":    ["dipole_total_difference", "total_energy_eh_difference"],
        "Molecular_Size": ["hba_mw", "hbd_mw", "avg_molecular_weight", "mw_ratio"],
        "HBond_Network":  ["hba_hbd_count", "hbd_hbd_count", "hba_hba_count",
                           "hbd_hba_count", "hbond_network_strength",
                           "donor_acceptor_ratio", "total_hbond_donors",
                           "total_hbond_acceptors"],
        "Interaction":    ["interaction_strength"],
    }

    C = dict(teal="#2E86AB", amber="#E07B39", slate="#3D405B", sage="#618B4A",
             rose="#C1666B", lavender="#7B6FA0", grid="#E8E8E8")


if FAST:   # smoke-test settings
    Config.n_repeats, Config.lc_subsamples = 2, 2
    Config.lc_fractions = [0.5, 1.0]
    Config.out_dir = "viscosity_output_v3_FAST"
    Config.fig_dir, Config.model_dir, Config.rep_dir = (
        f"{Config.out_dir}/figures", f"{Config.out_dir}/models", f"{Config.out_dir}/reports")

for _d in [Config.out_dir, Config.fig_dir, Config.model_dir, Config.rep_dir]:
    os.makedirs(_d, exist_ok=True)

CAT_PAL = {"Composition": "#2E86AB", "Temperature": "#E07B39",
           "Electronic_QM": "#3D405B", "Differences": "#C1666B",
           "Molecular_Size": "#618B4A", "HBond_Network": "#7B6FA0",
           "Interaction": "#B5838D", "Other": "#AAAAAA"}
CAT_LABEL = {"Composition": "Composition", "Temperature": "Temperature",
             "Electronic_QM": "Electronic (QM)", "Differences": "Differences",
             "Molecular_Size": "Molecular size", "HBond_Network": "H-bond network",
             "Interaction": "Interaction", "Other": "Other"}
PRETTY = {"inv_temperature": "1000/T (K⁻¹)",
          "weighted_lumo_energy_ev": "Weighted LUMO energy (eV)",
          "hbd_hbd_count": "HBD–HBD contact count",
          "hbd_hba_count": "HBD→HBA contact count",
          "hba_hbd_count": "HBA→HBD contact count",
          "hba_hba_count": "HBA–HBA contact count",
          "temp_x_solvation": "T × solvation energy",
          "temp_x_dipole": "T × dipole",
          "hbond_network_strength": "H-bond network strength",
          "weighted_solvation_energy_eh": "Weighted solvation energy (Eh)",
          "weighted_total_energy_eh": "Weighted total energy (Eh)",
          "weighted_dispersion_energy_eh": "Weighted dispersion energy (Eh)",
          "donor_acceptor_ratio": "Donor/acceptor ratio",
          "mw_ratio": "MW ratio (HBD/HBA)", "hbd_hba_ratio": "HBD:HBA molar ratio"}


def pretty(f):
    return PRETTY.get(f, f.replace("_", " "))


def get_category(feat):
    for cat, feats in Config.feature_categories.items():
        if feat in feats:
            return cat
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# METRICS & STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean(
        (np.log1p(np.asarray(y_true)) - np.log1p(np.clip(y_pred, 0, None))) ** 2)))


def mape_safe(y_true, y_pred, threshold=5.0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true > threshold
    return None if m.sum() == 0 else float(
        np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


def metrics_dict(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return dict(r2=float(r2_score(y_true, y_pred)),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                mae=float(mean_absolute_error(y_true, y_pred)),
                rmsle=rmsle(y_true, y_pred),
                mape=mape_safe(y_true, y_pred))


def msd(x):
    """mean and SAMPLE standard deviation (ddof=1)  [M3]"""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": Config.C["grid"], "grid.linewidth": 0.6,
        "figure.facecolor": "white", "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "legend.framealpha": 0.9, "lines.linewidth": 1.6})


def save_fig(fig, name, title=None):
    if Config.embed_titles and title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(f"{Config.fig_dir}/{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print(f"  ✔ {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING, CLEANING AND QC   [D1][D2]
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean():
    df = pd.read_csv(Config.data_path)
    n0 = len(df)
    tg, gc = Config.target_col, Config.group_col
    removed = []

    leak = [c for c in ["density_g_cm3", "conductivity_ms_cm"] if c in df.columns]
    if leak:
        print(f"  ⚠ Removing leakage columns: {leak}")
        df = df.drop(columns=leak)
    assert "temperature_c" in df.columns, "temperature_c is required"

    # 1. exact duplicate rows
    dup = df.duplicated(keep="first")
    removed.append(df[dup].assign(removal_reason="exact duplicate row"))
    df = df[~dup]
    n1 = len(df)

    # 2. same measurement, different descriptor sets
    key = [gc, "hbd_hba_ratio", "temperature_c", tg]
    keep = "first" if Config.conflict_policy == "first" else "last"
    conf = df.duplicated(subset=key, keep=keep)
    conflict_systems = sorted(df.loc[df.duplicated(subset=key, keep=False), gc].unique())
    removed.append(df[conf].assign(
        removal_reason=f"conflicting descriptor set (kept {keep})"))
    df = df[~conf]
    n2 = len(df)

    # 3. author decisions
    for (sys_, t), val in Config.value_corrections.items():
        m = (df[gc] == sys_) & (df["temperature_c"] == t)
        if val is None:
            removed.append(df[m].assign(removal_reason="value_corrections: dropped"))
            df = df[~m]
        else:
            df.loc[m, tg] = val
        print(f"  • value correction {sys_} @ {t} °C → {val}  ({m.sum()} row)")
    for sys_, ratio in Config.rows_to_drop:
        m = (df[gc] == sys_) & np.isclose(df["hbd_hba_ratio"], ratio)
        removed.append(df[m].assign(removal_reason="rows_to_drop"))
        df = df[~m]
    if Config.drop_temperatures:
        m = df["temperature_c"].isin(Config.drop_temperatures)
        removed.append(df[m].assign(removal_reason="drop_temperatures"))
        df = df[~m]

    df = df.reset_index(drop=True)
    pd.concat(removed).to_csv(f"{Config.rep_dir}/removed_rows.csv", index=False)
    log = dict(rows_original=n0,
               exact_duplicates_removed=n0 - n1,
               conflicting_descriptor_rows_removed=n1 - n2,
               systems_with_conflicting_descriptors=conflict_systems,
               other_rows_removed=n2 - len(df),
               rows_final=len(df), systems_final=int(df[gc].nunique()))
    print(f"  Rows: {n0} → {n1} (exact dups) → {n2} (conflicting descriptors) "
          f"→ {len(df)} final | systems: {df[gc].nunique()}")
    print(f"  Systems with conflicting descriptor sets: {conflict_systems}")
    return df, log


def data_qc(df):
    """[D2] Flags problems; never modifies data."""
    tg, gc = Config.target_col, Config.group_col
    rows = []
    for (sys_, ratio), sub in df.groupby([gc, "hbd_hba_ratio"]):
        sub = sub.sort_values("temperature_c")
        T, v = sub["temperature_c"].values, sub[tg].values
        for i in range(len(T) - 1):
            rows.append(dict(des_name=sys_, hbd_hba_ratio=ratio,
                             T_low=T[i], T_high=T[i + 1],
                             eta_low_T=v[i], eta_high_T=v[i + 1],
                             step_ratio=v[i] / v[i + 1] if v[i + 1] > 0 else np.nan))
    steps = pd.DataFrame(rows)
    steps["flag_nonmonotonic"] = steps["step_ratio"] < 1.0
    steps["flag_extreme_step"] = (steps["step_ratio"] > 5) | (steps["step_ratio"] < 0.9)
    steps.to_csv(f"{Config.rep_dir}/data_qc_temperature_steps.csv", index=False)

    per_T = (steps.groupby(["T_low", "T_high"])["step_ratio"]
             .agg(["median", "min", "max", "count"]).reset_index())
    per_T.to_csv(f"{Config.rep_dir}/data_qc_step_ratio_by_temperature.csv", index=False)

    # identical viscosity series recorded under different compositions/systems
    piv = df.pivot_table(index=[gc, "hbd_hba_ratio"], columns="temperature_c",
                         values=tg, aggfunc="first")
    same = piv[piv.duplicated(keep=False)].reset_index()
    same.to_csv(f"{Config.rep_dir}/data_qc_identical_series.csv", index=False)

    # composition-weighted descriptors identical at different molar ratios
    wcols = [c for c in df.columns if c.startswith("weighted_")]
    bad_w = []
    for sys_, sub in df.groupby(gc):
        if sub["hbd_hba_ratio"].nunique() > 1:
            if sub.groupby("hbd_hba_ratio")[wcols].first().nunique().max() == 1:
                bad_w.append(sys_)

    print("\n  DATA QC (see reports/data_qc_*.csv):")
    print(per_T.round(2).to_string(index=False))
    extreme = steps[steps["flag_extreme_step"]]
    if len(extreme):
        print(f"  ⚠ {len(extreme)} temperature steps outside 0.9–5× "
              f"(expected ≈2× per 5 °C). Most affected intervals:")
        print("   ", extreme.groupby(["T_low", "T_high"]).size().to_dict())
    nm = steps[steps["flag_nonmonotonic"] & (steps["T_low"] > min(steps["T_low"]))]
    for _, r in nm.iterrows():
        print(f"  ⚠ non-monotonic: {r.des_name} (ratio {r.hbd_hba_ratio}) "
              f"{r.T_low:.0f}→{r.T_high:.0f} °C: {r.eta_low_T} → {r.eta_high_T} cP")
    if len(same):
        print(f"  ⚠ identical viscosity series under different entries: "
              f"{same[gc].tolist()}")
    if bad_w:
        print(f"  ⚠ weighted descriptors do not change with molar ratio: {bad_w}")
    return steps, per_T


def build_xy(df):
    feat_cols = [c for c in df.columns if c not in [Config.group_col, Config.target_col]]
    X = df[feat_cols].astype(float)
    y = df[Config.target_col].astype(float)
    g = df[Config.group_col].astype(str)
    return X, y, g


# ─────────────────────────────────────────────────────────────────────────────
# FOLD-INTERNAL CORRELATION PRUNING   [M1]
# ─────────────────────────────────────────────────────────────────────────────
class CorrPruner(BaseEstimator, TransformerMixin):
    """Drops features with |r| > threshold against an already-retained feature.
    Features listed in drop_preference are considered last (so they are the
    ones dropped). Fitted on training data only."""

    def __init__(self, threshold=0.95, drop_preference=()):
        self.threshold = threshold
        self.drop_preference = drop_preference

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        corr = X.corr().abs().fillna(0.0)
        pref = [c for c in self.drop_preference if c in X.columns]
        order = [c for c in X.columns if c not in pref] + pref
        kept, self.dropped_ = [], {}
        for c in order:
            hits = [k for k in kept if corr.loc[c, k] > self.threshold]
            if hits:
                self.dropped_[c] = (hits[0], float(corr.loc[c, hits[0]]))
            else:
                kept.append(c)
        self.kept_ = [c for c in X.columns if c in kept]
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_)[self.kept_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.kept_)


def make_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("prune", CorrPruner(Config.corr_threshold, tuple(Config.drop_preference))),
        ("scale", StandardScaler()),
        ("model", estimator)])


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
def model_configs():
    rs = Config.random_state
    gbm_grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5],
                "learning_rate": [0.05, 0.10], "min_samples_leaf": [2, 4],
                "subsample": [0.8, 1.0]}
    rf_grid = {"n_estimators": [200, 300, 500], "max_depth": [8, 12, None],
               "max_features": [0.3, 0.5, 0.7, 1.0],        # [M7] extended
               "min_samples_leaf": [1, 2]}
    xgb_grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5],
                "learning_rate": [0.05, 0.10], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]}
    if FAST:
        gbm_grid = {"n_estimators": [100], "max_depth": [3, 5]}
        rf_grid = {"n_estimators": [100], "max_features": [0.5, 1.0]}
        xgb_grid = {"n_estimators": [100], "max_depth": [3, 5]}
    return {
        "Random Forest": (RandomForestRegressor(random_state=rs, n_jobs=1), rf_grid),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=rs), gbm_grid),
        "XGBoost": (XGBRegressor(random_state=rs, verbosity=0, n_jobs=1), xgb_grid),
        "Decision Tree": (DecisionTreeRegressor(max_depth=6, random_state=rs), None),
        "KNN (k=5)": (KNeighborsRegressor(n_neighbors=5), None),
        "SVR (RBF)": (SVR(kernel="rbf", C=10, gamma="scale"), None),
        "Ridge": (Ridge(alpha=10.0), None),
        "ElasticNet": (ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000), None),
        "Linear Regression": (LinearRegression(), None),
    }


TUNED_ENSEMBLES = ("Random Forest", "Gradient Boosting", "XGBoost")


# ─────────────────────────────────────────────────────────────────────────────
# FOLD GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def make_outer_folds(X, g, seed=None):
    """seed=None → deterministic GroupKFold (main analysis).
    seed=int → shuffled, size-balanced group k-fold (repeated CV)  [M5]."""
    k = Config.outer_folds
    if seed is None:
        return list(GroupKFold(n_splits=k).split(X, groups=g))
    rng = np.random.RandomState(seed)
    sizes = g.value_counts()
    groups = sizes.index.values.copy()
    rng.shuffle(groups)
    # greedy balancing on shuffled order (largest first, ties broken randomly)
    groups = sorted(groups, key=lambda s: -sizes[s])
    fold_of, load = {}, np.zeros(k)
    for s in groups:
        f = int(np.argmin(load + rng.uniform(0, 1e-6, k)))
        fold_of[s] = f
        load[f] += sizes[s]
    fid = g.map(fold_of).values
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)]


# ─────────────────────────────────────────────────────────────────────────────
# NESTED CV  (one routine for every model)   [M2]
# ─────────────────────────────────────────────────────────────────────────────
def run_nested(name, estimator, grid, X, y, g, df_meta, folds,
               compute_shap=False, fixed_params=None, verbose=True):
    y_log = np.log1p(y)
    inner = GroupKFold(n_splits=Config.inner_folds)
    records, oof, shap_store, fitted = [], [], [], []

    for f, (tr, te) in enumerate(folds, start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_log.iloc[tr], y_log.iloc[te]
        pipe = make_pipeline(copy.deepcopy(estimator))

        if grid is not None and fixed_params is None:
            gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                              cv=inner, scoring=Config.tuning_scoring,
                              n_jobs=Config.n_jobs)
            gs.fit(X_tr, y_tr, groups=g.iloc[tr])        # groups respected
            best, best_params, inner_score = (
                gs.best_estimator_,
                {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                float(gs.best_score_))
        else:
            if fixed_params:
                pipe.set_params(**{f"model__{k}": v for k, v in fixed_params.items()})
            best = pipe.fit(X_tr, y_tr)
            best_params, inner_score = (fixed_params or {}), None

        pred = np.expm1(best.predict(X_te))
        true = np.expm1(y_te.values)
        m = metrics_dict(true, pred)
        kept = best.named_steps["prune"].kept_
        records.append(dict(model=name, fold=f, **{f"te_{k}": v for k, v in m.items()},
                            best_params=json.dumps(best_params, default=str),
                            inner_score=inner_score,
                            n_test_systems=int(g.iloc[te].nunique()),
                            n_test_rows=int(len(te)), n_features=len(kept),
                            dropped_features=json.dumps(best.named_steps["prune"].dropped_)))
        meta = df_meta.iloc[te]
        oof.append(pd.DataFrame({
            "row_id": te, "des_name": meta[Config.group_col].values,
            "temperature_c": meta["temperature_c"].values,
            "hbd_hba_ratio": meta["hbd_hba_ratio"].values,
            "fold": f, "actual_cP": true, "predicted_cP": pred,
            "log_residual": np.log1p(true) - np.log1p(np.clip(pred, 0, None))}))
        fitted.append(best)

        if compute_shap:
            Xt = best[:-1].transform(X_te)
            sv = shap.TreeExplainer(best.named_steps["model"]).shap_values(Xt)
            full = pd.DataFrame(np.nan, index=range(len(te)), columns=X.columns)
            full[kept] = sv
            shap_store.append(dict(fold=f, shap=full, X_raw=X_te.reset_index(drop=True)))

        if verbose:
            print(f"    {name:18s} fold {f}: R²={m['r2']:.3f}  RMSLE={m['rmsle']:.3f}  "
                  f"RMSE={m['rmse']:.1f}  feats={len(kept)}  {best_params}")

    return dict(records=pd.DataFrame(records), oof=pd.concat(oof, ignore_index=True),
                shap=shap_store, fitted=fitted)


def summarise(results):
    rows = []
    for name, res in results.items():
        r = res["records"]
        r2m, r2s = msd(r["te_r2"]); rlm, rls = msd(r["te_rmsle"])
        rmm, rms = msd(r["te_rmse"]); mam, mas = msd(r["te_mae"])
        rows.append(dict(Model=name, R2_mean=r2m, R2_sd=r2s, RMSLE_mean=rlm,
                         RMSLE_sd=rls, RMSE_mean=rmm, RMSE_sd=rms, MAE_mean=mam,
                         MAE_sd=mas, Tuned=name in TUNED_ENSEMBLES,
                         R2_folds=r["te_r2"].round(4).tolist(),
                         RMSLE_folds=r["te_rmsle"].round(4).tolist()))
    return pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ARRHENIUS BASELINE (same folds)
# ─────────────────────────────────────────────────────────────────────────────
def arrhenius_cv(df, folds):
    tg, gc = Config.target_col, Config.group_col
    fr2, frl, true_all, pred_all = [], [], [], []
    for tr, te in folds:
        dtr = df.iloc[tr]
        A, B = [], []
        for _, sub in dtr.groupby(gc):
            if len(sub) >= 3:
                sl, ic, *_ = linregress(1000 / (sub["temperature_c"] + 273.15),
                                        np.log(sub[tg]))
                A.append(ic); B.append(sl)
        a, b = np.median(A), np.median(B)
        dte = df.iloc[te]
        pred = np.exp(a + b * 1000 / (dte["temperature_c"].values + 273.15))
        true = dte[tg].values
        fr2.append(float(r2_score(true, pred))); frl.append(rmsle(true, pred))
        true_all += true.tolist(); pred_all += pred.tolist()
    r2m, r2s = msd(fr2); rlm, rls = msd(frl)
    return dict(fold_r2=fr2, fold_rmsle=frl, mean_r2=r2m, sd_r2=r2s,
                mean_rmsle=rlm, sd_rmsle=rls, all_true=true_all, all_pred=pred_all)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNIFICANCE   [M4]
# ─────────────────────────────────────────────────────────────────────────────
def wilcoxon_tests(summary):
    rows = []
    for i, a in enumerate(TUNED_ENSEMBLES):
        for b in TUNED_ENSEMBLES[i + 1:]:
            ra = summary.set_index("Model").loc[a]; rb = summary.set_index("Model").loc[b]
            out = dict(Model_A=a, Model_B=b)
            for met in ["R2", "RMSLE"]:
                x, z = np.array(ra[f"{met}_folds"]), np.array(rb[f"{met}_folds"])
                try:
                    st, p = wilcoxon(x, z)
                except ValueError:
                    st, p = np.nan, np.nan
                out[f"{met}_mean_A"], out[f"{met}_mean_B"] = x.mean(), z.mean()
                out[f"{met}_A_better_folds"] = int(((x > z) if met == "R2" else (x < z)).sum())
                out[f"p_{met}"] = None if np.isnan(p) else float(p)
            rows.append(out)
    df = pd.DataFrame(rows)
    print("\n  Wilcoxon signed-rank (n = 5 paired folds; min attainable two-sided p = 0.0625):")
    print(df[["Model_A", "Model_B", "p_R2", "p_RMSLE"]].to_string(index=False))
    return df


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    d = np.asarray(d, float); J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


# ─────────────────────────────────────────────────────────────────────────────
# REPEATED CV WITH SHUFFLED GROUP PARTITIONS   [M5]
# ─────────────────────────────────────────────────────────────────────────────
def modal_params(records):
    ps = [json.loads(p) for p in records["best_params"]]
    out = {}
    for k in ps[0]:
        out[k] = Counter(json.dumps(p[k]) for p in ps).most_common(1)[0][0]
        out[k] = json.loads(out[k])
    return out


def repeated_cv(X, y, g, df, main_results):
    cfg = model_configs()
    rows = []
    print(f"\n  Repeated CV: {Config.n_repeats} × {Config.outer_folds}-fold shuffled "
          f"group partitions, tuning = {Config.repeat_tuning}")
    for r in range(Config.n_repeats):
        folds = make_outer_folds(X, g, seed=1000 + r)
        for name in TUNED_ENSEMBLES:
            est, grid = cfg[name]
            fixed = (modal_params(main_results[name]["records"])
                     if Config.repeat_tuning == "fixed" else None)
            res = run_nested(name, est, grid, X, y, g, df, folds,
                             fixed_params=fixed, verbose=False)
            rec = res["records"].assign(repeat=r + 1)
            rows.append(rec)
        last = pd.concat(rows)
        cur = last[last.repeat == r + 1].groupby("model")[["te_r2", "te_rmsle"]].mean()
        print(f"    repeat {r+1}: " + "  ".join(
            f"{m}: R²={cur.loc[m,'te_r2']:.3f}/RMSLE={cur.loc[m,'te_rmsle']:.3f}"
            for m in TUNED_ENSEMBLES))
    rep = pd.concat(rows, ignore_index=True)
    rep.to_csv(f"{Config.rep_dir}/repeated_cv_fold_scores.csv", index=False)

    per_rep = rep.groupby(["repeat", "model"])[["te_r2", "te_rmsle"]].mean().reset_index()
    per_rep["rank_r2"] = per_rep.groupby("repeat")["te_r2"].rank(ascending=False)
    per_rep["rank_rmsle"] = per_rep.groupby("repeat")["te_rmsle"].rank(ascending=True)
    stab = per_rep.groupby("model").agg(
        R2_mean=("te_r2", "mean"), R2_sd_over_repeats=("te_r2", "std"),
        RMSLE_mean=("te_rmsle", "mean"), RMSLE_sd_over_repeats=("te_rmsle", "std"),
        frac_rank1_R2=("rank_r2", lambda s: (s == 1).mean()),
        frac_rank1_RMSLE=("rank_rmsle", lambda s: (s == 1).mean())).reset_index()
    stab.to_csv(f"{Config.rep_dir}/repeated_cv_rank_stability.csv", index=False)

    n = len(X); n_test = n / Config.outer_folds; n_train = n - n_test
    tests = []
    for i, a in enumerate(TUNED_ENSEMBLES):
        for b in TUNED_ENSEMBLES[i + 1:]:
            A = rep[rep.model == a].sort_values(["repeat", "fold"])
            B = rep[rep.model == b].sort_values(["repeat", "fold"])
            out = dict(Model_A=a, Model_B=b, n_scores=len(A))
            for met in ["te_r2", "te_rmsle"]:
                t, p = corrected_ttest(A[met].values - B[met].values, n_train, n_test)
                out[f"t_{met}"], out[f"p_{met}"] = t, p
            tests.append(out)
    tests = pd.DataFrame(tests)
    tests.to_csv(f"{Config.rep_dir}/repeated_cv_corrected_ttests.csv", index=False)
    print("\n  Rank stability across repeats:\n" + stab.round(3).to_string(index=False))
    print("\n  Nadeau–Bengio corrected t-tests:\n" + tests.round(4).to_string(index=False))
    return rep, per_rep, stab, tests


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING CURVE   [M6]
# ─────────────────────────────────────────────────────────────────────────────
def learning_curve(X, y, g, folds, rf_records):
    est, _ = model_configs()["Random Forest"]
    y_log = np.log1p(y)
    rows = []
    for (tr, te), (_, rec) in zip(folds, rf_records.iterrows()):
        params = json.loads(rec["best_params"])
        sys_tr = g.iloc[tr].unique()
        for frac in Config.lc_fractions:
            reps = 1 if frac >= 1.0 else Config.lc_subsamples
            for s in range(reps):
                rng = np.random.RandomState(Config.random_state + s)
                pick = rng.choice(sys_tr, max(5, int(round(frac * len(sys_tr)))),
                                  replace=False)
                sub = tr[g.iloc[tr].isin(pick).values]
                pipe = make_pipeline(copy.deepcopy(est))
                pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
                pipe.fit(X.iloc[sub], y_log.iloc[sub])
                pred = np.expm1(pipe.predict(X.iloc[te]))
                true = y.iloc[te].values
                rows.append(dict(fold=int(rec["fold"]), fraction=frac, subsample=s,
                                 n_train_systems=len(pick), n_train_rows=len(sub),
                                 r2=r2_score(true, pred), rmsle=rmsle(true, pred)))
    lc = pd.DataFrame(rows)
    lc.to_csv(f"{Config.rep_dir}/learning_curve_raw.csv", index=False)
    per_fold = lc.groupby(["fraction", "fold"])[["r2", "rmsle", "n_train_systems"]].mean()
    summ = per_fold.groupby("fraction").agg(
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmsle_mean=("rmsle", "mean"), rmsle_sd=("rmsle", "std"),
        n_train_systems=("n_train_systems", "mean")).reset_index()
    summ.to_csv(f"{Config.rep_dir}/learning_curve_summary.csv", index=False)
    print("\n  Learning curve (RF):\n" + summ.round(3).to_string(index=False))
    return summ


# ─────────────────────────────────────────────────────────────────────────────
# SHAP AGGREGATION + SUPPLEMENTARY TABLES   [O2]
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_shap(store, label):
    all_shap = pd.concat([s["shap"] for s in store], ignore_index=True)
    all_X = pd.concat([s["X_raw"] for s in store], ignore_index=True)
    used = all_shap.columns[all_shap.notna().any()]
    mean_abs = (all_shap[used].abs().mean().rename("mean_abs_shap")
                .reset_index().rename(columns={"index": "feature"}))
    mean_abs["category"] = mean_abs["feature"].map(get_category)
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    mean_abs["rank"] = np.arange(1, len(mean_abs) + 1)

    per_fold = pd.DataFrame({f"fold{s['fold']}": s["shap"].abs().mean() for s in store})
    per_fold = per_fold.loc[mean_abs["feature"]]
    ranks = per_fold.rank(ascending=False)
    s1 = pd.DataFrame({
        "feature": per_fold.index,
        "pooled_mean_abs_shap": mean_abs.set_index("feature").loc[per_fold.index, "mean_abs_shap"].values,
        "fold_mean": per_fold.mean(axis=1).values,
        "fold_sd": per_fold.std(axis=1, ddof=1).values,
        "fold_cv_percent": (per_fold.std(axis=1, ddof=1) / per_fold.mean(axis=1) * 100).values,
        "rank_min": ranks.min(axis=1).values, "rank_max": ranks.max(axis=1).values})
    s1 = pd.concat([s1.reset_index(drop=True), per_fold.reset_index(drop=True)], axis=1)
    s1.to_csv(f"{Config.rep_dir}/tableS1_shap_fold_variance_{label}.csv", index=False)
    mean_abs.to_csv(f"{Config.out_dir}/shap_importance_{label}.csv", index=False)
    return all_shap[used], all_X, mean_abs


def fold_tables(df, folds):
    gc, tg = Config.group_col, Config.target_col
    mem, prof = [], []
    for f, (_, te) in enumerate(folds, start=1):
        d = df.iloc[te]
        for s in sorted(d[gc].unique()):
            mem.append(dict(fold=f, des_name=s, n_rows=int((d[gc] == s).sum())))
        prof.append(dict(
            fold=f, n_systems=d[gc].nunique(), n_rows=len(d),
            median_viscosity_cP=d[tg].median(),
            frac_rows_above_500cP=(d[tg] > 500).mean(),
            mean_ln_viscosity=np.log(d[tg]).mean(),
            mean_weighted_lumo_ev=d["weighted_lumo_energy_ev"].mean(),
            frac_lumo_below_minus6=(d["weighted_lumo_energy_ev"] < -6).mean(),
            mean_hbd_hbd_count=d["hbd_hbd_count"].mean(),
            mean_hbd_hba_count=d["hbd_hba_count"].mean(),
            molar_ratios=json.dumps(sorted(d["hbd_hba_ratio"].unique().tolist()))))
    pd.DataFrame(mem).to_csv(f"{Config.rep_dir}/tableS2_fold_membership.csv", index=False)
    prof = pd.DataFrame(prof)
    prof.to_csv(f"{Config.rep_dir}/tableS3_fold_descriptor_profile.csv", index=False)
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig1_data_landscape(df):
    set_style(); C = Config.C; tg, gc = Config.target_col, Config.group_col
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.38})
    y = df[tg]

    ax = axes[0]
    yl = np.log10(y); xs = np.linspace(yl.min() - 0.3, yl.max() + 0.3, 400)
    k = gaussian_kde(yl, bw_method=0.25)
    ax.fill_between(xs, k(xs), alpha=0.22, color=C["teal"]); ax.plot(xs, k(xs), color=C["teal"], lw=2.2)
    ax.axvline(np.log10(y.median()), color=C["amber"], ls="--", label=f"Median = {y.median():.0f} cP")
    ax.plot(yl, np.full(len(yl), -0.02), "|", color=C["slate"], alpha=0.2, ms=5)
    ax.set_xlabel("log₁₀(viscosity / cP)"); ax.set_ylabel("Probability density")
    ax.set_title(f"(A) Viscosity distribution\n{df[gc].nunique()} systems, {len(df)} measurements")
    ax.legend(loc="upper left")

    # [F2] 20 systems spread across the viscosity range
    ax = axes[1]
    nT = df["temperature_c"].nunique()
    full = df.groupby([gc, "hbd_hba_ratio"]).filter(lambda s: len(s) == nT)
    med = full.groupby([gc, "hbd_hba_ratio"])[tg].median().sort_values()
    pick = med.index[np.unique(np.linspace(0, len(med) - 1, min(20, len(med))).astype(int))]
    ratios = df["hbd_hba_ratio"]
    norm = mcolors.LogNorm(max(ratios.min(), 1e-3), ratios.max())
    cmap = plt.cm.viridis
    for sys_, rat in pick:
        s = full[(full[gc] == sys_) & (full["hbd_hba_ratio"] == rat)].sort_values("temperature_c")
        ax.plot(1000 / (s["temperature_c"] + 273.15), np.log(s[tg]), "-o", ms=2.5,
                color=cmap(norm(rat)), alpha=0.8, lw=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85).set_label("HBD:HBA molar ratio")
    ax.set_xlabel("1000/T (K⁻¹)"); ax.set_ylabel("ln(η / cP)")
    ax.set_title("(B) Arrhenius profiles\n20 systems spanning the viscosity range")

    ax = axes[2]
    temps = sorted(df["temperature_c"].unique())
    vp = ax.violinplot([df.loc[df["temperature_c"] == t, tg] for t in temps],
                       positions=temps, widths=3.5, showmedians=True, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(C["teal"]); b.set_alpha(0.45)
    vp["cmedians"].set_color(C["amber"]); vp["cmedians"].set_linewidth(2.2)
    ax.set_yscale("log"); ax.set_xticks(temps)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Viscosity (cP, log scale)")
    ax.set_title("(C) Viscosity by temperature")
    save_fig(fig, "fig1_data_landscape", "Figure 1 — Viscosity landscape")


def fig2_arrhenius_vs_ml(arr, rf):
    set_style(); C = Config.C
    rec, oof = rf["records"], rf["oof"]
    r2m, r2s = msd(rec["te_r2"]); rlm, rls = msd(rec["te_rmsle"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.35})
    lim = [0.5, 4000]

    ax = axes[0]
    ax.scatter(arr["all_true"], np.clip(arr["all_pred"], 0.1, None), s=14, alpha=0.35,
               color=C["rose"], edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.3); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual viscosity (cP)"); ax.set_ylabel("Predicted viscosity (cP)")
    ax.set_title(f"(A) Arrhenius baseline\nR² = {arr['mean_r2']:.3f} ± {arr['sd_r2']:.3f}   "
                 f"RMSLE = {arr['mean_rmsle']:.3f} ± {arr['sd_rmsle']:.3f}")

    ax = axes[1]
    ax.scatter(oof["actual_cP"], oof["predicted_cP"], s=14, alpha=0.45, color=C["teal"],
               edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.3); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual viscosity (cP)"); ax.set_ylabel("Predicted viscosity (cP)")
    ax.set_title(f"(B) Random Forest, nested CV\nR² = {r2m:.3f} ± {r2s:.3f}   "
                 f"RMSLE = {rlm:.3f} ± {rls:.3f}")

    ax = axes[2]
    f = np.arange(1, Config.outer_folds + 1); w = 0.34
    ax.bar(f - w / 2, arr["fold_r2"], w, color=C["rose"], alpha=0.8, label="Arrhenius baseline")
    ax.bar(f + w / 2, rec["te_r2"], w, color=C["teal"], alpha=0.85, label="Random Forest")
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(f)
    ax.set_xlabel("Outer CV fold"); ax.set_ylabel("Test R²")
    ax.set_title("(C) Per-fold R²")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    save_fig(fig, "fig2_arrhenius_vs_ml", "Figure 2 — Arrhenius baseline vs Random Forest")


def fig3_nested_cv_performance(rf):
    set_style(); C = Config.C
    rec, oof = rf["records"], rf["oof"]
    ot, op = oof["actual_cP"].values, oof["predicted_cP"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    xy = np.vstack([np.log10(ot + 1), np.log10(np.clip(op, 0.01, None) + 1)])
    dens = gaussian_kde(xy)(xy); o = dens.argsort()
    sc = ax.scatter(ot[o], op[o], c=dens[o], cmap="YlOrRd", s=18, alpha=0.85, edgecolors="none")
    fig.colorbar(sc, ax=ax, shrink=0.85).set_label("Local point density")
    ax.plot([0.5, 4000], [0.5, 4000], "k--", lw=1.3); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual viscosity (cP)"); ax.set_ylabel("Predicted viscosity (cP)")
    ax.set_title(f"(A) Pooled out-of-fold parity\nR² = {r2_score(ot, op):.3f}   RMSLE = {rmsle(ot, op):.3f}")

    ax = axes[1]; ax2 = ax.twinx()
    fo = rec["fold"].values
    ax.bar(fo - 0.18, rec["te_r2"], 0.35, color=C["teal"], alpha=0.85, label="R² (left axis)")
    ax2.bar(fo + 0.18, rec["te_rmsle"], 0.35, color=C["amber"], alpha=0.75, label="RMSLE (right axis)")
    ax.axhline(rec["te_r2"].mean(), color=C["teal"], ls="--", lw=1.1)
    ax2.axhline(rec["te_rmsle"].mean(), color=C["amber"], ls="--", lw=1.1)
    ax.set_ylim(0, 1.15); ax2.set_ylim(0, rec["te_rmsle"].max() * 1.6)
    ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_xticks(fo); ax.set_xlabel("Outer CV fold")
    ax.set_ylabel("R²", color=C["teal"]); ax2.set_ylabel("RMSLE", color=C["amber"])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=7.5)
    ax.set_title("(B) Per-fold metrics")

    ax = axes[2]
    res = oof["log_residual"].values
    ax.hist(res, bins=40, color=C["teal"], alpha=0.65, edgecolor="white", density=True)
    mu, sd = res.mean(), res.std(ddof=1)
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=C["amber"], lw=2,
            label=f"Normal fit (μ = {mu:.3f}, σ = {sd:.3f})")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Log residual, ln(1+η$_{true}$) − ln(1+η$_{pred}$)")   # [F3]
    ax.set_ylabel("Density"); ax.set_title("(C) Residual distribution"); ax.legend(loc="upper right")
    save_fig(fig, "fig3_nested_cv_performance", "Figure 3 — Nested CV performance")


def fig4_model_comparison(summary):
    set_style(); C = Config.C
    s = summary.sort_values("R2_mean").reset_index(drop=True)
    col = [C["amber"] if m == "Gradient Boosting" else C["lavender"] if m in TUNED_ENSEMBLES
           else C["teal"] for m in s["Model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, gridspec_kw={"wspace": 0.12})
    yp = np.arange(len(s))
    for ax, met, lab, title in [(axes[0], "R2", "Test R² (mean ± SD over folds)", "(A) R²"),
                                (axes[1], "RMSLE", "Test RMSLE (mean ± SD; lower is better)", "(B) RMSLE")]:
        m, sd = s[f"{met}_mean"], s[f"{met}_sd"]
        ax.errorbar(m, yp, xerr=sd, fmt="none", ecolor=C["slate"], capsize=4, lw=1.2, zorder=3)
        ax.scatter(m, yp, c=col, s=85, zorder=4)
        xmax = (m + sd).max(); span = xmax - (m - sd).min()
        for i in range(len(s)):                       # [F4] labels beyond whiskers
            ax.text(m[i] + sd[i] + 0.03 * span, i, f"{m[i]:.3f}", va="center", fontsize=8)
        ax.set_xlim((m - sd).min() - 0.05 * span, xmax + 0.18 * span)
        if met == "R2":
            ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(lab); ax.set_title(title)
    axes[0].set_yticks(yp); axes[0].set_yticklabels(s["Model"])
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color=C["amber"], label="Gradient Boosting (tuned)"),
                        mp.Patch(color=C["lavender"], label="RF / XGBoost (tuned)"),
                        mp.Patch(color=C["teal"], label="Fixed hyperparameters")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
    save_fig(fig, "fig4_model_comparison", "Figure 4 — Multi-model benchmark")


def fig5_shap_summary(all_shap, all_X, mean_abs):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.1, 0.8]})
    ax = axes[0]
    t = mean_abs.head(15).iloc[::-1]
    bars = ax.barh(range(len(t)), t["mean_abs_shap"], color=[CAT_PAL[c] for c in t["category"]], height=0.7)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t["feature"], fontsize=8.5)
    for b, v in zip(bars, t["mean_abs_shap"]):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, t["mean_abs_shap"].max() * 1.2)
    ax.set_xlabel("Mean |SHAP| (ln-viscosity units)"); ax.set_title("(A) Global importance, top 15")

    ax = axes[1]
    top = mean_abs.head(10)["feature"].values[::-1]
    rng = np.random.RandomState(0)
    for i, f in enumerate(top):
        sv = all_shap[f].values; fv = all_X[f].values; ok = ~np.isnan(sv)
        fvn = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-12)
        sc = ax.scatter(sv[ok], i + rng.uniform(-0.25, 0.25, ok.sum()), c=fvn[ok],
                        cmap="coolwarm", s=8, alpha=0.55, edgecolors="none", vmin=0, vmax=1)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8.5)
    ax.set_xlabel("SHAP value (positive = higher predicted viscosity)")
    ax.set_title("(B) SHAP values, top 10")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value")

    ax = axes[2]   # [F5] shares in legend, not on wedges
    cat = mean_abs.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=False)
    pct = cat / cat.sum() * 100
    ax.pie(cat.values, colors=[CAT_PAL[c] for c in cat.index], startangle=90,
           wedgeprops=dict(width=0.5, edgecolor="white"))
    ax.legend([f"{CAT_LABEL[c]}  {p:.1f}%" for c, p in pct.items()], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), fontsize=8, frameon=False)
    ax.set_title("(C) Share of total SHAP importance"); ax.axis("equal")
    save_fig(fig, "fig5_shap_summary", "Figure 5 — SHAP feature importance")


def fig6_shap_dependence(all_shap, all_X, mean_abs):
    set_style(); C = Config.C
    top4 = mean_abs.head(4)["feature"].values
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"wspace": 0.45})
    for ax, f in zip(axes, top4):
        sv, fv = all_shap[f].values, all_X[f].values
        cf = "inv_temperature" if f != "inv_temperature" else "weighted_lumo_energy_ev"
        sc = ax.scatter(fv, sv, c=all_X[cf].values, cmap="viridis", s=16, alpha=0.65, edgecolors="none")
        o = np.argsort(fv); w = max(10, len(fv) // 20)
        ax.plot(fv[o], pd.Series(sv[o]).rolling(w, center=True, min_periods=1).mean(),
                color=C["amber"], lw=2.4, label="Rolling mean")
        ax.axhline(0, color="black", ls="--", lw=0.9, alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.85).set_label(pretty(cf), fontsize=8)
        ax.set_xlabel(pretty(f)); ax.set_title(f)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("SHAP value")
    save_fig(fig, "fig6_shap_dependence", "Figure 6 — SHAP dependence plots")


def fig7_error_anatomy(oof):
    set_style(); C = Config.C
    ot, op = oof["actual_cP"].values, oof["predicted_cP"].values
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    hx = ax.hexbin(np.log10(ot + 1), np.log10(np.clip(op, 0.01, None) + 1), gridsize=25,
                   cmap="Blues", mincnt=1)
    fig.colorbar(hx, ax=ax, shrink=0.85).set_label("Count")
    ax.plot([-0.1, 3.6], [-0.1, 3.6], "--", color=C["amber"])
    ax.set_xlabel("log₁₀(actual + 1)"); ax.set_ylabel("log₁₀(predicted + 1)")
    ax.set_title(f"(A) Hexbin parity (R² = {r2_score(ot, op):.3f})")

    ax = axes[1]
    dec = pd.Series(pd.qcut(ot, 10, labels=False, duplicates="drop"))
    dv = sorted(dec.unique())
    dr = [rmsle(ot[dec.values == d], op[dec.values == d]) for d in dv]
    dc = [np.median(ot[dec.values == d]) for d in dv]
    ax.plot(dc, dr, "o-", color=C["teal"]); ax.set_xscale("log")
    ax.axhline(np.mean(dr), color=C["amber"], ls="--", label=f"Mean = {np.mean(dr):.3f}")
    ax.set_xlabel("Decile median viscosity (cP)"); ax.set_ylabel("RMSLE")
    ax.set_title("(B) RMSLE by viscosity decile"); ax.legend()

    ax = axes[2]
    (osm, osr), (sl, ic, r) = stats.probplot(oof["log_residual"].values, dist="norm")
    ax.scatter(osm, osr, color=C["teal"], s=14, alpha=0.6)
    ax.plot([osm[0], osm[-1]], [sl * osm[0] + ic, sl * osm[-1] + ic], color=C["amber"],
            label=f"r = {r:.3f}")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title("(C) Q–Q plot, log residuals"); ax.legend()

    ax = axes[3]
    sc = ax.scatter(ot, np.abs(ot - op), c=np.log10(ot + 1), cmap="plasma", s=14, alpha=0.65, edgecolors="none")
    fig.colorbar(sc, ax=ax, shrink=0.85).set_label("log₁₀(actual viscosity)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual viscosity (cP)"); ax.set_ylabel("|Prediction error| (cP)")
    ax.set_title("(D) Absolute error vs actual")
    save_fig(fig, "fig7_error_anatomy", "Figure 7 — Prediction error anatomy")


def fig8_partial_dependence(rf, X, folds, mean_abs, n_grid=60):
    set_style(); C = Config.C
    top4 = mean_abs.head(4)["feature"].values
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"wspace": 0.38})
    for ax, f in zip(axes, top4):
        lo = X[f].quantile(0.02); hi = X[f].quantile(0.98)
        grid = np.linspace(lo, hi, n_grid)
        curves, p10s, p90s = [], [], []
        for pipe, (tr, te) in zip(rf["fitted"], folds):
            Xte = X.iloc[te].copy()
            vals = []
            for v in grid:
                Xte[f] = v
                vals.append(np.expm1(pipe.predict(Xte)).mean())
            curves.append(vals)
            p10s.append(X[f].iloc[tr].quantile(0.10)); p90s.append(X[f].iloc[tr].quantile(0.90))
        cur = np.array(curves); m, s = cur.mean(0), cur.std(0, ddof=1)
        p10, p90 = np.mean(p10s), np.mean(p90s)          # [F7] TRAINING percentiles
        ax.axvspan(p10, p90, color=C["sage"], alpha=0.08, label="10–90th pct. of training data")
        ax.plot(grid, m, color=C["teal"], lw=2.3)
        ax.fill_between(grid, m - s, m + s, color=C["teal"], alpha=0.22, label="±1 SD across folds")
        ax.set_xlabel(pretty(f)); ax.set_title(f)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Mean predicted viscosity (cP)")
    save_fig(fig, "fig8_partial_dependence", "Figure 8 — Partial dependence")


def figS1_rf_vs_gbm(rank_cmp):
    set_style(); C = Config.C
    u = rank_cmp[(rank_cmp.rf_rank <= 10) | (rank_cmp.gbm_rank <= 10)].sort_values("rf_mean_abs_shap")
    rho = spearmanr(rank_cmp.rf_rank, rank_cmp.gbm_rank).correlation
    overlap = len(set(rank_cmp.nsmallest(10, "rf_rank").feature) & set(rank_cmp.nsmallest(10, "gbm_rank").feature))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    y = np.arange(len(u)); h = 0.38
    axes[0].barh(y + h / 2, u.rf_mean_abs_shap, h, color=C["lavender"], label="Random Forest")
    axes[0].barh(y - h / 2, u.gbm_mean_abs_shap, h, color=C["amber"], label="Gradient Boosting")
    axes[0].set_yticks(y); axes[0].set_yticklabels(u.feature); axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean |SHAP| (log axis)"); axes[0].legend(loc="lower right")
    axes[0].set_title("(A) Union of RF and GBM top-10 features")
    ax = axes[1]
    ax.scatter(rank_cmp.rf_rank, rank_cmp.gbm_rank, color=C["teal"], s=40)
    n = len(rank_cmp); ax.plot([0, n + 1], [0, n + 1], "k--", lw=1)
    for _, q in rank_cmp.iterrows():
        if abs(q.rf_rank - q.gbm_rank) >= 4:
            ax.annotate(q.feature, (q.rf_rank, q.gbm_rank), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(n + 1, 0); ax.set_ylim(n + 1, 0)
    ax.set_xlabel("Random Forest rank"); ax.set_ylabel("Gradient Boosting rank")
    ax.set_title(f"(B) Rank agreement: Spearman ρ = {rho:.2f}; top-10 overlap {overlap}/10")
    save_fig(fig, "figS1_shap_rf_vs_gbm")


def figS2_learning_curve(lc):
    set_style(); C = Config.C
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})
    for ax, met, lab in [(axes[0], "r2", "Test R²"), (axes[1], "rmsle", "Test RMSLE")]:
        ax.errorbar(lc["n_train_systems"], lc[f"{met}_mean"], yerr=lc[f"{met}_sd"],
                    fmt="o-", color=C["teal"], capsize=4)
        ax.set_xlabel("Training DES systems per outer fold"); ax.set_ylabel(lab)
    axes[0].set_title("(A) R² vs training-set size"); axes[1].set_title("(B) RMSLE vs training-set size")
    save_fig(fig, "figS2_learning_curve")


def figS3_repeated_cv(per_rep):
    set_style(); C = Config.C
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})
    cols = {"Random Forest": C["lavender"], "Gradient Boosting": C["amber"], "XGBoost": C["teal"]}
    for ax, met, lab in [(axes[0], "te_r2", "Mean test R² per repeat"),
                         (axes[1], "te_rmsle", "Mean test RMSLE per repeat")]:
        for i, m in enumerate(TUNED_ENSEMBLES):
            v = per_rep.loc[per_rep.model == m, met].values
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False)
            ax.scatter(np.full(len(v), i) + np.random.uniform(-0.1, 0.1, len(v)), v,
                       color=cols[m], s=20, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(TUNED_ENSEMBLES); ax.set_ylabel(lab)
    axes[0].set_title(f"(A) R² over {Config.n_repeats} repeated partitions")
    axes[1].set_title(f"(B) RMSLE over {Config.n_repeats} repeated partitions")
    save_fig(fig, "figS3_repeated_cv")


def figS4_data_qc(steps):
    """Diagnostic only (not for the paper): viscosity ratio for each 5 °C step."""
    set_style(); C = Config.C
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = []
    for i, ((a, b), sub) in enumerate(steps.groupby(["T_low", "T_high"])):
        v = sub["step_ratio"].values
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)), v, s=12, alpha=0.6,
                   color=C["rose"] if np.median(v) > 5 or np.median(v) < 0.9 else C["teal"])
        labels.append(f"{a:.0f}→{b:.0f}")
    ax.axhspan(0.9, 5, color=C["sage"], alpha=0.08)
    ax.set_yscale("log"); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("Temperature step (°C)"); ax.set_ylabel("η(T_low) / η(T_high)")
    ax.set_title("Data QC: viscosity ratio per 5 °C step (green = plausible 0.9–5×)")
    save_fig(fig, "figS4_data_qc_diagnostic")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = datetime.now()
    print("=" * 76); print(f"  DES VISCOSITY PIPELINE v3   {t0:%Y-%m-%d %H:%M}   FAST={FAST}"); print("=" * 76)

    print("\n[1] Load, deduplicate, QC")
    df, clean_log = load_and_clean()
    steps, _ = data_qc(df)
    X, y, g = build_xy(df)
    print(f"  Candidate features: {X.shape[1]} (pruned to ~{X.shape[1]-4} inside each fold)")
    print(f"  Viscosity {y.min():.2f}–{y.max():.1f} cP, median {y.median():.1f}, skew {y.skew():.2f}")

    folds = make_outer_folds(X, g, seed=None)
    fold_prof = fold_tables(df, folds)

    print("\n[2] Arrhenius baseline (same outer folds)")
    arr = arrhenius_cv(df, folds)
    print(f"  R² = {arr['mean_r2']:.3f} ± {arr['sd_r2']:.3f}   RMSLE = {arr['mean_rmsle']:.3f} ± {arr['sd_rmsle']:.3f}")

    print("\n[3] Nested CV for all nine models (identical folds)")
    results = {}
    for name, (est, grid) in model_configs().items():
        results[name] = run_nested(name, est, grid, X, y, g, df, folds,
                                   compute_shap=name in ("Random Forest", "Gradient Boosting"),
                                   verbose=name in TUNED_ENSEMBLES)
    summary = summarise(results)
    print("\n" + summary.drop(columns=["R2_folds", "RMSLE_folds"]).round(3).to_string(index=False))
    sig = wilcoxon_tests(summary)

    rf, gbm = results["Random Forest"], results["Gradient Boosting"]
    stab = pd.DataFrame([dict(fold=r.fold, n_features=r.n_features, dropped=r.dropped_features)
                         for _, r in rf["records"].iterrows()])
    print("\n  Pruning stability (RF outer folds):")
    print(stab.to_string(index=False))

    print("\n[4] SHAP (RF primary, GBM cross-check)")
    rf_shap, rf_X, rf_imp = aggregate_shap(rf["shap"], "rf")
    gb_shap, gb_X, gb_imp = aggregate_shap(gbm["shap"], "gbm")
    rank_cmp = (rf_imp[["feature", "mean_abs_shap", "rank"]]
                .rename(columns={"mean_abs_shap": "rf_mean_abs_shap", "rank": "rf_rank"})
                .merge(gb_imp[["feature", "mean_abs_shap", "rank"]]
                       .rename(columns={"mean_abs_shap": "gbm_mean_abs_shap", "rank": "gbm_rank"}),
                       on="feature", how="outer"))
    rank_cmp.to_csv(f"{Config.out_dir}/shap_rf_vs_gbm_rank_comparison.csv", index=False)
    rho = spearmanr(rank_cmp.rf_rank, rank_cmp.gbm_rank, nan_policy="omit").correlation
    print(rf_imp.head(10).round(4).to_string(index=False))
    print(f"  RF vs GBM Spearman ρ = {rho:.3f}")

    lc = per_rep = stab_rep = tt = None
    if Config.run_learning_curve:
        print("\n[5] Learning curve")
        lc = learning_curve(X, y, g, folds, rf["records"])
    if Config.run_repeated_cv:
        print("\n[6] Repeated system-disjoint CV")
        rep, per_rep, stab_rep, tt = repeated_cv(X, y, g, df, results)

    print("\n[7] Figures")
    fig1_data_landscape(df)
    fig2_arrhenius_vs_ml(arr, rf)
    fig3_nested_cv_performance(rf)
    fig4_model_comparison(summary)
    fig5_shap_summary(rf_shap, rf_X, rf_imp)
    fig6_shap_dependence(rf_shap, rf_X, rf_imp)
    fig7_error_anatomy(rf["oof"])
    fig8_partial_dependence(rf, X, folds, rf_imp)
    figS1_rf_vs_gbm(rank_cmp)
    if lc is not None: figS2_learning_curve(lc)
    if per_rep is not None: figS3_repeated_cv(per_rep)
    figS4_data_qc(steps)

    print("\n[8] Saving outputs")
    df.to_csv(f"{Config.out_dir}/viscosity_dataset_CLEANED.csv", index=False)
    summary.drop(columns=["R2_folds", "RMSLE_folds"]).to_csv(
        f"{Config.out_dir}/model_comparison_nested_cv.csv", index=False)
    summary[["Model", "R2_folds", "RMSLE_folds"]].to_json(
        f"{Config.out_dir}/model_comparison_fold_scores.json", orient="records", indent=2)
    pd.concat([r["records"] for r in results.values()]).to_csv(
        f"{Config.out_dir}/fold_metrics_all_models.csv", index=False)
    rf["records"].to_csv(f"{Config.out_dir}/nested_cv_fold_metrics_rf.csv", index=False)
    gbm["records"].to_csv(f"{Config.out_dir}/nested_cv_fold_metrics_gbm.csv", index=False)
    rf["oof"].to_csv(f"{Config.out_dir}/oof_predictions_rf.csv", index=False)
    gbm["oof"].to_csv(f"{Config.out_dir}/oof_predictions_gbm.csv", index=False)
    sig.to_csv(f"{Config.out_dir}/significance_tests.csv", index=False)
    pd.DataFrame({"fold": np.arange(1, 6), "r2": arr["fold_r2"], "rmsle": arr["fold_rmsle"]}).to_csv(
        f"{Config.out_dir}/arrhenius_fold_metrics.csv", index=False)
    sysres = (rf["oof"].groupby("des_name")["log_residual"]
              .agg(rms_log_residual=lambda r: np.sqrt((r ** 2).mean()), mean_log_residual="mean", n="size")
              .sort_values("rms_log_residual", ascending=False))
    sysres.to_csv(f"{Config.out_dir}/per_system_errors_rf.csv")
    joblib.dump(rf["fitted"], f"{Config.model_dir}/rf_outer_fold_pipelines.pkl")
    joblib.dump(gbm["fitted"], f"{Config.model_dir}/gbm_outer_fold_pipelines.pkl")

    oof_m = metrics_dict(rf["oof"]["actual_cP"], rf["oof"]["predicted_cP"])
    rfr = rf["records"]
    summ = dict(
        timestamp=datetime.now().isoformat(), runtime_min=(datetime.now() - t0).seconds / 60,
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                      xgboost=xgboost.__version__, shap=shap.__version__),
        data_cleaning=clean_log, drop_temperatures=Config.drop_temperatures,
        value_corrections={f"{k[0]}@{k[1]}": v for k, v in Config.value_corrections.items()},
        n_candidate_features=int(X.shape[1]),
        n_features_per_fold=rfr["n_features"].tolist(),
        sd_convention="sample SD (ddof=1) across outer folds",
        tuning_scoring=Config.tuning_scoring,
        rf=dict(oof=oof_m,
                fold_r2=msd(rfr["te_r2"]), fold_rmsle=msd(rfr["te_rmsle"]),
                fold_rmse=msd(rfr["te_rmse"]), fold_mae=msd(rfr["te_mae"])),
        arrhenius=dict(r2=[arr["mean_r2"], arr["sd_r2"]], rmsle=[arr["mean_rmsle"], arr["sd_rmsle"]]),
        delta_r2_rf_vs_arrhenius=msd(rfr["te_r2"])[0] - arr["mean_r2"],
        delta_rmsle_rf_vs_arrhenius=msd(rfr["te_rmsle"])[0] - arr["mean_rmsle"],
        rf_top10_shap=rf_imp.head(10)["feature"].tolist(),
        gbm_top10_shap=gb_imp.head(10)["feature"].tolist(),
        rf_gbm_shap_spearman=rho,
        worst_systems_rf=sysres.head(8).round(3).reset_index().to_dict("records"))
    with open(f"{Config.rep_dir}/model_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2, default=str)
    print(f"  Outputs → {Config.out_dir}/   (runtime {summ['runtime_min']:.1f} min)")
    print("  Done.")


if __name__ == "__main__":
    main()
