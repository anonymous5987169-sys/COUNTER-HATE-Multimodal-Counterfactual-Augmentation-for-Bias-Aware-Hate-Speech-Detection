"""
enhanced_statistical_tests.py
==============================
Comprehensive statistical tests for group-level FPR disparities across
identity groups for text, image, and fusion modalities.

Tests
-----
1. Per-group FPR computation (binary FP indicator on non-hate samples)
2. Chi-squared test of proportions across groups (PRIMARY omnibus test)
3. Kruskal-Wallis H-test across identity groups (secondary robustness check)
4. Pairwise Mann-Whitney U tests with Holm-Bonferroni & BH-FDR correction
5. Fisher's exact test for small-group pairwise comparisons
6. Clopper-Pearson exact binomial CIs for per-group FPR
7. Logistic regression: FP ~ C(group) * C(condition) (PRIMARY),
   OLS linear probability model (secondary)
8. Cohen's d effect size for group FPR differences
9. Cochran's Q test for paired multi-model comparison
10. DFPR logistic regression per modality

Outputs
-------
  analysis/results/enhanced_statistical_tests.json
  analysis/results/plots/group_fpr_comparison_boxplot.png
  analysis/results/plots/pairwise_significance_heatmap.png
  analysis/results/plots/clopper_pearson_ci_plot.png
"""

import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# -- Check optional deps -------------------------------------------------------
HAS_STATSMODELS = False
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols as sm_ols
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    pass

HAS_MULTIPLETESTS = False
try:
    from statsmodels.stats.multitest import multipletests as sm_multipletests
    HAS_MULTIPLETESTS = True
except ImportError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="covariance of constraints")
warnings.filterwarnings("ignore", message="Maximum Likelihood")

# -- Paths ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_CF_CSV = PROJECT_ROOT / "text_models" / "enhanced_results" / "predictions" / "hatebert_mlp_cf_predictions.csv"
TEXT_NCF_CSV = PROJECT_ROOT / "text_models" / "enhanced_results" / "predictions" / "hatebert_mlp_ncf_predictions.csv"
IMAGE_CF_CSV = PROJECT_ROOT / "image_models" / "results" / "predictions" / "efficientnet_grl_cf_predictions.csv"
PRED_DIR = PROJECT_ROOT / "cross_modal" / "results" / "predictions"
FUSION_CSV = PRED_DIR / "fusion_test_predictions.csv"

TARGET_GROUPS = [
    "race/ethnicity", "religion", "gender", "sexual_orientation",
    "national_origin/citizenship", "disability", "age", "multiple/none",
]

ALPHA = 0.05

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ==============================================================================
#  DATA LOADING
# ==============================================================================
def load_text_predictions():
    """Load text CF and nCF predictions, add FP indicator for non-hate samples."""
    cf = pd.read_csv(TEXT_CF_CSV)
    ncf = pd.read_csv(TEXT_NCF_CSV)
    cf["condition"] = "CF"
    ncf["condition"] = "nCF"
    for df in (cf, ncf):
        df["is_nonhate"] = (df["true_label"] == 0).astype(int)
        df["fp_indicator"] = ((df["true_label"] == 0) & (df["pred_label"] == 1)).astype(int)
    return cf, ncf


def load_image_predictions():
    """Load image predictions (test split only), add FP indicator."""
    df = pd.read_csv(IMAGE_CF_CSV)
    df = df[df["split"] == "test"].copy()
    df["is_nonhate"] = (df["true_label"] == 0).astype(int)
    df["fp_indicator"] = ((df["true_label"] == 0) & (df["pred_label"] == 1)).astype(int)
    df["condition"] = "CF"
    return df


def load_fusion_predictions():
    """Load fusion predictions, derive FP indicator using equal fusion threshold=0.5."""
    fusion_candidates = [
        FUSION_CSV,
        PRED_DIR / "fusion_test_predictions_cf_no_adv.csv",
        PRED_DIR / "fusion_test_predictions_cf.csv",
        PRED_DIR / "fusion_test_predictions_ncf.csv",
    ]
    fusion_path = next((p for p in fusion_candidates if p.exists()), None)
    if fusion_path is None:
        raise FileNotFoundError(
            "No fusion prediction CSV found. Expected one of: "
            + ", ".join(str(p) for p in fusion_candidates)
        )

    df = pd.read_csv(fusion_path)
    df.rename(columns={"target_group": "group_label"}, inplace=True)
    df["true_label"] = df["y_true"]
    df["pred_label"] = (df["p_equal_fusion"] >= 0.5).astype(int)
    df["is_nonhate"] = (df["true_label"] == 0).astype(int)
    df["fp_indicator"] = ((df["true_label"] == 0) & (df["pred_label"] == 1)).astype(int)
    df["condition"] = "fusion"
    return df


# ==============================================================================
#  FPR COMPUTATION
# ==============================================================================
def compute_group_fpr(df, group_col="group_label"):
    """Return dict {group: FPR} for non-hate samples only."""
    nonhate = df[df["true_label"] == 0]
    fpr = {}
    for grp, sub in nonhate.groupby(group_col):
        if grp not in TARGET_GROUPS:
            continue
        n = len(sub)
        if n == 0:
            continue
        fpr[grp] = sub["fp_indicator"].sum() / n
    return fpr


def get_fp_vectors(df, group_col="group_label"):
    """Return dict {group: array of per-sample FP indicators} for non-hate."""
    nonhate = df[df["true_label"] == 0]
    vectors = {}
    for grp, sub in nonhate.groupby(group_col):
        if grp not in TARGET_GROUPS:
            continue
        if len(sub) == 0:
            continue
        vectors[grp] = sub["fp_indicator"].values
    return vectors


# ==============================================================================
#  CHI-SQUARED TEST OF PROPORTIONS (PRIMARY OMNIBUS TEST)
# ==============================================================================
def chi2_proportions_test(fp_vectors):
    """Chi-squared test of proportions on a contingency table of group x FP/TN.

    This is the standard test for comparing proportions (here FPR) across
    multiple groups.  Constructs a k x 2 contingency table (FP count, TN count)
    and applies scipy.stats.chi2_contingency.
    """
    groups = sorted(fp_vectors.keys())
    if len(groups) < 2:
        return {"chi2": None, "p_value": None, "n_groups": len(groups),
                "note": "Need at least 2 groups"}

    table = []
    group_info = {}
    for g in groups:
        arr = fp_vectors[g]
        fp_count = int(arr.sum())
        tn_count = int(len(arr) - fp_count)
        table.append([fp_count, tn_count])
        group_info[g] = {"n": len(arr), "fp": fp_count, "tn": tn_count,
                         "fpr": round(fp_count / len(arr), 6) if len(arr) > 0 else 0.0}
    table = np.array(table)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, dof, expected = stats.chi2_contingency(table)

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "n_groups": len(groups),
        "significant": bool(p < ALPHA),
        "contingency_table": {g: group_info[g] for g in groups},
        "expected_counts": {g: {"fp": round(float(expected[i, 0]), 2),
                                "tn": round(float(expected[i, 1]), 2)}
                           for i, g in enumerate(groups)},
    }


# ==============================================================================
#  KRUSKAL-WALLIS H-TEST (secondary robustness check)
# ==============================================================================
def kruskal_wallis_test(fp_vectors):
    """Kruskal-Wallis H across groups on binary FP indicator.

    Retained as a secondary non-parametric robustness check alongside the
    primary chi-squared test of proportions.
    """
    groups = sorted(fp_vectors.keys())
    arrays = [fp_vectors[g] for g in groups]
    if len(arrays) < 2:
        return {"H_statistic": None, "p_value": None, "groups": groups}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H, p = stats.kruskal(*arrays)
    return {"H_statistic": float(H), "p_value": float(p), "n_groups": len(groups),
            "significant": bool(p < ALPHA)}


# ==============================================================================
#  MULTIPLE-TESTING CORRECTION HELPERS
# ==============================================================================
def _holm_bonferroni_manual(p_values):
    """Manual Holm-Bonferroni step-down correction."""
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    corrected = np.empty(m)
    for i in range(m):
        corrected[i] = sorted_p[i] * (m - i)
    for i in range(1, m):
        if corrected[i] < corrected[i - 1]:
            corrected[i] = corrected[i - 1]
    corrected = np.minimum(corrected, 1.0)
    result = np.empty(m)
    result[order] = corrected
    return result


def _bh_fdr_manual(p_values):
    """Manual Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    corrected = np.empty(m)
    for i in range(m):
        corrected[i] = sorted_p[i] * m / (i + 1)
    for i in range(m - 2, -1, -1):
        if corrected[i] > corrected[i + 1]:
            corrected[i] = corrected[i + 1]
    corrected = np.minimum(corrected, 1.0)
    result = np.empty(m)
    result[order] = corrected
    return result


def correct_pvalues(raw_p_values, method="holm"):
    """Apply multiple-testing correction (holm or fdr_bh)."""
    raw = np.asarray(raw_p_values, dtype=float)
    if len(raw) == 0:
        return raw
    if HAS_MULTIPLETESTS:
        _, corrected, _, _ = sm_multipletests(raw, alpha=ALPHA, method=method)
        return corrected
    if method == "holm":
        return _holm_bonferroni_manual(raw)
    elif method == "fdr_bh":
        return _bh_fdr_manual(raw)
    else:
        raise ValueError(f"Unsupported method: {method}")


# ==============================================================================
#  PAIRWISE MANN-WHITNEY U WITH HOLM-BONFERRONI & BH-FDR
# ==============================================================================
def pairwise_mann_whitney(fp_vectors):
    """Pairwise Mann-Whitney U with Holm-Bonferroni and BH-FDR corrections."""
    groups = sorted(fp_vectors.keys())
    pairs = list(combinations(groups, 2))
    n_comparisons = len(pairs)

    raw_pvals = []
    pair_info = []
    for g1, g2 in pairs:
        a, b = fp_vectors[g1], fp_vectors[g2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                U, p = np.nan, 1.0
        raw_pvals.append(p)
        pair_info.append({
            "group_1": g1, "group_2": g2,
            "U_statistic": float(U) if not np.isnan(U) else None,
            "p_value_raw": float(p),
            "n_1": len(a), "n_2": len(b),
        })

    p_holm = correct_pvalues(raw_pvals, method="holm")
    p_bh = correct_pvalues(raw_pvals, method="fdr_bh")

    results = []
    for i, info in enumerate(pair_info):
        info["p_value_holm"] = float(p_holm[i])
        info["p_value_bh_fdr"] = float(p_bh[i])
        info["significant_holm"] = bool(p_holm[i] < ALPHA)
        info["significant_bh_fdr"] = bool(p_bh[i] < ALPHA)
        # Legacy keys for backward compatibility
        info["p_value_bonferroni"] = info["p_value_holm"]
        info["significant_bonferroni"] = info["significant_holm"]
        results.append(info)

    return results, n_comparisons


# ==============================================================================
#  FISHER'S EXACT TEST FOR SMALL GROUPS
# ==============================================================================
def fishers_exact_per_group(fp_vectors):
    """Fisher's exact test for each pair of groups.

    Especially useful for groups with small n (disability n~23, age n~12)
    where chi-squared approximation may be unreliable.
    """
    groups = sorted(fp_vectors.keys())
    pairs = list(combinations(groups, 2))
    n_comparisons = len(pairs)

    raw_pvals = []
    pair_results = []
    for g1, g2 in pairs:
        a, b = fp_vectors[g1], fp_vectors[g2]
        fp1, tn1 = int(a.sum()), int(len(a) - a.sum())
        fp2, tn2 = int(b.sum()), int(len(b) - b.sum())
        table = np.array([[fp1, tn1], [fp2, tn2]])
        try:
            odds_ratio, p = stats.fisher_exact(table, alternative="two-sided")
        except Exception:
            odds_ratio, p = np.nan, 1.0
        raw_pvals.append(p)
        pair_results.append({
            "group_1": g1, "group_2": g2,
            "table": [[fp1, tn1], [fp2, tn2]],
            "odds_ratio": float(odds_ratio) if not np.isnan(odds_ratio) else None,
            "p_value_raw": float(p),
            "n_1": len(a), "n_2": len(b),
        })

    p_holm = correct_pvalues(raw_pvals, method="holm")
    p_bh = correct_pvalues(raw_pvals, method="fdr_bh")

    for i, r in enumerate(pair_results):
        r["p_value_holm"] = float(p_holm[i])
        r["p_value_bh_fdr"] = float(p_bh[i])
        r["significant_holm"] = bool(p_holm[i] < ALPHA)
        r["significant_bh_fdr"] = bool(p_bh[i] < ALPHA)

    return {"n_comparisons": n_comparisons, "pairs": pair_results}


# ==============================================================================
#  CLOPPER-PEARSON EXACT BINOMIAL CONFIDENCE INTERVALS
# ==============================================================================
def clopper_pearson_ci(successes, n, alpha=0.05):
    """Compute Clopper-Pearson exact binomial CI for a proportion."""
    if n == 0:
        return {"lower": 0.0, "upper": 1.0, "point_estimate": 0.0}
    point = successes / n
    lower = stats.beta.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return {"lower": float(lower), "upper": float(upper), "point_estimate": float(point)}


def clopper_pearson_all_groups(fp_vectors, alpha=0.05):
    """Compute Clopper-Pearson exact binomial CIs for FPR of every group."""
    result = {}
    for grp in sorted(fp_vectors.keys()):
        arr = fp_vectors[grp]
        n = len(arr)
        fp = int(arr.sum())
        ci = clopper_pearson_ci(fp, n, alpha=alpha)
        ci["n"] = n
        ci["fp_count"] = fp
        result[grp] = ci
    return result


# ==============================================================================
#  COHEN'S D EFFECT SIZE
# ==============================================================================
def cohens_d(a, b):
    """Pooled-variance Cohen's d."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def effect_sizes(fp_vectors):
    """Cohen's d for all group pairs."""
    groups = sorted(fp_vectors.keys())
    results = {}
    for g1, g2 in combinations(groups, 2):
        d = cohens_d(fp_vectors[g1], fp_vectors[g2])
        results[f"{g1} vs {g2}"] = round(d, 4)
    return results


# ==============================================================================
#  REGRESSION: LOGISTIC (PRIMARY) + OLS/LPM (SECONDARY)
# ==============================================================================
def regression_analysis(cf_df, ncf_df):
    """Logistic regression (PRIMARY) and OLS linear probability model (secondary)
    for FP_indicator ~ C(group_label) + C(condition) + C(group_label):C(condition).
    """
    combined = pd.concat([
        cf_df[cf_df["true_label"] == 0][["fp_indicator", "group_label", "condition"]],
        ncf_df[ncf_df["true_label"] == 0][["fp_indicator", "group_label", "condition"]],
    ], ignore_index=True)
    combined = combined[combined["group_label"].isin(TARGET_GROUPS)].copy()

    result = {"n_samples": len(combined)}

    if HAS_STATSMODELS:
        # PRIMARY: Logistic regression
        try:
            import statsmodels.formula.api as smf
            logit_model = smf.logit(
                "fp_indicator ~ C(group_label) + C(condition) + C(group_label):C(condition)",
                data=combined
            ).fit(disp=0, maxiter=300)

            logit_result = {
                "method": "logistic_regression",
                "pseudo_r_squared": float(logit_model.prsquared),
                "llr_p_value": float(logit_model.llr_pvalue),
                "aic": float(logit_model.aic),
                "bic": float(logit_model.bic),
            }
            coefs = {}
            for name, val in logit_model.params.items():
                pval = logit_model.pvalues[name]
                coefs[name] = {
                    "coefficient": round(float(val), 4),
                    "odds_ratio": round(float(np.exp(val)), 4),
                    "p_value": round(float(pval), 6),
                    "significant": bool(pval < ALPHA),
                }
            logit_result["coefficients"] = coefs
            interaction_coefs = {k: v for k, v in coefs.items() if ":" in k}
            if interaction_coefs:
                interaction_pvals = [v["p_value"] for v in interaction_coefs.values()]
                any_sig = any(v["significant"] for v in interaction_coefs.values())
                logit_result["interaction_any_significant"] = any_sig
                logit_result["interaction_min_p_value"] = min(interaction_pvals)
            logit_result["significant_overall"] = bool(logit_model.llr_pvalue < ALPHA)
            result["logistic_regression"] = logit_result
        except Exception as e:
            result["logistic_regression"] = {"method": "logistic_regression", "error": str(e)}

        # SECONDARY: OLS linear probability model
        try:
            ols_model = sm_ols(
                "fp_indicator ~ C(group_label) + C(condition) + C(group_label):C(condition)",
                data=combined
            ).fit()
            anova_table = anova_lm(ols_model, typ=2)
            ols_result = {
                "method": "OLS_linear_probability_model",
                "r_squared": float(ols_model.rsquared),
                "r_squared_adj": float(ols_model.rsquared_adj),
                "anova_table": {},
            }
            for idx_name in anova_table.index:
                row = anova_table.loc[idx_name]
                ols_result["anova_table"][idx_name] = {
                    "sum_sq": float(row.get("sum_sq", 0)),
                    "df": float(row.get("df", 0)),
                    "F": float(row.get("F", 0)) if not pd.isna(row.get("F", np.nan)) else None,
                    "PR(>F)": float(row.get("PR(>F)", 1)) if not pd.isna(row.get("PR(>F)", np.nan)) else None,
                }
            interaction_key = [k for k in ols_result["anova_table"] if ":" in k]
            if interaction_key:
                p_int = ols_result["anova_table"][interaction_key[0]].get("PR(>F)")
                ols_result["interaction_significant"] = bool(p_int is not None and p_int < ALPHA)
                ols_result["interaction_p_value"] = p_int
            result["ols_linear_probability_model"] = ols_result
        except Exception as e:
            result["ols_linear_probability_model"] = {"method": "OLS_linear_probability_model", "error": str(e)}
    else:
        result["logistic_regression"] = {"method": "logistic_regression", "error": "statsmodels not available"}
        from sklearn.linear_model import LinearRegression
        X_combined = pd.get_dummies(combined[["group_label", "condition"]], drop_first=True)
        grp_cols = [c for c in X_combined.columns if c.startswith("group_label_")]
        cond_cols = [c for c in X_combined.columns if c.startswith("condition_")]
        for gc in grp_cols:
            for cc in cond_cols:
                X_combined[f"{gc}:{cc}"] = X_combined[gc] * X_combined[cc]
        y = combined["fp_indicator"].values
        X_full = X_combined.values
        reg_full = LinearRegression().fit(X_full, y)
        ss_res_full = np.sum((y - reg_full.predict(X_full)) ** 2)
        interaction_cols = [c for c in X_combined.columns if ":" in c]
        X_reduced = X_combined.drop(columns=interaction_cols).values
        reg_red = LinearRegression().fit(X_reduced, y)
        ss_res_red = np.sum((y - reg_red.predict(X_reduced)) ** 2)
        df_num = len(interaction_cols)
        df_den = len(y) - X_full.shape[1] - 1
        ols_result = {"method": "scipy_fallback_OLS"}
        if df_den > 0 and ss_res_full > 0:
            F_stat = ((ss_res_red - ss_res_full) / df_num) / (ss_res_full / df_den)
            p_int = 1 - stats.f.cdf(F_stat, df_num, df_den)
            ols_result["F_interaction"] = float(F_stat)
            ols_result["interaction_p_value"] = float(p_int)
            ols_result["interaction_significant"] = bool(p_int < ALPHA)
        else:
            ols_result["F_interaction"] = None
            ols_result["interaction_p_value"] = None
        result["ols_linear_probability_model"] = ols_result

    return result


# Backward compatibility alias
def ols_regression_analysis(cf_df, ncf_df):
    """Backward-compatible wrapper -- delegates to regression_analysis."""
    return regression_analysis(cf_df, ncf_df)


# ==============================================================================
#  COCHRAN'S Q TEST
# ==============================================================================
def cochrans_q_test(cf_df, ncf_df):
    """Cochran's Q on paired FP indicators for CF vs nCF text."""
    cf_nh = cf_df[cf_df["true_label"] == 0][["sample_id", "fp_indicator"]].copy()
    ncf_nh = ncf_df[ncf_df["true_label"] == 0][["sample_id", "fp_indicator"]].copy()
    cf_nh.rename(columns={"fp_indicator": "fp_cf"}, inplace=True)
    ncf_nh.rename(columns={"fp_indicator": "fp_ncf"}, inplace=True)

    merged = cf_nh.merge(ncf_nh, on="sample_id", how="inner")
    if len(merged) < 5:
        return {"Q_statistic": None, "p_value": None, "n_paired": len(merged),
                "note": "Too few paired samples"}

    fp_matrix = merged[["fp_cf", "fp_ncf"]].values
    k = fp_matrix.shape[1]
    n = fp_matrix.shape[0]
    T_j = fp_matrix.sum(axis=0)
    T_i = fp_matrix.sum(axis=1)
    N = T_j.sum()
    T_j_sq_sum = np.sum(T_j ** 2)
    T_i_sq_sum = np.sum(T_i ** 2)
    numerator = (k - 1) * (k * T_j_sq_sum - N ** 2)
    denominator = k * N - T_i_sq_sum
    if denominator == 0:
        return {"Q_statistic": 0.0, "p_value": 1.0, "n_paired": n}
    Q = numerator / denominator
    p_value = 1 - stats.chi2.cdf(Q, df=k - 1)
    return {
        "Q_statistic": float(Q), "p_value": float(p_value),
        "n_paired": int(n), "significant": bool(p_value < ALPHA),
        "fp_rate_cf": float(T_j[0] / n), "fp_rate_ncf": float(T_j[1] / n),
    }


# ==============================================================================
#  DFPR REGRESSION
# ==============================================================================
def dfpr_regression(fp_vectors_dict, modality_name=""):
    """Formal DFPR analysis: FP ~ group using logistic regression."""
    rows = []
    for grp, arr in fp_vectors_dict.items():
        for val in arr:
            rows.append({"group_label": grp, "fp_indicator": int(val)})
    df = pd.DataFrame(rows)
    if len(df) < 10:
        return {"error": "Too few samples for DFPR regression"}

    result = {"modality": modality_name}

    if HAS_STATSMODELS:
        try:
            import statsmodels.formula.api as smf
            model = smf.logit("fp_indicator ~ C(group_label)", data=df).fit(disp=0, maxiter=200)
            result["method"] = "logistic_regression"
            result["pseudo_r_squared"] = float(model.prsquared)
            result["llr_p_value"] = float(model.llr_pvalue)
            result["significant_overall"] = bool(model.llr_pvalue < ALPHA)
            coefs = {}
            for name, val in model.params.items():
                pval = model.pvalues[name]
                coefs[name] = {
                    "coefficient": round(float(val), 4),
                    "p_value": round(float(pval), 6),
                    "significant": bool(pval < ALPHA),
                }
            result["coefficients"] = coefs
        except Exception as e:
            try:
                contingency = pd.crosstab(df["group_label"], df["fp_indicator"])
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                result["method"] = "chi2_fallback_after_logit_error"
                result["logit_error"] = str(e)
                result["chi2"] = float(chi2)
                result["p_value"] = float(p)
                result["dof"] = int(dof)
                result["significant_overall"] = bool(p < ALPHA)
            except Exception as e2:
                result["error"] = f"Logit: {e}; Chi2: {e2}"
                result["method"] = "failed"
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["group_encoded"] = le.fit_transform(df["group_label"])
        contingency = pd.crosstab(df["group_label"], df["fp_indicator"])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        result["method"] = "chi2_contingency"
        result["chi2"] = float(chi2)
        result["p_value"] = float(p)
        result["dof"] = int(dof)
        result["significant"] = bool(p < ALPHA)

    return result


# ==============================================================================
#  PLOTTING
# ==============================================================================
SHORT_NAMES = {
    "national_origin/citizenship": "nat_origin",
    "sexual_orientation": "sex_orient",
    "race/ethnicity": "race/eth",
    "multiple/none": "mult/none",
}


def plot_group_fpr_comparison(text_fpr, image_fpr, fusion_fpr, outpath):
    """Grouped bar plot of FPR per group for text, image, fusion."""
    groups = sorted(set(list(text_fpr) + list(image_fpr) + list(fusion_fpr)) & set(TARGET_GROUPS))
    labels = [SHORT_NAMES.get(g, g) for g in groups]
    modalities = [("Text (CF)", text_fpr), ("Image", image_fpr), ("Fusion", fusion_fpr)]
    n_groups = len(groups)
    if n_groups == 0:
        print("  [WARN] No FPR data for comparison plot.")
        return
    x = np.arange(n_groups)
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (mname, fpr_dict) in enumerate(modalities):
        vals = [fpr_dict.get(g, 0.0) for g in groups]
        ax.bar(x + idx * width, vals, width, label=mname, color=colors[idx],
               edgecolor="black", linewidth=0.6)
    ax.set_title("Per-Group False Positive Rate by Modality", fontsize=14, fontweight="bold")
    ax.set_xlabel("Identity Group", fontsize=12)
    ax.set_ylabel("False Positive Rate", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    max_fpr = max(max(d.values()) for d in [text_fpr, image_fpr, fusion_fpr] if d)
    ax.set_ylim(0, min(1.0, max_fpr * 1.3 + 0.02))
    ax.legend(title="Modality", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_pairwise_heatmap(pairwise_results, modality_name, outpath):
    """Heatmap of pairwise Mann-Whitney p-values (Holm-corrected)."""
    groups = sorted(set(
        [r["group_1"] for r in pairwise_results] + [r["group_2"] for r in pairwise_results]
    ))
    short_groups = [SHORT_NAMES.get(g, g) for g in groups]
    n = len(groups)
    matrix = np.ones((n, n))
    for r in pairwise_results:
        i = groups.index(r["group_1"])
        j = groups.index(r["group_2"])
        matrix[i, j] = r["p_value_holm"]
        matrix[j, i] = r["p_value_holm"]
    np.fill_diagonal(matrix, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Holm-Bonferroni p-value", shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_groups, rotation=30, ha="right")
    ax.set_yticklabels(short_groups)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "---", ha="center", va="center", fontsize=9, color="gray")
            else:
                val = matrix[i, j]
                color = "white" if val < 0.15 else "black"
                txt = f"{val:.3f}"
                if val < ALPHA:
                    txt += " *"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color,
                        fontweight="bold" if val < ALPHA else "normal")
    ax.set_title(f"Pairwise Mann-Whitney U p-values ({modality_name})\n"
                 f"* = significant after Holm-Bonferroni correction",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_clopper_pearson_ci(ci_dict_by_modality, outpath):
    """Forest plot of Clopper-Pearson exact CIs for per-group FPR."""
    modality_names = list(ci_dict_by_modality.keys())
    n_mods = len(modality_names)
    if n_mods == 0:
        return
    all_groups = sorted(set().union(*(ci_dict_by_modality[m].keys() for m in modality_names)))
    n_groups = len(all_groups)
    if n_groups == 0:
        return
    colors = {"text_cf": "#4C72B0", "text_ncf": "#6699CC",
              "image": "#DD8452", "fusion": "#55A868"}
    fig, ax = plt.subplots(figsize=(12, max(6, n_groups * 0.8)))
    y_positions = np.arange(n_groups)
    offset_step = 0.18
    offsets = np.linspace(-offset_step * (n_mods - 1) / 2, offset_step * (n_mods - 1) / 2, n_mods)
    for m_idx, mod_name in enumerate(modality_names):
        ci_data = ci_dict_by_modality[mod_name]
        for g_idx, grp in enumerate(all_groups):
            if grp not in ci_data:
                continue
            ci = ci_data[grp]
            y = y_positions[g_idx] + offsets[m_idx]
            c = colors.get(mod_name, "#999999")
            ax.errorbar(ci["point_estimate"], y,
                        xerr=[[ci["point_estimate"] - ci["lower"]],
                              [ci["upper"] - ci["point_estimate"]]],
                        fmt="o", color=c, markersize=5, capsize=3, linewidth=1.2,
                        label=mod_name if g_idx == 0 else None)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SHORT_NAMES.get(g, g) for g in all_groups])
    ax.set_xlabel("FPR (with 95% Clopper-Pearson CI)", fontsize=11)
    ax.set_title("Per-Group FPR with Exact Binomial Confidence Intervals",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Modality", loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================
def run_all():
    print("=" * 80)
    print("ENHANCED STATISTICAL TESTS: Group-Level FPR Disparity Analysis")
    print("=" * 80)

    all_results = {}

    # -- Load data --
    print("\n[1] Loading predictions...")
    text_cf, text_ncf = load_text_predictions()
    image_df = load_image_predictions()
    fusion_df = load_fusion_predictions()
    print(f"    Text CF:  {len(text_cf)} samples ({(text_cf['true_label']==0).sum()} non-hate)")
    print(f"    Text nCF: {len(text_ncf)} samples ({(text_ncf['true_label']==0).sum()} non-hate)")
    print(f"    Image:    {len(image_df)} samples ({(image_df['true_label']==0).sum()} non-hate)")
    print(f"    Fusion:   {len(fusion_df)} samples ({(fusion_df['true_label']==0).sum()} non-hate)")

    # -- Per-group FPR --
    print("\n[2] Computing per-group FPR...")
    modalities = {
        "text_cf": text_cf, "text_ncf": text_ncf,
        "image": image_df, "fusion": fusion_df,
    }
    fpr_results = {}
    fp_vecs = {}
    for name, df in modalities.items():
        fpr_results[name] = compute_group_fpr(df)
        fp_vecs[name] = get_fp_vectors(df)
        print(f"    {name:10s}: " + ", ".join(
            f"{g[:12]:12s}={v:.3f}" for g, v in sorted(fpr_results[name].items())))
    all_results["per_group_fpr"] = {k: {g: round(v, 6) for g, v in d.items()}
                                     for k, d in fpr_results.items()}

    # -- Clopper-Pearson exact CIs --
    print("\n[3] Clopper-Pearson exact binomial CIs for per-group FPR...")
    cp_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        cp = clopper_pearson_all_groups(fp_vecs[name])
        cp_results[name] = cp
        for grp in sorted(cp.keys()):
            ci = cp[grp]
            print(f"    {name:10s} | {grp:28s}: FPR={ci['point_estimate']:.4f} "
                  f"95%CI [{ci['lower']:.4f}, {ci['upper']:.4f}]  (n={ci['n']})")
    all_results["clopper_pearson_ci"] = cp_results

    # -- Chi-squared test of proportions (PRIMARY omnibus) --
    print("\n[4] Chi-squared test of proportions (PRIMARY omnibus test)...")
    chi2_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        chi2_res = chi2_proportions_test(fp_vecs[name])
        chi2_results[name] = chi2_res
        if chi2_res.get("chi2") is not None:
            sig = "YES" if chi2_res["significant"] else "no"
            print(f"    {name:10s}: chi2={chi2_res['chi2']:.4f}, "
                  f"p={chi2_res['p_value']:.6f}, dof={chi2_res['dof']}  significant={sig}")
        else:
            print(f"    {name:10s}: {chi2_res.get('note', 'N/A')}")
    all_results["chi2_proportions"] = chi2_results

    # -- Kruskal-Wallis (secondary) --
    print("\n[5] Kruskal-Wallis H-test (secondary robustness check)...")
    kw_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        kw = kruskal_wallis_test(fp_vecs[name])
        kw_results[name] = kw
        sig = "YES" if kw.get("significant") else "no"
        print(f"    {name:10s}: H={kw['H_statistic']:.4f}, p={kw['p_value']:.6f}  significant={sig}")
    all_results["kruskal_wallis"] = kw_results

    # -- Pairwise Mann-Whitney U (Holm + BH-FDR) --
    print("\n[6] Pairwise Mann-Whitney U tests (Holm-Bonferroni + BH-FDR)...")
    pw_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        pairs, n_comp = pairwise_mann_whitney(fp_vecs[name])
        pw_results[name] = {"n_comparisons": n_comp, "pairs": pairs}
        sig_holm = [p for p in pairs if p["significant_holm"]]
        sig_bh = [p for p in pairs if p["significant_bh_fdr"]]
        print(f"    {name:10s}: {len(sig_holm)}/{len(pairs)} sig (Holm), "
              f"{len(sig_bh)}/{len(pairs)} sig (BH-FDR), alpha={ALPHA}")
        if sig_holm:
            for sp in sig_holm:
                print(f"      -> {sp['group_1']} vs {sp['group_2']}: "
                      f"U={sp['U_statistic']:.1f}, p_holm={sp['p_value_holm']:.6f}, "
                      f"p_bh={sp['p_value_bh_fdr']:.6f}")
    all_results["pairwise_mann_whitney"] = pw_results

    # -- Fisher's exact test --
    print("\n[7] Fisher's exact test (pairwise, for small-group robustness)...")
    fisher_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        fe = fishers_exact_per_group(fp_vecs[name])
        fisher_results[name] = fe
        sig_pairs = [p for p in fe["pairs"] if p["significant_holm"]]
        small_groups = {"disability", "age"}
        small_sig = [p for p in fe["pairs"]
                     if (p["group_1"] in small_groups or p["group_2"] in small_groups)
                     and p["significant_holm"]]
        print(f"    {name:10s}: {len(sig_pairs)}/{fe['n_comparisons']} sig (Holm), "
              f"{len(small_sig)} involve small groups (disability/age)")
        for sp in small_sig:
            print(f"      -> {sp['group_1']} (n={sp['n_1']}) vs {sp['group_2']} (n={sp['n_2']}): "
                  f"OR={sp['odds_ratio']}, p_holm={sp['p_value_holm']:.6f}")
    all_results["fishers_exact"] = fisher_results

    # -- Cohen's d --
    print("\n[8] Cohen's d effect sizes (FPR group differences)...")
    es_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        es = effect_sizes(fp_vecs[name])
        es_results[name] = es
        top = sorted(es.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"    {name:10s} top-3: " + ", ".join(f"{k}: d={v:+.3f}" for k, v in top))
    all_results["cohens_d"] = es_results

    # -- Regression: Logistic (primary) + OLS/LPM (secondary) --
    print("\n[9] Regression: FP ~ C(group) * C(condition) [text CF vs nCF]...")
    reg_result = regression_analysis(text_cf, text_ncf)
    all_results["regression"] = reg_result
    all_results["ols_regression"] = reg_result  # backward compat key

    logit_info = reg_result.get("logistic_regression", {})
    print(f"    [PRIMARY] Logistic regression: {logit_info.get('method', 'N/A')}")
    if "error" not in logit_info:
        print(f"      pseudo-R2={logit_info.get('pseudo_r_squared', 'N/A'):.4f}, "
              f"LLR p={logit_info.get('llr_p_value', 'N/A'):.6f}")
        if logit_info.get("interaction_any_significant"):
            print(f"      Interaction: at least one significant "
                  f"(min p={logit_info['interaction_min_p_value']:.6f})")
        else:
            print(f"      Interaction: none significant "
                  f"(min p={logit_info.get('interaction_min_p_value', 'N/A')})")
    else:
        print(f"      Error: {logit_info['error']}")

    ols_info = reg_result.get("ols_linear_probability_model", {})
    print(f"    [SECONDARY] OLS linear probability model: {ols_info.get('method', 'N/A')}")
    if "r_squared" in ols_info:
        print(f"      R2={ols_info['r_squared']:.4f}, adj-R2={ols_info['r_squared_adj']:.4f}")
    if "interaction_p_value" in ols_info and ols_info["interaction_p_value"] is not None:
        sig = "YES" if ols_info.get("interaction_significant") else "no"
        print(f"      Interaction p={ols_info['interaction_p_value']:.6f}  significant={sig}")
    if "anova_table" in ols_info:
        print("      ANOVA Table:")
        for src, vals in ols_info["anova_table"].items():
            f_val = f"{vals['F']:.3f}" if vals["F"] is not None else "---"
            p_val = f"{vals['PR(>F)']:.6f}" if vals["PR(>F)"] is not None else "---"
            print(f"        {src:55s}  F={f_val:>10s}  p={p_val:>10s}")

    # -- DFPR Regression --
    print("\n[10] DFPR Logistic Regression: FP ~ C(group) per modality...")
    dfpr_results = {}
    for name in ["text_cf", "text_ncf", "image", "fusion"]:
        dfpr = dfpr_regression(fp_vecs[name], modality_name=name)
        dfpr_results[name] = dfpr
        if "llr_p_value" in dfpr:
            sig = "YES" if dfpr.get("significant_overall") else "no"
            print(f"    {name:10s}: LLR p={dfpr['llr_p_value']:.6f}, "
                  f"pseudo-R2={dfpr['pseudo_r_squared']:.4f}, significant={sig}")
            if dfpr.get("coefficients"):
                sig_coefs = {k: v for k, v in dfpr["coefficients"].items()
                             if v["significant"] and k != "Intercept"}
                if sig_coefs:
                    print(f"      Significant group coefficients ({len(sig_coefs)}):")
                    for cname, cval in sig_coefs.items():
                        print(f"        {cname}: b={cval['coefficient']:+.4f}, p={cval['p_value']:.6f}")
        elif "chi2" in dfpr:
            sig = "YES" if dfpr.get("significant") else "no"
            print(f"    {name:10s}: chi2={dfpr['chi2']:.4f}, p={dfpr['p_value']:.6f}, significant={sig}")
        elif "error" in dfpr:
            print(f"    {name:10s}: ERROR - {dfpr['error']}")
    all_results["dfpr_regression"] = dfpr_results

    # -- Cochran's Q test --
    print("\n[11] Cochran's Q test (paired CF vs nCF on text)...")
    cq = cochrans_q_test(text_cf, text_ncf)
    all_results["cochrans_q"] = cq
    if cq.get("Q_statistic") is not None:
        sig = "YES" if cq.get("significant") else "no"
        print(f"    Q={cq['Q_statistic']:.4f}, p={cq['p_value']:.6f}, "
              f"n_paired={cq['n_paired']}, significant={sig}")
        print(f"    FP rate CF={cq['fp_rate_cf']:.4f}, FP rate nCF={cq['fp_rate_ncf']:.4f}")
    else:
        print(f"    {cq.get('note', 'N/A')}")

    # -- Plots --
    print("\n[12] Generating plots...")
    plot_group_fpr_comparison(
        fpr_results["text_cf"], fpr_results["image"], fpr_results["fusion"],
        PLOTS_DIR / "group_fpr_comparison_boxplot.png"
    )
    plot_pairwise_heatmap(
        pw_results["text_cf"]["pairs"], "Text CF",
        PLOTS_DIR / "pairwise_significance_heatmap.png"
    )
    plot_clopper_pearson_ci(cp_results, PLOTS_DIR / "clopper_pearson_ci_plot.png")

    # -- Summary table --
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Per-Group FPR with 95% Clopper-Pearson CIs")
    print("=" * 80)
    header = (f"{'Group':<28s} {'Text CF':>9s} {'95% CI':>16s} "
              f"{'Text nCF':>9s} {'Image':>9s} {'Fusion':>9s}")
    print(header)
    print("-" * len(header))
    for grp in TARGET_GROUPS:
        vals = []
        for m in ["text_cf", "text_ncf", "image", "fusion"]:
            v = fpr_results[m].get(grp, float("nan"))
            vals.append(f"{v:>9.4f}")
        ci_str = ""
        if grp in cp_results.get("text_cf", {}):
            ci = cp_results["text_cf"][grp]
            ci_str = f"[{ci['lower']:.4f},{ci['upper']:.4f}]"
        line = f"{grp:<28s} {vals[0]} {ci_str:>16s} {'  '.join(vals[1:])}"
        print(line)

    print("\n" + "-" * 80)
    print("MAX FPR DISPARITY (max - min across groups):")
    for m in ["text_cf", "text_ncf", "image", "fusion"]:
        vals = list(fpr_results[m].values())
        if vals:
            disp = max(vals) - min(vals)
            print(f"  {m:10s}: dFPR = {disp:.4f}  (max={max(vals):.4f}, min={min(vals):.4f})")

    # -- Save results --
    outpath = RESULTS_DIR / "enhanced_statistical_tests.json"
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    all_results = make_serializable(all_results)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved: {outpath}")
    print(f"[OK] Plots: {PLOTS_DIR / 'group_fpr_comparison_boxplot.png'}")
    print(f"[OK] Plots: {PLOTS_DIR / 'pairwise_significance_heatmap.png'}")
    print(f"[OK] Plots: {PLOTS_DIR / 'clopper_pearson_ci_plot.png'}")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
