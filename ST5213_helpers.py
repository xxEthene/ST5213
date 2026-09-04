import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.stats import chi2

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from statsmodels.formula.api import glm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

def hat_diag(fm):
    X = fm.model.exog
    w = fm.model.weights  # final IRLS weights used by statsmodels
    Xw = X * np.sqrt(w)[:, None]
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    H = Xw @ XtX_inv @ Xw.T
    return np.diag(H)

def update_glm(model, formula=None, add=None, drop=None):
    """
    R-like update() for statsmodels formula models (GLM).
    
    Use ONE of:
      - formula="new_lhs ~ new_rhs"   (explicit replacement), or
      - add=" + ..." / drop=" - ..."  (simple RHS edits).
    """
    old = str(model.model.formula)

    if formula is not None:
        new_formula = formula
    else:
        lhs, rhs = old.split("~", 1)
        rhs = rhs.strip()

        if drop:
            rhs = f"({rhs}) - ({drop})"
        if add:
            rhs = f"({rhs}) + ({add})"

        new_formula = f"{lhs.strip()} ~ {rhs}"

    return smf.glm(
        formula=new_formula,
        data=model.model.data.frame,
        family=model.model.family
    ).fit()

def _get_deviance(m):
    # GLMResults (and many others) expose .deviance
    if hasattr(m, "deviance") and m.deviance is not None:
        return float(m.deviance)

    # MNLogitResults exposes .llf (log-likelihood), not deviance
    if hasattr(m, "llf") and m.llf is not None:
        return float(-2.0 * m.llf)

    raise AttributeError(
        "Model result has neither `.deviance` nor `.llf`; can't compute deviance."
    )

def anova_nested(*models):
    """Construct an Analysis of Deviance table for nested models.
    Pass models in ascending order of complexity: fm0, fm1, fm2, ...
    """
    df_resid = [float(m.df_resid) for m in models]
    deviance = [_get_deviance(m) for m in models]

    df_diff = [np.nan]
    dev_diff = [np.nan]
    p_values = [np.nan]

    for i in range(1, len(models)):
        d_df = df_resid[i - 1] - df_resid[i]
        d_dev = deviance[i - 1] - deviance[i]

        # Guard against numerical noise (occasionally negative by ~1e-12)
        if d_dev < 0 and abs(d_dev) < 1e-8:
            d_dev = 0.0

        p_val = chi2.sf(d_dev, d_df) if d_df > 0 else np.nan

        df_diff.append(d_df)
        dev_diff.append(d_dev)
        p_values.append(p_val)

    table = pd.DataFrame(
        {
            "Resid. Df": df_resid,
            "Resid. Dev": deviance,
            "Df": df_diff,
            "Deviance": dev_diff,
            "Pr(>Chi)": p_values,
        },
        index=[f"Model {i+1}" for i in range(len(models))],
    )
    return table


    """
    Sequential (Type I) Analysis of Deviance for a *single* fitted statsmodels GLM.

    Works best when the model was fit with the formula interface (statsmodels.formula.api),
    because we can rebuild nested models term-by-term in formula order.

    Prints a table similar to R's anova(glm, test="Chisq") / analysis of deviance.
    """
    # --- basic checks ---
    model = glm_res.model
    if not hasattr(model, "formula"):
        raise ValueError(
            "This function requires a formula-based GLM (fit via statsmodels.formula.api).\n"
            "Your model doesn't appear to have a .model.formula."
        )

    formula_full = model.formula  # e.g. 'y ~ Sex + Age + Sex:Age'
    data = model.data.frame       # original DataFrame used in fit

    # preserve common GLM extras if present
    family = model.family
    freq_weights = getattr(model, "freq_weights", None)
    var_weights  = getattr(model, "var_weights", None)
    exposure     = getattr(model, "exposure", None)
    offset       = getattr(model, "offset", None)

    # --- parse formula terms in order (Patsy) ---
    import patsy
    desc = patsy.ModelDesc.from_formula(formula_full)

    if len(desc.rhs_termlist) == 0:
        raise ValueError("No RHS terms found in formula.")

    # Identify response and RHS terms (excluding the intercept term if present)
    response = desc.lhs_termlist[0].factors[0].name()

    def term_to_str(term):
        # Convert a Patsy Term into something like "Sex" or "Sex:Age"
        return ":".join(f.name() for f in term.factors) if term.factors else "Intercept"

    rhs_terms = []
    for t in desc.rhs_termlist:
        tname = term_to_str(t)
        if tname.lower() == "intercept":
            continue
        rhs_terms.append(tname)

    # --- fit nested models: NULL, then add terms sequentially ---
    results = []

    # NULL model: response ~ 1
    f_null = f"{response} ~ 1"
    m_prev = smf_glm_fit(f_null, data, family, freq_weights, var_weights, exposure, offset)
    results.append(("NULL", m_prev.df_resid, m_prev.deviance, np.nan, np.nan, np.nan))

    current_terms = []
    for t in rhs_terms:
        current_terms.append(t)
        f_curr = f"{response} ~ 1 + " + " + ".join(current_terms)

        m_curr = smf_glm_fit(f_curr, data, family, freq_weights, var_weights, exposure, offset)

        df_diff = m_prev.df_resid - m_curr.df_resid
        dev_diff = m_prev.deviance - m_curr.deviance  # reduction in deviance
        pval = chi2.sf(dev_diff, df_diff) if (df_diff > 0) else np.nan

        results.append((t, m_curr.df_resid, m_curr.deviance, df_diff, dev_diff, pval))
        m_prev = m_curr

    out = pd.DataFrame(
        results,
        columns=["Term", "Resid. Df", "Resid. Dev", "Df", "Deviance", "Pr(>Chi)"]
        
    )
    out = out.set_index("Term")
    out.index.name = None   # or: out = out.rename_axis(None)
    return out

def _smf_glm_fit(formula, data, family, freq_weights=None, var_weights=None, exposure=None, offset=None):
    fit_kwargs = {}
    if freq_weights is not None:
        fit_kwargs["freq_weights"] = freq_weights
    if var_weights is not None:
        fit_kwargs["var_weights"] = var_weights
    if exposure is not None:
        fit_kwargs["exposure"] = exposure
    if offset is not None:
        fit_kwargs["offset"] = offset

    return smf.glm(formula=formula, data=data, family=family, **fit_kwargs).fit()

def anova_single(glm_res):
    """
    Sequential (Type I) Analysis of Deviance for a single fitted statsmodels GLM.
    Works with formula-based GLMs, including grouped binomial endog like: 'Cases + fail ~ ...'.
    """
    model = glm_res.model
    if not hasattr(model, "formula"):
        raise ValueError("Requires a formula-based GLM fit (statsmodels.formula.api.glm).")

    formula_full = model.formula
    data = model.data.frame

    family = model.family
    freq_weights = getattr(model, "freq_weights", None)
    var_weights  = getattr(model, "var_weights", None)
    exposure     = getattr(model, "exposure", None)
    offset       = getattr(model, "offset", None)

    # --- IMPORTANT: keep the full LHS, e.g. "Cases + fail" ---
    lhs = formula_full.split("~", 1)[0].strip()

    # Parse RHS term order from patsy (Type I / sequential)
    desc = patsy.ModelDesc.from_formula(formula_full)

    def term_to_str(term):
        return ":".join(f.name() for f in term.factors) if term.factors else "Intercept"

    rhs_terms = []
    for t in desc.rhs_termlist:
        tname = term_to_str(t)
        if tname.lower() != "intercept":
            rhs_terms.append(tname)

    results = []

    # NULL model
    f_null = f"{lhs} ~ 1"
    m_prev = _smf_glm_fit(f_null, data, family, freq_weights, var_weights, exposure, offset)
    results.append(("NULL", m_prev.df_resid, m_prev.deviance, np.nan, np.nan, np.nan))

    current_terms = []
    for t in rhs_terms:
        current_terms.append(t)
        f_curr = f"{lhs} ~ 1 + " + " + ".join(current_terms)

        m_curr = _smf_glm_fit(f_curr, data, family, freq_weights, var_weights, exposure, offset)

        df_diff = m_prev.df_resid - m_curr.df_resid
        dev_diff = m_prev.deviance - m_curr.deviance  # should be >= 0 for properly nested fits

        # Guard against tiny negative values due to numerical noise
        if dev_diff < 0 and abs(dev_diff) < 1e-8:
            dev_diff = 0.0

        pval = chi2.sf(dev_diff, df_diff) if (df_diff > 0 and dev_diff >= 0) else np.nan

        results.append((t, m_curr.df_resid, m_curr.deviance, df_diff, dev_diff, pval))
        m_prev = m_curr

    out = pd.DataFrame(
        results,
        columns=["Term", "Resid. Df", "Resid. Dev", "Df", "Deviance", "Pr(>Chi)"]
    ).set_index("Term")

    out.index.name = None
    return out

def drop1_lm(model, test="F"):
    """
    drop1-like table for statsmodels OLS, enforcing hierarchy:
    only terms that are NOT needed by any higher-order term are eligible to drop.

    Works best for models built via the formula API (smf.ols).
    """
    # --- full model info ---
    full_ssr = float(model.ssr)              # CHANGED: OLS uses SSR (RSS)
    full_df_resid = int(model.df_resid)
    full_aic = float(model.aic)

    # formula pieces
    formula_str = str(model.model.formula)
    lhs, _ = formula_str.split("~", 1)
    response = lhs.strip()

    # get term labels from patsy (pure strings)
    design_info = getattr(model.model.data, "design_info", None)
    if design_info is None:
        design_info = model.model.data.orig_exog.design_info
    all_terms = [str(t) for t in design_info.term_names if str(t) not in ("Intercept", "1")]

    # hierarchy via component sets (best for ':' interactions)
    term_comps = {t: set(t.split(":")) for t in all_terms}

    # droppable = "maximal" terms: no other term is a strict superset
    droppable = []
    for t, comps in term_comps.items():
        needed_by_higher = any((comps < other_comps) for other, other_comps in term_comps.items() if other != t)
        if not needed_by_higher:
            droppable.append(t)

    rows = [{
        "Term": "<none>",
        "Df": "",
        "RSS": full_ssr,                      # CHANGED: label/metric
        "AIC": full_aic,
        "F": "",
        "Pr(>F)": ""
    }]

    # refit reduced models
    for term in droppable:
        rhs_terms = [t for t in all_terms if t != term]
        rhs = " + ".join(rhs_terms) if rhs_terms else "1"
        new_formula = f"{response} ~ {rhs}"

        reduced = smf.ols(                    # CHANGED: smf.ols instead of smf.glm
            new_formula,
            data=model.model.data.frame
        ).fit()

        df_diff = int(reduced.df_resid - full_df_resid)

        # CHANGED: F-test based on RSS difference (extra sum-of-squares test)
        rss_diff = float(reduced.ssr - full_ssr)
        rss_diff = max(0.0, rss_diff)  # numerical guard

        fstat = (rss_diff / df_diff) / (full_ssr / full_df_resid) if df_diff > 0 else float("nan")
        pval = stats.f.sf(fstat, df_diff, full_df_resid) if (test == "F" and df_diff > 0) else None

        rows.append({
            "Term": term,
            "Df": df_diff,
            "RSS": float(reduced.ssr),
            "AIC": float(reduced.aic),
            "F": fstat,
            "Pr(>F)": pval
        })

    out = pd.DataFrame(rows).set_index("Term")
    out.index.name = None
    return out

def drop1_glm(model, test="Chisq"):
    """
    drop1-like table for statsmodels GLM, enforcing hierarchy:
    only terms that are NOT needed by any higher-order term are eligible to drop.

    Works best for models built via the formula API (smf.glm).
    """
    # --- full model info ---
    full_dev = float(model.deviance)
    full_df_resid = int(model.df_resid)
    full_aic = float(model.aic)

    # formula pieces
    formula_str = str(model.model.formula)
    lhs, _ = formula_str.split("~", 1)
    response = lhs.strip()

    # get term labels from patsy (pure strings)
    design_info = model.model.data.design_info
    all_terms = [str(t) for t in design_info.term_names if str(t) not in ("Intercept", "1")]

    # hierarchy via component sets (best for ':' interactions)
    term_comps = {t: set(t.split(":")) for t in all_terms}

    # droppable = "maximal" terms: no other term is a strict superset
    droppable = []
    for t, comps in term_comps.items():
        needed_by_higher = any((comps < other_comps) for other, other_comps in term_comps.items() if other != t)
        if not needed_by_higher:
            droppable.append(t)

    rows = [{
        "Term": "<none>",
        "Df": "",
        "Deviance": full_dev,
        "AIC": full_aic,
        "LRT": "",
        "Pr(>Chi)": ""
    }]

    # refit reduced models
    for term in droppable:
        rhs_terms = [t for t in all_terms if t != term]
        rhs = " + ".join(rhs_terms) if rhs_terms else "1"
        new_formula = f"{response} ~ {rhs}"

        reduced = smf.glm(
            new_formula,
            data=model.model.data.frame,
            family=model.model.family
        ).fit()

        df_diff = int(reduced.df_resid - full_df_resid)
        lrt = float(reduced.deviance - full_dev)
        lrt = max(0.0, lrt)  # numerical guard

        pval = stats.chi2.sf(lrt, df=df_diff) if test == "Chisq" else None

        rows.append({
            "Term": term,
            "Df": df_diff,
            "Deviance": float(reduced.deviance),
            "AIC": float(reduced.aic),
            "LRT": lrt,
            "Pr(>Chi)": pval
        })

    out = pd.DataFrame(rows).set_index("Term")
    out.index.name = None
    return out

def drop1_po(res_full, X_patsy, *, distr="logit", method="bfgs", disp=False):
    """
    Hierarchical drop1 for OrderedModel, using X_patsy.design_info for term->columns.
    X_patsy should be the ORIGINAL patsy dmatrix output (may include Intercept).
    """
    if not hasattr(X_patsy, "design_info"):
        raise ValueError("X_patsy must come from patsy.dmatrix(..., return_type='dataframe') so it has design_info")

    y = np.asarray(res_full.model.endog)

    # term -> columns
    di = X_patsy.design_info
    cols = list(X_patsy.columns)
    term_to_cols = {}
    for term, sl in di.term_name_slices.items():
        if term == "Intercept":
            continue
        term_to_cols[term] = cols[sl]

    # full model stats
    full_llf = float(res_full.llf)
    full_aic = float(res_full.aic)
    full_deviance = -2.0 * full_llf
    full_k_params = int(len(res_full.params))  # <-- robust df reference

    def deg(t): return t.count(":") + 1
    def bases(t): return tuple(p.strip() for p in t.split(":"))

    def allowed_drop(drop_term, kept_terms):
        # can't drop main if any kept interaction mentions it
        if deg(drop_term) == 1:
            for t in kept_terms:
                if deg(t) > 1 and drop_term in bases(t):
                    return False
        # can't drop lower-order if higher-order containing it remains
        drop_set = set(bases(drop_term))
        for t in kept_terms:
            if deg(t) > deg(drop_term) and drop_set.issubset(set(bases(t))):
                return False
        return True

    all_terms = list(term_to_cols.keys())

    rows = [{
        "Dropped": "<none>",
        "Deviance": full_deviance,
        "AIC": full_aic,
        "Df": np.nan,
        "LRT": np.nan,
        "Pr(>Chi)": np.nan
    }]

    for drop_term, drop_cols in term_to_cols.items():
        kept_terms = [t for t in all_terms if t != drop_term]
        if not allowed_drop(drop_term, kept_terms):
            rows.append({"Dropped": drop_term, "Deviance": np.nan, "AIC": np.nan, "Df": np.nan, "LRT": np.nan, "Pr(>Chi)": np.nan})
            continue

        # build reduced X (always remove Intercept for OrderedModel)
        keep_cols = [c for c in X_patsy.columns if c not in drop_cols and c != "Intercept"]
        X_red = X_patsy.loc[:, keep_cols].copy()

        # drop constant columns if any
        const_cols = X_red.columns[X_red.nunique(dropna=False) <= 1]
        if len(const_cols) > 0:
            X_red = X_red.drop(columns=const_cols)

        try:
            r = OrderedModel(y, X_red, distr=distr).fit(method=method, disp=disp)
        except Exception:
            rows.append({"Dropped": drop_term, "Deviance": np.nan, "AIC": np.nan, "Df": np.nan, "LRT": np.nan, "Pr(>Chi)": np.nan})
            continue

        lrt = 2.0 * (full_llf - float(r.llf))
        df_change = full_k_params - int(len(r.params))  # <-- FIX: df from parameter counts
        lr_p = float(chi2.sf(lrt, df_change)) if df_change > 0 else np.nan

        rows.append({
            "Dropped": drop_term,
            "Deviance": -2.0 * float(r.llf),
            "AIC": float(r.aic),
            "Df": df_change,
            "LRT": lrt,
            "Pr(>Chi)": lr_p
        })

    out = pd.DataFrame(rows) 
    out = out.dropna(subset=["AIC"]).reset_index(drop=True)

    # keep "<none>" first
    out = pd.concat(
        [out[out["Dropped"].eq("<none>")], out[~out["Dropped"].eq("<none>")]],
        ignore_index=True
    )

    # replace NaN with a single space
    out = out.fillna(" ")
    out = out[["Dropped", "Df", "Deviance", "AIC", "LRT", "Pr(>Chi)"]]

    return out

def drop1_MNL(res_full, X_patsy, *, method="newton", disp=False, maxiter=100, tol=1e-12):
    """
    Hierarchical drop1 for MNLogit, using X_patsy.design_info for term->columns.
    X_patsy should be the ORIGINAL patsy dmatrix output (may include Intercept).
    """
    if not hasattr(X_patsy, "design_info"):
        raise ValueError("X_patsy must come from patsy.dmatrix(..., return_type='dataframe') so it has design_info")

    # --- CHANGED: force y to be 1D class labels (match your manual MNLogit(y, X)) ---
    y = np.asarray(res_full.model.endog)
    if y.ndim == 2:
        # If it's an indicator matrix, convert to class labels.
        # (Assumes rows are one-hot / multinomial indicators.)
        y = np.asarray(y).argmax(axis=1)
    else:
        y = np.asarray(y).squeeze()
    # ------------------------------------------------------------------------------

    # term -> columns
    di = X_patsy.design_info
    cols = list(X_patsy.columns)
    term_to_cols = {}
    for term, sl in di.term_name_slices.items():
        if term == "Intercept":
            continue
        term_to_cols[term] = cols[sl]

    # full model stats
    full_llf = float(res_full.llf)
    full_aic = float(res_full.aic)
    full_deviance = -2.0 * full_llf
    full_k_params = int(res_full.params.size)  # MNLogit params are (k_exog, J-1)

    def deg(t): return t.count(":") + 1
    def bases(t): return tuple(p.strip() for p in t.split(":"))

    def allowed_drop(drop_term, kept_terms):
        # can't drop main if any kept interaction mentions it
        if deg(drop_term) == 1:
            for t in kept_terms:
                if deg(t) > 1 and drop_term in bases(t):
                    return False
        # can't drop lower-order if higher-order containing it remains
        drop_set = set(bases(drop_term))
        for t in kept_terms:
            if deg(t) > deg(drop_term) and drop_set.issubset(set(bases(t))):
                return False
        return True

    all_terms = list(term_to_cols.keys())

    rows = [{
        "Dropped": "<none>",
        "Deviance": full_deviance,
        "AIC": full_aic,
        "Df": np.nan,
        "LRT": np.nan,
        "Pr(>Chi)": np.nan
    }]

    for drop_term, drop_cols in term_to_cols.items():
        kept_terms = [t for t in all_terms if t != drop_term]
        if not allowed_drop(drop_term, kept_terms):
            rows.append({"Dropped": drop_term, "Deviance": np.nan, "AIC": np.nan, "Df": np.nan, "LRT": np.nan, "Pr(>Chi)": np.nan})
            continue

        # build reduced X (MNLogit can include Intercept; do NOT force-remove it)
        keep_cols = [c for c in X_patsy.columns if c not in drop_cols]
        X_red = X_patsy.loc[:, keep_cols].copy()

        # use restricted full-model params as start_params (helps match manual refit)
        start_params = None
        try:
            if hasattr(res_full.params, "reindex"):
                sp = res_full.params.reindex(index=keep_cols)
                start_params = np.asarray(sp).ravel(order="F")
            else:
                name_to_i = {n: i for i, n in enumerate(res_full.model.exog_names)}
                idx = [name_to_i[c] for c in keep_cols if c in name_to_i]
                sp = np.asarray(res_full.params)[idx, :]
                start_params = np.asarray(sp).ravel(order="F")
        except Exception:
            start_params = None

        try:
            # --- CHANGED: match your manual optimizer setup (newton + tight tol) ---
            r = sm.MNLogit(y, X_red).fit(
                method=method,
                disp=disp,
                start_params=start_params,
                maxiter=maxiter,
                tol=tol
            )
            # ----------------------------------------------------------------------
        except Exception:
            rows.append({"Dropped": drop_term, "Deviance": np.nan, "AIC": np.nan, "Df": np.nan, "LRT": np.nan, "Pr(>Chi)": np.nan})
            continue

        lrt = 2.0 * (full_llf - float(r.llf))
        df_change = full_k_params - int(r.params.size)
        lr_p = float(chi2.sf(lrt, df_change)) if df_change > 0 else np.nan

        rows.append({
            "Dropped": drop_term,
            "Deviance": -2.0 * float(r.llf),
            "AIC": float(r.aic),
            "Df": df_change,
            "LRT": lrt,
            "Pr(>Chi)": lr_p
        })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["AIC"]).reset_index(drop=True)

    # keep "<none>" first
    out = pd.concat(
        [out[out["Dropped"].eq("<none>")], out[~out["Dropped"].eq("<none>")]],
        ignore_index=True
    )

    out = out.fillna(" ")
    out = out[["Dropped", "Df", "Deviance", "AIC", "LRT", "Pr(>Chi)"]]
    return out


@dataclass
class StepAICResult:
    model: Any                       # statsmodels GLMResults
    selected_formula: str
    steps: List[Dict[str, Any]]      # trace of attempted drops / chosen drops

def _split_formula(formula: str) -> Tuple[str, str]:
    """
    Split 'lhs ~ rhs' at the first '~' only.
    Works when lhs is like 'successes + failures'.
    """
    parts = formula.split("~", 1)
    if len(parts) != 2:
        raise ValueError(f"Formula must contain '~': {formula!r}")
    lhs = parts[0].strip()
    rhs = parts[1].strip()
    return lhs, rhs

def _term_factors(term: str) -> Tuple[str, ...]:
    """
    Approximate Patsy 'factors' for hierarchy checks using the high-level term name.
    Examples:
      'x' -> ('x',)
      'C(a)' -> ('C(a)',)
      'x:C(a)' -> ('x', 'C(a)')
      'np.log(x):C(a)' -> ('np.log(x)', 'C(a)')
    """
    # statsmodels/patsy will use ':' to denote interactions in term_names
    return tuple(s.strip() for s in term.split(":") if s.strip())

def _is_required_by_hierarchy(candidate: str, terms: List[str]) -> bool:
    """
    Hierarchical principle (interaction hierarchy):
    If any remaining term is a strict superset interaction containing candidate's factors,
    then candidate cannot be removed.
    """
    cand_f = set(_term_factors(candidate))
    if not cand_f:
        return False

    for t in terms:
        if t == candidate:
            continue
        tf = set(_term_factors(t))
        # If t is an interaction (or higher-order) that contains candidate's factors,
        # candidate is required (e.g., x required if x:Z remains; Z required if x:Z remains).
        if len(tf) > len(cand_f) and cand_f.issubset(tf):
            return True

    return False

def _eligible_drops(terms: List[str]) -> List[str]:
    """Terms that can be dropped without violating hierarchy."""
    elig = []
    for t in terms:
        if not _is_required_by_hierarchy(t, terms):
            elig.append(t)
    return elig

def _refit_glm_like(original_fit, new_formula: str):
    """
    Refit a GLM using the same data + family and (where available) weights/offset/exposure.
    This supports binomial fits with lhs like 'successes + failures ~ ...' because we
    refit from the formula against the same dataframe.
    """
    model = original_fit.model

    # Formula models in statsmodels keep a pandas DataFrame here
    data = getattr(model.data, "frame", None)
    if data is None:
        raise ValueError(
            "This stepAIC implementation requires a formula-based GLM "
            "(so the original fit has model.data.frame)."
        )

    family = model.family
    offset = getattr(model, "offset", None)
    exposure = getattr(model, "exposure", None)

    # Weights live on the model; for GLM they can be present as arrays or None.
    freq_weights = getattr(model, "freq_weights", None)
    var_weights = getattr(model, "var_weights", None)

    # Build and fit the new model
    new_mod = smf.glm(
        formula=new_formula,
        data=data,
        family=family,
        offset=offset,
        exposure=exposure,
        freq_weights=freq_weights,
        var_weights=var_weights,
    )
    return new_mod.fit()

def stepAIC(
    fitted_glm,
    *,
    tol: float = 1e-10,
    verbose: bool = False,
) -> StepAICResult:
    """
    Backward-elimination AIC model selection for statsmodels GLMs, obeying the
    hierarchical principle for interactions (won't drop main effects needed by
    remaining interactions).

    Parameters
    ----------
    fitted_glm:
        A fitted statsmodels GLMResults (from smf.glm(...).fit()).
    tol:
        Minimum AIC improvement required to accept a drop.
    verbose:
        If True, prints progress.

    Returns
    -------
    StepAICResult with the best model, selected formula, and a step trace.

    Notes
    -----
    - Hierarchy enforcement here focuses on interaction hierarchy via ':' terms.
    - This refits via the original dataframe and formula, which also supports
      binomial formulas like 'successes + failures ~ x1 + x2'.
    """
    if not hasattr(fitted_glm, "model") or not hasattr(fitted_glm.model, "formula"):
        raise TypeError("Input must be a fitted statsmodels GLMResults from a formula model.")

    current_fit = fitted_glm
    current_formula = current_fit.model.formula
    lhs, _ = _split_formula(current_formula)

    # High-level terms from patsy design_info, excluding intercept
    design_info = current_fit.model.data.design_info
    current_terms = [t for t in design_info.term_names if t != "Intercept"]

    steps: List[Dict[str, Any]] = []
    current_aic = float(current_fit.aic)

    if verbose:
        print(f"Start AIC: {current_aic:.6f}")
        print(f"Start formula: {current_formula}")

    while True:
        elig = _eligible_drops(current_terms)

        # If nothing can be dropped (due to hierarchy), stop.
        if not elig:
            steps.append(
                {
                    "action": "stop",
                    "reason": "no eligible terms to drop (hierarchy constraint)",
                    "aic": current_aic,
                    "formula": current_formula,
                }
            )
            break

        # Try dropping each eligible term; choose the best AIC among candidates.
        tried = []
        best_candidate = None  # (aic, dropped_term, fit, terms, formula)

        for drop_term in elig:
            cand_terms = [t for t in current_terms if t != drop_term]
            rhs = " + ".join(cand_terms) if cand_terms else "1"
            cand_formula = f"{lhs} ~ {rhs}"

            try:
                cand_fit = _refit_glm_like(current_fit, cand_formula)
                cand_aic = float(cand_fit.aic)
                tried.append({"dropped": drop_term, "aic": cand_aic, "formula": cand_formula})

                if (best_candidate is None) or (cand_aic < best_candidate[0]):
                    best_candidate = (cand_aic, drop_term, cand_fit, cand_terms, cand_formula)
            except Exception as e:
                tried.append({"dropped": drop_term, "error": repr(e), "formula": cand_formula})

        steps.append(
            {
                "action": "try_drops",
                "current_aic": current_aic,
                "current_terms": list(current_terms),
                "eligible": list(elig),
                "tried": tried,
            }
        )

        if best_candidate is None:
            steps.append(
                {
                    "action": "stop",
                    "reason": "all candidate refits failed",
                    "aic": current_aic,
                    "formula": current_formula,
                }
            )
            break

        best_aic, dropped, best_fit, best_terms, best_formula = best_candidate
        improvement = current_aic - best_aic

        if verbose:
            print(f"Best drop: {dropped} -> AIC {best_aic:.6f} (improve {improvement:.6g})")

        if improvement > tol:
            # Accept this drop and continue
            current_fit = best_fit
            current_aic = best_aic
            current_terms = best_terms
            current_formula = best_formula

            steps.append(
                {
                    "action": "drop",
                    "dropped": dropped,
                    "new_aic": current_aic,
                    "formula": current_formula,
                }
            )
        else:
            # No meaningful improvement
            steps.append(
                {
                    "action": "stop",
                    "reason": "no AIC improvement beyond tol",
                    "aic": current_aic,
                    "formula": current_formula,
                }
            )
            break

    return StepAICResult(model=current_fit, selected_formula=current_formula, steps=steps)

