"""
src/utils/utils.py
-------------------
Faithful ports of the label-aggregation and standardization logic from
the original PTB-XL benchmark (helme/ecg_ptbxl_benchmarking, code/utils/
utils.py). These reproduce the original's exact filtering and lookup
behaviour and are kept separate from src/data.py, whose aggregate_labels()
and apply_standardizer() diverge from the original in ways that change
ground-truth labels and standardization granularity:

* aggregate_labels() filters on `likelihood > 0`. In PTB-XL, a likelihood
  of 0.0 means "not quantified", not "absent" -- the original never
  checks the value, only key membership in scp_codes.
* aggregate_labels() reads the `rhythm`/`form` columns of scp_statements
  as label values, but those columns are 0/1 flags, not class names --
  this collapses all rhythm/form codes into a single degenerate class.
* apply_standardizer() in data.py fits one mean/std per lead; the
  original fits a single global scalar mean/std across all leads,
  timesteps, and samples.

Use these when exact reproduction of Strodthoff et al. baseline numbers
matters. Rare-class filtering (`min_samples` in the original's
select_data, which trims subdiagnostic to 23 and rhythm to 12 classes)
is not yet ported.

References
----------
Strodthoff et al. (2021) Deep Learning for ECG Analysis: Benchmarks and
Insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics
25(5), 1519-1528.
Original source: https://github.com/helme/ecg_ptbxl_benchmarking
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# SCP statement tables
# ---------------------------------------------------------------------------


def load_scp_statements(data_dir: str | Path) -> pd.DataFrame:
    """Load scp_statements.csv with SCP code strings as the index."""
    return pd.read_csv(Path(data_dir) / "scp_statements.csv", index_col=0)


def diagnostic_agg_table(scp_statements: pd.DataFrame) -> pd.DataFrame:
    """Subset of scp_statements where `diagnostic == 1.0`.

    Original: ``diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]``
    """
    return scp_statements[scp_statements.diagnostic == 1.0]


def rhythm_agg_table(scp_statements: pd.DataFrame) -> pd.DataFrame:
    """Subset of scp_statements where `rhythm == 1.0`.

    Original: ``rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]``
    """
    return scp_statements[scp_statements.rhythm == 1.0]


# ---------------------------------------------------------------------------
# Per-record label aggregation
# ---------------------------------------------------------------------------


def aggregate_superdiagnostic(
    scp_codes: Dict[str, float], diag_agg_df: pd.DataFrame
) -> List[str]:
    """Map one record's SCP codes to diagnostic superclasses (5 classes).

    Faithful port of ``aggregate_diagnostic`` in the original utils.py.
    Every key in `scp_codes` is considered regardless of its likelihood
    value, matching the original.

    Parameters
    ----------
    scp_codes:
        The record's ``scp_codes`` dict, e.g. ``{'NORM': 100.0, 'SR': 0.0}``.
    diag_agg_df:
        Output of :func:`diagnostic_agg_table`.

    Returns
    -------
    List of unique superclass labels, e.g. ``['NORM']``. Empty if none apply.
    """
    labels = []
    for code in scp_codes.keys():
        if code in diag_agg_df.index:
            cls = diag_agg_df.loc[code].diagnostic_class
            if str(cls) != "nan":
                labels.append(cls)
    return list(set(labels))


def aggregate_subdiagnostic(
    scp_codes: Dict[str, float], diag_agg_df: pd.DataFrame
) -> List[str]:
    """Map one record's SCP codes to diagnostic subclasses (23 classes).

    Faithful port of ``aggregate_subdiagnostic`` in the original utils.py.
    Same filtering as :func:`aggregate_superdiagnostic`, reading
    ``diagnostic_subclass`` instead of ``diagnostic_class``.
    """
    labels = []
    for code in scp_codes.keys():
        if code in diag_agg_df.index:
            cls = diag_agg_df.loc[code].diagnostic_subclass
            if str(cls) != "nan":
                labels.append(cls)
    return list(set(labels))


def aggregate_rhythm(
    scp_codes: Dict[str, float], rhythm_agg_df: pd.DataFrame
) -> List[str]:
    """Map one record's SCP codes to rhythm classes.

    Faithful port of ``aggregate_rhythm`` in the original utils.py. Unlike
    the diagnostic aggregations, there is no separate label column: the
    SCP code itself is the class label. `rhythm_agg_df` only gates which
    codes count as rhythm statements.

    Note
    ----
    The original benchmark additionally drops rhythm classes with too
    few samples (``min_samples`` in ``select_data``) to arrive at exactly
    12 classes. This function reproduces only the per-record mapping
    step; rare-class filtering is not yet ported.
    """
    labels = []
    for code in scp_codes.keys():
        if code in rhythm_agg_df.index:
            if str(code) != "nan":
                labels.append(code)
    return list(set(labels))


# ---------------------------------------------------------------------------
# Standardization (single global scalar mean/std, not per-lead)
# ---------------------------------------------------------------------------


def fit_standardizer(X_train: np.ndarray) -> StandardScaler:
    """Fit a single global mean/std over all leads, timesteps, and samples.

    Faithful port of the fit step in the original ``preprocess_signals``:
    ``ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))``.
    One scalar mean and one scalar std are used for every value in every
    channel -- not one pair per lead.

    Parameters
    ----------
    X_train:
        Array of shape ``(n_train, n_samples, n_leads)``.

    Returns
    -------
    StandardScaler fitted on the flattened training data.
    """
    flat = np.vstack(X_train).flatten()[:, np.newaxis].astype(float)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_standardizer(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a global scalar standardizer to each recording.

    Faithful port of ``apply_standardizer`` in the original utils.py: each
    recording is flattened, transformed, and reshaped back, so the same
    global mean/std is applied to every lead and timestep.

    Parameters
    ----------
    X:
        Array of shape ``(n, n_samples, n_leads)``.
    scaler:
        A StandardScaler previously returned by :func:`fit_standardizer`.

    Returns
    -------
    np.ndarray of the same shape as X.
    """
    out = []
    for x in X:
        shape = x.shape
        out.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(shape))
    return np.array(out)
