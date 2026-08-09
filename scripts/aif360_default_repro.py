#!/usr/bin/env python3.11
"""Minimal reproduction: AIF360's favorable-label default inverts a DI ratio.

Two groups of 1,772 applicants; the privileged group defaults 111 times, the
unprivileged 134. On a target named `loan_default`, `BinaryLabelDataset`
defaults `favorable_label=1.0` -- asserting that defaulting is the favorable
outcome -- and reports DI 1.2072 (the unprivileged group "favored"). Stating
`favorable_label=0.0` gives the oriented value, 0.9862. Both numbers come from
the same table; the sign of the conclusion is set by a constructor default.

Scope: this is one constructor in a two-API library. `StandardDataset`
requires `favorable_classes` and cannot be constructed without answering the
question. Fairlearn's `demographic_parity_ratio` exposes no favorable-label
parameter at all: it takes the min/max ratio of `y_pred`'s selection rate as
encoded, so on this table it reports 0.8284 -- a ratio of *default* rates,
a number about the wrong outcome -- and the orientation question is delegated
entirely to whoever encoded the column.

Install: pip install -r requirements-repro.txt
Run:     python3.11 scripts/aif360_default_repro.py
Output verbatim in artifacts/consolidated/aif360_default_repro.txt.
"""
import aif360
import fairlearn
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.metrics import demographic_parity_ratio

N, K_PRIV, K_UNPRIV = 1772, 111, 134  # defaults per group of N

df = pd.DataFrame(
    {
        "loan_default": [1] * K_PRIV + [0] * (N - K_PRIV)
        + [1] * K_UNPRIV + [0] * (N - K_UNPRIV),
        "group": [1] * N + [0] * N,  # 1 = privileged
    }
)

kwargs = dict(
    df=df, label_names=["loan_default"], protected_attribute_names=["group"]
)
groups = dict(privileged_groups=[{"group": 1}], unprivileged_groups=[{"group": 0}])

di_default = BinaryLabelDatasetMetric(
    BinaryLabelDataset(**kwargs), **groups  # favorable_label defaults to 1.0
).disparate_impact()
di_oriented = BinaryLabelDatasetMetric(
    BinaryLabelDataset(favorable_label=0.0, unfavorable_label=1.0, **kwargs),
    **groups,
).disparate_impact()
dpr_as_encoded = demographic_parity_ratio(
    y_true=df["loan_default"], y_pred=df["loan_default"], sensitive_features=df["group"]
)
dpr_oriented = demographic_parity_ratio(
    y_true=1 - df["loan_default"], y_pred=1 - df["loan_default"],
    sensitive_features=df["group"],
)

print(f"aif360=={aif360.__version__} fairlearn=={fairlearn.__version__} "
      f"numpy=={np.__version__} pandas=={pd.__version__}")
print(f"default favorable_label=1.0 on 'loan_default':  DI = {di_default:.4f}")
print(f"oriented favorable_label=0.0:                   DI = {di_oriented:.4f}")
print(f"fairlearn demographic_parity_ratio on the column as encoded: "
      f"{dpr_as_encoded:.4f} (a min/max ratio of default rates)")
print(f"fairlearn demographic_parity_ratio, column re-encoded by hand: "
      f"{dpr_oriented:.4f} (no favorable-label parameter exists to say which)")

assert round(di_default, 4) == 1.2072
assert round(di_oriented, 4) == 0.9862
assert round(dpr_as_encoded, 4) == 0.8284  # a ratio about the wrong outcome
assert round(dpr_oriented, 4) == 0.9862
