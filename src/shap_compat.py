"""Preserve sklearn RF routing in SHAP's interventional float32 threshold kernel.

sklearn casts RF inputs to float32 but stores split thresholds as float64.
For ANY float32 value x, x <= t64 iff x <= floor_float32(t64). Rounding a
threshold upward instead changes routing at the float32 boundary. We floor
only the SHAP representation's thresholds, not the fitted forest. All masked
values are quantized to float32 first. This preserves every possible RF input
decision and makes the interventional kernel consistent with sklearn.
"""
from common import *
import shap

POLICY='sklearn_float32_inputs_and_floor32_thresholds_v1'

def rf_probability_explainer(model, background):
    b=np.asarray(background,dtype=np.float32)
    explainer=shap.TreeExplainer(model,data=b,model_output='probability',feature_perturbation='interventional')
    assert explainer.data.dtype==np.float32
    old=explainer.model.thresholds.copy()
    rounded=old.astype(np.float32)
    rounded_up=rounded.astype(np.float64)>old
    floored=np.where(rounded_up,np.nextafter(rounded,np.float32(-np.inf)),rounded).astype(np.float64)
    # A deterministic regression check at both adjacent representable boundaries.
    finite=np.isfinite(old)
    above=np.nextafter(floored.astype(np.float32),np.float32(np.inf)).astype(np.float64)
    assert np.all(floored[finite]<=old[finite])
    assert np.all(above[finite]>old[finite])
    explainer.model.thresholds=floored
    # The real model is never mutated. Check parser fidelity on all background rows.
    parsed=explainer.model.predict(b)
    reference=model.predict_proba(b)
    error=float(np.max(np.abs(parsed-reference)))
    assert error<1e-10, f'RF routing mismatch after numeric alignment: {error}'
    meta=dict(policy=POLICY,background_dtype=str(b.dtype),changed_shap_thresholds=int(np.count_nonzero(old!=floored)),
        background_prediction_error=error,max_threshold_adjustment=float(np.max(np.abs(old-floored))),
        fitted_model_unchanged=True,shap_values_rescaled=False,all_float32_threshold_boundaries_verified=True,
        rationale='For float32 RF inputs, threshold comparisons against t64 equal comparisons against the greatest float32 <= t64.')
    return explainer,meta
