# -*- coding: utf-8 -*-
"""
Tests for dif_analysis.py — the "Deconvoluting Ischemic Factors" cardiac
electrophysiology + ML study.

NOTE ON NAMING: the repository is named `DIF`, but the code it contains is NOT
Differential Item Functioning (psychometrics). It is a computational cardiology
study: Luo-Rudy (1991) and Ten Tusscher (2006) myocardial ion-channel ODE models
are simulated under varying ischemic conditions (hyperkalemia Ko, acidosis pH_i,
I_K(ATP) conductance), action-potential features are extracted, and RandomForest
regressors are trained to recover the ischemic factors from those features.

These tests therefore validate the actual code: ODE simulation, feature
extraction invariants, and an end-to-end "planted-signal" check analogous to a
planted-effect test — a severely ischemic cell must be distinguishable from a
healthy one and the trained regressor must recover the correct Ko/pH ordering,
while a no-perturbation (healthy) case stays at physiological baseline.
"""

import numpy as np
import pandas as pd
import pytest

import dif_analysis as d

FEATURE_COLS = ['rmp', 'peak_v', 'apa', 'dvdt_max', 'apd90', 'apd50']
BASE = {'duration': 400, 'stim_start': 50.0, 'stim_end': 51.0, 'stim_amp': -40.0}


def _features(Ko, pH_i, G_K_ATP, model='lrd', stim_amp=None, duration=400):
    """Run one simulation + feature extraction for a given cell condition."""
    params = dict(BASE, Ko=Ko, pH_i=pH_i, G_K_ATP=G_K_ATP, duration=duration)
    if stim_amp is not None:
        params['stim_amp'] = stim_amp
    t, V = d.run_simulation(params, model_type=model)
    return t, V, d.extract_features(t, V)


# --------------------------------------------------------------------------- #
# Import / side-effect safety
# --------------------------------------------------------------------------- #

def test_import_is_side_effect_free():
    """Importing the module must not run the heavy workflow."""
    assert hasattr(d, 'main')
    for name in ('run_simulation', 'extract_features',
                 'luo_rudy_1991_model', 'ten_tusscher_2006_model'):
        assert callable(getattr(d, name))


def test_matplotlib_headless_backend():
    """The Agg backend must be active so the module is importable headless."""
    assert d.matplotlib.get_backend().lower() == 'agg'


# --------------------------------------------------------------------------- #
# Simulation invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model,stim", [('lrd', -40.0), ('tt06', -80.0)])
def test_simulation_shapes_and_finite(model, stim):
    t, V, _ = _features(5.4, 7.4, 0.0, model=model, stim_amp=stim)
    assert t.shape == V.shape
    assert t.size > 0
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(t))


def test_healthy_action_potential_is_physiological():
    """A healthy LRd cell upstrokes from a negative resting potential to a
    positive peak (a real action potential), and is flagged valid."""
    _, V, f = _features(5.4, 7.4, 0.0)
    assert f['is_valid'] == 1
    assert f['rmp'] < -60.0          # resting membrane potential negative
    assert f['peak_v'] > 0.0         # depolarizes past 0 mV
    assert f['apa'] > 80.0           # physiological amplitude
    assert f['dvdt_max'] > 0.0       # rising upstroke


# --------------------------------------------------------------------------- #
# Feature-extraction property checks
# --------------------------------------------------------------------------- #

def test_extract_features_amplitude_consistency():
    """apa must equal peak_v - rmp, and apd50 <= apd90 (50% repol is earlier)."""
    _, _, f = _features(5.4, 7.4, 0.0)
    assert f['is_valid'] == 1
    assert f['apa'] == pytest.approx(f['peak_v'] - f['rmp'], abs=1e-9)
    assert f['apd50'] <= f['apd90'] + 1e-9
    assert f['apd90'] > 0.0


def test_flat_trace_is_invalid():
    """A flat sub-threshold trace (no action potential) must be rejected."""
    t = np.linspace(0, 100, 500)
    V = np.full_like(t, -85.0)
    assert d.extract_features(t, V) == {"is_valid": 0}


def test_extract_features_no_divide_by_zero_on_constant_time():
    """diff(t) appears in a denominator; the +1e-12 guard must prevent inf."""
    t = np.zeros(50)
    V = np.linspace(-85.0, 40.0, 50)
    out = d.extract_features(t, V)
    # Either flagged invalid or, if it computes, dvdt must stay finite.
    if out.get('is_valid'):
        assert np.isfinite(out['dvdt_max'])


# --------------------------------------------------------------------------- #
# Planted-signal: ischemia is distinguishable from health
# --------------------------------------------------------------------------- #

def test_ischemia_changes_action_potential():
    """A severely ischemic cell (high Ko, low pH, active I_K(ATP)) must produce
    a measurably different action-potential feature vector than a healthy cell.
    This is the 'planted effect' the ML stage is meant to detect."""
    _, _, healthy = _features(5.4, 7.4, 0.0)
    _, _, severe = _features(11.5, 6.55, 0.28)
    assert healthy['is_valid'] == 1 and severe['is_valid'] == 1
    # APD90 is the dominant ischemia-sensitive feature in this model.
    rel_change = abs(severe['apd90'] - healthy['apd90']) / max(healthy['apd90'], 1e-9)
    assert rel_change > 0.5, (
        f"ischemic APD90 ({severe['apd90']:.2f}) indistinguishable from "
        f"healthy ({healthy['apd90']:.2f})"
    )


def _build_training_set(seed=42, n_per_class=12):
    rng = np.random.RandomState(seed)
    class_defs = {
        0: {'Ko': (5.4, 5.4), 'pH_i': (7.4, 7.4), 'G_K_ATP': (0.0, 0.0)},
        1: {'Ko': (6.5, 9.5), 'pH_i': (6.7, 7.1), 'G_K_ATP': (0.05, 0.2)},
        2: {'Ko': (9.0, 12.5), 'pH_i': (6.5, 6.8), 'G_K_ATP': (0.15, 0.3)},
    }
    rows = []
    for spec in class_defs.values():
        for _ in range(n_per_class):
            p = dict(BASE)
            p.update({k: rng.uniform(*v) for k, v in spec.items()})
            t, V = d.run_simulation(p, model_type='lrd')
            f = d.extract_features(t, V)
            if f.get('is_valid'):
                f.update(Ko_actual=p['Ko'], pH_actual=p['pH_i'])
                rows.append(f)
    return pd.DataFrame(rows)


def test_regressor_recovers_planted_factors():
    """End-to-end: train the same RandomForest regressors the script uses on
    LRd-simulated data, then confirm they recover the correct ordering on a
    fresh healthy vs. severe cell. The clean (healthy) case must stay near
    baseline; the ischemic case must be flagged as more hyperkalemic / acidotic.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = _build_training_set()
    assert len(df) >= 20, "too few valid training samples generated"

    X = df[FEATURE_COLS]
    ko_reg = RandomForestRegressor(n_estimators=60, random_state=0).fit(X, df['Ko_actual'])
    ph_reg = RandomForestRegressor(n_estimators=60, random_state=0).fit(X, df['pH_actual'])

    _, _, healthy = _features(5.4, 7.4, 0.0)
    _, _, severe = _features(11.5, 6.55, 0.28)
    Xh = pd.DataFrame([healthy])[FEATURE_COLS]
    Xs = pd.DataFrame([severe])[FEATURE_COLS]

    ko_h, ko_s = ko_reg.predict(Xh)[0], ko_reg.predict(Xs)[0]
    ph_h, ph_s = ph_reg.predict(Xh)[0], ph_reg.predict(Xs)[0]

    # Ordering: severe ischemia => higher Ko, lower pH than the healthy cell.
    assert ko_s > ko_h, f"Ko ordering failed: severe {ko_s:.2f} <= healthy {ko_h:.2f}"
    assert ph_s < ph_h, f"pH ordering failed: severe {ph_s:.2f} >= healthy {ph_h:.2f}"

    # Clean (healthy) case stays near the planted baseline (Ko 5.4, pH 7.4).
    assert ko_h == pytest.approx(5.4, abs=1.0)
    assert ph_h == pytest.approx(7.4, abs=0.2)

    # Predictions stay inside the physiologically planted ranges (no runaway).
    assert 5.0 <= ko_s <= 13.0
    assert 6.4 <= ph_s <= 7.5


def test_no_dif_symmetry_two_healthy_cells_match():
    """Symmetry / 'no planted effect' control: two independent healthy cells
    yield essentially identical features (the model is deterministic given
    fixed parameters, so this also pins determinism)."""
    _, _, a = _features(5.4, 7.4, 0.0)
    _, _, b = _features(5.4, 7.4, 0.0)
    for col in FEATURE_COLS:
        assert a[col] == pytest.approx(b[col], rel=1e-9, abs=1e-9)
