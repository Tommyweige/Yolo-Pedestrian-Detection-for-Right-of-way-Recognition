"""Synthetic tests for yield_learn (numpy only, no files, no randomness)."""
import json
import warnings

import numpy as np
import pytest

from yield_learn import (
    Model,
    Scaler,
    best_threshold,
    fit_logistic,
    label_candidates,
    leave_one_group_out,
    predict_proba,
    train,
)


def _synthetic_groups(names=("v1", "v2", "v3", "v4"), per_group=10):
    """Feature 0 decides the label; the other features are harmless noise."""
    rows, y, groups = [], [], []
    for gi, name in enumerate(names):
        for j in range(per_group):
            positive = j % 2 == 0
            base = 1.0 if positive else -1.0
            rows.append([base + 0.01 * j, 0.3 * ((j + gi) % 3) - 0.2 * gi, -0.1 * gi])
            y.append(1 if positive else 0)
            groups.append(name)
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=int), groups


def test_label_candidates_greedy_pairing_and_ties():
    assert label_candidates([1.0, 1.2, 5.0], [1.1], 1.0).tolist() == [1, 0, 0]
    assert label_candidates([1.0], [0.5, 1.4], 1.0).tolist() == [1]
    assert label_candidates([], [], 1.0).tolist() == []
    assert label_candidates([10.0], [1.0], 0.5).tolist() == [0]
    # Both candidates tie at distance 0.05: the lower candidate index wins.
    assert label_candidates([1.0, 1.1, 1.2], [1.05], 0.1).tolist() == [1, 0, 0]
    assert label_candidates([1.0, 1.5], [1.1, 1.6], 0.2).tolist() == [1, 1]


def test_label_candidates_returns_int_array():
    labels = label_candidates([1.0, 2.0], [1.0], 0.1)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype.kind in "iu"
    assert labels.shape == (2,)


def test_scaler_zero_std_column_is_finite():
    X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    scaler = Scaler().fit(X)
    Z = scaler.transform(X)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 0], 0.0)
    assert np.allclose(Z[:, 1], (X[:, 1] - X[:, 1].mean()) / X[:, 1].std())


def test_fit_logistic_separable_signs_and_degenerate_labels():
    X = np.array([[2.0, 0.1], [1.5, -0.2], [1.2, 0.3], [2.2, 0.0],
                  [-2.0, -0.1], [-1.5, 0.2], [-1.2, -0.3], [-2.2, 0.0]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    w, b = fit_logistic(X, y)
    scores = predict_proba(X, w, b)
    assert (scores[:4] > 0.5).all()
    assert (scores[4:] < 0.5).all()

    w0, b0 = fit_logistic(X, np.zeros(len(X)))
    assert np.all(w0 == 0)
    assert b0 <= -10.0

    w1, b1 = fit_logistic(X, np.ones(len(X)))
    assert np.all(w1 == 0)
    assert b1 >= 10.0


def test_predict_proba_extreme_logits_do_not_overflow():
    X = np.array([[1.0], [-1.0]])
    w = np.array([1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = predict_proba(X, w, 0.0)
    assert isinstance(scores, np.ndarray)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_model_dict_roundtrip_is_json_serialisable():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 3.0]])
    scaler = Scaler().fit(X)
    model = Model(scaler, np.array([0.4, -0.7]), 0.2, 0.55, ["a", "b"])
    payload = model.to_dict()
    json.dumps(payload)
    restored = Model.from_dict(payload)
    assert restored.feature_names == ["a", "b"]
    assert restored.threshold == pytest.approx(0.55)
    assert np.allclose(restored.score(X), model.score(X))


def test_best_threshold_hand_computed():
    scores = np.array([0.9, 0.8, 0.6, 0.4, 0.3])
    y = np.array([1, 1, 0, 1, 0])
    threshold = best_threshold(scores, y, ["a"] * 5, {"a": 3})
    assert threshold == pytest.approx(0.4)


def test_best_threshold_without_positive_labels_returns_one():
    scores = np.array([0.9, 0.4, 0.2])
    assert best_threshold(scores, np.zeros(3, dtype=int), ["a"] * 3, {"a": 3}) == 1.0


def test_leave_one_group_out_high_f1_on_synthetic_groups():
    X, y, groups = _synthetic_groups()
    truth_counts = {name: int(y[np.asarray(groups) == name].sum()) for name in set(groups)}
    result = leave_one_group_out(X, y, groups, truth_counts)
    assert set(result["per_group"]) == set(truth_counts)
    for name, row in result["per_group"].items():
        assert row["scores"] and len(row["scores"]) == int(np.sum(np.asarray(groups) == name))
        assert row["labelled"] == truth_counts[name]
    assert result["total"]["f1"] >= 0.9


def test_leave_one_group_out_includes_labelled_group_without_candidates():
    X, y, groups = _synthetic_groups(names=("v1", "v2", "v3"), per_group=10)
    truth_counts = {"v1": 5, "v2": 5, "v3": 5, "v4": 4}
    result = leave_one_group_out(X, y, groups, truth_counts)
    assert result["per_group"]["v4"]["predicted"] == 0
    assert result["per_group"]["v4"]["scores"] == []
    assert result["per_group"]["v4"]["labelled"] == 4
    assert result["total"]["labelled"] == 19


def test_train_handles_single_group_and_is_deterministic():
    X, y, groups = _synthetic_groups(names=("only",), per_group=10)
    truth_counts = {"only": int(y.sum())}
    model_a = train(X, y, groups, truth_counts, feature_names=["f0", "f1", "f2"])
    model_b = train(X, y, groups, truth_counts, feature_names=["f0", "f1", "f2"])
    assert model_a.threshold == model_b.threshold
    assert np.allclose(model_a.score(X), model_b.score(X))
    assert model_a.feature_names == ["f0", "f1", "f2"]


def test_leave_one_group_out_repeatable():
    X, y, groups = _synthetic_groups()
    truth_counts = {name: int(y[np.asarray(groups) == name].sum()) for name in set(groups)}
    first = leave_one_group_out(X, y, groups, truth_counts)
    second = leave_one_group_out(X, y, groups, truth_counts)
    assert first == second
