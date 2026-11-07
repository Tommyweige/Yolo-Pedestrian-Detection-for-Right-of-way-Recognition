"""Learn a regularised logistic scorer for candidate events, with leave-one-group-out validation.

Only numpy and the standard library are used. The data sets are tiny (about 100 rows,
20 features, 15 positives), so full-batch deterministic optimisation is enough. There is
no randomness anywhere: the same inputs always produce the same model.
"""
import numpy as np

EXTREME_LOGIT = 10.0


def _groups_in_order(groups):
    """Unique groups in first-appearance order (groups are hashable strings or ints)."""
    ordered, seen = [], set()
    for group in groups:
        if group not in seen:
            seen.add(group)
            ordered.append(group)
    return ordered


def _as_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _sigmoid(z):
    """Numerically stable sigmoid: never evaluates exp() on a large positive value."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        return float(_sigmoid(z.reshape(1))[0])
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _sigmoid_derivative(z):
    """sigma(z) * sigma(-z), computed without overflow for very large |z|."""
    e = np.exp(-np.abs(np.asarray(z, dtype=float)))
    return e / (1.0 + e) ** 2


def label_candidates(times, truth_times, tolerance):
    """Greedy one-to-one matching of candidates to manual labels.

    Every (candidate, truth) pair within ``tolerance`` is a candidate pair. Pairs are
    consumed in order of increasing distance, then candidate index, then truth index.
    A matched candidate is labelled 1, everything else 0.
    """
    times = np.asarray(times, dtype=float).ravel()
    truth_times = np.asarray(truth_times, dtype=float).ravel()
    labels = np.zeros(times.size, dtype=int)
    if times.size == 0 or truth_times.size == 0:
        return labels

    pairs = []
    for candidate_index, time in enumerate(times):
        for truth_index, truth_time in enumerate(truth_times):
            distance = abs(time - truth_time)
            if distance <= tolerance:
                # Rounding makes mathematically equal distances (e.g. 1.2-1.1 vs
                # 1.1-1.0) compare equal instead of being decided by float noise.
                pairs.append((round(distance, 12), candidate_index, truth_index))
    pairs.sort()

    used_candidates, used_truths = set(), set()
    for _, candidate_index, truth_index in pairs:
        if candidate_index in used_candidates or truth_index in used_truths:
            continue
        used_candidates.add(candidate_index)
        used_truths.add(truth_index)
        labels[candidate_index] = 1
    return labels


class Scaler:
    """Column-wise standardiser; constant columns use std=1 so they become 0, not NaN."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

    def fit(self, X):
        X = _as_2d(X)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X):
        return (_as_2d(X) - self.mean) / self.std


def fit_logistic(X, y, l2=1.0, iterations=500, class_weight="balanced"):
    """Fit an L2-regularised logistic model with a deterministic damped Newton (IRLS) loop.

    ``loss = weighted mean logistic loss + 0.5 * l2 * ||w||^2 / n`` (the intercept is not
    regularised). Returns ``(w, b)``. Degenerate labels return zero weights and an extreme
    intercept instead of raising.
    """
    X = _as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    n, n_features = X.shape

    n_pos = int(np.sum(y >= 0.5))
    n_neg = n - n_pos
    if n_pos == 0:
        return np.zeros(n_features), -EXTREME_LOGIT
    if n_neg == 0:
        return np.zeros(n_features), EXTREME_LOGIT

    if class_weight == "balanced":
        positive_weight = n / (2.0 * n_pos)
        negative_weight = n / (2.0 * n_neg)
    elif class_weight is None:
        positive_weight = negative_weight = 1.0
    else:
        raise ValueError("class_weight must be 'balanced' or None")

    weights = np.where(y >= 0.5, positive_weight, negative_weight)
    signs = np.where(y >= 0.5, 1.0, -1.0)

    design = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(n_features + 1)
    regularisation = float(l2) / n
    regularised_diagonal = np.eye(n_features + 1)
    regularised_diagonal[n_features, n_features] = 0.0  # the intercept is not penalised

    def objective(value):
        margin = design @ value
        data_term = np.sum(weights * np.logaddexp(0.0, -signs * margin)) / n
        penalty = 0.5 * float(l2) * np.dot(value[:n_features], value[:n_features]) / n
        return data_term + penalty

    current = objective(theta)
    for _ in range(int(iterations)):
        margin = design @ theta
        probability = _sigmoid(-signs * margin)
        gradient = -(design.T @ (weights * signs * probability)) / n
        gradient[:n_features] += regularisation * theta[:n_features]

        curvature = weights * _sigmoid_derivative(margin)
        hessian = (design.T * curvature) @ design / n
        hessian += regularisation * regularised_diagonal
        hessian += np.eye(n_features + 1) * 1e-10  # keeps a saturated Hessian solvable

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)) or np.linalg.norm(step) < 1e-12:
            break

        # Backtracking keeps Newton stable; every iteration remains deterministic.
        slope = float(np.dot(gradient, step))
        accepted = False
        alpha = 1.0
        for _ in range(30):
            candidate = theta - alpha * step
            candidate_objective = objective(candidate)
            if np.isfinite(candidate_objective) and candidate_objective <= current - 1e-4 * alpha * slope:
                theta = candidate
                current = candidate_objective
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    return theta[:n_features].copy(), float(theta[n_features])


def predict_proba(X, w, b):
    """Sigmoid of ``X @ w + b``; stable for logits of any magnitude."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    w = np.asarray(w, dtype=float).ravel()
    return _sigmoid(X @ w + float(b))


class Model:
    """A fitted scorer: scaler + logistic weights + decision threshold + feature names."""

    def __init__(self, scaler, w, b, threshold, feature_names):
        self.scaler = scaler
        self.w = np.asarray(w, dtype=float).ravel()
        self.b = float(b)
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)

    def score(self, X):
        return predict_proba(self.scaler.transform(X), self.w, self.b)

    def to_dict(self):
        return {
            "mean": np.asarray(self.scaler.mean, dtype=float).tolist(),
            "std": np.asarray(self.scaler.std, dtype=float).tolist(),
            "w": self.w.tolist(),
            "b": float(self.b),
            "threshold": float(self.threshold),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, d):
        scaler = Scaler(mean=d["mean"], std=d["std"])
        return cls(
            scaler,
            np.asarray(d["w"], dtype=float),
            float(d["b"]),
            float(d["threshold"]),
            list(d["feature_names"]),
        )


def best_threshold(scores, y, groups, truth_counts):
    """Event-level F1-maximising threshold; ties go to the larger (more conservative) value."""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    labelled = int(sum(truth_counts.values()))
    if scores.size == 0 or np.sum(y >= 0.5) == 0:
        return 1.0

    best_value, best_f1 = 1.0, -np.inf
    for threshold in np.unique(scores)[::-1]:
        selected = scores >= threshold
        true_positives = float(np.sum(y[selected] >= 0.5))
        predicted = float(np.sum(selected))
        f1 = 2.0 * true_positives / (predicted + labelled)
        if f1 > best_f1:
            best_value, best_f1 = float(threshold), f1
    return best_value


def _fit_scaled(X, y, l2):
    """Fit a scaler on X, then logistic weights on the standardised X."""
    scaler = Scaler().fit(X)
    w, b = fit_logistic(scaler.transform(X), y, l2=l2)
    return scaler, w, b


def train(X, y, groups, truth_counts, l2=1.0, feature_names=None):
    """Train the final model, picking its threshold from inner leave-one-group-out scores."""
    X = _as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups, dtype=object).tolist()

    unique_groups = _groups_in_order(groups)
    if len(unique_groups) <= 1:
        scaler, w, b = _fit_scaled(X, y, l2)
        threshold = best_threshold(predict_proba(scaler.transform(X), w, b), y, groups, truth_counts)
    else:
        out_of_fold = np.empty(X.shape[0], dtype=float)
        group_array = np.asarray(groups, dtype=object)
        for group in unique_groups:
            held_out = group_array == group
            training = ~held_out
            fold_scaler, fold_w, fold_b = _fit_scaled(X[training], y[training], l2)
            out_of_fold[held_out] = predict_proba(fold_scaler.transform(X[held_out]), fold_w, fold_b)
        threshold = best_threshold(out_of_fold, y, groups, truth_counts)

        scaler, w, b = _fit_scaled(X, y, l2)

    names = [] if feature_names is None else list(feature_names)
    return Model(scaler, w, b, threshold, names)


def leave_one_group_out(X, y, groups, truth_counts, l2=1.0):
    """Leave one whole video (group) out, train on the rest, and score the held-out video."""
    X = _as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups, dtype=object).tolist()
    group_array = np.asarray(groups, dtype=object)

    ordered_groups = _groups_in_order(groups)
    for group in truth_counts:
        if group not in ordered_groups:
            ordered_groups.append(group)

    per_group = {}
    total_predicted = total_true_positives = total_labelled = 0
    for group in ordered_groups:
        held_out = group_array == group
        training = ~held_out
        other_truth = {key: value for key, value in truth_counts.items() if key != group}

        if np.any(training):
            model = train(X[training], y[training], group_array[training].tolist(), other_truth, l2=l2)
        else:
            model = train(X, y, groups, truth_counts, l2=l2)

        if np.any(held_out):
            scores = np.asarray(model.score(X[held_out]), dtype=float)
            hits = scores >= model.threshold
            predicted = int(np.sum(hits))
            true_positives = int(np.sum((y[held_out] >= 0.5) & hits))
        else:
            scores = np.empty(0, dtype=float)
            predicted = true_positives = 0

        labelled = int(truth_counts.get(group, 0))
        per_group[group] = {
            "predicted": predicted,
            "true_positive": true_positives,
            "labelled": labelled,
            "scores": scores.tolist(),
        }
        total_predicted += predicted
        total_true_positives += true_positives
        total_labelled += labelled

    denominator = total_predicted + total_labelled
    total = {
        "predicted": total_predicted,
        "true_positive": total_true_positives,
        "labelled": total_labelled,
        "precision": total_true_positives / total_predicted if total_predicted else None,
        "recall": total_true_positives / total_labelled if total_labelled else None,
        "f1": 2.0 * total_true_positives / denominator if denominator else None,
    }
    return {"per_group": per_group, "total": total}
