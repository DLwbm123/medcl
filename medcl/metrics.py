"""Pure NumPy metrics; no file access, inferred labels, or imputed results.

Ranges are a contiguous half-open partition of the sample axis. Matrix rows
and columns must already follow the protocol's task order. Scores are fractions
(not percentages); TRE spacing must convert coordinates to the protocol's mm.
"""

import math

import numpy as np


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _mean(values):
    values = [_finite(value, "score") for value in values]
    if not values:
        return None
    try:
        return _finite(math.fsum(value / len(values) for value in values), "mean")
    except OverflowError as exc:
        raise ValueError("mean exceeds the finite numeric range") from exc


def _direction(direction):
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be 'higher' or 'lower'")
    return 1 if direction == "higher" else -1


def _kind(kind):
    if kind not in ("classification", "segmentation", "registration"):
        raise ValueError("unknown task kind")


def score_cases(kind, prediction, target, case_ranges, classes=(), spacing=None):
    """Score each case; empty cases remain present with a None score.

    Segmentation averages exactly the explicit classes; callers include label
    0 when background belongs in the primary score. Dice uses eps=1e-5 in both
    numerator and denominator. Registration requires floating-point, already
    transformed fixed-space correspondences and explicit positive spacing.
    """
    _kind(kind)
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    allowed_ndim = {"classification": (1,), "segmentation": (3, 4), "registration": (3,)}
    if prediction.ndim not in allowed_ndim[kind]:
        raise ValueError(f"invalid {kind} array dimensions")
    expected_dtype = "f" if kind == "registration" else "iu"
    if prediction.dtype.kind not in expected_dtype or target.dtype.kind not in expected_dtype:
        raise ValueError("registration needs floats; classification/segmentation need integers")
    if any(size == 0 for size in prediction.shape[1:]):
        raise ValueError("spatial, landmark, and coordinate axes must be nonempty")
    if kind == "registration" and (not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("prediction and target must be finite")

    ranges = np.asarray(case_ranges, dtype=object)
    if ranges.shape == (0,):
        ranges = ranges.reshape(0, 2)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("case_ranges must contain (start, end) pairs")
    validated_ranges, previous = [], 0
    for start, end in ranges:
        start, end = _integer(start, "range start"), _integer(end, "range end")
        if start != previous or end < start or end > len(prediction):
            raise ValueError("case_ranges must form a contiguous, nonoverlapping partition")
        validated_ranges.append((start, end))
        previous = end
    if previous != len(prediction):
        raise ValueError("case_ranges must cover every sample exactly once")

    if kind == "segmentation":
        class_array = np.asarray(classes, dtype=object)
        if class_array.ndim != 1 or not len(class_array):
            raise ValueError("segmentation requires explicit nonempty classes")
        classes = tuple(_integer(label, "class ID") for label in class_array)
        if len(set(classes)) != len(classes):
            raise ValueError("class IDs must be unique")
    if kind == "registration":
        spacing_array = np.asarray(spacing, dtype=object)
        if spacing_array.shape != (prediction.shape[-1],):
            raise ValueError("registration requires one spacing value per coordinate axis")
        spacing = np.array([_finite(value, "spacing") for value in spacing_array])
        if np.any(spacing <= 0):
            raise ValueError("spacing must be positive")

    results = []
    for case_index, (start, end) in enumerate(validated_ranges):
        record = {"case_index": case_index, "n_samples": end - start, "score": None, "per_class": None}
        if start != end:
            predicted, expected = prediction[start:end], target[start:end]
            if kind == "classification":
                record["score"] = float(np.count_nonzero(predicted == expected) / (end - start))
            elif kind == "segmentation":
                per_class = {}
                for label in classes:
                    predicted_mask, expected_mask = predicted == label, expected == label
                    intersection = int(np.count_nonzero(predicted_mask & expected_mask))
                    denominator = int(np.count_nonzero(predicted_mask)) + int(np.count_nonzero(expected_mask))
                    per_class[str(label)] = (2 * intersection + 1e-5) / (denominator + 1e-5)
                record.update(score=_mean(per_class.values()), per_class=per_class)
            else:
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        delta = (predicted.astype(np.float64) - expected.astype(np.float64)) * spacing
                        distances = np.hypot.reduce(delta, axis=-1)
                except FloatingPointError as exc:
                    raise ValueError("TRE exceeds the finite numeric range") from exc
                record["score"] = _mean(distances.ravel())
        results.append(record)
    return results


def continual_summary(matrix, direction="higher", random_baseline=None, allow_unseen=False):
    """Summarize an ordered stage-by-task matrix without dropping missing stages.

    Forgetting excludes the final stage from its historical optimum and can be
    negative. BWTR is mean((final - acquisition) / acquisition), excluding the
    last task. FWT uses each task's immediately preceding stage and random
    baseline; task zero has no pre-acquisition stage and is not in FWT.
    """
    sign = _direction(direction)
    if not isinstance(allow_unseen, (bool, np.bool_)):
        raise ValueError("allow_unseen must be boolean")
    matrix = np.asarray(matrix, dtype=object)
    if matrix.shape == (0,):
        matrix = matrix.reshape(0, 0)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square, preserving every protocol stage")
    size = len(matrix)
    matrix = [[None if value is None else _finite(value, "matrix value") for value in row] for row in matrix]
    baseline = None
    if random_baseline is not None:
        baseline = np.asarray(random_baseline, dtype=object)
        if baseline.shape != (size,):
            raise ValueError("random_baseline must have one entry per task")
        baseline = [None if value is None else _finite(value, "random baseline") for value in baseline]

    result = {key: {"value": None, "reason": "没有任务"} for key in
              ("Final average", "BWT", "Forgetting", "FWT", "BWTR")}
    if size == 0:
        return result
    final = matrix[-1]
    result["Final average"] = {"value": None, "reason": "最终阶段缺少任务结果"}
    if all(value is not None for value in final):
        result["Final average"] = {"value": _mean(final), "reason": "最终阶段全部任务等权平均"}
    if size == 1:
        for key in ("BWT", "Forgetting", "FWT", "BWTR"):
            result[key]["reason"] = "至少需要两个任务"
        return result

    diagonal = [matrix[index][index] for index in range(size - 1)]
    endpoints_ready = all(value is not None for value in diagonal + final[:-1])
    result["BWT"]["reason"] = "缺少获得阶段或最终结果"
    if endpoints_ready:
        result["BWT"] = {"value": _mean(sign * (final[index] - diagonal[index]) for index in range(size - 1)),
                         "reason": "正值表示后续学习带来提升"}

    history = [matrix[stage][task] for task in range(size - 1) for stage in range(task, size - 1)]
    result["Forgetting"]["reason"] = "缺少完整已见历史或最终结果"
    if all(value is not None for value in history + final[:-1]):
        changes = []
        for task in range(size - 1):
            seen = [matrix[stage][task] for stage in range(task, size - 1)]
            best = max(seen) if sign == 1 else min(seen)
            changes.append(sign * (best - final[task]))
        result["Forgetting"] = {"value": _mean(changes), "reason": "正值为下降，负值为超过历史最佳"}

    if not allow_unseen:
        result["FWT"]["reason"] = "协议不允许未见任务评估"
    elif baseline is None or any(value is None for value in baseline[1:]):
        result["FWT"]["reason"] = "缺少待评估任务的随机初始化基线"
    elif any(matrix[task - 1][task] is None for task in range(1, size)):
        result["FWT"]["reason"] = "缺少学习前一阶段的未见任务结果"
    else:
        result["FWT"] = {"value": _mean(sign * (matrix[task - 1][task] - baseline[task]) for task in range(1, size)),
                         "reason": "正值表示优于随机初始化基线"}

    if sign != 1:
        result["BWTR"]["reason"] = "仅支持高值更优指标"
    elif not endpoints_ready:
        result["BWTR"]["reason"] = "缺少获得阶段或最终结果"
    elif any(value <= 0 for value in diagonal):
        result["BWTR"]["reason"] = "获得阶段分母必须大于0"
    else:
        result["BWTR"] = {"value": _mean((final[index] - diagonal[index]) / diagonal[index] for index in range(size - 1)),
                          "reason": "相对获得阶段的变化比例，非百分数"}
    return result


def federated_summary(client_scores, sample_counts, kind, direction="higher"):
    """Aggregate available clients; classification weighting uses sample counts.

    Per-client scores must already be protocol-macro scores, except that
    classification weighted accuracy requires each client's sample accuracy.
    Standard deviation is the population statistic (ddof=0).
    """
    _kind(kind)
    sign = _direction(direction)
    scores, counts = np.asarray(client_scores, dtype=object), np.asarray(sample_counts, dtype=object)
    if scores.ndim != 1 or counts.shape != scores.shape:
        raise ValueError("client_scores and sample_counts must be matching one-dimensional sequences")
    scores = [None if score is None else _finite(score, "client score") for score in scores]
    counts = [_integer(count, "sample count") for count in counts]
    if any(count < 0 for count in counts):
        raise ValueError("sample counts must be nonnegative")
    available = [(score, count) for score, count in zip(scores, counts) if score is not None and count > 0]
    result = {"client_macro": None, "sample_weighted_accuracy": None, "worst_client": None,
              "client_std": None, "client_gap": None, "available_clients": len(available), "total_clients": len(scores)}
    if not available:
        return result
    scores, counts = zip(*available)
    mean = _mean(scores)
    divisor = math.sqrt(len(scores))
    result.update(client_macro=mean, worst_client=min(scores) if sign == 1 else max(scores),
                  client_std=_finite(math.hypot(*(score / divisor - mean / divisor for score in scores)), "client std"),
                  client_gap=_finite(max(scores) - min(scores), "client gap"))
    if kind == "classification":
        total = sum(counts)
        result["sample_weighted_accuracy"] = _finite(math.fsum(score * (count / total) for score, count in zip(scores, counts)), "weighted accuracy")
    return result
