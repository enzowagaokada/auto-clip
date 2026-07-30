"""Classification metrics for the imbalanced clip classifier.

Accuracy is intentionally omitted — it is misleading under class imbalance.
"""

import numpy as np


def confusion_counts(labels, preds, threshold=0.5):
    pred_pos = preds >= threshold
    label_pos = labels >= 0.5

    tp = int(np.sum(pred_pos & label_pos))
    fp = int(np.sum(pred_pos & ~label_pos))
    fn = int(np.sum(~pred_pos & label_pos))
    tn = int(np.sum(~pred_pos & ~label_pos))
    return tp, fp, fn, tn


def precision_recall_f1(labels, preds, threshold=0.5):
    tp, fp, fn, _ = confusion_counts(labels, preds, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_auc(labels, preds):
    """AUC via rank statistic (Mann-Whitney U). Threshold-independent."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n_pos = int(np.sum(labels >= 0.5))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(preds, kind="mergesort")
    ranks = np.empty(len(preds), dtype=np.float64)
    ranks[order] = np.arange(1, len(preds) + 1)

    # Average ranks for ties.
    sorted_preds = preds[order]
    i = 0
    while i < len(sorted_preds):
        j = i
        while j + 1 < len(sorted_preds) and sorted_preds[j + 1] == sorted_preds[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    sum_ranks_pos = np.sum(ranks[labels >= 0.5])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(labels, preds):
    """Average precision, a prevalence-aware summary of the PR curve."""
    labels = np.asarray(labels) >= 0.5
    preds = np.asarray(preds)
    n_pos = int(np.sum(labels))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-preds, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    precision_at_rank = true_positives / ranks
    return float(np.sum(precision_at_rank[sorted_labels]) / n_pos)


def find_best_threshold(labels, preds, minimum=0.05, maximum=0.95, steps=181):
    """Choose the F1-maximizing threshold, breaking ties with precision."""
    best_threshold = 0.5
    best_pair = None
    for threshold in np.linspace(minimum, maximum, steps):
        precision, _, f1 = precision_recall_f1(
            labels,
            preds,
            threshold=float(threshold),
        )
        pair = (f1, precision)
        if best_pair is None or pair > best_pair:
            best_threshold = float(threshold)
            best_pair = pair
    return best_threshold, evaluate(labels, preds, threshold=best_threshold)


def evaluate(labels, preds, threshold=0.5):
    """Return a dict of all metrics for a set of predictions."""
    tp, fp, fn, tn = confusion_counts(labels, preds, threshold)
    precision, recall, f1 = precision_recall_f1(labels, preds, threshold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(labels, preds),
        "average_precision": average_precision(labels, preds),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def format_metrics(m):
    c = m["confusion"]
    return (
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"AUC={m['auc']:.3f} AP={m['average_precision']:.3f} | "
        f"TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}"
    )
