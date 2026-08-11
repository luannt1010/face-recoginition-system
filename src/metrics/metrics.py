import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def classification_metrics(y_true, y_preds):
    p = precision_score(y_true, y_preds, average="macro", zero_division=0.0)
    r = recall_score(y_true, y_preds, average="macro", zero_division=0.0)
    f1 = f1_score(y_true, y_preds, average="macro", zero_division=0.0)
    acc = accuracy_score(y_true, y_preds)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}

def create_verification_pairs(embeddings, labels, num_samples=2_000, seed=42):
    rng = random.Random(seed)
    labels_np = np.asarray(labels)
    # num_samples = min(num_samples, len(labels_np))
    # selected_indices = list(range(len(labels_np)))
    # rng.shuffle(selected_indices)
    # selected_indices = selected_indices[:num_samples]
    positive_pairs, negative_pairs = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            # original_i = selected_indices[i]
            # original_j = selected_indices[j]
            if labels_np[i] == labels_np[j]:
                positive_pairs.append((i, j))
            else:
                negative_pairs.append((i, j))

    rng.shuffle(positive_pairs)
    rng.shuffle(negative_pairs)

    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    pair_labels = np.array([1]*len(positive_pairs) + [0]*len(negative_pairs), dtype=np.int64)
    all_pairs = positive_pairs + negative_pairs
    left_indices = [left for left, _ in all_pairs]
    right_indices = [right for _, right in all_pairs]
    return embeddings[left_indices], embeddings[right_indices], pair_labels


def create_verification_pairs_v2(embeddings, labels, seed=42):
    rng = random.Random(seed)
    labels_np = np.asarray(labels)  
    positive_pairs, negative_pairs = [], []
    all_indices = np.arange(len(labels_np))
    for i in range(len(labels_np)):
        pos_indices = all_indices[(labels_np==labels_np[i]) & (all_indices!=i)]
        neg_indices = all_indices[(labels_np!=labels_np[i])]
        if len(pos_indices) > 0:
            pos_idx = rng.choice(pos_indices)
            positive_pairs.append((i, pos_idx))
        if len(neg_indices) > 0:
            neg_idx = rng.choice(neg_indices)
            negative_pairs.append((i, neg_idx))

    rng.shuffle(positive_pairs)
    rng.shuffle(negative_pairs)

    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    pair_labels = np.array([1]*len(positive_pairs) + [0]*len(negative_pairs), dtype=np.int64)
    all_pairs = positive_pairs + negative_pairs
    left_indices = [left for left, _ in all_pairs]
    right_indices = [right for _, right in all_pairs]
    return embeddings[left_indices], embeddings[right_indices], pair_labels

def create_verification_pairs_v3(embeddings, labels, seed=42):
    rng = random.Random(seed)
    labels_np = np.asarray(labels)
    indices_by_label = defaultdict(list)
    for idx, label in enumerate(labels_np):
        indices_by_label[label].append(idx)
    unique_labels = list(indices_by_label.keys())
    positive_pairs = []
    negative_pairs = []
    for i, current_label in enumerate(labels_np):
        # Positive candidates thuộc cùng class, loại chính i
        same_class_indices = indices_by_label[current_label]
        if len(same_class_indices) > 1:
            pos_idx = rng.choice(same_class_indices)
            while pos_idx == i:
                pos_idx = rng.choice(same_class_indices)
            positive_pairs.append((i, pos_idx))

        # Chọn một class khác trước, rồi chọn sample trong class đó
        if len(unique_labels) > 1:
            negative_label = rng.choice(unique_labels)
            while negative_label == current_label:
                negative_label = rng.choice(unique_labels)
            neg_idx = rng.choice(indices_by_label[negative_label])
            negative_pairs.append((i, neg_idx))

    rng.shuffle(positive_pairs)
    rng.shuffle(negative_pairs)

    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    pair_labels = np.array([1]*len(positive_pairs) + [0]*len(negative_pairs), dtype=np.int64)
    all_pairs = positive_pairs + negative_pairs
    left_indices = [left for left, _ in all_pairs]
    right_indices = [right for _, right in all_pairs]
    return embeddings[left_indices], embeddings[right_indices], pair_labels

def l2_normalize(embedding, eps=1e-6):
    return np.divide(embedding, np.maximum(np.linalg.norm(embedding, axis=-1, keepdims=True), eps))

def calculate_cosine_similarity(emb1, emb2, normalize=True):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    if normalize:
        emb1 = l2_normalize(emb1)
        emb2 = l2_normalize(emb2)
    return np.sum(emb1 * emb2, axis=-1)

def calculate_euclid_distance(emb1, emb2, normalize=True):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    if normalize:
        emb1 = l2_normalize(emb1)
        emb2 = l2_normalize(emb2)
    return np.sqrt(np.sum((emb1 - emb2)**2, keepdims=True, axis=-1)).reshape(-1)

def calculate_tp_tn_fp_fn(cosine_scores, pair_labels, threshold, eps=1e-6):
    cosine_scores = np.asarray(cosine_scores)
    pair_labels = np.asarray(pair_labels, dtype=bool)
    scores = (cosine_scores >= threshold)

    tp = np.logical_and(pair_labels, scores).sum().item()
    tn = np.logical_and(~pair_labels, ~scores).sum().item()
    fp = np.logical_and(~pair_labels, scores).sum().item()
    fn = np.logical_and(pair_labels, ~scores).sum().item()
    return tp, tn, fp, fn

def calculate_verification_metrics(cosine_scores, pair_labels, threshold, eps=1e-6):
    tp, tn, fp, fn = calculate_tp_tn_fp_fn(cosine_scores, pair_labels, threshold, eps)

    far = fp / (fp + tn + eps)
    frr = fn / (fn + tp + eps)

    err = (far + frr) / 2
    tar = 1 - frr

    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = (2 * p * r) / (p + r + eps)

    return {"FAR": far, "FRR": frr, "ERR": err, "TAR": tar, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1}

def calculate_tar_at_far(metrics, thresholds, target_far=0.01):
    tar = metrics[:, 3]
    far = metrics[:, 0]
    valid_indices = np.where(far<=target_far)[0]
    valid_far = far[valid_indices]
    best_index = valid_indices[np.argmax(valid_far)]
    return {"threshold@target_far": thresholds[best_index].item(), "far@target_far": far[best_index].item(), "tar@target_far": tar[best_index].item()}

def calculate_eer(metrics, thresholds):
    far = metrics[:, 0]
    frr = metrics[:, 1]
    best_idx = np.argmin(np.abs(far - frr))
    eer = (far[best_idx] + frr[best_idx]) / 2
    return {"threshold@eer": thresholds[best_idx].item(), "far@eer": far[best_idx].item(), "frr@eer": frr[best_idx], "eer": eer}

def calculate_auc(metrics):
    tar = metrics[:, 3][::-1]
    far = metrics[:, 0][::-1]
    auc = np.trapezoid(tar, far)
    return auc

def compute_verify_acc_p_r_f1(metrics):
    results = []
    far = metrics[:, 0]
    frr = metrics[:, 1]
    best_idx = np.argmin(np.abs(far - frr))
    for i in [4, 5, 6, 7]:
        results.append(metrics[best_idx, i])
    return results

def plot_roc_curve(metrics, title="ROC Curve"):
    far = metrics[:, 0]
    tar = metrics[:, 3]
    # ROC cần đi theo chiều FAR tăng dần
    sorted_indices = np.argsort(far)
    far_sorted = far[sorted_indices]
    tar_sorted = tar[sorted_indices]
    auc = calculate_auc(metrics)

    plt.figure(figsize=(7, 6))

    plt.plot(far_sorted, tar_sorted, linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (TAR)")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1.01)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return float(auc)

def calculate_verification_metrics_multi_thresholds(cosine_scores, pair_labels, num_thresholds: int=400, eps=1e-6):
    thresholds = np.linspace(cosine_scores.min(), cosine_scores.max(), num_thresholds)
    metrics = np.zeros((num_thresholds, 8))
    for i in range(num_thresholds):
        results = calculate_verification_metrics(cosine_scores, pair_labels, thresholds[i], eps)
        for j, key in enumerate(results.keys()):
            metrics[i][j] = results[key]
    return metrics, thresholds

def verification_metrics_report(embeddings, labels, num_thresholds: int=400, target_at_far=0.01):
    embedding_left, embedding_right, pair_labels = create_verification_pairs_v3(embeddings, labels)
    scores = calculate_cosine_similarity(embedding_left, embedding_right)
    metrics, thresholds = calculate_verification_metrics_multi_thresholds(scores, pair_labels, num_thresholds)
    tar_at_far = calculate_tar_at_far(metrics, thresholds, target_at_far)
    metrics_at_eer = calculate_eer(metrics, thresholds)
    acc_p_r_f1 = compute_verify_acc_p_r_f1(metrics)
    return {"scores": scores, "pair_labels": pair_labels, "thresholds": thresholds, "metrics": metrics, "tar_at_far": tar_at_far,
            "metrics_at_eer": metrics_at_eer, "auc": calculate_auc(metrics), "acc_p_r_f1": acc_p_r_f1}
