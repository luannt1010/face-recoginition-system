import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.configs import MODEL_MAP, LOSS_MAP
from src.dataset import SubsetFaceDataset
from src.metrics import (calculate_cosine_similarity, classification_metrics,
                         l2_normalize, verification_metrics_report)

def load_model(model_type="iresnet", model_size=18, embedding_dim=512, dropout_rate=0.3, sd_path=None):
    if model_type not in MODEL_MAP:
        choices = ", ".join(MODEL_MAP)
        raise ValueError(f"Unknown model type {model_type!r}. Choose one of: {choices}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "iresnet":
        model = MODEL_MAP[model_type](model_size=model_size, embedding_dim=embedding_dim, dropout=dropout_rate)
    elif model_type == "base":
        model = MODEL_MAP[model_type](embedding_dim=embedding_dim, dropout=dropout_rate)
    elif model_type == "mobile":
        model = MODEL_MAP[model_type](embedding_dim=embedding_dim)
    if sd_path is not None:
        sd = torch.load(sd_path, map_location=device)
        model.load_state_dict(sd["model"])
    return model.to(device)

def create_loss(loss_type, num_classes, embedding_dim=512, margin=0.4, scale=64.0, t_alpha=0.01):
    if loss_type not in LOSS_MAP:
        choices = ", ".join(LOSS_MAP)
        raise ValueError(f"Unknown loss type {loss_type!r}. Choose one of: {choices}")
    if loss_type == "ada":
        return LOSS_MAP[loss_type](num_classes=num_classes, embedding_dim=embedding_dim, m=margin, s=scale, t_alpha=t_alpha)
    if loss_type == "arc":
        return LOSS_MAP[loss_type]( num_classes=num_classes, embedding_dim=embedding_dim, m=margin, s=scale)
    return LOSS_MAP[loss_type](margin=margin)

def create_data_splits(dataset, val_factor):
    length = len(dataset)
    val_size = int(length * val_factor)
    train_size = length - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def crop_face(img, bbox=None, show=True, return_numpy=False):
    if bbox is None:
        from src.detection import FaceDetector
        results = FaceDetector().detect(img)
        bbox = results["bbox"]
    elif bbox is not None:
        bbox = np.asarray(bbox).astype(int)
    else:
        raise ValueError("BBox is invalid.")

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    else:
        img = Image.fromarray(img).convert("RGB")
    img_cropped = img.crop(bbox)
    if show:
        img_cropped.show()
    return img_cropped if not return_numpy else np.asarray(img_cropped)

def define_transform():
    train_transform = transforms.Compose([transforms.Resize((112, 112)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
                                          transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
                                          transforms.RandomGrayscale(p=0.05), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return train_transform, val_transform

def evaluate(model, loss_fn, loader, device, test=False, num_thresholds=400, target_at_far=0.01, epsilon=1e-6):
    model.eval()
    loss_fn.eval()
    running_loss = 0
    all_preds = []
    all_labels = []
    all_embeddings = []
    pbar = tqdm(loader, desc=f"[EVALUATING]", leave=False if not test else True)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            embedding = model(images)
            loss = loss_fn(embedding, labels)
            running_loss += loss.item()

            w = loss_fn.W.detach().cpu().numpy()
            embedding_np = embedding.detach().cpu().numpy()
            w_norm = l2_normalize(w, epsilon)
            emb_norm = l2_normalize(embedding_np, epsilon)
            preds = np.argmax(np.dot(emb_norm, w_norm.T), axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_embeddings.append(embedding.detach().cpu())

    total_loss = running_loss / len(loader)
    classify_metrics = classification_metrics(all_labels, all_preds)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    verify_metrics = verification_metrics_report(all_embeddings, all_labels, num_thresholds=num_thresholds,
                                                 target_at_far=target_at_far)
    return total_loss, classify_metrics, verify_metrics

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, num_thresholds=400, target_at_far=0.01, epsilon=1e-6):
    model.train()
    loss_fn.train()
    train_running_loss = 0
    train_preds, train_labels = [], []
    all_embeddings = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embedding = model(images)
        loss = loss_fn(embedding, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            w = loss_fn.W.detach().cpu().numpy()
            embedding_np = embedding.detach().cpu().numpy()
            w_norm = l2_normalize(w, epsilon)
            emb_norm = l2_normalize(embedding_np, epsilon)
            preds = np.argmax(np.dot(emb_norm, w_norm.T), axis=1)
            train_preds.extend(preds.tolist())
            train_labels.extend(labels.detach().cpu().tolist())
            all_embeddings.append(embedding.detach().cpu())
        train_running_loss += loss.item()
    train_epoch_loss = train_running_loss / len(train_loader)
    train_metrics = classification_metrics(train_labels, train_preds)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    verify_metrics = verification_metrics_report(all_embeddings, train_labels, num_thresholds=num_thresholds,
                                                 target_at_far=target_at_far)
    return train_epoch_loss, train_metrics, verify_metrics

def create_empty_his():
    return {
        "train_loss": [], "val_loss": [],
        "val_classify_results": [], "train_classify_results": [],

        "train_eer_results": [], "train_thresholds": [],
        "train_tar_at_far_results": [], "train_raw_results": [],
        "train_auc": [], "train_verify_acc_p_r_f1": [],

        "val_eer_results": [], "val_thresholds": [],
        "val_tar_at_far_results": [], "val_raw_results": [],
        "val_auc": [], "val_verify_acc_p_r_f1": [],

        "learning_rate": [],
        "total_time": 0
    }

def save_history(history, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_history(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return create_empty_his()

def train(model, train_loader, val_loader, test_loader, epochs, optimizer, loss_fn, save_path, device, scheduler,
          num_thresholds: int=400, target_at_far=0.01, epsilon=1e-6, resume=False, start_epoch=0, best_score=float("-inf")):
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "checkpoints")
    report_path = os.path.join(save_path, "reports")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)

    best_save_path = os.path.join(checkpoint_path, "best.pth")
    last_save_path = os.path.join(checkpoint_path, "last.pth")
    his_save_path = os.path.join(report_path, "history.json")
    test_res_save_path = os.path.join(report_path, "test_results.json")

    history = load_history(his_save_path) if resume else create_empty_his()

    for epoch in range(start_epoch, epochs):
        start = time.perf_counter()
        #====================TRAINING====================
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        train_epoch_loss, train_classify_metrics, train_verify_metrics = train_one_epoch(model, train_pbar, optimizer,
                                                                                         loss_fn, device, num_thresholds, target_at_far, epsilon)

        #====================VALIDATING====================
        val_epoch_loss, val_classify_metrics, val_verify_metrics = evaluate(model, loss_fn, val_loader, device, test=False,
                                                                            num_thresholds=num_thresholds, target_at_far=target_at_far, epsilon=epsilon)

        end = time.perf_counter()
        epoch_time = (end - start) / 60
        history["total_time"] += epoch_time

        # ==========EXTRACT TRAIN METRICS==========
        train_tar_at_far = train_verify_metrics["tar_at_far"]
        train_eer_results = train_verify_metrics["metrics_at_eer"]
        train_auc = train_verify_metrics["auc"]
        train_eer = train_eer_results["eer"]
        train_v_acc, train_v_p, train_v_r, train_v_f1 = train_verify_metrics["acc_p_r_f1"]

        train_acc = train_classify_metrics["accuracy"]
        train_precision = train_classify_metrics["precision"]
        train_recall = train_classify_metrics["recall"]
        train_f1 = train_classify_metrics["f1"]

        #==========EXTRACT VAL METRICS==========
        val_tar_at_far = val_verify_metrics["tar_at_far"]
        val_eer_results = val_verify_metrics["metrics_at_eer"]
        val_auc = val_verify_metrics["auc"]
        val_eer = val_eer_results["eer"]
        val_v_acc, val_v_p, val_v_r, val_v_f1 = val_verify_metrics["acc_p_r_f1"]

        val_acc = val_classify_metrics["accuracy"]
        val_precision = val_classify_metrics["precision"]
        val_recall = val_classify_metrics["recall"]
        val_f1 = val_classify_metrics["f1"]

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{epochs} - {epoch_time:.4f}m: TrLoss={train_epoch_loss:.4f} | ValLoss={val_epoch_loss:.4f} - LR={current_lr}")


        print(f"    - Classification: (TrAcc={train_acc:.4f} TrP={train_precision:.4f} TrR={train_recall:.4f} TrF1={train_f1:.4f} | ValAcc={val_acc:.4f} ValP={val_precision:.4f} ValR={val_recall:.4f} ValF1={val_f1:.4f})")
        print(f"    - Verification:\n"
              f"        - Train: (TAR@FAR{target_at_far}={train_tar_at_far['tar@target_far']:.4f} FAR={train_tar_at_far['far@target_far']:.4f} Threshold={train_tar_at_far['threshold@target_far']:.4f} | EER={train_eer:.4f} Threshold={train_eer_results['threshold@eer']:.4f} AUC={train_auc:.4f} | Acc={train_v_acc:.4f} P={train_v_p:.4f} R={train_v_r:.4f} F1={train_v_f1:.4f}) - NumPairs={len(train_verify_metrics['pair_labels'])}\n"
              f"        - Val:   (TAR@FAR{target_at_far}={val_tar_at_far['tar@target_far']:.4f} FAR={val_tar_at_far['far@target_far']:.4f} Threshold={val_tar_at_far['threshold@target_far']:.4f} | EER={val_eer:.4f} Threshold={val_eer_results['threshold@eer']:.4f} AUC={val_auc:.4f} | Acc={val_v_acc:.4f} P={val_v_p:.4f} R={val_v_r:.4f} F1={val_v_f1:.4f}) - NumPairs={len(val_verify_metrics['pair_labels'])}")


        # Save Train metrics
        history["train_auc"].append(train_auc)
        history["train_eer_results"].append(train_eer_results)
        history["train_tar_at_far_results"].append(train_tar_at_far)
        history["train_thresholds"].append(train_verify_metrics["thresholds"].tolist())
        history["train_raw_results"].append(train_verify_metrics["metrics"].tolist())
        history["train_verify_acc_p_r_f1"].append([train_v_acc, train_v_p, train_v_r, train_v_f1])

        history["train_loss"].append(train_epoch_loss)
        history["train_classify_results"].append([train_acc, train_precision, train_recall, train_f1])

        # Save Val metrics
        history["val_auc"].append(val_auc)
        history["val_eer_results"].append(val_eer_results)
        history["val_tar_at_far_results"].append(val_tar_at_far)
        history["val_thresholds"].append(val_verify_metrics["thresholds"].tolist())
        history["val_raw_results"].append(val_verify_metrics["metrics"].tolist())
        history["val_verify_acc_p_r_f1"].append([val_v_acc, val_v_p, val_v_r, val_v_f1])

        history["val_loss"].append(val_epoch_loss)
        history["val_classify_results"].append([val_acc, val_precision, val_recall, val_f1])

        history["learning_rate"].append(current_lr)

        is_best = val_tar_at_far["tar@target_far"] > best_score
        if is_best:
            best_score = val_tar_at_far["tar@target_far"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step()

        checkpoints = {"model": model.state_dict(),
                       "loss_fn": loss_fn.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict() if scheduler is not None else None,
                       "epoch": epoch, "best_score": best_score,
                       "train_indices": train_loader.dataset.subset.indices,
                       "val_indices": val_loader.dataset.subset.indices,}

        save_history(history, his_save_path)
        if is_best:
            torch.save(checkpoints, best_save_path)
        torch.save(checkpoints, last_save_path)

    print("History is saved!")
    print("Training completed!\n")

    best_checkpoint = torch.load(best_save_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    loss_fn.load_state_dict(best_checkpoint["loss_fn"])
    test_loss, test_classify_metrics, test_verify_metrics = evaluate(model, loss_fn, loader=test_loader, epsilon=epsilon,
                                                                     test=True, device=device, num_thresholds=num_thresholds, target_at_far=target_at_far)
    # Verification metrics
    test_tar_at_far_results = test_verify_metrics["tar_at_far"]
    test_eer_results = test_verify_metrics["metrics_at_eer"]
    test_auc = test_verify_metrics["auc"]
    # Raw verification curve
    test_thresholds = test_verify_metrics["thresholds"].tolist()
    test_raw_results = test_verify_metrics["metrics"].tolist()
    # Verification Acc / Precision / Recall / F1
    test_verify_acc, test_verify_precision, test_verify_recall, test_verify_f1 = test_verify_metrics["acc_p_r_f1"]

    test_history = {"test_loss": test_loss, "test_classify_results": test_classify_metrics,
                    "test_eer_results": test_eer_results, "test_tar_at_far_results": test_tar_at_far_results,
                    "test_thresholds": test_thresholds, "test_raw_results": test_raw_results,"test_auc": test_auc,
                    "test_verify_acc_p_r_f1": [test_verify_acc, test_verify_precision, test_verify_recall, test_verify_f1]}
    save_history(test_history, test_res_save_path)
    print("Test results are saved!")
    print("Testing completed!")
    return history

def plot_train_val(ax, epochs, train_values, val_values, title, ylabel, best="max",
                   train_label="Train", val_label="Validation", annotate=True):

    train_values = np.asarray(train_values)
    val_values = np.asarray(val_values)
    ax.plot(epochs, train_values, label=train_label)
    ax.plot(epochs, val_values, label=val_label)

    if annotate:
        if best == "max":
            idx = np.argmax(val_values)
        else:
            idx = np.argmin(val_values)
        best_epoch = epochs[idx]
        best_value = val_values[idx]
        ax.scatter(best_epoch, best_value, s=50, zorder=5)
        ax.annotate(f"Best Val\nEpoch {best_epoch}: {best_value:.4f}",
                    xy=(best_epoch, best_value), xytext=(20, 20), color="red",
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()

def plot_history(history, sp=None):

    train_classify_results = np.asarray(history["train_classify_results"])
    val_classify_results = np.asarray(history["val_classify_results"])
    train_tar_at_far_results = history["train_tar_at_far_results"]
    val_tar_at_far_results = history["val_tar_at_far_results"]
    train_eer_results = history["train_eer_results"]
    val_eer_results = history["val_eer_results"]
    train_verify_acc_p_r_f1 = np.asarray(history["train_verify_acc_p_r_f1"])
    val_verify_acc_p_r_f1 = np.asarray(history["val_verify_acc_p_r_f1"])

    p_classify_train = train_classify_results[:, 1]
    p_classify_val = val_classify_results[:, 1]
    r_classify_train = train_classify_results[:, 2]
    r_classify_val = val_classify_results[:, 2]
    f1_classify_train = train_classify_results[:, 3]
    f1_classify_val = val_classify_results[:, 3]
    train_classify_acc = train_classify_results[:, 0]
    val_classify_acc = val_classify_results[:, 0]

    p_verify_train = train_verify_acc_p_r_f1[:, 1]
    p_verify_val = val_verify_acc_p_r_f1[:, 1]
    r_verify_train = train_verify_acc_p_r_f1[:, 2]
    r_verify_val = val_verify_acc_p_r_f1[:, 2]
    f1_verify_train = train_verify_acc_p_r_f1[:, 3]
    f1_verify_val = val_verify_acc_p_r_f1[:, 3]
    train_verify_acc = train_verify_acc_p_r_f1[:, 0]
    val_verify_acc = val_verify_acc_p_r_f1[:, 0]

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    train_eers, val_eers = [], []
    for tr_res, val_res in zip(train_eer_results, val_eer_results):
        train_eers.append(tr_res["eer"])
        val_eers.append(val_res["eer"])
    tr_fars, val_fars, tr_tars, val_tars = [], [], [], []
    for tr_res, val_res in zip(train_tar_at_far_results, val_tar_at_far_results):
        tr_fars.append(tr_res["far@target_far"])
        tr_tars.append(tr_res["tar@target_far"])
        val_fars.append(val_res["far@target_far"])
        val_tars.append(val_res["tar@target_far"])
    train_auc = history["train_auc"]
    val_auc = history["val_auc"]
    learning_rate = history["learning_rate"]
    figs = []
    epochs = [i + 1 for i in range(len(train_loss))]

    # 1. LOSS + LEARNING RATE
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_train_val(ax[0], epochs, train_loss, val_loss, title="Training & Validation Loss",
                   ylabel="Loss", best="min", train_label="Train Loss", val_label="Val Loss")
    ax[1].plot(epochs, learning_rate, label="Learning Rate")
    ax[1].set_title("Learning Rate Schedule")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Learning Rate")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    figs.append(("loss_and_lr", fig))

    # 2. CLASSIFICATION METRICS
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    plot_train_val(ax[0, 0], epochs, train_classify_acc, val_classify_acc,
                   title="Classification Accuracy", ylabel="Accuracy")
    plot_train_val(ax[0, 1], epochs, p_classify_train, p_classify_val,
                   title="Classification Precision", ylabel="Precision")
    plot_train_val(ax[1, 0], epochs, r_classify_train, r_classify_val,
                   title="Classification Recall", ylabel="Recall")
    plot_train_val(ax[1, 1], epochs, f1_classify_train, f1_classify_val,
                   title="Classification F1-Score", ylabel="F1-Score")

    fig.suptitle("Classification Metrics", fontsize=16)
    fig.tight_layout()
    figs.append(("classification_metrics", fig))

    # 3. VERIFICATION ACC / P / R / F1
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    plot_train_val(ax[0, 0], epochs, train_verify_acc, val_verify_acc,
                   title="Verification Accuracy", ylabel="Accuracy")
    plot_train_val(ax[0, 1], epochs, p_verify_train, p_verify_val,
                   title="Verification Precision", ylabel="Precision")
    plot_train_val(ax[1, 0], epochs, r_verify_train, r_verify_val,
                   title="Verification Recall", ylabel="Recall")
    plot_train_val(ax[1, 1], epochs, f1_verify_train, f1_verify_val,
                   title="Verification F1-Score", ylabel="F1-Score")

    fig.suptitle("Verification Classification Metrics", fontsize=16)
    fig.tight_layout()
    figs.append(("verification_classification_metrics", fig))

    # 4. EER + AUC
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_train_val(ax[0], epochs, train_eers, val_eers, title="Equal Error Rate (EER)", ylabel="EER", best="min",
                   train_label="Train EER", val_label="Val EER")
    plot_train_val(ax[1], epochs, train_auc, val_auc, title="ROC AUC", ylabel="AUC", best="max",
                   train_label="Train AUC", val_label="Val AUC")

    fig.suptitle("Verification Performance", fontsize=16)
    fig.tight_layout()
    figs.append(("eer_auc", fig))

    # 5. FAR + TAR
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_train_val(ax[0], epochs, tr_fars, val_fars, title="FAR at Target FAR", ylabel="FAR",
                   train_label="Train FAR", val_label="Val FAR", annotate=False)
    plot_train_val(ax[1], epochs, tr_tars, val_tars, title="TAR at Target FAR", ylabel="TAR", best="max",
                   train_label="Train TAR", val_label="Val TAR")

    fig.suptitle( "TAR / FAR Performance", fontsize=16)
    fig.tight_layout()
    figs.append(("tar_far", fig))

    # SAVE
    if sp is not None:
        report_path = os.path.join(sp, "reports")
        os.makedirs(report_path, exist_ok=True)
        for name, fig in figs:
            fig.savefig(os.path.join( report_path, f"{name}.png"), dpi=300, bbox_inches="tight")

    plt.show()

def extract_embedding(model, img, bbox=None, show=True, crop=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if crop:
        pil_img = crop_face(img=img, show=show, bbox=bbox)
    else:
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        else:
            pil_img = Image.fromarray(img).convert("RGB")
        if show:
            pil_img.show()
    img = trans(pil_img)
    img = img.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        embedding = model(img.unsqueeze(0)).cpu()
    return l2_normalize(embedding).squeeze(0).numpy()

def face_verification(img_path1, img_path2, model, show=True):
    embedding1 = extract_embedding(model, img_path1, None, show, True)
    embedding2 = extract_embedding(model, img_path2, None, show, True)
    result = calculate_cosine_similarity(embedding1, embedding2, normalize=False)
    return result, embedding1, embedding2

def export_onnx(model, sp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 112, 112, device=device)
    model.eval()
    torch.onnx.export(model, dummy_input, sp, export_params=True, opset_version=17, do_constant_folding=True, external_data=False,
                      input_names=["images"], output_names=["predictions"], dynamic_axes={"images": {0: "batch_size"}, "predictions": {0: "batch_size"}})

