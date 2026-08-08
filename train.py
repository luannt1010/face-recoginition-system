import argparse
import torch
from torch.utils.data import DataLoader
from src import (FaceDataset, SubsetFaceDataset, create_data_splits, create_loss, define_transform,
                 load_model, plot_history, train)


def get_args():
    parser = argparse.ArgumentParser(description="Train AdaFace ResNet Encoder")
    # Paths
    parser.add_argument("--train_dir", type=str, default=r".\datasets\webface\webface_112x112")
    parser.add_argument("--test_dir", type=str, default=r".\datasets\webface\webface_112x112")
    parser.add_argument("--save_path", type=str, default=r".\checkpoints\final")
    # Dataset
    parser.add_argument("--val_factor", type=float, default=0.3)
    # DataLoader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    # Model
    parser.add_argument("--model_type", choices=("base", "mobile", "iresnet"), type=str, default="mobile")
    parser.add_argument("--model_size", type=int, default=18)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    # Loss
    parser.add_argument("--loss_type", choices=("ada", "arc", "triplet"), type=str, default="arc")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--t_alpha", type=float, default=0.01)
    # Training
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_thresholds", type=int, default=400)
    parser.add_argument("--target_at_far", type=float, default=0.01)
    return parser.parse_args()

def main():
    args = get_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    save_path = args.save_path
    val_factor = args.val_factor
    batch_size = args.batch_size
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor
    model_type = args.model_type
    model_size = args.model_size
    embedding_dim = args.embedding_dim
    dropout_rate = args.dropout_rate
    loss_type = args.loss_type
    margin = args.margin
    scale = args.scale
    t_alpha = args.t_alpha
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    num_thresholds = args.num_thresholds
    target_at_far = args.target_at_far
    resume = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{model_type.title()} model is training on {device} with {loss_type.title()} Loss.")

    train_transform, val_transform = define_transform()
    train_orig_dataset = FaceDataset(root_dir=train_dir)
    test_dataset = FaceDataset(root_dir=test_dir, transform=val_transform)
    train_dataset, val_dataset = create_data_splits(train_orig_dataset, val_factor=val_factor)
    train_dataset = SubsetFaceDataset(train_dataset, train_transform)
    val_dataset = SubsetFaceDataset(val_dataset, val_transform)
    print(f" Size of train dataset: {len(train_dataset):,}")
    print(f" Size of val dataset: {len(val_dataset):,}")
    print(f" Size of test dataset: {len(test_dataset):,}")
    print(f" Total train classes: {len(train_orig_dataset.classes)}")
    print(f" Total test classes: {len(test_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor)
    print("Create dataloader successfully!")

    model = load_model(model_type, model_size, embedding_dim, dropout_rate, None)
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = create_loss(loss_type, num_classes=len(train_orig_dataset.classes), embedding_dim=embedding_dim,
                            margin=margin, scale=scale, t_alpha=t_alpha)
    optimizer = torch.optim.AdamW(params=list(model.parameters()) + list(criterion.parameters()),
                                  lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    history = train(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, resume=resume,
                    save_path=save_path, scheduler=scheduler, device=device, num_thresholds=num_thresholds, target_at_far=target_at_far)
    plot_history(history, save_path)


if __name__ == "__main__":
    main()

