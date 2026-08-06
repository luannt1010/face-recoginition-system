import argparse
import torch
from src.utils import face_verification, load_model


def get_args():
    parser = argparse.ArgumentParser(description="Verify two faces with an embedding model")
    parser.add_argument("--img_path1", type=str, required=True)
    parser.add_argument("--img_path2", type=str, required=True)
    parser.add_argument("--cp_path", type=str, default=r".\checkpoints\final\checkpoints\best.pth")
    parser.add_argument("--model_type", choices=("iresnet", "base", "mobile"), default="mobile")
    parser.add_argument("--model_size", choices=(18, 34, 50, 100, 200), default=18, type=int)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--mode", choices=("cosine", "euclid"), default="cosine")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    model = load_model(model_type=args.model_type, model_size=args.model_size, embedding_dim=args.embedding_dim, 
                       dropout_rate=args.dropout_rate, sd_path=args.cp_path)
    print("Checkpoint loaded successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is inferring on {device}.")

    cosine, _, _ = face_verification(img_path1=args.img_path1, img_path2=args.img_path2, model=model, mode=args.mode, show=args.show)
    cosine = cosine.item()
    if cosine >= args.threshold:
        print(f"Same identity: score={cosine:.4f}, threshold={args.threshold}")
    else:
        print(f"Different identities: score={cosine:.4f}, threshold={args.threshold}")

if __name__ == "__main__":
    main()
