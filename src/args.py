import argparse
import yaml

def parse_args():
    # 1) Define your CLI flags (with sensible defaults)
    parser = argparse.ArgumentParser(description="Siamese training")
    parser.add_argument("--config", type=str, help="path to YAML config file")
    parser.add_argument("--epochs",      type=int,   default=100,    help="number of epochs")
    parser.add_argument("--batch_size",  type=int,   default=256,    help="batch size")
    parser.add_argument("--lr",          type=float, default=1e-3,   help="learning rate")
    parser.add_argument("--margin",      type=float, default=1.0,    help="triplet margin")
    parser.add_argument("--alpha",       type=int,   default=50,     help="Gaussian width")
    parser.add_argument("--scheduler_stepsize", type=int, default=100, help="LR step size")
    parser.add_argument("--device",      type=str,   default="cuda", help="cpu or cuda")
    parser.add_argument("--folds",       type=int,   default=5,      help="CV folds")
    parser.add_argument("--seed",        type=int,   default=42,     help="random seed")
    parser.add_argument("--spectra_path",type=str,   default="data/",help="where .npz lives")
    parser.add_argument("--pca_components", type=int,default=30,     help="PCA dims")
    parser.add_argument("--hard_weight", type=float, default=2.0,   help="hard‐mining weight")
    # …add whatever else you need…

    args = parser.parse_args()

    # 2) If a YAML was provided, load it and overwrite
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    return args
