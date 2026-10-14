import argparse
import os
import torch
import sys
import numpy as np
from data_prior_config import get_prior_config
from tqdm import tqdm

def str2bool(value):
    return value.lower() == "true"

def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def parse_argv(argv):
    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--dataset_type", type=str, choices=["tabiclv1", "tabiclv2", "mothernet"], help="Type of dataset to generate",
                         default="tabiclv1")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=256, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=1, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument(
        "--binary_class_p",
        type=float,
        default=0.8,
        help="Fraction of generated tasks that are binary (2 classes); the rest get 2..max_classes classes. Default 0.8.",
    )
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--padding", default=False, type=str2bool, help="Whether padding to max features")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size", type=train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=train_size_type, default=0.9, help="Maximum training size position/ratio"
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="prior_bag",
        choices=["prior_bag", "prior_bag_plus_tree", "tree_only"],
        help="Type of prior to use",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )
    parser.add_argument("--dataset_config_path", type=str, default=None,
                        help="Path to JSON config file for tabiclv2 dataset generation")
    parser.add_argument("--delete_after_load", default=False, type=str2bool,
                        help="Delete batch files after loading during training")
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = parse_argv(sys.argv[1:])
    os.makedirs(args.save_dir, exist_ok=True)

    # Set seeds
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.torch_seed)

    if args.dataset_type == "tabiclv1" or args.dataset_type == "mothernet":
        from prior.tabiclv1.dataloader import generate_one_batch
        overall_config = get_prior_config(mlp_prior_type=args.dataset_type, max_features=args.max_features)
        # Build the prior directly (same logic as get_dataloader, but without PriorDataLoader wrapper)
        prior_config = overall_config['prior']
        prior_config['num_features'] = args.max_features
        prior_config['batch_size_per_subgroup'] = args.batch_size_per_gp
        prior_config['classification']['binary_class_p'] = args.binary_class_p
        worker_config = {
                    'prior_config': prior_config,
                    'prior_type': args.prior_type,
                    'batch_size_per_gp': args.batch_size_per_gp,
                    'padding': args.padding,
                    'batch_size': args.batch_size,
                    'max_features': args.max_features,
                    'device': args.device,
                    'save_dir': args.save_dir,
                    'min_seq_len': args.min_seq_len,
                    'max_seq_len': args.max_seq_len,
                    'log_seq_len': args.log_seq_len,
                    'min_train_size': args.min_train_size,
                    'max_train_size': args.max_train_size,
                    'max_retries': 20,
                }
    
        n_jobs = args.n_jobs
        if n_jobs > 1 and args.device == 'cpu':
            import joblib
            print(f"Using joblib with n_jobs={n_jobs} for parallel generation")
            results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                joblib.delayed(generate_one_batch)(i, worker_config)
                for i in range(args.resume_from, args.num_batches)
            )
            total_skipped = sum(1 for _, shape in results if shape is None)
            for batch_idx, shape in results[:3]:
                if shape is not None:
                    print(f"[batch {batch_idx}] x.shape={shape}")
            print(f"Done! Saved batches to {args.save_dir}")
            if total_skipped:
                print(f"Skipped: {total_skipped}")
        else:
            # Single-threaded fallback
            for batch_idx in tqdm(range(args.num_batches), desc="Generating"):
                batch_idx, shape = generate_one_batch(batch_idx, worker_config)
    

    elif args.dataset_type == "tabiclv2":
        from prior.tabiclv2.dataloader import generate_one_batch_tabiclv2
        if args.dataset_config_path is None:
            raise ValueError("--dataset_config_path is required for tabiclv2")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset_config_path = os.path.join(cur_dir, args.dataset_config_path)
        print(f"Generating {args.num_batches} batches (batch_size={args.batch_size}, "
                f"n_jobs={args.n_jobs})")
        print(f"Dataset config: {args.dataset_config_path}")
        print(f"Saving to {args.save_dir}")

        worker_config = {
            'dataset_config_path': args.dataset_config_path,
            'save_dir': args.save_dir,
            'seed': args.np_seed,
            'max_retries': 20,
        }

        if args.n_jobs > 1:
            import joblib
            print(f"Using joblib with n_jobs={args.n_jobs} for parallel generation")
            try:
                results = joblib.Parallel(n_jobs=args.n_jobs, backend='multiprocessing')( # loky
                    joblib.delayed(generate_one_batch_tabiclv2)(i, worker_config)
                    for i in range(args.resume_from, args.num_batches)
                )
            finally:
                # Ensure workers are cleaned up on macOS
                import gc; gc.collect()
            total_skipped = sum(1 for _, size in results if size is None)
            for batch_idx, size in results[:5]:
                if size is not None:
                    print(f"[batch {batch_idx:06d}] valid datasets: {size}")
            print(f"Done! Saved {args.num_batches - args.resume_from - total_skipped} batches to {args.save_dir}")
            if total_skipped:
                print(f"Skipped: {total_skipped} batches")
        else:
            from tqdm import tqdm
            for batch_idx in tqdm(range(args.resume_from, args.num_batches), desc="Generating"):
                idx, size = generate_one_batch_tabiclv2(batch_idx, worker_config)
                if batch_idx == args.resume_from and size is not None:
                    print(f"[batch {batch_idx:06d}] valid datasets: {size}")
            print(f"Done! Saved to {args.save_dir}")
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")