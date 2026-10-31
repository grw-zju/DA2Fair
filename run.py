import os
import sys
import yaml
import json
import time
import copy
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_utils import load_dataset
from models.trainer import run_experiment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'configs', 'default.yaml')


def _deep_update(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    if path == 'configs/default.yaml':
        path = DEFAULT_CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device(gpu_id=0):
    if gpu_id < 0:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='DA2Fair: Distribution Alignment and Nash Equilibrium for Two-Sided Fairness-aware Recommendation')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        choices=['Epinion', 'Gowalla', 'demo'])
    parser.add_argument('--backbone', type=str, default='LightGCN',
                        choices=['MF', 'NeuMF', 'VAECF', 'LightGCN',
                                 'mf', 'neumf', 'vaecf', 'lightgcn'])
    parser.add_argument('--method', type=str, default='Standard',
                        choices=['Standard', 'DA2Fair', 'CPFair', 'MultiFR',
                                 'Ada2Fair', 'FAIR', 'FairSort', 'Ablation'])
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--lambda_fair', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--emb_dim', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--n_neg', type=int, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--user_active_ratio', type=float, default=None)
    parser.add_argument('--item_hot_ratio', type=float, default=None)
    parser.add_argument('--align_sides', type=str, default='both',
                        choices=['user', 'item', 'both'])
    parser.add_argument('--align_terms', type=str, default='GL',
                        choices=['G', 'L', 'GL'])
    parser.add_argument('--use_uniform_marginal', action='store_true')
    parser.add_argument('--grad_coordinator', type=str, default='nemo',
                        choices=['equal', 'nash_mtl', 'nemo'])
    parser.add_argument('--track_nemo_decisions', action='store_true')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--output_suffix', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    model_cfg = config.get('model', {})
    da2fair_cfg = config.get('da2fair', {})
    eval_cfg = config.get('evaluation', {})
    baseline_cfg = config.get('baseline', {})

    epochs = args.epochs if args.epochs is not None else model_cfg.get('max_epochs', 200)
    lr = args.lr if args.lr is not None else model_cfg.get('learning_rate', 1e-4)
    emb_dim = args.emb_dim if args.emb_dim is not None else model_cfg.get('embedding_dim', 64)
    batch_size = args.batch_size if args.batch_size is not None else model_cfg.get('batch_size', 2048)
    K = args.K if args.K is not None else eval_cfg.get('top_k', 10)
    n_neg = args.n_neg if args.n_neg is not None else model_cfg.get('n_neg', 99)
    wd = args.wd if args.wd is not None else model_cfg.get('weight_decay', 1e-5)
    user_ratio = args.user_active_ratio if args.user_active_ratio is not None else da2fair_cfg.get('user_active_ratio', 0.05)
    item_ratio = args.item_hot_ratio if args.item_hot_ratio is not None else da2fair_cfg.get('item_hot_ratio', 0.20)
    lambda_fair = args.lambda_fair if args.lambda_fair is not None else baseline_cfg.get(args.method.lower(), {}).get('lambda_fair', 0.005)

    device = get_device(args.gpu)
    print(f"Using device: {device}")

    if args.seeds and len(args.seeds) > 1:
        import subprocess
        from scripts.run_multi_seed import aggregate_runs
        aggregate_results = []
        for model_seed in args.seeds:
            print(f"\n=== Running seed {model_seed} ===")
            cmd = [sys.executable, os.path.abspath(__file__),
                   '--dataset', args.dataset,
                   '--backbone', args.backbone,
                   '--method', args.method,
                   '--config', args.config,
                   '--seed', str(model_seed),
                   '--split_seed', str(args.split_seed),
                   '--gpu', str(args.gpu),
                   '--results_dir', args.results_dir,
                   '--save_dir', args.save_dir]
            for flag, val in [('--epochs', epochs), ('--lr', lr), ('--emb_dim', emb_dim),
                              ('--batch_size', batch_size), ('--K', K), ('--n_neg', n_neg),
                              ('--wd', wd), ('--lambda_fair', lambda_fair),
                              ('--user_active_ratio', user_ratio),
                              ('--item_hot_ratio', item_ratio)]:
                cmd.extend([flag, str(val)])
            if args.use_uniform_marginal:
                cmd.append('--use_uniform_marginal')
            if args.track_nemo_decisions:
                cmd.append('--track_nemo_decisions')
            if args.align_sides != 'both':
                cmd.extend(['--align_sides', args.align_sides])
            if args.align_terms != 'GL':
                cmd.extend(['--align_terms', args.align_terms])
            if args.grad_coordinator != 'nemo':
                cmd.extend(['--grad_coordinator', args.grad_coordinator])
            subprocess.run(cmd, check=True)

            result_path = os.path.join(args.results_dir, args.dataset, args.backbone,
                                       args.method, f'seed_{model_seed}.json')
            with open(result_path, 'r') as f:
                aggregate_results.append(json.load(f))

        summary = aggregate_runs(aggregate_results)
        summary.update({
            'dataset': args.dataset,
            'backbone': args.backbone,
            'method': args.method,
            'seeds': args.seeds,
        })
        summary_path = os.path.join(args.results_dir, args.dataset, args.backbone,
                                    args.method, 'aggregate.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nMulti-seed summary saved to {summary_path}")
        return

    backbone_map = {'mf': 'MF', 'neumf': 'NeuMF', 'vaecf': 'VAECF', 'lightgcn': 'LightGCN'}
    args.backbone = backbone_map.get(args.backbone.lower(), args.backbone)

    print(f"\n===== {args.dataset} | {args.backbone} | {args.method} =====")
    result = run_experiment(
        dataset_name=args.dataset,
        model_name=args.backbone,
        method=args.method,
        data_dir='data',
        epochs=epochs,
        lr=lr,
        emb_dim=emb_dim,
        batch_size=batch_size,
        K=K,
        n_neg=n_neg,
        seed=args.seed,
        wd=wd,
        device=device,
        user_active_ratio=user_ratio,
        item_hot_ratio=item_ratio,
        lambda_fair=lambda_fair,
        align_sides=args.align_sides,
        align_terms=args.align_terms,
        use_uniform_marginal=args.use_uniform_marginal,
        grad_coordinator=args.grad_coordinator,
        track_nemo_decisions=args.track_nemo_decisions,
    )

    result['dataset'] = args.dataset
    result['backbone'] = args.backbone
    result['method'] = args.method
    result['seed'] = args.seed
    result['split_seed'] = args.split_seed

    run_results_dir = os.path.join(args.results_dir, args.dataset, args.backbone, args.method)
    os.makedirs(run_results_dir, exist_ok=True)
    results_path = os.path.join(run_results_dir, f'seed_{args.seed}{args.output_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nFinal Results:")
    print(f"  NDCG@10: All={result['NDCG_all']:.4f}, Adv={result['NDCG_adv']:.4f}, Dis={result['NDCG_dis']:.4f}")
    print(f"  DUF={result['DUF']:.4f}")
    print(f"  Cov={result['Cov']:.4f}, Hot={result['Hot']:.4f}, Cold={result['Cold']:.4f}")
    print(f"  DIF={result['DIF']:.6f}")
    if 'Delta' in result:
        print(f"  Delta(%)={result['Delta']:.2f}")
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
