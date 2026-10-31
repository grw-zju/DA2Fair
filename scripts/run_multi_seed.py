import json
import numpy as np


def aggregate_runs(results_list):
    if not results_list:
        return {}

    metric_keys = [k for k in results_list[0]
                   if isinstance(results_list[0][k], (int, float))]
    summary = {}
    for key in metric_keys:
        values = [r.get(key, 0) for r in results_list if key in r]
        if values:
            summary[key + '_mean'] = float(np.mean(values))
            summary[key + '_std'] = float(np.std(values))
            summary[key + '_values'] = values

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate multi-seed results')
    parser.add_argument('--input', nargs='+', required=True,
                        help='Paths to seed result JSON files')
    parser.add_argument('--output', type=str, default='aggregate.json')
    args = parser.parse_args()

    results = []
    for path in args.input:
        with open(path, 'r') as f:
            results.append(json.load(f))

    summary = aggregate_runs(results)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved to {args.output}')
