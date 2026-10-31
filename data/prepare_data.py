import os
import numpy as np
import pickle
import argparse
from collections import defaultdict


def _ratio_label(value):
    percent = value * 100
    if abs(percent - round(percent)) < 1e-8:
        return f'{int(round(percent))}%'
    return f'{percent:.2f}'.rstrip('0').rstrip('.') + '%'


def _split_dir_name(user_active_ratio, item_hot_ratio):
    return f'{_ratio_label(user_active_ratio)}active_{_ratio_label(item_hot_ratio)}hot'


def _save_processed(out_dir, user2item, item2user, user_list, item_list,
                    active_users, static_users, advantaged_items,
                    disadvantaged_items, user_map=None, item_map=None):
    os.makedirs(out_dir, exist_ok=True)

    pickle.dump(dict(user2item), open(os.path.join(out_dir, 'user2item.pkl'), 'wb'))
    pickle.dump(dict(item2user), open(os.path.join(out_dir, 'item2user.pkl'), 'wb'))
    pickle.dump(user_list, open(os.path.join(out_dir, 'user.pkl'), 'wb'))
    pickle.dump(item_list, open(os.path.join(out_dir, 'item.pkl'), 'wb'))
    pickle.dump(active_users, open(os.path.join(out_dir, 'active_user.pkl'), 'wb'))
    pickle.dump(static_users, open(os.path.join(out_dir, 'static_user.pkl'), 'wb'))
    pickle.dump(advantaged_items, open(os.path.join(out_dir, 'advantaged_item.pkl'), 'wb'))
    pickle.dump(disadvantaged_items, open(os.path.join(out_dir, 'disadvantaged_item.pkl'), 'wb'))
    if user_map is not None:
        pickle.dump(user_map, open(os.path.join(out_dir, 'user_map.pkl'), 'wb'))
    if item_map is not None:
        pickle.dump(item_map, open(os.path.join(out_dir, 'item_map.pkl'), 'wb'))


def _filter_users(user2item, item2user, user_list, item_list, min_interactions=3):
    user_set = {uid for uid, items in user2item.items() if len(items) >= min_interactions}
    item_set = set()
    for uid in user_set:
        item_set.update(user2item[uid])
    new_user2item = defaultdict(list)
    new_item2user = defaultdict(list)
    for uid in user_set:
        for iid in user2item[uid]:
            if iid in item_set:
                new_user2item[uid].append(iid)
                new_item2user[iid].append(uid)
    new_user_list = sorted(new_user2item.keys())
    new_item_list = sorted(new_item2user.keys())
    return new_user2item, new_item2user, new_user_list, new_item_list


def _build_groups(user2item, item2user, user_list, item_list,
                  user_active_ratio, item_hot_ratio):
    user_interactions = [(uid, len(user2item[uid])) for uid in user_list]
    user_interactions.sort(key=lambda x: x[1], reverse=True)
    active_cutoff = max(1, round(len(user_interactions) * user_active_ratio))
    active_users = [uid for uid, _ in user_interactions[:active_cutoff]]
    static_users = [uid for uid, _ in user_interactions[active_cutoff:]]

    item_interactions = [(iid, len(item2user[iid])) for iid in item_list]
    item_interactions.sort(key=lambda x: x[1], reverse=True)
    hot_cutoff = max(1, round(len(item_interactions) * item_hot_ratio))
    advantaged_items = [iid for iid, _ in item_interactions[:hot_cutoff]]
    disadvantaged_items = [iid for iid, _ in item_interactions[hot_cutoff:]]
    return active_users, static_users, advantaged_items, disadvantaged_items


def process_epinion(data_dir='data/Epinion', user_active_ratio=0.05,
                    item_hot_ratio=0.20):
    inters_file = os.path.join(data_dir, 'Epinion_inters.txt')
    data = np.loadtxt(inters_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    user2item = defaultdict(list)
    item2user = defaultdict(list)
    for row in data:
        uid = int(row[0])
        iid = int(row[1])
        user2item[uid].append(iid)
        item2user[iid].append(uid)

    user_list = sorted(user2item.keys())
    item_list = sorted(item2user.keys())
    user2item, item2user, user_list, item_list = _filter_users(
        user2item, item2user, user_list, item_list, min_interactions=3)

    active_users, static_users, advantaged_items, disadvantaged_items = _build_groups(
        user2item, item2user, user_list, item_list, user_active_ratio, item_hot_ratio)

    out_dir = os.path.join(data_dir, _split_dir_name(user_active_ratio, item_hot_ratio))
    _save_processed(out_dir, user2item, item2user, user_list, item_list,
                    active_users, static_users, advantaged_items, disadvantaged_items)

    print(f'Epinion: {len(user_list)} users, {len(item_list)} items')
    print(f'Active: {len(active_users)}, Static: {len(static_users)}')
    print(f'Hot: {len(advantaged_items)}, Cold: {len(disadvantaged_items)}')
    print(f'Saved to {out_dir}')


def process_amazon(dataset_name, data_dir, user_active_ratio=0.05,
                   item_hot_ratio=0.20):
    import gzip
    import json

    name_map = {
        'Beauty': 'reviews_Beauty_5.json.gz',
        'Health': 'reviews_Health_and_Personal_Care_5.json.gz',
        'Grocery': 'reviews_Grocery_and_Gourmet_Food_5.json.gz',
    }
    gz_name = name_map.get(dataset_name)
    if gz_name is None:
        print(f'No mapping for {dataset_name}')
        return

    gz_file = os.path.join(data_dir, dataset_name, gz_name)

    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    user_map = {}
    item_map = {}
    uc = 0
    ic = 0
    for entry in data:
        if entry['reviewerID'] not in user_map:
            user_map[entry['reviewerID']] = uc
            uc += 1
        if entry['asin'] not in item_map:
            item_map[entry['asin']] = ic
            ic += 1

    user2item = defaultdict(list)
    item2user = defaultdict(list)
    for entry in data:
        uid = user_map[entry['reviewerID']]
        iid = item_map[entry['asin']]
        user2item[uid].append(iid)
        item2user[iid].append(uid)

    user_list = list(range(uc))
    item_list = list(range(ic))
    user2item, item2user, user_list, item_list = _filter_users(
        user2item, item2user, user_list, item_list, min_interactions=3)

    active_users, static_users, advantaged_items, disadvantaged_items = _build_groups(
        user2item, item2user, user_list, item_list, user_active_ratio, item_hot_ratio)

    out_dir = os.path.join(data_dir, dataset_name,
                           _split_dir_name(user_active_ratio, item_hot_ratio))
    _save_processed(out_dir, user2item, item2user, user_list, item_list,
                    active_users, static_users, advantaged_items,
                    disadvantaged_items, user_map, item_map)

    print(f'{dataset_name}: {uc} users, {ic} items')
    print(f'Active: {len(active_users)}, Static: {len(static_users)}')
    print(f'Hot: {len(advantaged_items)}, Cold: {len(disadvantaged_items)}')


def _read_group_file(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def process_gowalla(data_dir='data/Gowalla',
                    source_file='Gowalla_inters_processed.txt',
                    user_active_ratio=0.05, item_hot_ratio=0.20):
    inters_file = os.path.join(data_dir, source_file)
    if not os.path.exists(inters_file):
        raise FileNotFoundError(f'Gowalla source file not found: {inters_file}')

    raw_pairs = []
    with open(inters_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            raw_pairs.append((int(parts[0]), int(parts[1])))

    user_ids = sorted({uid for uid, _ in raw_pairs})
    item_ids = sorted({iid for _, iid in raw_pairs})
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    user2item = defaultdict(list)
    item2user = defaultdict(list)
    seen = set()
    for raw_uid, raw_iid in raw_pairs:
        uid = user_map[raw_uid]
        iid = item_map[raw_iid]
        if (uid, iid) in seen:
            continue
        seen.add((uid, iid))
        user2item[uid].append(iid)
        item2user[iid].append(uid)

    n_users = len(user_ids)
    n_items = len(item_ids)
    user_list = list(range(n_users))
    item_list = list(range(n_items))
    user2item, item2user, user_list, item_list = _filter_users(
        user2item, item2user, user_list, item_list, min_interactions=3)

    active_raw = inactive_raw = hot_raw = cold_raw = None
    use_official_default_groups = (
        source_file == 'Gowalla_data.txt' and
        abs(user_active_ratio - 0.05) < 1e-8 and
        abs(item_hot_ratio - 0.20) < 1e-8
    )
    if use_official_default_groups:
        active_raw = _read_group_file(os.path.join(
            data_dir, 'groups', 'users', '005', 'active_ids.txt'))
        inactive_raw = _read_group_file(os.path.join(
            data_dir, 'groups', 'users', '005', 'inactive_ids.txt'))
        hot_raw = _read_group_file(os.path.join(
            data_dir, 'groups', 'items', '020', 'shorthead_items.txt'))
        cold_raw = _read_group_file(os.path.join(
            data_dir, 'groups', 'items', '020', 'longtail_items.txt'))

    if active_raw is not None and inactive_raw is not None:
        active_users = [user_map[uid] for uid in active_raw if uid in user_map]
        static_users = [user_map[uid] for uid in inactive_raw if uid in user_map]
        covered = set(active_users) | set(static_users)
        static_users.extend([uid for uid in user_list if uid not in covered])
    else:
        user_interactions = [(uid, len(user2item[uid])) for uid in user_list]
        user_interactions.sort(key=lambda x: x[1], reverse=True)
        active_cutoff = max(1, round(n_users * user_active_ratio))
        active_users = [uid for uid, _ in user_interactions[:active_cutoff]]
        static_users = [uid for uid, _ in user_interactions[active_cutoff:]]

    if hot_raw is not None and cold_raw is not None:
        advantaged_items = [item_map[iid] for iid in hot_raw if iid in item_map]
        disadvantaged_items = [item_map[iid] for iid in cold_raw if iid in item_map]
        covered = set(advantaged_items) | set(disadvantaged_items)
        disadvantaged_items.extend([iid for iid in item_list if iid not in covered])
    else:
        item_interactions = [(iid, len(item2user[iid])) for iid in item_list]
        item_interactions.sort(key=lambda x: x[1], reverse=True)
        hot_cutoff = max(1, round(n_items * item_hot_ratio))
        advantaged_items = [iid for iid, _ in item_interactions[:hot_cutoff]]
        disadvantaged_items = [iid for iid, _ in item_interactions[hot_cutoff:]]

    out_dir = os.path.join(data_dir, _split_dir_name(user_active_ratio, item_hot_ratio))
    _save_processed(out_dir, user2item, item2user, user_list, item_list,
                    active_users, static_users, advantaged_items,
                    disadvantaged_items, user_map, item_map)

    print(f'Gowalla: {n_users} users, {n_items} items, {len(seen)} interactions')
    print(f'Active: {len(active_users)}, Static: {len(static_users)}')
    print(f'Hot: {len(advantaged_items)}, Cold: {len(disadvantaged_items)}')
    print(f'Saved to {out_dir}')


def process_demo(data_dir='data', user_active_ratio=0.05, item_hot_ratio=0.20):
    import csv
    csv_path = os.path.join(data_dir, 'demo', 'interactions.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Demo data not found: {csv_path}')

    raw_user2item = defaultdict(list)
    raw_item2user = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            if uid not in raw_user2item or iid not in raw_user2item[uid]:
                raw_user2item[uid].append(iid)
                raw_item2user[iid].append(uid)

    user_list = sorted(raw_user2item.keys())
    item_list = sorted(raw_item2user.keys())
    user_map = {uid: idx for idx, uid in enumerate(user_list)}
    item_map = {iid: idx for idx, iid in enumerate(item_list)}

    user2item = defaultdict(list)
    item2user = defaultdict(list)
    for uid in user_list:
        for iid in raw_user2item[uid]:
            new_uid = user_map[uid]
            new_iid = item_map[iid]
            user2item[new_uid].append(new_iid)
            item2user[new_iid].append(new_uid)

    user_list = sorted(user2item.keys())
    item_list = sorted(item2user.keys())
    user2item, item2user, user_list, item_list = _filter_users(
        user2item, item2user, user_list, item_list, min_interactions=3)

    active_users, static_users, advantaged_items, disadvantaged_items = _build_groups(
        user2item, item2user, user_list, item_list, user_active_ratio, item_hot_ratio)

    out_dir = os.path.join(data_dir, 'demo',
                           _split_dir_name(user_active_ratio, item_hot_ratio))
    _save_processed(out_dir, user2item, item2user, user_list, item_list,
                    active_users, static_users, advantaged_items, disadvantaged_items)
    print(f'Demo: {len(user_list)} users, {len(item_list)} items')
    print(f'Active: {len(active_users)}, Static: {len(static_users)}')
    print(f'Hot: {len(advantaged_items)}, Cold: {len(disadvantaged_items)}')
    print(f'Saved to {out_dir}')


def prepare_dataset(dataset_name, data_dir='data', user_active_ratio=0.05,
                    item_hot_ratio=0.20, **kwargs):
    if dataset_name == 'Epinion':
        process_epinion(os.path.join(data_dir, 'Epinion'),
                        user_active_ratio, item_hot_ratio)
    elif dataset_name == 'Gowalla':
        process_gowalla(os.path.join(data_dir, 'Gowalla'),
                        kwargs.get('gowalla_source', 'Gowalla_inters_processed.txt'),
                        user_active_ratio, item_hot_ratio)
    elif dataset_name == 'demo':
        process_demo(data_dir, user_active_ratio, item_hot_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare DA2Fair datasets')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--datasets', nargs='+',
                        default=['Epinion', 'Gowalla', 'Beauty', 'Health', 'Grocery'],
                        choices=['Epinion', 'Gowalla', 'Beauty', 'Health', 'Grocery', 'demo'])
    parser.add_argument('--user_active_ratio', type=float, default=0.05)
    parser.add_argument('--item_hot_ratio', type=float, default=0.20)
    parser.add_argument('--gowalla_source', type=str,
                        default='Gowalla_inters_processed.txt')
    args = parser.parse_args()

    if not (0 < args.user_active_ratio < 1 and 0 < args.item_hot_ratio < 1):
        raise ValueError('Ratios must be in (0, 1)')

    if 'Epinion' in args.datasets:
        process_epinion(os.path.join(args.data_dir, 'Epinion'),
                        args.user_active_ratio, args.item_hot_ratio)
    if 'Gowalla' in args.datasets:
        process_gowalla(os.path.join(args.data_dir, 'Gowalla'),
                        args.gowalla_source, args.user_active_ratio,
                        args.item_hot_ratio)
    if 'demo' in args.datasets:
        process_demo(args.data_dir, args.user_active_ratio, args.item_hot_ratio)
    for ds in ['Beauty', 'Health', 'Grocery']:
        if ds in args.datasets:
            process_amazon(ds, args.data_dir, args.user_active_ratio,
                           args.item_hot_ratio)

