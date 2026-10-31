import numpy as np
import pickle
import os
from sklearn.metrics import ndcg_score
import torch


class Dataset:
    def __init__(self, dataset_name, data_dir='data', n_neg=99, neg_per_pos=4,
                 seed=42, user_active_ratio=0.05, item_hot_ratio=0.20):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.n_neg = n_neg
        self.neg_per_pos = neg_per_pos
        self.seed = seed

        base_dir = os.path.join(
            data_dir, dataset_name,
            self._split_dir_name(user_active_ratio, item_hot_ratio))
        if not os.path.exists(base_dir):
            raise FileNotFoundError(
                f'Processed data not found: {base_dir}. Run data/prepare_data.py '
                f'with --user_active_ratio {user_active_ratio} '
                f'--item_hot_ratio {item_hot_ratio}.')
        self.user2item = pickle.load(open(os.path.join(base_dir, 'user2item.pkl'), 'rb'))
        self.item2user = pickle.load(open(os.path.join(base_dir, 'item2user.pkl'), 'rb'))
        self.user_list = pickle.load(open(os.path.join(base_dir, 'user.pkl'), 'rb'))
        self.item_list = pickle.load(open(os.path.join(base_dir, 'item.pkl'), 'rb'))
        self.active_users = pickle.load(open(os.path.join(base_dir, 'active_user.pkl'), 'rb'))
        self.static_users = pickle.load(open(os.path.join(base_dir, 'static_user.pkl'), 'rb'))
        self.advantaged_items = pickle.load(open(os.path.join(base_dir, 'advantaged_item.pkl'), 'rb'))
        self.disadvantaged_items = pickle.load(open(os.path.join(base_dir, 'disadvantaged_item.pkl'), 'rb'))

        self.n_users = len(self.user_list)
        self.n_items = len(self.item_list)

        self.adv_users = set(self.active_users)
        self.dis_users = set(self.static_users)
        self.hot_items = set(self.advantaged_items)
        self.cold_items = set(self.disadvantaged_items)

        split_dir = os.path.join(base_dir, f'neg{n_neg}_npp{neg_per_pos}_seed{seed}')
        os.makedirs(split_dir, exist_ok=True)

        if os.path.exists(os.path.join(split_dir, 'test_nega.pkl')):
            self.train_data = pickle.load(open(os.path.join(split_dir, 'train.pkl'), 'rb'))
            self.val_data = pickle.load(open(os.path.join(split_dir, 'vali.pkl'), 'rb'))
            self.test_data = pickle.load(open(os.path.join(split_dir, 'test.pkl'), 'rb'))
            self.test_neg = pickle.load(open(os.path.join(split_dir, 'test_nega.pkl'), 'rb'))
            self.val_neg = pickle.load(open(os.path.join(split_dir, 'vali_nega.pkl'), 'rb'))
            self.train_neg = pickle.load(open(os.path.join(split_dir, 'train_nega.pkl'), 'rb'))
        else:
            self._loo_split_and_neg_sample(split_dir)

    @staticmethod
    def _ratio_label(value):
        percent = value * 100
        if abs(percent - round(percent)) < 1e-8:
            return f'{int(round(percent))}%'
        return f'{percent:.2f}'.rstrip('0').rstrip('.') + '%'

    @classmethod
    def _split_dir_name(cls, user_active_ratio, item_hot_ratio):
        return f'{cls._ratio_label(user_active_ratio)}active_{cls._ratio_label(item_hot_ratio)}hot'

    def _loo_split_and_neg_sample(self, split_dir):
        np.random.seed(self.seed)
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        self.test_neg = {}
        self.val_neg = {}
        self.train_neg = {}

        all_item_set = set(self.item_list)

        for uid in self.user_list:
            items = self.user2item[uid]
            items = items.copy()
            if len(items) >= 2:
                np.random.shuffle(items)
                self.val_data[uid] = items[0]
                self.test_data[uid] = items[1]
                self.train_data[uid] = items[2:]
            elif len(items) == 1:
                self.val_data[uid] = items[0]
                self.test_data[uid] = items[0]
                self.train_data[uid] = [items[0]]
            else:
                self.val_data[uid] = -1
                self.test_data[uid] = -1
                self.train_data[uid] = []

            pos_items = set(items)
            neg_pool = list(all_item_set - pos_items)
            np.random.shuffle(neg_pool)
            train_neg_num = len(self.train_data[uid]) * self.neg_per_pos
            self.test_neg[uid] = neg_pool[:self.n_neg]
            self.val_neg[uid] = neg_pool[self.n_neg:2 * self.n_neg]
            self.train_neg[uid] = neg_pool[2 * self.n_neg:2 * self.n_neg + train_neg_num]

        pickle.dump(self.train_data, open(os.path.join(split_dir, 'train.pkl'), 'wb'))
        pickle.dump(self.test_data, open(os.path.join(split_dir, 'test.pkl'), 'wb'))
        pickle.dump(self.val_data, open(os.path.join(split_dir, 'vali.pkl'), 'wb'))
        pickle.dump(self.train_neg, open(os.path.join(split_dir, 'train_nega.pkl'), 'wb'))
        pickle.dump(self.val_neg, open(os.path.join(split_dir, 'vali_nega.pkl'), 'wb'))
        pickle.dump(self.test_neg, open(os.path.join(split_dir, 'test_nega.pkl'), 'wb'))

    def get_train_batches(self):
        users, items, labels = [], [], []
        for uid in self.user_list:
            for iid in self.train_data[uid]:
                users.append(uid)
                items.append(iid)
                labels.append(1.0)
            for iid in self.train_neg[uid]:
                users.append(uid)
                items.append(iid)
                labels.append(0.0)
        return users, items, labels

    def get_eval_data(self, mode='test'):
        if mode == 'test':
            pos = self.test_data
            neg = self.test_neg
        else:
            pos = self.val_data
            neg = self.val_neg

        all_uids, all_iids, all_labels = [], [], []
        sample_per_user = 1 + self.n_neg
        valid_uids = []
        for uid in self.user_list:
            if pos[uid] < 0:
                continue
            valid_uids.append(uid)
            all_uids.append(uid)
            all_iids.append(pos[uid])
            all_labels.append(1.0)
            for nid in neg[uid]:
                all_uids.append(uid)
                all_iids.append(nid)
                all_labels.append(0.0)
        return all_uids, all_iids, all_labels, valid_uids, sample_per_user

    def get_interaction_matrix(self):
        R = np.zeros((self.n_users, self.n_items))
        for uid, items in self.train_data.items():
            for iid in items:
                R[uid][iid] = 1
        return R

    def get_graph(self):
        R = self.get_interaction_matrix()
        rows, cols = R.nonzero()
        src = list(rows) + list(cols + self.n_users)
        dst = list(cols + self.n_users) + list(rows)
        return torch.LongTensor([src, dst])

    def get_train_interactions(self):
        users, pos_items = [], []
        for uid, items in self.train_data.items():
            for iid in items:
                users.append(uid)
                pos_items.append(iid)
        return users, pos_items

    def get_infonce_batches(self):
        batch_users, batch_pos_items, batch_neg_items = [], [], []
        neg_per_pos = self.neg_per_pos
        for uid in self.user_list:
            pos_items = self.train_data[uid]
            if len(pos_items) == 0:
                continue
            neg_items = self.train_neg[uid]
            n_neg_total = len(neg_items)
            for idx, pid in enumerate(pos_items):
                start_neg = idx * neg_per_pos
                end_neg = min(start_neg + neg_per_pos, n_neg_total)
                if end_neg <= start_neg:
                    end_neg = min(neg_per_pos, n_neg_total)
                    start_neg = 0
                nids = neg_items[start_neg:end_neg]
                if len(nids) == 0:
                    continue
                batch_users.append(uid)
                batch_pos_items.append(pid)
                batch_neg_items.append(nids)
        return batch_users, batch_pos_items, batch_neg_items


def load_dataset(dataset_name, data_dir='data', n_neg=99, seed=42,
                 user_active_ratio=0.05, item_hot_ratio=0.20, **kwargs):
    return Dataset(dataset_name, data_dir=data_dir, n_neg=n_neg, seed=seed,
                   user_active_ratio=user_active_ratio,
                   item_hot_ratio=item_hot_ratio)
