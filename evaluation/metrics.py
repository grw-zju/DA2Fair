import numpy as np
import torch
from sklearn.metrics import ndcg_score


def _compute_group_ndcg(ndcg_users, valid_uids, adv_users, dis_users):
    adv_ndcg, dis_ndcg = [], []
    for i, uid in enumerate(valid_uids):
        if uid in adv_users:
            adv_ndcg.append(ndcg_users[i])
        else:
            dis_ndcg.append(ndcg_users[i])
    avg_adv = np.mean(adv_ndcg) if adv_ndcg else 0
    avg_dis = np.mean(dis_ndcg) if dis_ndcg else 0
    duf = abs(avg_adv - avg_dis)
    return avg_adv, avg_dis, duf, adv_ndcg, dis_ndcg


def _compute_item_exposure(topk_items_per_user, hot_items, cold_items, n_items, K=10):
    discount = 1.0 / np.log2(np.arange(2, K + 2))

    item_exposure = np.zeros(n_items, dtype=np.float64)
    for rec_items in topk_items_per_user:
        for rank, iid in enumerate(rec_items[:K]):
            item_exposure[iid] += discount[rank]

    total_exp = item_exposure.sum()
    if total_exp > 0:
        item_exposure_norm = item_exposure / total_exp
    else:
        item_exposure_norm = item_exposure

    hot_indices = list(hot_items)
    cold_indices = list(cold_items)

    hot_share = item_exposure_norm[hot_indices].sum() if hot_indices else 0
    cold_share = item_exposure_norm[cold_indices].sum() if cold_indices else 0

    hot_per_item = (item_exposure_norm[hot_indices].mean()
                    if hot_indices else 0)
    cold_per_item = (item_exposure_norm[cold_indices].mean()
                     if cold_indices else 0)
    dif = abs(hot_per_item - cold_per_item)

    recommended = set()
    for rec_items in topk_items_per_user:
        recommended.update(rec_items[:K])
    cov = len(recommended) / n_items

    return dif, cov, hot_share, cold_share


def evaluate_recommendation(model, dataset, K=10, mode='test',
                           model_name='MF', edge_index=None, eval_batch_size=512):
    all_uids, all_iids, all_labels, valid_uids, sample_per_user = \
        dataset.get_eval_data(mode)

    if len(all_uids) == 0:
        return _empty_result()

    device = next(model.parameters()).device
    model.eval()
    scores_chunks = []
    with torch.no_grad():
        all_emb = None
        if model_name == 'LightGCN' and edge_index is not None:
            edge_index = edge_index.to(device)
            all_emb = model.get_final_embed(edge_index)

        for start in range(0, len(all_uids), eval_batch_size):
            end = min(start + eval_batch_size, len(all_uids))
            user_tensor = torch.LongTensor(all_uids[start:end]).to(device)
            item_tensor = torch.LongTensor(all_iids[start:end]).to(device)
            if model_name == 'LightGCN' and all_emb is not None:
                batch_scores = model.score_from_embeddings(
                    all_emb, user_tensor, item_tensor)
            else:
                batch_scores = model(user_tensor, item_tensor)
            scores_chunks.append(batch_scores.cpu().numpy())

    scores = np.concatenate(scores_chunks, axis=0)

    n_users_total = len(valid_uids)
    labels_arr = np.array(all_labels[:n_users_total * sample_per_user]).reshape(
        n_users_total, sample_per_user)
    scores_arr = scores.reshape(n_users_total, sample_per_user)

    hits_all = []
    ndcg_users = []
    topk_items_per_user = []
    for i in range(n_users_total):
        topk_idx = np.argsort(-scores_arr[i])[:K]
        topk_items = [all_iids[i * sample_per_user + idx] for idx in topk_idx]
        topk_items_per_user.append(topk_items)
        hits_all.append(int(np.sum(labels_arr[i][topk_idx]) > 0))
        ndcg_users.append(ndcg_score(
            labels_arr[i:i + 1], scores_arr[i:i + 1], k=K, ignore_ties=True))

    avg_adv, avg_dis, duf, _, _ = _compute_group_ndcg(
        ndcg_users, valid_uids, dataset.adv_users, dataset.dis_users)

    adv_hr, dis_hr = [], []
    for i, uid in enumerate(valid_uids):
        if uid in dataset.adv_users:
            adv_hr.append(hits_all[i])
        else:
            dis_hr.append(hits_all[i])

    dif, cov, hot_share, cold_share = _compute_item_exposure(
        topk_items_per_user, dataset.hot_items, dataset.cold_items,
        dataset.n_items, K)

    return {
        'NDCG_all': np.mean(ndcg_users),
        'NDCG_adv': avg_adv,
        'NDCG_dis': avg_dis,
        'DUF': duf,
        'Cov': cov,
        'Hot': hot_share,
        'Cold': cold_share,
        'DIF': dif,
        'HR_all': np.mean(hits_all),
        'HR_adv': np.mean(adv_hr) if adv_hr else 0,
        'HR_dis': np.mean(dis_hr) if dis_hr else 0,
    }


def evaluate_full_ranking(model, dataset, K=10, mode='test',
                          model_name='MF', edge_index=None,
                          eval_batch_size=512):
    valid_uids = [uid for uid in dataset.user_list
                  if dataset.test_data.get(uid, -1) >= 0] if mode == 'test' \
        else [uid for uid in dataset.user_list
              if dataset.val_data.get(uid, -1) >= 0]

    if not valid_uids:
        return _empty_result()

    device = next(model.parameters()).device
    model.eval()

    all_emb = None
    if model_name == 'LightGCN' and edge_index is not None:
        edge_index = edge_index.to(device)
        with torch.no_grad():
            all_emb = model.get_final_embed(edge_index)

    pos_data = dataset.test_data if mode == 'test' else dataset.val_data
    train_data = dataset.train_data
    val_data = dataset.val_data if mode == 'test' else dataset.test_data

    ndcg_users = []
    hits_all = []
    topk_items_per_user = []
    adv_ndcg, dis_ndcg = [], []
    adv_hr, dis_hr = [], []

    all_item_ids = torch.arange(dataset.n_items, device=device)

    with torch.no_grad():
        for start in range(0, len(valid_uids), eval_batch_size):
            end = min(start + eval_batch_size, len(valid_uids))
            batch_uids = valid_uids[start:end]

            for uid in batch_uids:
                pos_item = pos_data[uid]
                train_items = set(train_data.get(uid, []))
                if mode == 'test':
                    val_item = val_data.get(uid, -1)
                    if val_item >= 0:
                        train_items.add(val_item)
                else:
                    test_item = dataset.test_data.get(uid, -1)
                    if test_item >= 0:
                        train_items.add(test_item)

                candidate_mask = np.ones(dataset.n_items, dtype=bool)
                for ti in train_items:
                    if 0 <= ti < dataset.n_items:
                        candidate_mask[ti] = False

                u_t = torch.LongTensor([uid]).to(device)
                i_t = all_item_ids
                if model_name == 'LightGCN' and all_emb is not None:
                    scores = model.score_from_embeddings(all_emb, u_t, i_t)
                else:
                    u_t_rep = u_t.repeat(dataset.n_items)
                    scores = model(u_t_rep, i_t)
                scores_np = scores.cpu().numpy()

                scores_np[~candidate_mask] = -np.inf
                topk_idx = np.argsort(-scores_np)[:K]
                topk_items = topk_idx.tolist()
                topk_items_per_user.append(topk_items)

                relevance = [1.0 if iid == pos_item else 0.0
                             for iid in topk_items]
                dcg = sum(rel / np.log2(j + 2)
                          for j, rel in enumerate(relevance))
                idcg = 1.0 / np.log2(2)
                ndcg_u = dcg / idcg if idcg > 0 else 0
                hr_u = int(1.0 in relevance)

                ndcg_users.append(ndcg_u)
                hits_all.append(hr_u)
                if uid in dataset.adv_users:
                    adv_ndcg.append(ndcg_u)
                    adv_hr.append(hr_u)
                else:
                    dis_ndcg.append(ndcg_u)
                    dis_hr.append(hr_u)

    avg_adv = np.mean(adv_ndcg) if adv_ndcg else 0
    avg_dis = np.mean(dis_ndcg) if dis_ndcg else 0
    duf = abs(avg_adv - avg_dis)

    dif, cov, hot_share, cold_share = _compute_item_exposure(
        topk_items_per_user, dataset.hot_items, dataset.cold_items,
        dataset.n_items, K)

    return {
        'NDCG_all': np.mean(ndcg_users),
        'NDCG_adv': avg_adv,
        'NDCG_dis': avg_dis,
        'DUF': duf,
        'Cov': cov,
        'Hot': hot_share,
        'Cold': cold_share,
        'DIF': dif,
        'HR_all': np.mean(hits_all),
        'HR_adv': np.mean(adv_hr) if adv_hr else 0,
        'HR_dis': np.mean(dis_hr) if dis_hr else 0,
    }


def evaluate_recommendation_vaecf(model, dataset, K=10, mode='test',
                                  eval_batch_size=512):
    valid_uids = [uid for uid in dataset.user_list
                  if dataset.test_data.get(uid, -1) >= 0] if mode == 'test' \
        else [uid for uid in dataset.user_list
              if dataset.val_data.get(uid, -1) >= 0]

    if not valid_uids:
        return _empty_result()

    device = next(model.parameters()).device
    model.eval()

    pos_data = dataset.test_data if mode == 'test' else dataset.val_data
    train_data = dataset.train_data
    val_data = dataset.val_data if mode == 'test' else dataset.test_data

    ndcg_users = []
    hits_all = []
    topk_items_per_user = []
    adv_ndcg, dis_ndcg = [], []
    adv_hr, dis_hr = [], []

    with torch.no_grad():
        for start in range(0, len(valid_uids), eval_batch_size):
            end = min(start + eval_batch_size, len(valid_uids))
            batch_uids = valid_uids[start:end]

            R_tensor = _build_interaction_batch_eval(
                dataset, batch_uids, device)
            recon, _, _, _ = model(R_tensor)
            recon_np = recon.cpu().numpy()

            for i, uid in enumerate(batch_uids):
                pos_item = pos_data[uid]
                train_items = set(train_data.get(uid, []))
                if mode == 'test':
                    val_item = val_data.get(uid, -1)
                    if val_item >= 0:
                        train_items.add(val_item)
                else:
                    test_item = dataset.test_data.get(uid, -1)
                    if test_item >= 0:
                        train_items.add(test_item)

                user_scores = recon_np[i].copy()
                for ti in train_items:
                    if 0 <= ti < dataset.n_items:
                        user_scores[ti] = -np.inf

                topk_idx = np.argsort(-user_scores)[:K]
                topk_items = topk_idx.tolist()
                topk_items_per_user.append(topk_items)

                relevance = [1.0 if iid == pos_item else 0.0
                             for iid in topk_items]
                dcg = sum(rel / np.log2(j + 2)
                          for j, rel in enumerate(relevance))
                idcg = 1.0 / np.log2(2)
                ndcg_u = dcg / idcg if idcg > 0 else 0
                hr_u = int(1.0 in relevance)

                ndcg_users.append(ndcg_u)
                hits_all.append(hr_u)
                if uid in dataset.adv_users:
                    adv_ndcg.append(ndcg_u)
                    adv_hr.append(hr_u)
                else:
                    dis_ndcg.append(ndcg_u)
                    dis_hr.append(hr_u)

    avg_adv = np.mean(adv_ndcg) if adv_ndcg else 0
    avg_dis = np.mean(dis_ndcg) if dis_ndcg else 0
    duf = abs(avg_adv - avg_dis)

    dif, cov, hot_share, cold_share = _compute_item_exposure(
        topk_items_per_user, dataset.hot_items, dataset.cold_items,
        dataset.n_items, K)

    return {
        'NDCG_all': np.mean(ndcg_users),
        'NDCG_adv': avg_adv,
        'NDCG_dis': avg_dis,
        'DUF': duf,
        'Cov': cov,
        'Hot': hot_share,
        'Cold': cold_share,
        'DIF': dif,
        'HR_all': np.mean(hits_all),
        'HR_adv': np.mean(adv_hr) if adv_hr else 0,
        'HR_dis': np.mean(dis_hr) if dis_hr else 0,
    }


def _build_interaction_batch_eval(dataset, batch_users, device):
    R = np.zeros((len(batch_users), dataset.n_items), dtype=np.float32)
    for i, uid in enumerate(batch_users):
        for iid in dataset.train_data.get(uid, []):
            if 0 <= iid < dataset.n_items:
                R[i][iid] = 1.0
    return torch.from_numpy(R).to(device)


def compute_delta(result, baseline_result):
    duf_std = baseline_result['DUF']
    dif_std = baseline_result['DIF']
    if duf_std == 0 or dif_std == 0:
        return 0.0
    duf_imp = (duf_std - result['DUF']) / duf_std
    dif_imp = (dif_std - result['DIF']) / dif_std
    return (duf_imp + dif_imp) / 2 * 100


def _empty_result():
    return {
        'NDCG_all': 0, 'NDCG_adv': 0, 'NDCG_dis': 0,
        'DUF': 0, 'Cov': 0, 'Hot': 0, 'Cold': 0, 'DIF': 0,
        'HR_all': 0, 'HR_adv': 0, 'HR_dis': 0,
    }
