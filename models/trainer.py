import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from data.dataset_utils import Dataset
from evaluation.metrics import (evaluate_full_ranking, evaluate_recommendation_vaecf,
                           compute_delta)
from models.da2fair.nemo import NEMO
from models.da2fair.uid_alignm import compute_alignment_loss, compute_infoot_mi
from baselines.cpfair import CPFair
from baselines.fairsort import FairSort
from models.backbone import get_backbone


def create_backbone(model_name, n_users, n_items, emb_dim):
    return get_backbone(model_name, n_users, n_items, emb_dim)


def _get_user_emb(model, model_name, u_ids, edge_index=None, dataset=None):
    if model_name == 'MF':
        return model.user_emb(u_ids)
    elif model_name == 'NeuMF':
        mf_u = model.mf_user_emb(u_ids)
        mlp_u = model.mlp_user_emb(u_ids)
        return torch.cat([mf_u, mlp_u], dim=1)
    elif model_name == 'LightGCN':
        all_emb = model.get_final_embed(edge_index)
        return all_emb[:model.n_users][u_ids]
    elif model_name == 'VAECF':
        return model.get_user_emb_batch(u_ids, dataset)
    else:
        raise ValueError(f"Cannot get user embeddings for {model_name}")


def _get_item_emb(model, model_name, i_ids, edge_index=None, dataset=None):
    if model_name == 'MF':
        return model.item_emb(i_ids)
    elif model_name == 'NeuMF':
        mf_i = model.mf_item_emb(i_ids)
        mlp_i = model.mlp_item_emb(i_ids)
        return torch.cat([mf_i, mlp_i], dim=1)
    elif model_name == 'LightGCN':
        all_emb = model.get_final_embed(edge_index)
        return all_emb[model.n_users:][i_ids]
    elif model_name == 'VAECF':
        return model.get_item_emb_batch(i_ids)
    else:
        raise ValueError(f"Cannot get item embeddings for {model_name}")


def _get_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _long_tensor(values, device):
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _zero_loss(model):
    device = _get_device(model)
    return torch.zeros((), device=device)


def _compute_scores(model, model_name, u_ids, i_ids, edge_index=None):
    if model_name == 'LightGCN':
        return model(edge_index, u_ids, i_ids)
    else:
        return model(u_ids, i_ids)


def _build_interaction_batch(dataset, batch_users, device):
    R_batch = np.zeros((len(batch_users), dataset.n_items), dtype=np.float32)
    for i, uid in enumerate(batch_users):
        for iid in dataset.train_data[uid]:
            if 0 <= iid < dataset.n_items:
                R_batch[i][iid] = 1.0
    return torch.from_numpy(R_batch).to(device)


def _iter_batches(n_samples, batch_size):
    return math.ceil(n_samples / batch_size) if n_samples > 0 else 0


def _clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _model_scores_for_batch(model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index):
    if model_name == 'LightGCN':
        all_emb = model.get_final_embed(edge_index)
        pos_scores = model.score_from_embeddings(all_emb, u_ids, pos_i_ids)
        neg_scores = model.score_from_embeddings(
            all_emb,
            u_ids.repeat_interleave(neg_i_ids.shape[1]),
            neg_i_ids.reshape(-1)
        ).view(u_ids.shape[0], neg_i_ids.shape[1])
        return pos_scores, neg_scores

    pos_scores = _compute_scores(model, model_name, u_ids, pos_i_ids, edge_index)
    neg_scores_list = []
    for j in range(neg_i_ids.shape[1]):
        neg_j = neg_i_ids[:, j]
        neg_s = _compute_scores(model, model_name, u_ids, neg_j, edge_index)
        neg_scores_list.append(neg_s.unsqueeze(1))
    neg_scores = torch.cat(neg_scores_list, dim=1)
    return pos_scores, neg_scores


def _make_cf_batch(batch_users, batch_pos_items, batch_neg_items, batch_idx, device):
    batch_uids = [batch_users[i] for i in batch_idx]
    u_ids = _long_tensor(batch_uids, device)
    pos_i_ids = _long_tensor([batch_pos_items[i] for i in batch_idx], device)
    max_neg_len = max(len(batch_neg_items[i]) for i in batch_idx)
    neg_i_ids = _long_tensor(
        [[batch_neg_items[i][j] if j < len(batch_neg_items[i]) else batch_neg_items[i][0]
          for j in range(max_neg_len)] for i in batch_idx],
        device)
    return batch_uids, u_ids, pos_i_ids, neg_i_ids


def _evaluate_model(model, dataset, model_name, K, mode='test', edge_index=None):
    if model_name == 'VAECF':
        return evaluate_recommendation_vaecf(model, dataset, K, mode)
    return evaluate_full_ranking(model, dataset, K, mode, model_name, edge_index)


def _sample_group_ids(group_ids, sample_size):
    ids = list(group_ids)
    if not ids:
        return []
    n_sample = min(sample_size, len(ids))
    return list(np.random.choice(ids, n_sample, replace=False))


def bpr_loss(pos_scores, neg_scores):
    diff = pos_scores.unsqueeze(1) - neg_scores
    return -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))


def bce_loss(pos_scores, neg_scores):
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    all_scores = torch.cat([pos_scores, neg_scores.flatten()])
    all_labels = torch.cat([pos_labels, neg_labels.flatten()])
    return F.binary_cross_entropy_with_logits(all_scores, all_labels)


def compute_group_alignment_loss(model, dataset, model_name, edge_index,
                                 side='user', sample_size=64):
    device = _get_device(model)
    if side == 'user':
        src_ids = _sample_group_ids(dataset.dis_users, sample_size)
        tgt_ids = _sample_group_ids(dataset.adv_users, sample_size)
        if not src_ids or not tgt_ids:
            return _zero_loss(model)
        src_emb = _get_user_emb(model, model_name, _long_tensor(src_ids, device),
                                edge_index, dataset)
        tgt_emb = _get_user_emb(model, model_name, _long_tensor(tgt_ids, device),
                                edge_index, dataset)
    else:
        src_ids = _sample_group_ids(dataset.cold_items, sample_size)
        tgt_ids = _sample_group_ids(dataset.hot_items, sample_size)
        if not src_ids or not tgt_ids:
            return _zero_loss(model)
        src_emb = _get_item_emb(model, model_name, _long_tensor(src_ids, device),
                                edge_index, dataset)
        tgt_emb = _get_item_emb(model, model_name, _long_tensor(tgt_ids, device),
                                edge_index, dataset)

    return F.mse_loss(src_emb.mean(dim=0), tgt_emb.mean(dim=0))


def compute_adaptive_sample_weights(dataset, batch_uids, pos_i_ids):
    device = pos_i_ids.device
    weights = []
    for uid, iid in zip(batch_uids, pos_i_ids.detach().cpu().tolist()):
        user_weight = 2.0 if uid in dataset.dis_users else 1.0
        item_weight = 2.0 if iid in dataset.cold_items else 1.0
        weights.append(user_weight * item_weight)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean().clamp_min(1e-8)


def compute_ada2fair_dynamic_weights(model, dataset, model_name, edge_index,
                                     topk=100, eta=2.0, delta=1e-6,
                                     eval_batch_size=4096):
    device = _get_device(model)
    model.eval()
    item_exposure = np.zeros(dataset.n_items, dtype=np.float64)
    user_utility = np.zeros(dataset.n_users, dtype=np.float64)
    discount = 1.0 / np.log2(np.arange(2, topk + 2))

    with torch.no_grad():
        all_emb = None
        if model_name == 'LightGCN':
            all_emb = model.get_final_embed(edge_index)

        item_ids = torch.arange(dataset.n_items, device=device)
        for start in range(0, dataset.n_users, eval_batch_size):
            end = min(start + eval_batch_size, dataset.n_users)
            user_ids = torch.arange(start, end, device=device)
            if model_name == 'MF':
                scores = torch.matmul(
                    model.user_emb(user_ids),
                    model.item_emb.weight.t())
            elif model_name == 'NeuMF':
                rows = []
                for uid in user_ids:
                    u = uid.repeat(dataset.n_items)
                    rows.append(model(u, item_ids).unsqueeze(0))
                scores = torch.cat(rows, dim=0)
            elif model_name == 'LightGCN':
                user_emb = all_emb[:model.n_users][user_ids]
                item_emb = all_emb[model.n_users:]
                scores = torch.matmul(user_emb, item_emb.t())
            else:
                R = _build_interaction_batch(
                    dataset, list(range(start, end)), device)
                scores, _, _, _ = model(R)

            k = min(topk, dataset.n_items)
            top_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
            for row_idx, rec_items in enumerate(top_items):
                uid = start + row_idx
                hist = set(dataset.train_data.get(uid, []))
                util = 0.0
                for rank, iid in enumerate(rec_items):
                    item_exposure[iid] += discount[rank]
                    if iid in hist:
                        util += discount[rank]
                if hist:
                    user_utility[uid] = util / len(hist)

    item_group_exposure = np.zeros(dataset.n_items, dtype=np.float64)
    hot_mean = item_exposure[list(dataset.hot_items)].mean() if dataset.hot_items else 0.0
    cold_mean = item_exposure[list(dataset.cold_items)].mean() if dataset.cold_items else 0.0
    for iid in range(dataset.n_items):
        item_group_exposure[iid] = cold_mean if iid in dataset.cold_items else hot_mean

    item_weights = 1.0 / (np.power(item_group_exposure, eta) + delta)
    user_weights = 1.0 / (np.power(user_utility, eta) + delta)

    train_items = [iid for items in dataset.train_data.values() for iid in items]
    if train_items:
        item_weights = item_weights / max(item_weights[train_items].mean(), delta)
    train_users = [uid for uid, items in dataset.train_data.items() for _ in items]
    if train_users:
        user_weights = user_weights / max(user_weights[train_users].mean(), delta)

    model.train()
    return (torch.tensor(user_weights, dtype=torch.float32, device=device),
            torch.tensor(item_weights, dtype=torch.float32, device=device))


def compute_ada2fair_sample_weights(dynamic_weights, batch_uids, pos_i_ids):
    user_weights, item_weights = dynamic_weights
    u_t = torch.as_tensor(batch_uids, dtype=torch.long, device=pos_i_ids.device)
    weights = user_weights[u_t] * item_weights[pos_i_ids]
    return weights / weights.mean().clamp_min(1e-8)


def weighted_bpr_loss(pos_scores, neg_scores, weights):
    diff = pos_scores.unsqueeze(1) - neg_scores
    per_sample = -torch.log(torch.sigmoid(diff) + 1e-8)
    return (per_sample.mean(dim=1) * weights).mean()


def compute_multifr_losses(model, dataset, model_name, edge_index, sample_size=64):
    user_loss = compute_group_alignment_loss(
        model, dataset, model_name, edge_index, side='user', sample_size=sample_size)
    item_loss = compute_group_alignment_loss(
        model, dataset, model_name, edge_index, side='item', sample_size=sample_size)
    return user_loss, item_loss


def _get_activity_counts(dataset, side, ids):
    if side == 'user':
        return np.array([len(dataset.train_data.get(uid, [])) for uid in ids],
                        dtype=np.float64)
    else:
        return np.array([len(dataset.item2user.get(iid, [])) for iid in ids],
                        dtype=np.float64)


def compute_infoot_alignment_loss(model, dataset, model_name, edge_index,
                                  side='user', batch_src_ids=None,
                                  n_sample=128, h=0.5, lam=0.1, eps=0.05,
                                  gamma=0.5, delta=1.0, J=3, S=20,
                                  align_terms='GL'):
    device = _get_device(model)

    if side == 'user':
        tgt_ids_all = list(dataset.adv_users)
        src_ids_all = batch_src_ids if batch_src_ids else list(dataset.dis_users)
    else:
        tgt_ids_all = list(dataset.hot_items)
        src_ids_all = list(dataset.cold_items)

    if not tgt_ids_all or not src_ids_all:
        return _zero_loss(model)

    n_src = min(n_sample, len(src_ids_all))
    n_tgt = min(n_sample, len(tgt_ids_all))

    sampled_src_ids = list(np.random.choice(src_ids_all, n_src, replace=False))
    sampled_tgt_ids = list(np.random.choice(tgt_ids_all, n_tgt, replace=False))

    src_ids_t = _long_tensor(sampled_src_ids, device)
    tgt_ids_t = _long_tensor(sampled_tgt_ids, device)

    if side == 'user':
        src_emb = _get_user_emb(model, model_name, src_ids_t, edge_index, dataset)
        tgt_emb = _get_user_emb(model, model_name, tgt_ids_t, edge_index, dataset)
    else:
        src_emb = _get_item_emb(model, model_name, src_ids_t, edge_index, dataset)
        tgt_emb = _get_item_emb(model, model_name, tgt_ids_t, edge_index, dataset)

    src_emb_norm = F.normalize(src_emb, dim=1)
    tgt_emb_norm = F.normalize(tgt_emb, dim=1)

    nu_s = _get_activity_counts(dataset, side, sampled_src_ids)

    src_np = src_emb_norm.detach().cpu().numpy()
    tgt_np = tgt_emb_norm.detach().cpu().numpy()

    try:
        ot_loss_val, proj_source, P, Kx_np, Ky_np, p_np, q_np, mi_hat_np = \
            compute_alignment_loss(
                src_np, tgt_np, nu_s=nu_s, delta=delta, gamma=gamma,
                lam=lam, eps=eps, h=h, J=J, S=S, align_terms=align_terms)
    except Exception as e:
        print(f'[WARN] OT alignment failed: {e}')
        ot_loss_val = 0.0
        proj_source = tgt_np.mean(axis=0).reshape(1, -1).repeat(n_src, axis=0)
        P = None
        Kx_np = Ky_np = p_np = q_np = None
        mi_hat_np = 0.0

    C_theta = 1.0 - torch.mm(src_emb_norm, tgt_emb_norm.t())
    P_t = torch.FloatTensor(P).to(device) if P is not None else None
    if P_t is not None:
        if align_terms == 'G':
            align_loss = torch.sum(P_t * C_theta)
        elif align_terms == 'L':
            src_sq = (src_emb_norm ** 2).sum(dim=1, keepdim=True)
            tgt_sq = (tgt_emb_norm ** 2).sum(dim=1, keepdim=True)
            dist_x = torch.sqrt(torch.clamp(src_sq + src_sq.t() - 2 * torch.mm(src_emb_norm, src_emb_norm.t()), min=0))
            dist_y = torch.sqrt(torch.clamp(tgt_sq + tgt_sq.t() - 2 * torch.mm(tgt_emb_norm, tgt_emb_norm.t()), min=0))
            std_x = torch.sqrt((dist_x ** 2).mean() / 2)
            std_y = torch.sqrt((dist_y ** 2).mean() / 2)
            h1 = torch.clamp(h * std_x, min=1e-8)
            h2 = torch.clamp(h * std_y, min=1e-8)
            Kx_t = torch.exp(-(dist_x / h1) ** 2 / 2)
            Ky_t = torch.exp(-(dist_y / h2) ** 2 / 2)

            p_t = torch.FloatTensor(p_np).to(device)
            q_t = torch.FloatTensor(q_np).to(device)

            F_Gamma_t = Kx_t @ P_t @ Ky_t.t()
            M_X_t = Kx_t @ p_t
            M_Y_t = Ky_t @ q_t
            denom_t = torch.outer(M_X_t, M_Y_t)
            R_t = F_Gamma_t / (denom_t + 1e-30)
            mi_loss = torch.sum(P_t * torch.log(R_t + 1e-30))

            align_loss = -lam * mi_loss
        else:
            geo_loss = torch.sum(P_t * C_theta)

            src_sq = (src_emb_norm ** 2).sum(dim=1, keepdim=True)
            tgt_sq = (tgt_emb_norm ** 2).sum(dim=1, keepdim=True)
            dist_x = torch.sqrt(torch.clamp(src_sq + src_sq.t() - 2 * torch.mm(src_emb_norm, src_emb_norm.t()), min=0))
            dist_y = torch.sqrt(torch.clamp(tgt_sq + tgt_sq.t() - 2 * torch.mm(tgt_emb_norm, tgt_emb_norm.t()), min=0))
            std_x = torch.sqrt((dist_x ** 2).mean() / 2)
            std_y = torch.sqrt((dist_y ** 2).mean() / 2)
            h1 = torch.clamp(h * std_x, min=1e-8)
            h2 = torch.clamp(h * std_y, min=1e-8)
            Kx_t = torch.exp(-(dist_x / h1) ** 2 / 2)
            Ky_t = torch.exp(-(dist_y / h2) ** 2 / 2)

            p_t = torch.FloatTensor(p_np).to(device)
            q_t = torch.FloatTensor(q_np).to(device)

            F_Gamma_t = Kx_t @ P_t @ Ky_t.t()
            M_X_t = Kx_t @ p_t
            M_Y_t = Ky_t @ q_t
            denom_t = torch.outer(M_X_t, M_Y_t)
            R_t = F_Gamma_t / (denom_t + 1e-30)
            mi_loss = torch.sum(P_t * torch.log(R_t + 1e-30))

            align_loss = geo_loss - lam * mi_loss
    else:
        align_loss = torch.sum(C_theta * 0.0)

    return align_loss


def train_backbone(model, dataset, model_name, epochs=200, lr=1e-4,
                   batch_size=2048, wd=1e-5, patience=20, n_neg_per_pos=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)

    if model_name == 'VAECF':
        model.set_train_data(dataset.train_data)
        return train_vaecf(model, dataset, epochs, lr, batch_size, wd, patience)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = _iter_batches(n_samples, batch_size)
        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            _, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)

            if model_name == 'NeuMF':
                loss = bce_loss(pos_scores, neg_scores)
            else:
                loss = bpr_loss(pos_scores, neg_scores[:, :n_neg_per_pos])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        print(f'Epoch {epoch+1}, Loss: {total_loss/n_batches:.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_vaecf(model, dataset, epochs, lr, batch_size, wd, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)
    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = _iter_batches(dataset.n_users, batch_size)
        kl_weight = min(0.2, 0.2 * (epoch + 1) / 20.0) if epoch < 20 else 0.2

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, dataset.n_users)
            batch_users = dataset.user_list[start:end]

            R_tensor = _build_interaction_batch(dataset, batch_users, device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(R_tensor)
            x_sum = R_tensor.sum(dim=1, keepdim=True).clamp(min=1)
            x_normalized = R_tensor / x_sum
            log_softmax = F.log_softmax(recon, dim=1)
            recon_loss = -(x_normalized * log_softmax).sum(dim=1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss = recon_loss.mean() + kl_weight * kl_loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = evaluate_recommendation_vaecf(model, dataset, K=10, mode='val')
        ndcg = result['NDCG_all']
        print(f'VAECF Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_vaecf_with_fairness(model, dataset, epochs, lr, batch_size, wd,
                              lambda_fair=0.005, adaptive=False,
                              method_name='VAECF-Fair', patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)
    model.set_train_data(dataset.train_data)
    edge_index = None
    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        user_indices = np.random.permutation(dataset.n_users)
        n_batches = _iter_batches(dataset.n_users, batch_size)
        kl_weight = min(0.2, 0.2 * (epoch + 1) / 20.0) if epoch < 20 else 0.2

        for b in range(n_batches):
            batch_pos = user_indices[b * batch_size: (b + 1) * batch_size]
            batch_users = [dataset.user_list[i] for i in batch_pos]
            R_tensor = _build_interaction_batch(dataset, batch_users, device)

            recon, mu, logvar, _ = model(R_tensor)
            x_sum = R_tensor.sum(dim=1, keepdim=True).clamp(min=1)
            x_normalized = R_tensor / x_sum
            log_softmax = F.log_softmax(recon, dim=1)
            recon_loss = -(x_normalized * log_softmax).sum(dim=1)
            if adaptive:
                user_weights = torch.tensor(
                    [2.0 if uid in dataset.dis_users else 1.0 for uid in batch_users],
                    dtype=torch.float32, device=device)
                user_weights = user_weights / user_weights.mean().clamp_min(1e-8)
                recon_loss = recon_loss * user_weights
            recon_loss = recon_loss.mean()
            kl_loss = -0.5 * torch.mean(torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            user_fair, item_fair = compute_multifr_losses(
                model, dataset, 'VAECF', edge_index)
            loss = recon_loss + kl_weight * kl_loss + lambda_fair * (user_fair + item_fair)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = evaluate_recommendation_vaecf(model, dataset, K=10, mode='val')
        ndcg = result['NDCG_all']
        print(f'{method_name} Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_da2fair(model, dataset, model_name, epochs=200, lr=1e-4,
                  batch_size=2048, wd=1e-5, patience=20,
                  n_sample=128, h=0.5, lam=0.1, eps=0.05,
                  gamma=0.5, delta=1.0, J=3, S=20,
                  tau=1e-3, xi=1e-3):
    if model_name == 'VAECF':
        return train_da2fair_vaecf(
            model, dataset, epochs, lr, batch_size, wd, patience,
            n_sample, h, lam, eps, gamma, delta, J, S, tau, xi)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    nemo = NEMO(max_iter=20, tau=tau, xi=xi)
    device = _get_device(model)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = _iter_batches(n_samples, batch_size)
        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            batch_uids, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)

            if model_name == 'NeuMF':
                rec_loss = bce_loss(pos_scores, neg_scores)
            else:
                rec_loss = bpr_loss(pos_scores, neg_scores[:, :1])

            inactive_uids = [uid for uid in batch_uids
                             if uid not in dataset.adv_users]

            if len(inactive_uids) > 0:
                all_emb = model.get_final_embed(edge_index) if model_name == 'LightGCN' else None

                user_align_loss = compute_infoot_alignment_loss(
                    model, dataset, model_name, edge_index,
                    side='user', batch_src_ids=inactive_uids,
                    n_sample=n_sample, h=h, lam=lam, eps=eps,
                    gamma=gamma, delta=delta, J=J, S=S)

                item_align_loss = compute_infoot_alignment_loss(
                    model, dataset, model_name, edge_index,
                    side='item', batch_src_ids=inactive_uids,
                    n_sample=n_sample, h=h, lam=lam, eps=eps,
                    gamma=gamma, delta=delta, J=J, S=S)

                optimizer.zero_grad()
                rec_loss.backward()
                g_rec = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                   else torch.zeros(p.numel(), device=device)
                                   for p in model.parameters()])

                optimizer.zero_grad()
                user_align_loss.backward(retain_graph=(model_name == 'LightGCN'))
                g_user = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                    else torch.zeros(p.numel(), device=device)
                                    for p in model.parameters()])

                optimizer.zero_grad()
                item_align_loss.backward()
                g_item = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                    else torch.zeros(p.numel(), device=device)
                                    for p in model.parameters()])

                g_nemo = nemo.coordinate([g_rec, g_user, g_item])

                if g_nemo is not None:
                    optimizer.zero_grad()
                    idx_p = 0
                    for p in model.parameters():
                        sz = p.numel()
                        p.grad = g_nemo[idx_p:idx_p+sz].view(p.shape).clone()
                        idx_p += sz
                    optimizer.step()
            else:
                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()

            total_loss += rec_loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        print(f'DA2Fair Epoch {epoch+1}, Loss: {total_loss/n_batches:.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_da2fair_vaecf(model, dataset, epochs=200, lr=1e-4,
                        batch_size=512, wd=1e-5, patience=20,
                        n_sample=128, h=0.5, lam=0.1, eps=0.05,
                        gamma=0.5, delta=1.0, J=3, S=20,
                        tau=1e-3, xi=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    nemo = NEMO(max_iter=20, tau=tau, xi=xi)
    device = _get_device(model)
    model.set_train_data(dataset.train_data)
    edge_index = None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        user_indices = np.random.permutation(dataset.n_users)
        n_batches = _iter_batches(dataset.n_users, batch_size)
        kl_weight = min(0.2, 0.2 * (epoch + 1) / 20.0) if epoch < 20 else 0.2

        for b in range(n_batches):
            batch_pos = user_indices[b * batch_size: (b + 1) * batch_size]
            batch_users = [dataset.user_list[i] for i in batch_pos]
            R_tensor = _build_interaction_batch(dataset, batch_users, device)

            recon, mu, logvar, _ = model(R_tensor)
            x_sum = R_tensor.sum(dim=1, keepdim=True).clamp(min=1)
            x_normalized = R_tensor / x_sum
            log_softmax = F.log_softmax(recon, dim=1)
            recon_loss = -(x_normalized * log_softmax).sum(dim=1) / max(len(batch_users), 1)
            kl_loss = -0.5 * torch.mean(torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            rec_loss = recon_loss.mean() + kl_weight * kl_loss

            inactive_uids = [uid for uid in batch_users if uid not in dataset.adv_users]
            if inactive_uids:
                user_align_loss = compute_infoot_alignment_loss(
                    model, dataset, 'VAECF', edge_index,
                    side='user', batch_src_ids=inactive_uids,
                    n_sample=n_sample, h=h, lam=lam, eps=eps,
                    gamma=gamma, delta=delta, J=J, S=S)
                item_align_loss = compute_infoot_alignment_loss(
                    model, dataset, 'VAECF', edge_index,
                    side='item', batch_src_ids=inactive_uids,
                    n_sample=n_sample, h=h, lam=lam, eps=eps,
                    gamma=gamma, delta=delta, J=J, S=S)

                optimizer.zero_grad()
                rec_loss.backward(retain_graph=True)
                g_rec = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                   else torch.zeros(p.numel(), device=device)
                                   for p in model.parameters()])

                optimizer.zero_grad()
                user_align_loss.backward(retain_graph=True)
                g_user = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                    else torch.zeros(p.numel(), device=device)
                                    for p in model.parameters()])

                optimizer.zero_grad()
                item_align_loss.backward()
                g_item = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                    else torch.zeros(p.numel(), device=device)
                                    for p in model.parameters()])

                g_nemo = nemo.coordinate([g_rec, g_user, g_item])

                if g_nemo is not None:
                    optimizer.zero_grad()
                    idx_p = 0
                    for p in model.parameters():
                        sz = p.numel()
                        p.grad = g_nemo[idx_p:idx_p + sz].view(p.shape).clone()
                        idx_p += sz
                    optimizer.step()
            else:
                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()

            total_loss += rec_loss.item()

        result = evaluate_recommendation_vaecf(model, dataset, K=10, mode='val')
        ndcg = result['NDCG_all']
        print(f'DA2Fair Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_multifr(model, dataset, model_name, epochs=200, lr=1e-4,
                  batch_size=2048, wd=1e-5, lambda_fair=0.005, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)

    if model_name == 'VAECF':
        model.set_train_data(dataset.train_data)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = _iter_batches(n_samples, batch_size)

        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            _, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)
            if model_name == 'NeuMF':
                rec_loss = bce_loss(pos_scores, neg_scores)
            else:
                rec_loss = bpr_loss(pos_scores, neg_scores[:, :1])
            user_fair, item_fair = compute_multifr_losses(
                model, dataset, model_name, edge_index)
            loss = rec_loss + lambda_fair * (user_fair + item_fair)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        print(f'MultiFR Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_ada2fair(model, dataset, model_name, epochs=200, lr=1e-4,
                   batch_size=2048, wd=1e-5, lambda_fair=0.005, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)

    if model_name == 'VAECF':
        model.set_train_data(dataset.train_data)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = _iter_batches(n_samples, batch_size)
        dynamic_weights = None
        if model_name in ('MF', 'LightGCN'):
            dynamic_weights = compute_ada2fair_dynamic_weights(
                model, dataset, model_name, edge_index)
            model.train()

        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            batch_uids, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)
            if dynamic_weights is None:
                weights = compute_adaptive_sample_weights(dataset, batch_uids, pos_i_ids)
            else:
                weights = compute_ada2fair_sample_weights(
                    dynamic_weights, batch_uids, pos_i_ids)
            if model_name == 'NeuMF':
                rec_loss = bce_loss(pos_scores, neg_scores)
            else:
                rec_loss = weighted_bpr_loss(pos_scores, neg_scores[:, :1], weights)
            user_fair, item_fair = compute_multifr_losses(
                model, dataset, model_name, edge_index)
            loss = rec_loss + lambda_fair * (user_fair + item_fair)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        print(f'Ada2Fair Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_fair(model, dataset, model_name, epochs=200, lr=1e-4,
               batch_size=2048, wd=1e-5, lambda_fair=0.005, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)

    if model_name == 'VAECF':
        model.set_train_data(dataset.train_data)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    constraint_user = 0.0
    constraint_item = 0.0
    constraint_lr = 0.01

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = _iter_batches(n_samples, batch_size)

        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            _, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)
            if model_name == 'NeuMF':
                rec_loss = bce_loss(pos_scores, neg_scores)
            else:
                rec_loss = bpr_loss(pos_scores, neg_scores[:, :1])

            user_fair, item_fair = compute_multifr_losses(
                model, dataset, model_name, edge_index)

            constraint_user = constraint_user + constraint_lr * user_fair.item()
            constraint_item = constraint_item + constraint_lr * item_fair.item()

            loss = rec_loss + lambda_fair * (
                constraint_user * user_fair + constraint_item * item_fair)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        print(f'FAIR Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_with_reranking(model, reranker, dataset, model_name, K=10):
    valid_uids = [uid for uid in dataset.user_list
                  if dataset.test_data.get(uid, -1) >= 0]

    if not valid_uids:
        return {'NDCG_all': 0, 'NDCG_adv': 0, 'NDCG_dis': 0,
                'DUF': 0, 'Cov': 0, 'Hot': 0, 'Cold': 0, 'DIF': 0,
                'HR_all': 0, 'HR_adv': 0, 'HR_dis': 0}

    model.eval()
    device = _get_device(model)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None
    lightgcn_emb = None
    if model_name == 'LightGCN' and edge_index is not None:
        with torch.no_grad():
            lightgcn_emb = model.get_final_embed(edge_index)

    n_users_total = len(valid_uids)
    ndcg_list, hr_list = [], []
    adv_ndcg, dis_ndcg = [], []
    adv_hr, dis_hr = [], []
    topk_items_per_user = []

    all_item_ids = torch.arange(dataset.n_items, device=device)

    for uid in valid_uids:
        pos_item = dataset.test_data[uid]
        train_items = set(dataset.train_data.get(uid, []))
        val_item = dataset.val_data.get(uid, -1)
        if val_item >= 0:
            train_items.add(val_item)

        candidate_mask = np.ones(dataset.n_items, dtype=bool)
        for ti in train_items:
            if 0 <= ti < dataset.n_items:
                candidate_mask[ti] = False
        candidate_indices = np.where(candidate_mask)[0].tolist()

        with torch.no_grad():
            u_t = _long_tensor([uid] * len(candidate_indices), device)
            i_t = _long_tensor(candidate_indices, device)
            if model_name == 'LightGCN' and lightgcn_emb is not None:
                raw_scores = model.score_from_embeddings(
                    lightgcn_emb, u_t, i_t).cpu().numpy()
            elif model_name == 'VAECF':
                R = _build_interaction_batch(dataset, [uid], device)
                recon, _, _, _ = model(R)
                raw_scores = recon.squeeze(0).cpu().numpy()
                raw_scores = raw_scores[candidate_indices]
            else:
                raw_scores = model(u_t, i_t).cpu().numpy()

        reranked_items = reranker.rerank(uid, candidate_indices, raw_scores)
        topk_items_per_user.append(reranked_items[:K])

        relevance = [1.0 if item == pos_item else 0.0 for item in reranked_items[:K]]
        dcg = sum(rel / np.log2(j + 2) for j, rel in enumerate(relevance))
        idcg = 1.0 / np.log2(2)
        ndcg_u = dcg / idcg if idcg > 0 else 0
        hr_u = int(1.0 in relevance)

        ndcg_list.append(ndcg_u)
        hr_list.append(hr_u)
        if uid in dataset.adv_users:
            adv_ndcg.append(ndcg_u); adv_hr.append(hr_u)
        else:
            dis_ndcg.append(ndcg_u); dis_hr.append(hr_u)

    from evaluation.metrics import _compute_item_exposure
    dif, cov, hot_share, cold_share = _compute_item_exposure(
        topk_items_per_user, dataset.hot_items, dataset.cold_items,
        dataset.n_items, K)

    duf = abs(np.mean(adv_ndcg) - np.mean(dis_ndcg)) if adv_ndcg and dis_ndcg else 0

    return {
        'NDCG_all': np.mean(ndcg_list), 'HR_all': np.mean(hr_list),
        'NDCG_adv': np.mean(adv_ndcg) if adv_ndcg else 0,
        'NDCG_dis': np.mean(dis_ndcg) if dis_ndcg else 0,
        'HR_adv': np.mean(adv_hr) if adv_hr else 0,
        'HR_dis': np.mean(dis_hr) if dis_hr else 0,
        'DUF': duf, 'Cov': cov, 'Hot': hot_share, 'Cold': cold_share,
        'DIF': dif}


def _compute_align_loss_for_side(model, dataset, model_name, edge_index,
                                  side, batch_src_ids, n_sample, h, lam, eps,
                                  gamma, delta, J, S, align_terms):
    return compute_infoot_alignment_loss(
        model, dataset, model_name, edge_index,
        side=side, batch_src_ids=batch_src_ids,
        n_sample=n_sample, h=h, lam=lam, eps=eps,
        gamma=gamma, delta=delta, J=J, S=S,
        align_terms=align_terms)


def _coordinate_and_step(optimizer, model, gradients, device,
                          grad_coordinator, nemo_obj, track_decisions=None):
    K = len(gradients)
    if grad_coordinator == 'equal':
        g_agg = sum(gradients) / K
        optimizer.zero_grad()
        idx_p = 0
        for p in model.parameters():
            sz = p.numel()
            p.grad = g_agg[idx_p:idx_p+sz].view(p.shape).clone()
            idx_p += sz
        optimizer.step()
        if track_decisions is not None:
            track_decisions['equal'] = track_decisions.get('equal', 0) + 1
    elif grad_coordinator in ('nash_mtl', 'nemo'):
        g_nemo = nemo_obj.coordinate(gradients)
        if g_nemo is not None:
            optimizer.zero_grad()
            idx_p = 0
            for p in model.parameters():
                sz = p.numel()
                p.grad = g_nemo[idx_p:idx_p+sz].view(p.shape).clone()
                idx_p += sz
            optimizer.step()
        if track_decisions is not None:
            track_decisions[nemo_obj.decision] = \
                track_decisions.get(nemo_obj.decision, 0) + 1
    else:
        raise ValueError(f"Unknown grad_coordinator: {grad_coordinator}")


def train_da2fair_ablation(model, dataset, model_name, epochs=200, lr=1e-4,
                            batch_size=2048, wd=1e-5, patience=20,
                            n_sample=128, h=0.5, lam=0.1, eps=0.05,
                            gamma=0.5, delta=1.0, J=3, S=20,
                            tau=1e-3, xi=1e-3,
                            align_sides='both', align_terms='GL',
                            use_uniform_marginal=False,
                            grad_coordinator='nemo',
                            track_nemo_decisions=False):
    eff_gamma = 0.0 if use_uniform_marginal else gamma
    nemo = NEMO(max_iter=20, tau=tau, xi=xi,
                vanilla=(grad_coordinator == 'nash_mtl'))

    if model_name == 'VAECF':
        return train_da2fair_ablation_vaecf(
            model, dataset, epochs, lr, batch_size, wd, patience,
            n_sample, h, lam, eps, eff_gamma, delta, J, S, tau, xi,
            align_sides, align_terms, grad_coordinator, track_nemo_decisions)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = _get_device(model)

    batch_users, batch_pos_items, batch_neg_items = dataset.get_infonce_batches()
    n_samples = len(batch_users)
    edge_index = dataset.get_graph().to(device) if model_name == 'LightGCN' else None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0
    nemo_decisions_log = {} if track_nemo_decisions else None
    epoch_objectives = [] if track_nemo_decisions else None

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        total_rec = 0
        total_user = 0
        total_item = 0
        n_batches = _iter_batches(n_samples, batch_size)
        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            batch_uids, u_ids, pos_i_ids, neg_i_ids = _make_cf_batch(
                batch_users, batch_pos_items, batch_neg_items, batch_idx, device)
            pos_scores, neg_scores = _model_scores_for_batch(
                model, model_name, u_ids, pos_i_ids, neg_i_ids, edge_index)

            if model_name == 'NeuMF':
                rec_loss = bce_loss(pos_scores, neg_scores)
            else:
                rec_loss = bpr_loss(pos_scores, neg_scores[:, :1])

            inactive_uids = [uid for uid in batch_uids
                             if uid not in dataset.adv_users]

            if len(inactive_uids) > 0:
                gradients = []
                optimizer.zero_grad()
                rec_loss.backward()
                g_rec = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                   else torch.zeros(p.numel(), device=device)
                                   for p in model.parameters()])
                gradients.append(g_rec)
                total_rec += rec_loss.item()

                if align_sides in ('user', 'both'):
                    user_align_loss = _compute_align_loss_for_side(
                        model, dataset, model_name, edge_index,
                        'user', inactive_uids, n_sample, h, lam, eps,
                        eff_gamma, delta, J, S, align_terms)
                    optimizer.zero_grad()
                    user_align_loss.backward(retain_graph=(model_name == 'LightGCN'))
                    g_user = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                        else torch.zeros(p.numel(), device=device)
                                        for p in model.parameters()])
                    gradients.append(g_user)
                    total_user += user_align_loss.item()
                else:
                    gradients.append(torch.zeros_like(g_rec))

                if align_sides in ('item', 'both'):
                    item_align_loss = _compute_align_loss_for_side(
                        model, dataset, model_name, edge_index,
                        'item', inactive_uids, n_sample, h, lam, eps,
                        eff_gamma, delta, J, S, align_terms)
                    optimizer.zero_grad()
                    item_align_loss.backward()
                    g_item = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                        else torch.zeros(p.numel(), device=device)
                                        for p in model.parameters()])
                    gradients.append(g_item)
                    total_item += item_align_loss.item()
                else:
                    gradients.append(torch.zeros_like(g_rec))

                active_count = 1
                if align_sides in ('user', 'both'):
                    active_count += 1
                if align_sides in ('item', 'both'):
                    active_count += 1

                if active_count == 1:
                    optimizer.zero_grad()
                    rec_loss.backward()
                    optimizer.step()
                elif grad_coordinator == 'equal':
                    g_agg = sum(gradients[:active_count]) / active_count
                    optimizer.zero_grad()
                    idx_p = 0
                    for p in model.parameters():
                        sz = p.numel()
                        p.grad = g_agg[idx_p:idx_p+sz].view(p.shape).clone()
                        idx_p += sz
                    optimizer.step()
                else:
                    active_grads = gradients[:active_count]
                    g_nemo = nemo.coordinate(active_grads)
                    if g_nemo is not None:
                        optimizer.zero_grad()
                        idx_p = 0
                        for p in model.parameters():
                            sz = p.numel()
                            p.grad = g_nemo[idx_p:idx_p+sz].view(p.shape).clone()
                            idx_p += sz
                        optimizer.step()
                    if nemo_decisions_log is not None:
                        nemo_decisions_log[nemo.decision] = \
                            nemo_decisions_log.get(nemo.decision, 0) + 1
            else:
                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
                total_rec += rec_loss.item()

            total_loss += rec_loss.item()

        result = _evaluate_model(model, dataset, model_name, K=10,
                                 mode='val', edge_index=edge_index)
        ndcg = result['NDCG_all']
        if epoch_objectives is not None:
            epoch_objectives.append({
                'rec': total_rec / max(n_batches, 1),
                'user': total_user / max(n_batches, 1),
                'item': total_item / max(n_batches, 1),
                'ndcg': ndcg,
            })
        print(f'Ablation Epoch {epoch+1}, Loss: {total_loss/n_batches:.4f}, '
              f'Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, nemo_decisions_log, epoch_objectives


def train_da2fair_ablation_vaecf(model, dataset, epochs, lr, batch_size, wd,
                                  patience, n_sample, h, lam, eps,
                                  gamma, delta, J, S, tau, xi,
                                  align_sides, align_terms, grad_coordinator,
                                  track_nemo_decisions):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    nemo = NEMO(max_iter=20, tau=tau, xi=xi,
                vanilla=(grad_coordinator == 'nash_mtl'))
    device = _get_device(model)
    model.set_train_data(dataset.train_data)
    edge_index = None

    best_ndcg = 0
    best_state = None
    epochs_no_improve = 0
    nemo_decisions_log = {} if track_nemo_decisions else None
    epoch_objectives = [] if track_nemo_decisions else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_rec = 0
        total_user = 0
        total_item = 0
        user_indices = np.random.permutation(dataset.n_users)
        n_batches = _iter_batches(dataset.n_users, batch_size)
        kl_weight = min(0.2, 0.2 * (epoch + 1) / 20.0) if epoch < 20 else 0.2

        for b in range(n_batches):
            batch_pos = user_indices[b * batch_size: (b + 1) * batch_size]
            batch_users = [dataset.user_list[i] for i in batch_pos]
            R_tensor = _build_interaction_batch(dataset, batch_users, device)

            recon, mu, logvar, _ = model(R_tensor)
            x_sum = R_tensor.sum(dim=1, keepdim=True).clamp(min=1)
            x_normalized = R_tensor / x_sum
            log_softmax = F.log_softmax(recon, dim=1)
            recon_loss = -(x_normalized * log_softmax).sum(dim=1) / max(len(batch_users), 1)
            kl_loss = -0.5 * torch.mean(torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            rec_loss = recon_loss.mean() + kl_weight * kl_loss

            inactive_uids = [uid for uid in batch_users if uid not in dataset.adv_users]
            if inactive_uids:
                gradients = []
                optimizer.zero_grad()
                rec_loss.backward(retain_graph=True)
                g_rec = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                   else torch.zeros(p.numel(), device=device)
                                   for p in model.parameters()])
                gradients.append(g_rec)
                total_rec += rec_loss.item()

                if align_sides in ('user', 'both'):
                    user_align_loss = _compute_align_loss_for_side(
                        model, dataset, 'VAECF', edge_index,
                        'user', inactive_uids, n_sample, h, lam, eps,
                        gamma, delta, J, S, align_terms)
                    optimizer.zero_grad()
                    user_align_loss.backward(retain_graph=True)
                    g_user = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                        else torch.zeros(p.numel(), device=device)
                                        for p in model.parameters()])
                    gradients.append(g_user)
                    total_user += user_align_loss.item()
                else:
                    gradients.append(torch.zeros_like(g_rec))

                if align_sides in ('item', 'both'):
                    item_align_loss = _compute_align_loss_for_side(
                        model, dataset, 'VAECF', edge_index,
                        'item', inactive_uids, n_sample, h, lam, eps,
                        gamma, delta, J, S, align_terms)
                    optimizer.zero_grad()
                    item_align_loss.backward()
                    g_item = torch.cat([p.grad.clone().view(-1) if p.grad is not None
                                        else torch.zeros(p.numel(), device=device)
                                        for p in model.parameters()])
                    gradients.append(g_item)
                    total_item += item_align_loss.item()
                else:
                    gradients.append(torch.zeros_like(g_rec))

                active_count = 1
                if align_sides in ('user', 'both'):
                    active_count += 1
                if align_sides in ('item', 'both'):
                    active_count += 1

                if active_count == 1:
                    optimizer.zero_grad()
                    rec_loss.backward()
                    optimizer.step()
                elif grad_coordinator == 'equal':
                    g_agg = sum(gradients[:active_count]) / active_count
                    optimizer.zero_grad()
                    idx_p = 0
                    for p in model.parameters():
                        sz = p.numel()
                        p.grad = g_agg[idx_p:idx_p+sz].view(p.shape).clone()
                        idx_p += sz
                    optimizer.step()
                else:
                    active_grads = gradients[:active_count]
                    g_nemo = nemo.coordinate(active_grads)
                    if g_nemo is not None:
                        optimizer.zero_grad()
                        idx_p = 0
                        for p in model.parameters():
                            sz = p.numel()
                            p.grad = g_nemo[idx_p:idx_p+sz].view(p.shape).clone()
                            idx_p += sz
                        optimizer.step()
                    if nemo_decisions_log is not None:
                        nemo_decisions_log[nemo.decision] = \
                            nemo_decisions_log.get(nemo.decision, 0) + 1
            else:
                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
                total_rec += rec_loss.item()

            total_loss += rec_loss.item()

        result = evaluate_recommendation_vaecf(model, dataset, K=10, mode='val')
        ndcg = result['NDCG_all']
        if epoch_objectives is not None:
            epoch_objectives.append({
                'rec': total_rec / max(n_batches, 1),
                'user': total_user / max(n_batches, 1),
                'item': total_item / max(n_batches, 1),
                'ndcg': ndcg,
            })
        print(f'Ablation Epoch {epoch+1}, Loss: {total_loss/max(n_batches,1):.4f}, '
              f'Val NDCG@10: {ndcg:.4f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = _clone_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, nemo_decisions_log, epoch_objectives


def run_experiment(dataset_name, model_name, method, data_dir='data',
                   epochs=200, lr=1e-4, emb_dim=64, batch_size=2048, K=10,
                   n_neg=99, seed=42, wd=1e-5, device=None,
                   user_active_ratio=0.05, item_hot_ratio=0.20,
                   lambda_fair=0.005,
                   align_sides='both', align_terms='GL',
                   use_uniform_marginal=False,
                   grad_coordinator='nemo',
                   track_nemo_decisions=False):
    print(f'\n===== {dataset_name} | {model_name} | {method} =====')
    dataset = Dataset(dataset_name, data_dir=data_dir, n_neg=n_neg, seed=seed,
                      user_active_ratio=user_active_ratio,
                      item_hot_ratio=item_hot_ratio)
    print(f'Dataset: {dataset.n_users} users, {dataset.n_items} items')
    print(f'Adv: {len(dataset.adv_users)}, Dis: {len(dataset.dis_users)}')
    print(f'Hot: {len(dataset.hot_items)}, Cold: {len(dataset.cold_items)}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    vaecf_batch = 512 if model_name == 'VAECF' else batch_size

    model = None
    baseline_result = None

    def train_standard_for_delta():
        np.random.seed(seed)
        torch.manual_seed(seed)
        baseline_model = create_backbone(
            model_name, dataset.n_users, dataset.n_items, emb_dim).to(device)
        baseline_model = train_backbone(
            baseline_model, dataset, model_name, epochs, lr,
            vaecf_batch if model_name == 'VAECF' else batch_size, wd)
        baseline_edge_index = dataset.get_graph() if model_name == 'LightGCN' else None
        return _evaluate_model(
            baseline_model, dataset, model_name, K, 'test', baseline_edge_index)

    def evaluate_current(trained_model):
        current_edge_index = dataset.get_graph() if model_name == 'LightGCN' else None
        return _evaluate_model(
            trained_model, dataset, model_name, K, 'test', current_edge_index)

    def create_seeded_model():
        np.random.seed(seed)
        torch.manual_seed(seed)
        return create_backbone(model_name, dataset.n_users, dataset.n_items, emb_dim).to(device)

    if method == 'Standard':
        model = create_seeded_model()
        model = train_backbone(model, dataset, model_name, epochs, lr,
                               vaecf_batch if model_name == 'VAECF' else batch_size, wd)
        result = evaluate_current(model)
        baseline_result = result
    elif method == 'DA2Fair':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        model = train_da2fair(model, dataset, model_name, epochs, lr,
                              vaecf_batch if model_name == 'VAECF' else batch_size, wd)
        result = evaluate_current(model)
    elif method == 'Ablation':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        model, nemo_log, epoch_objs = train_da2fair_ablation(
            model, dataset, model_name, epochs, lr,
            vaecf_batch if model_name == 'VAECF' else batch_size, wd,
            align_sides=align_sides, align_terms=align_terms,
            use_uniform_marginal=use_uniform_marginal,
            grad_coordinator=grad_coordinator,
            track_nemo_decisions=track_nemo_decisions)
        result = evaluate_current(model)
        if nemo_log is not None:
            result['nemo_decisions'] = nemo_log
        if epoch_objs is not None:
            result['epoch_objectives'] = epoch_objs
    elif method == 'CPFair':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        model = train_backbone(model, dataset, model_name, epochs, lr,
                               vaecf_batch if model_name == 'VAECF' else batch_size, wd)
        cpfair = CPFair(model, dataset, K)
        result = evaluate_with_reranking(model, cpfair, dataset, model_name, K)
    elif method == 'FairSort':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        model = train_backbone(model, dataset, model_name, epochs, lr,
                               vaecf_batch if model_name == 'VAECF' else batch_size, wd)
        fsort = FairSort(model, dataset, K)
        result = evaluate_with_reranking(model, fsort, dataset, model_name, K)
    elif method == 'MultiFR':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        if model_name == 'VAECF':
            model = train_vaecf_with_fairness(
                model, dataset, epochs, lr, vaecf_batch, wd, lambda_fair,
                adaptive=False, method_name='MultiFR')
        else:
            model = train_multifr(
                model, dataset, model_name, epochs, lr, batch_size, wd, lambda_fair)
        result = evaluate_current(model)
    elif method == 'Ada2Fair':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        if model_name == 'VAECF':
            model = train_vaecf_with_fairness(
                model, dataset, epochs, lr, vaecf_batch, wd, lambda_fair,
                adaptive=True, method_name='Ada2Fair')
        else:
            model = train_ada2fair(
                model, dataset, model_name, epochs, lr, batch_size, wd, lambda_fair)
        result = evaluate_current(model)
    elif method == 'FAIR':
        baseline_result = train_standard_for_delta()
        model = create_seeded_model()
        if model_name == 'VAECF':
            model = train_vaecf_with_fairness(
                model, dataset, epochs, lr, vaecf_batch, wd, lambda_fair,
                adaptive=False, method_name='FAIR')
        else:
            model = train_fair(
                model, dataset, model_name, epochs, lr, batch_size, wd, lambda_fair)
        result = evaluate_current(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    delta = compute_delta(result, baseline_result) if baseline_result else 0.0
    result['Delta'] = delta

    print(f'Results:')
    print(f'  NDCG@10: All={result["NDCG_all"]:.4f}, Adv={result["NDCG_adv"]:.4f}, Dis={result["NDCG_dis"]:.4f}')
    print(f'  DUF={result["DUF"]:.4f}')
    print(f'  Cov={result["Cov"]:.4f}, Hot={result["Hot"]:.4f}, Cold={result["Cold"]:.4f}')
    print(f'  DIF={result["DIF"]:.6f}')
    print(f'  Delta(%)={result["Delta"]:.2f}')
    return result
