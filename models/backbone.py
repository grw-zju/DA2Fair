import torch
import torch.nn as nn
import numpy as np


class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return torch.sum(u * i, dim=1)

    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            device = self.user_emb.weight.device
            u_t = torch.LongTensor([user_id] * len(item_ids)).to(device)
            i_t = torch.LongTensor(item_ids).to(device)
            return self.forward(u_t, i_t).cpu().numpy()

    def get_user_embeddings(self):
        return self.user_emb.weight.detach().cpu().numpy()

    def get_item_embeddings(self):
        return self.item_emb.weight.detach().cpu().numpy()


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, mlp_layers=[128, 64, 32]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.mf_user_emb = nn.Embedding(n_users, emb_dim)
        self.mf_item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.mf_user_emb.weight, std=0.01)
        nn.init.normal_(self.mf_item_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_user_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_item_emb.weight, std=0.01)

        mlp_input_dim = emb_dim * 2
        self.mlp_layers_list = nn.ModuleList()
        for dim in mlp_layers:
            self.mlp_layers_list.append(nn.Linear(mlp_input_dim, dim))
            mlp_input_dim = dim
        self.pred_layer = nn.Linear(mlp_layers[-1] + 1, 1)
        nn.init.normal_(self.pred_layer.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        mf_u = self.mf_user_emb(user_ids)
        mf_i = self.mf_item_emb(item_ids)
        mf_out = torch.sum(mf_u * mf_i, dim=1)

        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        for layer in self.mlp_layers_list:
            mlp_in = torch.relu(layer(mlp_in))

        out = torch.cat([mf_out.unsqueeze(1), mlp_in], dim=1)
        pred = self.pred_layer(out).squeeze(1)
        return pred

    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            u_t = torch.LongTensor([user_id] * len(item_ids)).to(device)
            i_t = torch.LongTensor(item_ids).to(device)
            return self.forward(u_t, i_t).cpu().numpy()

    def get_user_embeddings(self):
        mf_u = self.mf_user_emb.weight.detach().cpu().numpy()
        mlp_u = self.mlp_user_emb.weight.detach().cpu().numpy()
        return np.concatenate([mf_u, mlp_u], axis=1)

    def get_item_embeddings(self):
        mf_i = self.mf_item_emb.weight.detach().cpu().numpy()
        mlp_i = self.mlp_item_emb.weight.detach().cpu().numpy()
        return np.concatenate([mf_i, mlp_i], axis=1)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.edge_index_needed = True

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, edge_index, user_ids, item_ids):
        all_emb = self.get_final_embed(edge_index)
        return self.score_from_embeddings(all_emb, user_ids, item_ids)

    def score_from_embeddings(self, all_emb, user_ids, item_ids):
        u = all_emb[:self.n_users][user_ids]
        i = all_emb[self.n_users:][item_ids]
        return torch.sum(u * i, dim=1)

    def _sparse_propagate(self, edge_index, all_emb):
        edge_index = edge_index.to(all_emb.device)
        src, dst = edge_index
        deg = torch.zeros(all_emb.shape[0], device=all_emb.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        src_emb = all_emb[src]
        messages = src_emb * norm.unsqueeze(1)

        out = torch.zeros_like(all_emb)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        return out

    def predict(self, user_id, item_ids):
        u = self.user_emb.weight[user_id].detach().cpu().numpy()
        items = self.item_emb.weight.detach().cpu().numpy()
        return np.dot(items[item_ids], u)

    def get_final_embed(self, edge_index):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self._sparse_propagate(edge_index, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        return embs

    def get_user_embeddings(self):
        return self.user_emb.weight.detach().cpu().numpy()

    def get_item_embeddings(self):
        return self.item_emb.weight.detach().cpu().numpy()


class VAECF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(nn.Linear(n_items, hidden_dim), nn.Tanh())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_items))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            R = self._get_rating_vec(user_id)
            device = next(self.parameters()).device
            recon, mu, _, _ = self.forward(torch.FloatTensor(R).unsqueeze(0).to(device))
            scores = recon.squeeze(0).cpu().numpy()
            return scores[item_ids]

    def _get_rating_vec(self, user_id):
        R = np.zeros(self.n_items)
        if hasattr(self, 'train_data') and user_id in self.train_data:
            for iid in self.train_data[user_id]:
                if 0 <= iid < self.n_items:
                    R[iid] = 1
        return R

    def set_train_data(self, train_data):
        self.train_data = train_data

    def get_user_emb_batch(self, u_ids, dataset=None):
        if dataset is None:
            dataset = self.train_data_ref if hasattr(self, 'train_data_ref') else None
        embs = []
        for uid in u_ids.cpu().numpy():
            R = self._get_rating_vec(uid)
            device = next(self.parameters()).device
            mu, _ = self.encode(torch.FloatTensor(R).unsqueeze(0).to(device))
            embs.append(mu.squeeze(0))
        return torch.stack(embs, dim=0)

    def get_item_emb_batch(self, i_ids):
        return self.decoder[-1].weight[i_ids]

    def get_all_user_emb(self, dataset):
        self.eval()
        embs = []
        for uid in range(self.n_users):
            R = self._get_rating_vec(uid)
            with torch.no_grad():
                device = next(self.parameters()).device
                mu, _ = self.encode(torch.FloatTensor(R).unsqueeze(0).to(device))
            embs.append(mu.squeeze(0))
        return torch.stack(embs, dim=0)

    def get_all_item_emb(self, dataset=None):
        return self.decoder[-1].weight

    def get_user_embeddings(self):
        self.eval()
        embs = []
        for uid in range(self.n_users):
            R = self._get_rating_vec(uid)
            device = next(self.parameters()).device
            mu, _ = self.encode(torch.FloatTensor(R).unsqueeze(0).to(device))
            embs.append(mu.squeeze(0).detach().cpu().numpy())
        return np.array(embs)

    def get_item_embeddings(self):
        return self.decoder[-1].weight.detach().cpu().numpy()


def get_backbone(name, n_users, n_items, embedding_dim=64, **kwargs):
    name = name.lower()
    if name == 'mf':
        return MF(n_users, n_items, embedding_dim)
    elif name == 'neumf':
        return NeuMF(n_users, n_items, embedding_dim, **kwargs)
    elif name == 'lightgcn':
        return LightGCN(n_users, n_items, embedding_dim, **kwargs)
    elif name == 'vaecf':
        return VAECF(n_users, n_items, emb_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
