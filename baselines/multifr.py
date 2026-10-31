import torch
import torch.nn.functional as F


class MultiFR:
    def __init__(self, model, dataset, lambda_fair=0.1):
        self.model = model
        self.dataset = dataset
        self.lambda_fair = lambda_fair

    def compute_fair_loss(self, user_ids, item_ids):
        adv_users = self.dataset.adv_users
        dis_users = self.dataset.dis_users
        hot_items = self.dataset.hot_items
        cold_items = self.dataset.cold_items

        device = next(self.model.parameters()).device
        user_fair_loss = torch.zeros((), device=device)
        item_fair_loss = torch.zeros((), device=device)

        if hasattr(self.model, 'user_emb') and hasattr(self.model, 'item_emb'):
            u_emb = self.model.user_emb.weight
            i_emb = self.model.item_emb.weight

            adv_emb = u_emb[list(adv_users)]
            dis_emb = u_emb[list(dis_users)]
            if len(dis_emb) > 0 and len(adv_emb) > 0:
                user_fair_loss = F.mse_loss(dis_emb.mean(dim=0), adv_emb.mean(dim=0))

            hot_emb = i_emb[list(hot_items)]
            cold_emb = i_emb[list(cold_items)]
            if len(cold_emb) > 0 and len(hot_emb) > 0:
                item_fair_loss = F.mse_loss(cold_emb.mean(dim=0), hot_emb.mean(dim=0))

        return self.lambda_fair * (user_fair_loss + item_fair_loss)

    def predict(self, user_id, item_ids):
        return self.model.predict(user_id, item_ids)
