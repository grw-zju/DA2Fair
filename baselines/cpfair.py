import numpy as np


class CPFair:
    def __init__(self, model, dataset, K=10, alpha=0.5):
        self.model = model
        self.dataset = dataset
        self.K = K
        self.alpha = alpha

    def rerank(self, user_id, candidate_items, scores):
        ranked_indices = np.argsort(-np.array(scores))
        ranked_items = [candidate_items[i] for i in ranked_indices]

        hot_items = self.dataset.hot_items
        cold_items = self.dataset.cold_items
        adv_users = self.dataset.adv_users
        dis_users = self.dataset.dis_users

        n_cold_target = max(1, int(self.K * self.alpha))

        final_list = []
        hot_buffer = []
        cold_buffer = []

        for item in ranked_items:
            if item in cold_items:
                cold_buffer.append(item)
            else:
                hot_buffer.append(item)

        cold_idx = 0
        hot_idx = 0
        for pos in range(min(self.K, len(ranked_items))):
            cold_needed = n_cold_target - len([i for i in final_list if i in cold_items])
            if cold_needed > 0 and cold_idx < len(cold_buffer):
                final_list.append(cold_buffer[cold_idx])
                cold_idx += 1
            elif hot_idx < len(hot_buffer):
                final_list.append(hot_buffer[hot_idx])
                hot_idx += 1
            elif cold_idx < len(cold_buffer):
                final_list.append(cold_buffer[cold_idx])
                cold_idx += 1

        for item in ranked_items:
            if item not in final_list:
                final_list.append(item)

        return final_list

    def predict(self, user_id, item_ids):
        return self.model.predict(user_id, item_ids)
