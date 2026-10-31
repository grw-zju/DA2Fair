import numpy as np


class FairSort:
    def __init__(self, model, dataset, K=10, p_user=0.5, p_item=0.5):
        self.model = model
        self.dataset = dataset
        self.K = K
        self.p_user = p_user
        self.p_item = p_item

    def rerank(self, user_id, candidate_items, scores):
        ranked_indices = np.argsort(-np.array(scores))
        ranked_items = [candidate_items[i] for i in ranked_indices]

        hot_items = self.dataset.hot_items
        cold_items = self.dataset.cold_items

        n_cold_target = max(1, int(self.K * self.p_item))

        final_list = []
        for pos in range(min(self.K, len(ranked_items))):
            cold_count = len([i for i in final_list if i in cold_items])
            added = False

            if cold_count < n_cold_target:
                for item in ranked_items:
                    if item not in final_list and item in cold_items:
                        final_list.append(item)
                        added = True
                        break
            if not added:
                for item in ranked_items:
                    if item not in final_list:
                        final_list.append(item)
                        added = True
                        break
            if not added:
                break

        for item in ranked_items:
            if item not in final_list:
                final_list.append(item)

        return final_list

    def predict(self, user_id, item_ids):
        return self.model.predict(user_id, item_ids)
