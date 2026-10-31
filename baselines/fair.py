import numpy as np


class FAIR:
    def __init__(self, model, dataset, lambda_fair=0.1, learning_rate=0.01):
        self.model = model
        self.dataset = dataset
        self.lambda_fair = lambda_fair
        self.learning_rate = learning_rate
        self.constraint_values = {'user': 0.0, 'item': 0.0}

    def update_constraints(self, user_ndcg_adv, user_ndcg_dis, hot_exp, cold_exp):
        user_violation = abs(user_ndcg_adv - user_ndcg_dis)
        item_violation = abs(hot_exp - cold_exp)

        self.constraint_values['user'] += self.learning_rate * user_violation
        self.constraint_values['item'] += self.learning_rate * item_violation

    def compute_fair_penalty(self):
        return self.lambda_fair * (self.constraint_values['user'] + self.constraint_values['item'])

    def predict(self, user_id, item_ids):
        return self.model.predict(user_id, item_ids)
