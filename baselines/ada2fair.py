class Ada2Fair:
    def __init__(self, model, dataset, lambda_fair=0.1):
        self.model = model
        self.dataset = dataset
        self.lambda_fair = lambda_fair
        self.sample_weights = None

    def compute_adaptive_weights(self):
        train_data = self.dataset.train_data
        dis_users = self.dataset.dis_users
        cold_items = self.dataset.cold_items

        self.sample_weights = {}
        for uid, items in train_data.items():
            for iid in items:
                if uid in dis_users:
                    user_weight = 2.0
                else:
                    user_weight = 1.0

                if iid in cold_items:
                    item_weight = 2.0
                else:
                    item_weight = 1.0

                self.sample_weights[(uid, iid)] = user_weight * item_weight
        return self.sample_weights

    def get_weight(self, user_id, item_id):
        if self.sample_weights is None:
            self.compute_adaptive_weights()
        return self.sample_weights.get((user_id, item_id), 1.0)

    def predict(self, user_id, item_ids):
        return self.model.predict(user_id, item_ids)
