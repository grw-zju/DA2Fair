import numpy as np
import torch
from evaluation.metrics import evaluate_full_ranking, evaluate_recommendation_vaecf


class Evaluator:
    def __init__(self, dataset, k=10, device=None, split='test',
                 model_name='MF', edge_index=None):
        self.dataset = dataset
        self.k = k
        self.device = device or torch.device('cpu')
        self.split = split
        self.model_name = model_name
        self.edge_index = edge_index

    def evaluate(self, model, mode='full'):
        if self.model_name == 'VAECF':
            return evaluate_recommendation_vaecf(model, self.dataset, self.k, self.split)
        return evaluate_full_ranking(model, self.dataset, self.k, self.split,
                                      self.model_name, self.edge_index)
