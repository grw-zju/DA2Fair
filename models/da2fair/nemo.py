import torch
import numpy as np
from itertools import combinations


class NEMO:
    def __init__(self, max_iter=20, tau=1e-3, xi=1e-3,
                 eps_rho=1e-12, eps_r=1e-12, norm_eps=1e-8,
                 simplex_tol=1e-10, vanilla=False):
        self.max_iter = max_iter
        self.tau = tau
        self.xi = xi
        self.eps_rho = eps_rho
        self.eps_r = eps_r
        self.norm_eps = norm_eps
        self.simplex_tol = simplex_tol
        self.vanilla = vanilla
        self.decision = 'skip'

    def compute_min_norm_weights(self, gradients):
        K = len(gradients)
        G = torch.stack([g.view(-1) for g in gradients])
        GG = torch.mm(G, G.T).cpu().numpy().astype(np.float64)

        best_alpha = np.ones(K) / K
        best_norm_sq = float('inf')

        for size in range(1, K + 1):
            for subset in combinations(range(K), size):
                idx = list(subset)
                s = len(idx)

                if s == 1:
                    alpha = np.zeros(K, dtype=np.float64)
                    alpha[idx[0]] = 1.0
                elif s == K:
                    ones = np.ones(K, dtype=np.float64)
                    try:
                        inv_GG = np.linalg.inv(GG)
                    except np.linalg.LinAlgError:
                        continue
                    raw = inv_GG @ ones
                    ssum = ones @ raw
                    if abs(ssum) < 1e-30:
                        continue
                    alpha = raw / ssum
                else:
                    sub_GG = GG[np.ix_(idx, idx)]
                    ones_s = np.ones(s, dtype=np.float64)
                    try:
                        inv_sub = np.linalg.inv(sub_GG)
                    except np.linalg.LinAlgError:
                        continue
                    raw = inv_sub @ ones_s
                    ssum = ones_s @ raw
                    if abs(ssum) < 1e-30:
                        continue
                    sub_alpha = raw / ssum
                    alpha = np.zeros(K, dtype=np.float64)
                    alpha[idx] = sub_alpha

                if np.all(alpha >= -self.simplex_tol) and \
                   abs(alpha.sum() - 1.0) < self.simplex_tol:
                    alpha = np.clip(alpha, 0, None)
                    ssum = alpha.sum()
                    if ssum > 0:
                        alpha = alpha / ssum
                    else:
                        continue
                    norm_sq = float(alpha @ (GG @ alpha))
                    if norm_sq < best_norm_sq:
                        best_norm_sq = norm_sq
                        best_alpha = alpha

        return torch.tensor(best_alpha, dtype=gradients[0].dtype,
                            device=gradients[0].device)

    def compute_nash_weights(self, gradients):
        K = len(gradients)
        G = torch.stack([g.view(-1) for g in gradients])
        GG = torch.mm(G, G.T)
        GG_np = GG.cpu().numpy().astype(np.float64)

        alpha = np.ones(K, dtype=np.float64) / K
        for _ in range(self.max_iter):
            denom = GG_np @ alpha + 1e-30
            new_alpha = alpha / denom
            s = new_alpha.sum()
            if s > 0:
                new_alpha = new_alpha / s
            else:
                new_alpha = np.ones(K, dtype=np.float64) / K
                break
            if np.allclose(alpha, new_alpha, atol=1e-10):
                alpha = new_alpha
                break
            alpha = new_alpha

        return torch.tensor(alpha, dtype=gradients[0].dtype,
                            device=gradients[0].device)

    def coordinate(self, gradients):
        self.decision = 'skip'
        K = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        G = torch.stack([g.view(-1) for g in gradients])

        alpha_F = self.compute_min_norm_weights(gradients)
        g_min = sum(alpha_F[k] * gradients[k] for k in range(K))

        g_min_norm = torch.norm(g_min).item()
        sum_norms = sum(torch.norm(gradients[k]).item()
                        for k in range(K))
        rho = g_min_norm / (sum_norms / (K - 1) + self.eps_rho)

        if self.vanilla:
            alpha_N = self.compute_nash_weights(gradients)
            g_N = sum(alpha_N[k] * gradients[k] for k in range(K))
            self.decision = 'nash'
            return g_N

        if rho <= self.tau:
            self.decision = 'skip'
            return None

        alpha_N = self.compute_nash_weights(gradients)

        G_np = G.cpu().numpy().astype(np.float64)
        GG_np = G_np @ G_np.T
        alpha_N_np = alpha_N.cpu().numpy().astype(np.float64)

        lhs = GG_np @ alpha_N_np
        rhs = 1.0 / (alpha_N_np + 1e-30)
        residual = np.max(np.abs(lhs - rhs))
        denom_r = np.max(np.abs(rhs)) + self.eps_r
        r_N = residual / denom_r

        if not np.all(np.isfinite(alpha_N_np)):
            self.decision = 'fallback'
            return g_min

        if np.any(alpha_N_np <= 0):
            self.decision = 'fallback'
            return g_min

        if r_N > self.xi:
            self.decision = 'fallback'
            return g_min

        g_N = sum(alpha_N[k] * gradients[k] for k in range(K))

        all_descent = True
        for k in range(K):
            dot = torch.dot(gradients[k].view(-1), g_N.view(-1)).item()
            if dot <= 0:
                all_descent = False
                break

        if not all_descent:
            self.decision = 'fallback'
            return g_min

        self.decision = 'nash'
        return g_N

    def aggregate_gradients(self, gradients, weights):
        result = torch.zeros_like(gradients[0])
        for k in range(len(gradients)):
            result = result + weights[k] * gradients[k]
        return result
