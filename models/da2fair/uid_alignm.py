import numpy as np
import ot
from sklearn.metrics.pairwise import cosine_similarity


def cosine_dissimilarity(X, Y):
    sim = cosine_similarity(X, Y)
    return 1.0 - sim


def compute_kernel_from_dist(Cx, Cy, h):
    std1 = np.sqrt((Cx ** 2).mean() / 2)
    std2 = np.sqrt((Cy ** 2).mean() / 2)
    h1 = max(h * std1, 1e-8)
    h2 = max(h * std2, 1e-8)
    Kx = np.exp(-(Cx / h1) ** 2 / 2)
    Ky = np.exp(-(Cy / h2) ** 2 / 2)
    return Kx, Ky


def compute_kernel_from_emb(X, Y, h):
    Cx = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    Cy = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    return compute_kernel_from_dist(Cx, Cy, h)


def activity_aware_marginals(nu_s, n_tgt, delta=1.0, gamma=0.5):
    n = len(nu_s)
    raw = (nu_s + delta) ** (-gamma)
    p = raw / raw.sum()
    q = np.ones(n_tgt) / n_tgt
    return p, q


def compute_infoot_mi(P, Kx, Ky, p, q):
    F_Gamma = Kx @ P @ Ky.T
    M_X = Kx @ p
    M_Y = Ky @ q
    denom = np.outer(M_X, M_Y)
    R = F_Gamma / (denom + 1e-30)
    mi_hat = np.sum(P * np.log(R + 1e-30))
    return mi_hat


def compute_infoot_mi_gradient(P, Kx, Ky, p, q):
    F_Gamma = Kx @ P @ Ky.T
    M_X = Kx @ p
    M_Y = Ky @ q
    denom = np.outer(M_X, M_Y)
    R = F_Gamma / (denom + 1e-30)
    grad = np.log(R + 1e-30)
    P_over_F = P / (F_Gamma + 1e-30)
    grad = grad + Kx.T @ P_over_F @ Ky
    return -grad


def solve_fused_infoot(X_s, Y_t, nu_s=None, delta=1.0, gamma=0.5,
                       lam=0.1, eps=0.05, h=0.5,
                       J=3, S=20, align_terms='GL'):
    n_s = X_s.shape[0]
    n_t = Y_t.shape[0]

    X_norm = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-12)
    Y_norm = Y_t / (np.linalg.norm(Y_t, axis=1, keepdims=True) + 1e-12)

    C = cosine_dissimilarity(X_norm, Y_norm)

    if align_terms == 'L':
        C = np.zeros_like(C)

    if nu_s is not None and gamma > 0:
        p, q = activity_aware_marginals(nu_s, n_t, delta, gamma)
    else:
        p = np.ones(n_s) / n_s
        q = np.ones(n_t) / n_t

    Kx, Ky = compute_kernel_from_emb(X_norm, Y_norm, h)

    P = np.outer(p, q)

    if align_terms == 'G':
        P = ot.bregman.sinkhorn(
            p, q, C, reg=eps, numItermax=S, warn=False)
    else:
        for ell in range(J):
            grad_mi = compute_infoot_mi_gradient(P, Kx, Ky, p, q)
            C_tilde = C + lam * grad_mi
            P = ot.bregman.sinkhorn(
                p, q, C_tilde, reg=eps, numItermax=S, warn=False)

    return P, C, Kx, Ky, p, q


def compute_alignment_loss(source_emb_np, target_emb_np, nu_s=None,
                           delta=1.0, gamma=0.5, lam=0.1, eps=0.05,
                           h=0.5, J=3, S=20, align_terms='GL'):
    P, C, Kx, Ky, p, q = solve_fused_infoot(
        source_emb_np, target_emb_np, nu_s, delta, gamma, lam, eps, h, J, S,
        align_terms=align_terms)

    geo_loss = np.sum(P * C)
    mi_hat = compute_infoot_mi(P, Kx, Ky, p, q)

    if align_terms == 'G':
        total_loss = geo_loss
    elif align_terms == 'L':
        total_loss = -lam * mi_hat
    else:
        total_loss = geo_loss - lam * mi_hat

    weights = np.sum(P, axis=1)
    proj_source = np.matmul(P, target_emb_np) / weights[:, None]

    return float(total_loss), proj_source, P, Kx, Ky, p, q, mi_hat


def compute_infoot_alignment_loss(source_emb_np, target_emb_np, h=0.5,
                                  lam=0.1, reg=0.05, numIter=3,
                                  nu_s=None, delta=1.0, gamma=0.5,
                                  align_terms='GL'):
    return compute_alignment_loss(
        source_emb_np, target_emb_np, nu_s=nu_s,
        delta=delta, gamma=gamma, lam=lam, eps=reg, h=h, J=numIter, S=20,
        align_terms=align_terms)[:3]


class FusedInfoOT:
    def __init__(self, Xs, Xt, h=0.5, Ys=None, lam=0.1, reg=0.05,
                 nu_s=None, delta=1.0, gamma=0.5):
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.h = h
        self.lam = lam
        self.reg = reg
        self.nu_s = nu_s
        self.delta = delta
        self.gamma = gamma

        X_norm = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
        Y_norm = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)

        self.C = cosine_dissimilarity(X_norm, Y_norm)

        if nu_s is not None and gamma > 0:
            self.p, self.q = activity_aware_marginals(nu_s, len(Xt), delta, gamma)
        else:
            n_s = len(Xs)
            n_t = len(Xt)
            self.p = np.ones(n_s) / n_s
            self.q = np.ones(n_t) / n_t

        self.Ks, self.Kt = compute_kernel_from_emb(X_norm, Y_norm, h)
        self.P = None

    def solve(self, numIter=3, S=20):
        P = np.outer(self.p, self.q)
        for i in range(numIter):
            grad_mi = compute_infoot_mi_gradient(
                P, self.Ks, self.Kt, self.p, self.q)
            C_tilde = self.C + self.lam * grad_mi
            P = ot.bregman.sinkhorn(
                self.p, self.q, C_tilde, reg=self.reg,
                numItermax=S, warn=False)
        self.P = P
        return P

    def project(self, X=None, method='barycentric', h=None):
        if self.P is None:
            raise Exception('please run FusedInfoOT.solve() first')
        weights = np.sum(self.P, axis=1)
        return np.matmul(self.P, self.Xt) / weights[:, None]


class InfoOT:
    def __init__(self, Xs, Xt, h=0.5, reg=0.05, nu_s=None,
                 delta=1.0, gamma=0.5):
        self.Xs = Xs
        self.Xt = Xt
        self.h = h
        self.reg = reg
        self.nu_s = nu_s
        self.delta = delta
        self.gamma = gamma

        X_norm = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
        Y_norm = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)

        self.C = cosine_dissimilarity(X_norm, Y_norm)

        if nu_s is not None and gamma > 0:
            self.p, self.q = activity_aware_marginals(nu_s, len(Xt), delta, gamma)
        else:
            n_s = len(Xs)
            n_t = len(Xt)
            self.p = np.ones(n_s) / n_s
            self.q = np.ones(n_t) / n_t

        self.Ks, self.Kt = compute_kernel_from_emb(X_norm, Y_norm, h)
        self.P = None

    def solve(self, numIter=3, S=20):
        P = np.outer(self.p, self.q)
        for i in range(numIter):
            grad_mi = compute_infoot_mi_gradient(
                P, self.Ks, self.Kt, self.p, self.q)
            P = ot.bregman.sinkhorn(
                self.p, self.q, grad_mi, reg=self.reg,
                numItermax=S, warn=False)
        self.P = P
        return P

    def project(self, X=None, method='barycentric', h=None):
        if self.P is None:
            raise Exception('please run InfoOT.solve() first')
        weights = np.sum(self.P, axis=1)
        return np.matmul(self.P, self.Xt) / weights[:, None]
