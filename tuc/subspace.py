# tuc/subspace.py
import numpy as np

def fit_space(X, k=96, whiten=False):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:k].T                      # [D, k]
    W = (basis / S[:k]) if whiten else basis
    return {"mean": mu.astype("float32"), "basis": W.astype("float32"), "k": int(k), "whiten": bool(whiten)}

def apply_space(vecs, space):
    mu = space["mean"]
    W  = space["basis"]
    Z = (vecs - mu) @ W                   # [N, k]
    # L2 normalize row-wise
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z
