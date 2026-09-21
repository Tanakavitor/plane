"""The ANN used throughout the Intelligent Autopilot System (Baomar & Bentley 2021, Sec. 3.B):
a small fully-connected feed-forward network with ONE hidden layer, tanh activations (Eq. 1),
trained offline by back-propagation (Eq. 2) with learning rate 0.1 and momentum 0.9 (Eq. 3).

Pure NumPy on purpose: the models are a handful of coefficients that can be stored as JSON,
inspected, plotted and verified - the paper's argument against black-box deep models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEARNING_RATE = 0.1
MOMENTUM = 0.9
TARGET_SPAN = 0.9        # targets are scaled into [-0.9, 0.9] so tanh never has to saturate
INPUT_PCT = (0.5, 99.5)  # inputs are clipped to the demonstrated range: outside it the ANN holds its
                         # edge value instead of extrapolating (per-ANN override: Spec.input_pct)


class Scaler:
    """Affine map of a physical range onto [-1, 1] (inputs) or [-0.9, 0.9] (targets)."""

    def __init__(self, lo, hi, span: float = 1.0):
        self.lo, self.hi, self.span = np.asarray(lo, float), np.asarray(hi, float), span

    def fwd(self, v):
        return (2.0 * (np.asarray(v, float) - self.lo) / (self.hi - self.lo) - 1.0) * self.span

    def inv(self, s):
        return (np.asarray(s, float) / self.span + 1.0) / 2.0 * (self.hi - self.lo) + self.lo

    def to_json(self):
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist(), span=self.span)

    @classmethod
    def from_json(cls, d):
        return cls(d["lo"], d["hi"], d["span"])


class ANN:
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1, seed: int = 0, odd: bool = False):
        # odd=True: no bias terms and symmetric scalers -> f(-x) = -f(x) exactly. Used for the lateral
        # ANNs, where "no error" must mean "no command" and left/right must be treated alike.
        self.odd = odd
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1, 1, (n_in, n_hidden)) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.uniform(-1, 1, (n_hidden, n_out)) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_out)
        self.in_scaler: Scaler | None = None
        self.out_scaler: Scaler | None = None
        self.meta: dict = {}

    # ---- forward passes ------------------------------------------------------------------
    def _forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)                     # Eq. (1)
        return H, np.tanh(H @ self.W2 + self.b2)

    def predict(self, x):
        """Physical units in -> physical units out. Inputs are clipped to the demonstrated
        range so the network is never asked to extrapolate beyond what the pilot showed."""
        X = np.clip(self.in_scaler.fwd(np.atleast_2d(x)), -1.0, 1.0)
        return self.out_scaler.inv(self._forward(X)[1])

    def __call__(self, *x) -> float:
        return float(self.predict(np.asarray(x, float))[0, 0])

    # ---- training ------------------------------------------------------------------------
    def fit(self, x, y, epochs: int = 300, batch: int = 32, target_mse: float = 0.01,
            min_epochs: int = 40, seed: int = 0, verbose: bool = False, input_pct=INPUT_PCT,
            val: tuple | None = None) -> list[float]:
        """Returns the training MSE per epoch (scaled units). With val=(x, y) the validation MSE per
        epoch is also recorded in self.meta["val_history"]."""
        x, y = np.atleast_2d(x.T).T.astype(float), np.atleast_2d(y.T).T.astype(float)
        lo, hi = np.percentile(x, input_pct[0], axis=0), np.percentile(x, input_pct[1], axis=0)
        ylo, yhi = np.percentile(y, 0.5, axis=0), np.percentile(y, 99.5, axis=0)
        if self.odd:
            hi, yhi = np.maximum(np.abs(lo), np.abs(hi)), np.maximum(np.abs(ylo), np.abs(yhi))
            lo, ylo = -hi, -yhi
        self.in_scaler = Scaler(lo, hi)
        self.out_scaler = Scaler(ylo, yhi, TARGET_SPAN)
        X = np.clip(self.in_scaler.fwd(x), -1, 1)
        Y = np.clip(self.out_scaler.fwd(y), -TARGET_SPAN, TARGET_SPAN)

        if val is not None:
            xv, yv = np.atleast_2d(val[0].T).T.astype(float), np.atleast_2d(val[1].T).T.astype(float)
            XV, YV = np.clip(self.in_scaler.fwd(xv), -1, 1), np.clip(self.out_scaler.fwd(yv), -TARGET_SPAN, TARGET_SPAN)
        rng = np.random.default_rng(seed)
        vel = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        history, val_history = [], []
        for ep in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = order[i:i + batch]
                xb, yb = X[idx], Y[idx]
                H, O = self._forward(xb)
                d_out = (O - yb) * (1.0 - O ** 2)                # Eq. (2): tanh' = 1 - tanh^2
                d_hid = (d_out @ self.W2.T) * (1.0 - H ** 2)
                grads = (xb.T @ d_hid / len(idx), d_hid.mean(0), H.T @ d_out / len(idx), d_out.mean(0))
                for k, (p, v, g) in enumerate(zip((self.W1, self.b1, self.W2, self.b2), vel, grads)):
                    if self.odd and k in (1, 3):
                        continue                                 # biases stay at zero
                    v *= MOMENTUM                                # Eq. (3)
                    v -= LEARNING_RATE * g
                    p += v
            mse = float(np.mean((self._forward(X)[1] - Y) ** 2))
            history.append(mse)
            if val is not None:
                val_history.append(float(np.mean((self._forward(XV)[1] - YV) ** 2)))
            if verbose and ep % 20 == 0:
                print(f"    epoch {ep:4d}  mse {mse:.5f}")
            if ep + 1 >= min_epochs and (mse < target_mse and abs(history[-10] - mse) < 1e-5):
                break
        self.meta.update(mse=history[-1], epochs=len(history), n_samples=int(len(X)), history=history, val_history=val_history)
        return history

    # ---- persistence ---------------------------------------------------------------------
    def to_json(self) -> dict:
        return dict(W1=self.W1.tolist(), b1=self.b1.tolist(), W2=self.W2.tolist(), b2=self.b2.tolist(),
                    in_scaler=self.in_scaler.to_json(), out_scaler=self.out_scaler.to_json(), odd=self.odd, meta=self.meta)

    @classmethod
    def from_json(cls, d: dict) -> "ANN":
        W1 = np.array(d["W1"])
        net = cls(W1.shape[0], W1.shape[1], len(d["b2"]), odd=d.get("odd", False))
        net.W1, net.b1, net.W2, net.b2 = W1, np.array(d["b1"]), np.array(d["W2"]), np.array(d["b2"])
        net.in_scaler, net.out_scaler = Scaler.from_json(d["in_scaler"]), Scaler.from_json(d["out_scaler"])
        net.meta = d.get("meta", {})
        return net


def save_models(models: dict[str, ANN], path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(extra=extra or {}, anns={k: m.to_json() for k, m in models.items()}), indent=1))


def load_models(path: Path) -> tuple[dict[str, ANN], dict]:
    d = json.loads(Path(path).read_text())
    return {k: ANN.from_json(v) for k, v in d["anns"].items()}, d.get("extra", {})
