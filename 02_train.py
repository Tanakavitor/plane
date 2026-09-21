"""Step 2 - offline training of every ANN of the Intelligent Autopilot System (paper Sec. 3.B).

Outputs  models/ias_models.json   (weights, biases, scalers, set-points)
         reports/02_ann_fits.png  (what each ANN learned, over the demonstration data)
         reports/02_training.md   (MSE table, incl. held-out demonstrations)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ias.ann import ANN, save_models
from ias.config import MODELS_DIR, REPORTS_DIR
from ias.data import load_dataset
from ias.networks import SPECS, estimate_parameters

HOLD_OUT = ["voo-normal3", "voo-com-rajada", "turbulencia05"]   # calm / crosswind+gust / turbulence
EPOCHS = 150


def training_set(spec, frames, params):
    parts = [spec.build(d, params) for d in frames.values()]
    parts = [(x, y) for x, y in parts if len(y)]
    return np.vstack([x for x, _ in parts]), np.concatenate([y for _, y in parts])


def noise_floor(x, y, net, bins=25):
    """Irreducible MSE: variance of the (scaled) target around its conditional mean, estimated by
    binning the inputs. No function of these inputs can do better than this."""
    X, Y = np.clip(net.in_scaler.fwd(x), -1, 1), np.clip(net.out_scaler.fwd(y).ravel(), -0.9, 0.9)
    key = np.zeros(len(X), int)
    for j in range(X.shape[1]):
        key = key * bins + np.clip(((X[:, j] + 1) / 2 * bins).astype(int), 0, bins - 1)
    resid = Y - pd.Series(Y).groupby(key).transform("mean").to_numpy()
    return float(np.mean(resid ** 2))


def scaled_mse(net, x, y):
    pred = net.out_scaler.fwd(net.predict(x)[:, 0])
    return float(np.mean((pred - np.clip(net.out_scaler.fwd(y), -0.9, 0.9)) ** 2))


def main():
    meta, frames = load_dataset()
    params = estimate_parameters(frames, meta)
    print("set-points read from the demonstrations:", {k: round(v, 2) for k, v in params.items()})

    train = {k: v for k, v in frames.items() if k not in HOLD_OUT}
    held = {k: v for k, v in frames.items() if k in HOLD_OUT}

    models, rows, curves = {}, [], {}
    fig, axes = plt.subplots(4, 4, figsize=(22, 17))
    for spec, ax in zip(SPECS, axes.ravel()):
        # (a) generalisation check: train without the held-out demonstrations
        xt, yt = training_set(spec, train, params)
        xv, yv = training_set(spec, held, params)
        probe = ANN(xt.shape[1], spec.n_hidden, seed=1, odd=spec.odd)
        probe.fit(xt, yt, epochs=EPOCHS, input_pct=spec.input_pct, val=(xv, yv), min_epochs=EPOCHS)
        val = scaled_mse(probe, xv, yv)
        curves[spec.name] = (probe.meta["history"], probe.meta["val_history"], noise_floor(xt, yt, probe))
        # (b) deployed model: all demonstrations
        x, y = training_set(spec, frames, params)
        net = ANN(x.shape[1], spec.n_hidden, seed=1, odd=spec.odd)
        net.fit(x, y, epochs=EPOCHS, input_pct=spec.input_pct)
        net.meta.update(inputs=spec.inputs, output=spec.output, phase=spec.phase, val_mse=val)
        models[spec.name] = net
        rows.append((spec.name, spec.phase, " , ".join(spec.inputs), spec.output, len(y), net.meta["mse"], val, curves[spec.name][2]))
        print(f"{spec.name:20s} n={len(y):6d}  mse={net.meta['mse']:.4f}  held-out mse={val:.4f}")

        if x.shape[1] == 1:
            ax.scatter(x[:, 0], y, s=1, alpha=0.15, color="tab:gray")
            g = np.linspace(*np.percentile(x[:, 0], [0.5, 99.5]), 200)
            ax.plot(g, net.predict(g[:, None])[:, 0], color="tab:red", lw=2.5)
            ax.set_ylim(*np.percentile(y, [0.5, 99.5]))
            ax.set_xlabel(spec.inputs[0])
        else:
            for v, c in zip(np.percentile(x[:, 1], [10, 50, 90]), ("tab:blue", "tab:orange", "tab:red")):
                g = np.linspace(*np.percentile(x[:, 0], [0.5, 99.5]), 200)
                ax.plot(g, net.predict(np.c_[g, np.full_like(g, v)])[:, 0], color=c, lw=2.5, label=f"{spec.inputs[1]} = {v:.0f}")
            ax.set_ylim(*np.percentile(y, [0.5, 99.5]))
            ax.scatter(x[:, 0], y, s=1, alpha=0.1, color="tab:gray")
            ax.legend(); ax.set_xlabel(spec.inputs[0])
        ax.set_ylabel(spec.output); ax.grid(True)
        ax.set_title(f"{spec.name}  (mse {net.meta['mse']:.4f})")
    for ax in axes.ravel()[len(SPECS):]:
        ax.axis("off")
    fig.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / "02_ann_fits.png", dpi=60)

    fig, axes = plt.subplots(3, 5, figsize=(22, 11))
    for (name, (tr, va, floor)), ax in zip(curves.items(), axes.ravel()):
        ax.plot(tr, label="training (13 demos)"); ax.plot(va, label="validation (3 held-out demos)")
        ax.axhline(floor, color="0.4", ls="--", label="noise floor of the labels")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)"); ax.set_ylim(0, max(max(tr[:5]), max(va[:5]), floor) * 1.1); ax.grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Learning curves - MSE per epoch on the normalised targets")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "02_learning_curves.png", dpi=60)

    save_models(models, MODELS_DIR / "ias_models.json", extra=params)
    with open(REPORTS_DIR / "02_training.md", "w") as f:
        f.write("| ANN | phase | input(s) | output | samples | MSE (all demos) | MSE (held-out demos) | noise floor of labels |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} |\n")
        f.write("\nMSE is on targets scaled to [-0.9, 0.9]. 'Noise floor' = variance of the label around its conditional mean "
                "(inputs binned into 25 cells per dimension): the best MSE any function of these inputs could reach. "
                "Learning curves: reports/02_learning_curves.png\n")
        f.write(f"\nHeld-out demonstrations: {', '.join(HOLD_OUT)}\n\nSet-points: {params}\n")
    print("saved", MODELS_DIR / "ias_models.json")


if __name__ == "__main__":
    main()
