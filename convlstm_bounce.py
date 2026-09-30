import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_sequences(n_total=2200, T=10, size=32, r=2, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_total, T, size, size), np.float32)
    for s in range(n_total):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
            if y < r or y > size - r:
                vy = -vy
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
    return seqs[..., None]


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)]
        )
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x):
        b, _, _, hsize, wsize = x.shape
        states = [
            (
                torch.zeros(b, c.hid, hsize, wsize, device=x.device),
                torch.zeros(b, c.hid, hsize, wsize, device=x.device),
            )
            for c in self.cells
        ]
        for t in range(x.shape[1]):
            xt = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]
        return self.out(states[-1][0]).squeeze(1)


class FlattenLSTM(nn.Module):
    def __init__(self, size=32, hidden=256, num_layers=1):
        super().__init__()
        self.size = size
        in_features = size * size
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden, in_features)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, _, h, w = x.shape
        flat = x[:, :, 0].reshape(b, t, h * w)
        out, _ = self.lstm(flat)
        out = self.out(out[:, -1])
        return out.reshape(b, h, w)


def make_model(name):
    if name == "baseline":
        return ConvLSTM(in_ch=1, hid=32, k=3, layers=1)
    if name == "exp1":
        return ConvLSTM(in_ch=1, hid=32, k=3, layers=2)
    if name == "exp2":
        return ConvLSTM(in_ch=1, hid=64, k=3, layers=1)
    if name == "exp3":
        return ConvLSTM(in_ch=1, hid=32, k=5, layers=1)
    if name == "exp4":
        return ConvLSTM(in_ch=1, hid=32, k=3, layers=1)
    if name == "exp5":
        return FlattenLSTM(size=32, hidden=256, num_layers=1)
    if name == "exp6":
        return ConvLSTM(in_ch=1, hid=32, k=3, layers=1)
    if name == "exp7":
        return ConvLSTM(in_ch=1, hid=32, k=5, layers=2)
    raise ValueError(name)


def eval_model(model, xs, ys, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, xs.shape[0], batch_size):
            preds.append(model(xs[i : i + batch_size]))
    pred = torch.cat(preds, dim=0)
    mse = (pred - ys).pow(2).mean().item()
    mae = (pred - ys).abs().mean().item()
    return pred, mse, mae


def train_model(model, train_x, train_y, test_x, test_y, loss_fn, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
            batches += idx.shape[0]
        train_loss = total_loss / batches
        pred, mse, mae = eval_model(model, test_x, test_y)
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": float(train_loss),
                "test_mse": float(mse),
                "test_mae": float(mae),
            }
        )
        print(
            "epoch %d train_loss %.6f test_MSE %.6f test_MAE %.6f"
            % (ep + 1, train_loss, mse, mae),
            flush=True,
        )
    elapsed = time.time() - t0
    return pred, history, elapsed


def plot_curves(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_mse = [h["test_mse"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(epochs, train_loss, "o-", color="#2b6cb0", label="训练损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax2 = ax1.twinx()
    ax2.plot(epochs, test_mse, "s--", color="#c53030", label="测试 MSE")
    ax2.set_ylabel("测试 MSE", color="#c53030")
    ax2.tick_params(axis="y", labelcolor="#c53030")
    fig.suptitle("训练损失与测试 MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prediction_comparison(xs, ys, pred, path, sample_indices=(0, 67, 133)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    sample_indices = list(range(min(3, xs.shape[0])))
    fig, axes = plt.subplots(3, 6, figsize=(13.5, 7.5))
    for row, si in enumerate(sample_indices):
        input_frames = xs[si, -4:, 0].cpu().numpy()
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(input_frames[col], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"输入帧 {col + 1}", fontsize=10)
            ax.axis("off")
        mse = (pred[si] - ys[si]).pow(2).mean().item()
        axes[row, 4].imshow(ys[si].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 4].set_title("真实帧", fontsize=10)
        axes[row, 4].axis("off")
        axes[row, 5].imshow(pred[si].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 5].set_title(f"预测帧  MSE={mse:.4f}", fontsize=10)
        axes[row, 5].axis("off")
    fig.suptitle("输入帧 | 真实帧 | 预测帧 对比")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_data(n_train, n_test, k_in):
    data = make_sequences(n_total=n_train + n_test)
    train_x = torch.from_numpy(data[:n_train, :k_in]).permute(0, 1, 4, 2, 3)
    train_y = torch.from_numpy(data[:n_train, k_in, ..., 0])
    test_x = torch.from_numpy(data[n_train : n_train + n_test, :k_in]).permute(0, 1, 4, 2, 3)
    test_y = torch.from_numpy(data[n_train : n_train + n_test, k_in, ..., 0])
    return train_x, train_y, test_x, test_y


def run_experiment(name, out_dir, epochs, n_train, n_test, bs):
    configs = {
        "baseline": {"loss": "mse", "k_in": 4},
        "exp1": {"loss": "mse", "k_in": 4},
        "exp2": {"loss": "mse", "k_in": 4},
        "exp3": {"loss": "mse", "k_in": 4},
        "exp4": {"loss": "mse", "k_in": 8},
        "exp5": {"loss": "mse", "k_in": 4},
        "exp6": {"loss": "l1", "k_in": 4},
        "exp7": {"loss": "mse", "k_in": 4},
    }
    cfg = configs[name]
    k_in = cfg["k_in"]
    train_x, train_y, test_x, test_y = prepare_data(n_train, n_test, k_in)
    model = make_model(name)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params={params} k_in={k_in}", flush=True)
    loss_fn = nn.L1Loss() if cfg["loss"] == "l1" else nn.MSELoss()
    pred, history, elapsed = train_model(
        model, train_x, train_y, test_x, test_y, loss_fn, epochs, bs, 0.001, 42
    )
    final = history[-1]
    result = {
        "name": name,
        "k_in": k_in,
        "loss": cfg["loss"],
        "params": params,
        "elapsed_seconds": elapsed,
        "test_mse": final["test_mse"],
        "test_mae": final["test_mae"],
        "history": history,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    curve_name = "result_baseline.png" if name == "baseline" else f"{name}_curve.png"
    if name == "baseline":
        curve_name = "result_baseline.png"
    elif name == "exp7":
        curve_name = "result_exp7.png"
    else:
        curve_name = f"result_{name}.png"
    plot_curves(history, str(out_path / curve_name))
    if name in ("baseline", "exp7"):
        plot_prediction_comparison(test_x, test_y, pred, str(out_path / "result_pred.png"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    threads = args.threads or max(1, min(os.cpu_count() or 1, 8))
    torch.set_num_threads(threads)
    names = [s.strip() for s in args.experiments.split(",") if s.strip()]
    all_results = {}
    for name in names:
        result = run_experiment(
            name,
            args.out_dir,
            args.epochs,
            args.n_train,
            args.n_test,
            args.batch_size,
        )
        all_results[name] = {
            "params": result["params"],
            "k_in": result["k_in"],
            "loss": result["loss"],
            "elapsed_seconds": result["elapsed_seconds"],
            "test_mse": result["test_mse"],
            "test_mae": result["test_mae"],
        }
        print(
            f"DONE {name} MSE={result['test_mse']:.6f} MAE={result['test_mae']:.6f} "
            f"elapsed={result['elapsed_seconds']:.1f}s params={result['params']}",
            flush=True,
        )
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
