# ConvLSTM 弹跳小球时空预测实验

这是《人工智能——深度学习大模型智能体》实验作业二的代码仓库，实现并复现了基于 ConvLSTM 的弹跳小球下一帧预测实验。

## 实验内容

代码生成 2200 条 32×32 弹跳小球序列，其中 2000 条用于训练、200 条用于测试，每条 10 帧。模型观察前若干帧并预测下一帧，同时完成以下对照实验：

| 组号 | 实验名称 | 改动内容 |
|---|---|---|
| 0 | 基线 Baseline | 1 层，C_h=32，K=3，看 4 帧，MSE |
| 1 | 加深层数 | 2 层 ConvLSTM |
| 2 | 隐藏通道 | C_h 32 → 64 |
| 3 | 卷积核 | K=3 → 5 |
| 4 | 输入帧数 | 看 4 帧 → 看 8 帧 |
| 5 | 结构对照 | 全连接 LSTM（展平输入） |
| 6 | 损失函数 | MSE → L1 |
| 7 | 最优组合 | 本次采用最佳完成配置 K=5、单层、C_h=32、看 4 帧 |

## 环境

- Python 3.10 及以上
- PyTorch 2.0 及以上
- NumPy / Matplotlib

CPU 版 PyTorch 安装可参考：

```bash
python -m venv venv_lab02
venv_lab02\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib
```

## 运行

运行基线：

```bash
python convlstm_bounce.py --experiments baseline --epochs 5 --n-train 2000 --n-test 200 --batch-size 64 --out-dir output
```

依次运行 6 组对照实验：

```bash
python convlstm_bounce.py --experiments exp1,exp2,exp3,exp4,exp5,exp6 --epochs 5 --n-train 2000 --n-test 200 --batch-size 64 --out-dir output
```

参数说明：

- `--experiments`：实验名，多个实验用英文逗号分隔。
- `--epochs`：训练轮数。
- `--n-train`：训练序列数。
- `--n-test`：测试序列数。
- `--batch-size`：批大小。
- `--threads`：PyTorch CPU 线程数，`0` 表示自动。
- `--out-dir`：结果输出目录。

## 结果文件

运行后 `output/` 目录会生成：

- `summary.json`：各组实验汇总指标。
- `*_metrics.json`：每个实验的分轮训练损失和测试 MSE/MAE。
- `result_baseline.png`、`result_exp1.png` 到 `result_exp6.png`：训练损失与测试 MSE 曲线。
- `result_pred.png`：输入帧、真实帧与预测帧对比图。

