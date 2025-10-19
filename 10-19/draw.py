#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法一（自动找最新目录）：
    python plot_results.py

用法二（命令行手动指定）：
    python plot_results.py \
        --eval_xlsx  result_cur/train_10_19_1935/eval.xlsx \
        --train_xlsx result_cur/train_10_19_1935/train.xlsx \
        --out_dir    result_cur/train_10_19_1935 \
        --show

用法三（脚本内手动设置路径）：
    在本文件顶部把 USE_MANUAL_PATHS = True，并填写 EVAL_PATH / TRAIN_PATH / OUT_DIR，
    然后直接运行：python plot_results.py
依赖：
    pip install pandas matplotlib openpyxl
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ========== 方式B：脚本内手动设置路径 ==========
USE_MANUAL_PATHS = False
EVAL_PATH  = r"result_cur/train_10_19_1935/eval.xlsx"
TRAIN_PATH = r"result_cur/train_10_19_1935/train.xlsx"
OUT_DIR    = r"result_cur/train_10_19_1935"

# ========== 基础工具 ==========
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def plot_series(x, y, xlabel, ylabel, title, out_path, show=False):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close()

def find_latest_run_dir(root="result_cur") -> str | None:
    """在 root 下查找匹配 train_* 的目录，按修改时间降序返回最新一个。"""
    if not os.path.isdir(root):
        return None
    cand = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and re.match(r"^train_", name):
            cand.append((path, os.path.getmtime(path)))
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def infer_defaults(eval_xlsx: str | None, train_xlsx: str | None, out_dir: str | None):
    """
    路径推断逻辑：
      * 若命令行传了 eval/train，则优先用命令行；out_dir 默认=eval.xlsx 所在目录
      * 若没传，则自动找 result_cur 下最新的 train_* 目录
    """
    if eval_xlsx and train_xlsx:
        out_dir = out_dir or os.path.dirname(os.path.abspath(eval_xlsx))
        return eval_xlsx, train_xlsx, out_dir

    latest_dir = find_latest_run_dir("result_cur")
    if latest_dir is None:
        raise FileNotFoundError(
            "未找到默认目录：result_cur 下没有 train_* 结果目录。"
            "请用 --eval_xlsx/--train_xlsx 指定，或在脚本顶部开启 USE_MANUAL_PATHS。"
        )
    inferred_eval  = os.path.join(latest_dir, "eval.xlsx")
    inferred_train = os.path.join(latest_dir, "train.xlsx")

    if not os.path.isfile(inferred_eval):
        raise FileNotFoundError(f"自动推断失败，找不到文件：{inferred_eval}")
    if not os.path.isfile(inferred_train):
        raise FileNotFoundError(f"自动推断失败，找不到文件：{inferred_train}")

    out_dir = out_dir or latest_dir
    return inferred_eval, inferred_train, out_dir

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="Load two Excel files and plot 4 figures.")
    parser.add_argument("--eval_xlsx",  type=str, default=None, help="Path to eval.xlsx (epoch, val_return, eval_time, train_time).")
    parser.add_argument("--train_xlsx", type=str, default=None, help="Path to train.xlsx (epoch, update, reward, loss).")
    parser.add_argument("--out_dir",    type=str, default=None, help="Output directory for pngs.")
    parser.add_argument("--show",       action="store_true",    help="画完每张图后弹窗显示。")
    args = parser.parse_args()

    # 0) 如果脚本内手动路径开关打开，则直接使用手动路径
    if USE_MANUAL_PATHS:
        eval_path  = EVAL_PATH
        train_path = TRAIN_PATH
        out_dir    = OUT_DIR or os.path.dirname(os.path.abspath(eval_path))
    else:
        # 1) 路径推断/检查：命令行优先，缺省则自动找最新目录
        eval_path, train_path, out_dir = infer_defaults(args.eval_xlsx, args.train_xlsx, args.out_dir)

    print(f"Using eval_xlsx : {eval_path}")
    print(f"Using train_xlsx: {train_path}")
    print(f"Output to       : {out_dir}")
    ensure_dir(out_dir)

    # 2) 读取 Excel
    df_eval  = pd.read_excel(eval_path)
    df_train = pd.read_excel(train_path)

    # 3) 列检查
    need_eval_cols_base = {"epoch", "val_return"}
    if not need_eval_cols_base.issubset(df_eval.columns):
        raise ValueError(f"eval.xlsx 至少需要列: {need_eval_cols_base}，当前列：{set(df_eval.columns)}")

    has_eval_time = "eval_time" in df_eval.columns
    if not has_eval_time:
        print("⚠️ 提示：eval.xlsx 中未发现列 'eval_time'，将跳过 Evaluation Time 曲线绘制。")

    need_train_cols = {"epoch", "reward", "loss"}
    if not need_train_cols.issubset(df_train.columns):
        raise ValueError(f"train.xlsx 需要包含列: {need_train_cols}，当前列：{set(df_train.columns)}")

    # 4) 聚合训练数据到每个 epoch 的均值
    df_ep = df_train.groupby("epoch", as_index=False).agg(
        mean_reward=("reward", "mean"),
        mean_loss=("loss", "mean")
    )

    # 5) 画图并保存（四张图）
    plot_series(
        x=df_eval["epoch"],
        y=df_eval["val_return"],
        xlabel="epoch",
        ylabel="val_return (greedy on fixed-100)",
        title="Validation Return over Epochs",
        out_path=os.path.join(out_dir, "val_return_curve.png"),
        show=args.show
    )

    if has_eval_time:
        plot_series(
            x=df_eval["epoch"],
            y=df_eval["eval_time"],
            xlabel="epoch",
            ylabel="eval_time (s)",
            title="Evaluation Time over Epochs",
            out_path=os.path.join(out_dir, "eval_time_curve.png"),
            show=args.show
        )

    plot_series(
        x=df_ep["epoch"],
        y=df_ep["mean_reward"],
        xlabel="epoch",
        ylabel="mean_reward (per-epoch avg)",
        title="Training Mean Reward per Epoch",
        out_path=os.path.join(out_dir, "train_mean_reward_curve.png"),
        show=args.show
    )

    plot_series(
        x=df_ep["epoch"],
        y=df_ep["mean_loss"],
        xlabel="epoch",
        ylabel="mean_loss (per-epoch avg)",
        title="Training Mean Loss per Epoch",
        out_path=os.path.join(out_dir, "train_mean_loss_curve.png"),
        show=args.show
    )

    print("All figures saved.")

if __name__ == "__main__":
    main()
