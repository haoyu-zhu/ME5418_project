import os
import time
import math
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from gym_env import PointsEnv
from network import AttentionNet

from valid import build_fixed_eval_envs

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import glob


# ========== 可调超参 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAPH_TASKS = 100         # 每个图的任务点数量（你的 env 参数）
MAX_DIST = 3.0             # 每个图的最大里程
BATCH_SIZE = 48            # 每次 update 用 64 个独立图/episode
N_EPOCHS = 300             # 一共训练 100 个 epoch
UPDATES_PER_EPOCH = 10   # 每个 epoch 进行 1000 次反向传播（每次 64 图）

LR = 1e-4
GAMMA = 1
ENTROPY_COEF = 0.0         # 与 Kool 主干一致，通常设 0；需要探索可>0
MAX_GRAD_NORM = 1
BETA = 0.8                 # EMA baseline 平滑
VAL_EPISODES = 100         # 验证时的样本数量（可调）
SAVE_DIR = "checkpoints_kool_style"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 工具：导入模型 ==========
def find_latest_ckpt(model_dir: str) -> str:
    """在指定文件夹里按修改时间找最新的 .pt / .pth"""
    cand = glob.glob(os.path.join(model_dir, "*.pt")) + glob.glob(os.path.join(model_dir, "*.pth"))
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def try_resume_from_ckpt(policy: nn.Module,
                         optimizer: optim.Optimizer,
                         baseline: 'ExponentialBaseline',
                         path: str,
                         device: str = "cpu"):   # ← 默认用cpu
    """
    从 checkpoint 恢复，返回 (start_epoch, best_val)。
    先强制CPU加载，避免 ROCm 下 amdsmi 初始化错误；随后把权重和优化器状态迁回 device。
    """
    print(f"🔁 Loading checkpoint (CPU-safe): {path}")
    # 注意：为了能恢复 optimizer，我们这里仍需要 weights_only=False
    ckpt = torch.load(path, map_location="cpu")  # ★ 关键：强制到CPU

    # 纯 state_dict 兼容
    if isinstance(ckpt, dict) and ("model" not in ckpt) and \
       ("state_dict" in ckpt or any(isinstance(v, torch.Tensor) for v in ckpt.values())):
        state = ckpt.get("state_dict", ckpt)
        policy.load_state_dict(state)
        policy.to(device)
        print("  ✅ Loaded model weights (state_dict only).")
        return 0, -1e9

    # 标准保存格式
    if "model" in ckpt:
        policy.load_state_dict(ckpt["model"])
        policy.to(device)
        print("  ✅ Loaded model weights.")

    # 恢复优化器（先在CPU上load，再把state里的张量迁到device）
    if "opt" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["opt"])
            # 把优化器state里的张量搬到目标device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            # 覆盖学习率为当前脚本设定
            for pg in optimizer.param_groups:
                pg["lr"] = LR
            print(f"  ✅ Restored optimizer (moved to {device}, lr={LR}).")
        except Exception as e:
            print(f"  ⚠️ Optimizer state not loaded: {e}")

    # 恢复 EMA baseline（标量）
    if "baseline" in ckpt:
        try:
            baseline._b = torch.tensor(float(ckpt["baseline"]), dtype=torch.float32, device=device)
            print(f"  ✅ Restored baseline EMA: {baseline.value():.4f}")
        except Exception as e:
            print(f"  ⚠️ Baseline not restored: {e}")

    start_epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", -1e9))
    print(f"  ▶️ Resume from epoch={start_epoch}, best_val={best_val:.4f}")
    return start_epoch, best_val


# ===== Resume / Load =====
MODEL_DIR = "model"          # 你存放已训练模型的文件夹
RESUME = True                # True=尝试继续训练；False=从头开始
# RESUME_PATH = None           # 手动指定 ckpt 路径（例如 "model/10-17-best_model_384_5000.pt"）
#                              # 若为 None 且 RESUME=True，则自动在 MODEL_DIR 按修改时间找最新 *.pt/*.pth

RESUME_PATH = "model/10-21/10-21-last_model_900_560.pt"


# ========== 工具：绘图 ==========
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def now_tag():
    # 形如 10_19_1935（MM_DD_HHMM）
    return datetime.now().strftime("%m_%d_%H%M")

def init_result_dir():
    base_dir = os.path.join("result_cur", f"train_{now_tag()}")
    ensure_dir(base_dir)
    print(f"📁 results will be saved under: {base_dir}")
    return base_dir

def save_excels(base_dir, train_rows, eval_rows):
    """持续覆盖式保存到两份 Excel。"""
    df_train = pd.DataFrame(train_rows)  # 列: epoch, update, reward, loss
    df_eval  = pd.DataFrame(eval_rows)   # 列: epoch, val_return, eval_time, train_time
    # 两个 Excel：一个评测结果&耗时，一个训练 reward/loss
    eval_xlsx  = os.path.join(base_dir, "eval.xlsx")
    train_xlsx = os.path.join(base_dir, "train.xlsx")
    # 使用 openpyxl 写 xlsx（pandas 默认即可）
    df_eval.to_excel(eval_xlsx, index=False)
    df_train.to_excel(train_xlsx, index=False)
    return eval_xlsx, train_xlsx

def plot_curves(base_dir, train_rows, eval_rows):
    """训练结束（或中断）后画 4 张图：val_return、eval_time、epoch均值reward、epoch均值loss。"""
    if len(eval_rows) > 0:
        df_eval = pd.DataFrame(eval_rows)
        # 1) val_return
        plt.figure()
        plt.plot(df_eval["epoch"], df_eval["val_return"])
        plt.xlabel("epoch"); plt.ylabel("val_return (greedy on fixed-100)")
        plt.title("Validation Return over Epochs")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "val_return_curve.png"), dpi=150)
        plt.close()

        # 2) eval_time
        plt.figure()
        plt.plot(df_eval["epoch"], df_eval["eval_time"])
        plt.xlabel("epoch"); plt.ylabel("eval_time (s)")
        plt.title("Evaluation Time over Epochs")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "eval_time_curve.png"), dpi=150)
        plt.close()

    if len(train_rows) > 0:
        df_train = pd.DataFrame(train_rows)  # 每个 update 一行
        # 聚合到 epoch 均值，避免曲线太抖
        df_ep = df_train.groupby("epoch", as_index=False).agg(
            mean_reward=("reward", "mean"),
            mean_loss=("loss", "mean")
        )

        # 3) mean_reward per epoch
        plt.figure()
        plt.plot(df_ep["epoch"], df_ep["mean_reward"])
        plt.xlabel("epoch"); plt.ylabel("mean_reward (per-epoch avg)")
        plt.title("Training Mean Reward per Epoch")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "train_mean_reward_curve.png"), dpi=150)
        plt.close()

        # 4) mean_loss per epoch
        plt.figure()
        plt.plot(df_ep["epoch"], df_ep["mean_loss"])
        plt.xlabel("epoch"); plt.ylabel("mean_loss (per-epoch avg)")
        plt.title("Training Mean Loss per Epoch")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "train_mean_loss_curve.png"), dpi=150)
        plt.close()

# ========== 工具：指数滑动基线 ==========
class ExponentialBaseline:
    def __init__(self, beta: float = 0.8):
        self.beta = beta
        self._b = None

    @torch.no_grad()
    def eval_and_update(self, returns: torch.Tensor) -> torch.Tensor:
        """returns: [B]，返回标量 baseline（常数基线，不逐样本）"""
        val = returns.mean()  # 批均值
        if self._b is None:
            self._b = val
        else:
            self._b = self.beta * self._b + (1 - self.beta) * val
        return self._b

    def value(self) -> float:
        return float(self._b) if self._b is not None else 0.0

# ========== 单个 env 上滚动一个 episode，返回“序列 logp 之和”和“折扣回报” ==========
def run_one_episode_collect_seq(policy: AttentionNet, env: PointsEnv) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    返回:
      seq_logp_sum: shape [], 当前策略在整条轨迹上的 logπ 之和（标量张量）
      Rt:           shape [], 该条轨迹的折扣回报（标量张量）
      ep_return:    float, 未折扣总回报（用于打印统计）
    说明：
      - 整条轨迹按“采样”选动作
      - 每一步将 log_prob(action) 累加
      - 折扣回报 Rt 用 GAMMA 计算
    """
    env.reset(env.world.rewards, env.world.start_depot_index)
    done = False

    step_logps: List[torch.Tensor] = []
    step_rewards: List[float] = []
    ep_return = 0.0

    while not done:
        points, agent_idx, remaining_distance, valid_mask = env.observe()
        # 组装张量（单条轨迹，用 batch 维度=1）
        points_t = torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0)            # [1, N, 3]
        agent_t  = torch.tensor([agent_idx], dtype=torch.long, device=DEVICE)                       # [1]
        rem_t    = torch.tensor([[remaining_distance]], dtype=torch.float32, device=DEVICE)         # [1, 1]
        mask_t   = torch.tensor(valid_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)        # [1, N]

        # 前向得到分布（训练=采样）
        dist = policy(points_t, agent_t, rem_t, mask_t.bool())
        action = dist.sample()                                 # [1]
        logp   = dist.log_prob(action).squeeze(0)              # []
        step_logps.append(logp)

        action_int = int(action.item())
        _, reward, done = env.step(action_int)

        step_rewards.append(float(reward))
        ep_return += float(reward)

    # 折扣回报（reward-to-go 的首项，等价整条折扣和）
    Rt = 0.0
    for r in reversed(step_rewards):
        Rt = r + GAMMA * Rt
    Rt = torch.tensor(Rt, dtype=torch.float32, device=DEVICE)  # []

    # 序列 log 概率之和
    seq_logp_sum = torch.stack(step_logps).sum()               # []

    return seq_logp_sum, Rt, ep_return

# ========== 采样一个 batch（64 条独立图）并计算一次 REINFORCE 损失 ==========
def reinforce_update(policy: AttentionNet, optimizer, baseline: ExponentialBaseline) -> Tuple[float, float]:
    """
    进行一次参数更新（一次反向传播），基于 BATCH_SIZE=64 条独立图/episode。
    返回:
      mean_cost:  本次 batch 未折扣总回报的负值均值（当作“代价”，仅统计）
      loss_val:   本次用于反向传播的 loss 标量（float）
    """
    policy.train()
    seq_logps = []
    returns = []
    ep_returns = []

    # 为加速，这里重复使用 64 个独立 env，每个 env 只跑一条轨迹
    envs = [PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST) for _ in range(BATCH_SIZE)]

    for env in envs:
        seq_logp_sum, Rt, ep_ret = run_one_episode_collect_seq(policy, env)
        seq_logps.append(seq_logp_sum)  # []
        returns.append(Rt)              # []
        ep_returns.append(ep_ret)

    seq_logps_t = torch.stack(seq_logps)    # [B]
    returns_t   = torch.stack(returns)      # [B]

    # 基线（常数基线）：EMA(returns.mean)
    b = baseline.eval_and_update(returns_t)  # 标量
    adv = returns_t - b                      # [B]

    # 可选：标准化 advantage（通常对稳定训练有益）
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # REINFORCE： ((R - b) * logπ).mean()，我们最小化 -该期望
    reinforce_loss = -(adv * seq_logps_t).mean()

    # 可选：加熵正则（论文主干通常不用）
    # 这里没有逐步 entropy，所以先不加；若你要加，在收集时把 step entropy 累加一起平均即可。
    loss = reinforce_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    mean_cost = float(np.mean(ep_returns))  # 仅统计显示（把“回报”取负，当作 cost）
    return mean_cost, float(loss.item())

# ========== 贪心评估 ==========
#@torch.no_grad()
# def evaluate_greedy(policy: AttentionNet, n_eval: int = VAL_EPISODES) -> float:
#     """
#     用贪心策略（argmax）在 n_eval 个独立图上评估平均“未折扣总回报”。
#     """
#     policy.eval()
#     rets = []
#     for _ in range(n_eval):
#         env = PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST)
#         env.reset(env.world.rewards, env.world.start_depot_index)
#         done = False
#         ep_ret = 0.0
#         while not done:
#             points, agent_idx, remaining_distance, valid_mask = env.observe()
#             points_t = torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#             agent_t  = torch.tensor([agent_idx], dtype=torch.long, device=DEVICE)
#             rem_t    = torch.tensor([[remaining_distance]], dtype=torch.float32, device=DEVICE)
#             mask_t   = torch.tensor(valid_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#             dist = policy(points_t, agent_t, rem_t, mask_t.bool())
#             action = torch.argmax(dist.probs, dim=-1).item()
#             _, r, done = env.step(int(action))
#             ep_ret += float(r)
#         rets.append(ep_ret)
#     return float(np.mean(rets))

# ===== 用固定 100 张图做贪心评测 =====
@torch.no_grad()
def evaluate_greedy_fixed(policy: AttentionNet, eval_envs) -> float:
    """
    在传入的固定评测环境列表 eval_envs 上做贪心评测。
    每次评测都会把各自环境 reset 回到同一张图（同奖励&起点）。
    """
    policy.eval()
    rets = []
    for env in eval_envs:
        # 确保回到同一张图：使用该 env 自己的 world 参数重置
        env.reset(env.world.rewards, env.world.start_depot_index)

        done = False
        ep_ret = 0.0
        while not done:
            points, agent_idx, remaining_distance, valid_mask = env.observe()

            points_t = torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            agent_t  = torch.tensor([agent_idx], dtype=torch.long, device=DEVICE)
            rem_t    = torch.tensor([[remaining_distance]], dtype=torch.float32, device=DEVICE)
            mask_t   = torch.tensor(valid_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            dist = policy(points_t, agent_t, rem_t, mask_t.bool())
            #action = dist.sample()
            action = torch.argmax(dist.probs, dim=-1).item()

            _, r, done = env.step(int(action))
            ep_ret += float(r)

        rets.append(ep_ret)
    return float(np.mean(rets))


# ========== 主训练 ==========
# def main():
#     torch.backends.cudnn.deterministic = True
#
#     policy = AttentionNet().to(DEVICE)
#     optimizer = optim.Adam(policy.parameters(), lr=LR)
#     baseline = ExponentialBaseline(beta=BETA)
#
#     best_val = -1e9
#     ckpt_best = os.path.join(SAVE_DIR, "best.pt")
#
#     #100eval_envs
#     eval_envs = build_fixed_eval_envs(n_envs=100, tasks=GRAPH_TASKS, max_dist=MAX_DIST, seed=2025)
#
#     val_results = []
#     val_times = []
#
#     for epoch in range(1, N_EPOCHS + 1):
#         t0 = time.time()
#         loss_meter = []
#         cost_meter = []
#
#         # —— 每个 epoch 做 1000 次更新，每次基于 64 个独立图 —— #
#         for u in range(1, UPDATES_PER_EPOCH + 1):
#             mean_reward, loss_val = reinforce_update(policy, optimizer, baseline)
#             loss_meter.append(loss_val)
#             cost_meter.append(mean_reward)
#
#             # 也可以每隔一定步数打印一次
#             if u % 1 == 0:
#                 print(f"update{u:4d}/{UPDATES_PER_EPOCH}, reward={mean_reward}  loss={loss_val:.4f}")
#
#         # 验证（贪心）
#         t_dur = time.time() - t0
#
#         t0_v = time.time()
#         #val_ret = evaluate_greedy(policy, n_eval=VAL_EPISODES)
#         val_ret = evaluate_greedy_fixed(policy, eval_envs)
#         v_dur = time.time() - t0_v
#
#         val_results.append(val_ret)
#         val_times.append(v_dur)
#
#
#         print(
#             f"[Epoch {epoch:03d}/{N_EPOCHS}] "
#               f"updates={UPDATES_PER_EPOCH}  "
#               f"train_loss={np.mean(loss_meter):.4f}  "
#               f"train_reward(neg_ret)={np.mean(cost_meter):.2f}  "
#               f"val_return(greedy)={val_ret:.2f}  "
#               f"baseline={baseline.value():.2f}  "
#               )
#
#         print(
#               f"train_one_epoch_time={t_dur:.2f}   "
#               f"validate_one_epoch_time={v_dur:.2f}"
#         )
#
#
#         # 保存最好
#         if val_ret > best_val:
#             best_val = val_ret
#             torch.save({"model": policy.state_dict(),
#                         "opt": optimizer.state_dict(),
#                         "baseline": baseline.value(),
#                         "epoch": epoch}, ckpt_best)
#             print(f"  ✅ Saved best to {ckpt_best} (val_return={best_val:.2f})")
#
#         print(f"best_val={best_val:.2f}")
#
#     # 期末再存一份
#     ckpt_last = os.path.join(SAVE_DIR, "last.pt")
#     torch.save({"model": policy.state_dict(),
#                 "opt": optimizer.state_dict(),
#                 "baseline": baseline.value(),
#                 "epoch": N_EPOCHS}, ckpt_last)
#     print(f"Training done. Last saved to {ckpt_last}")
#     print(f"best_val={best_val:.2f}")

def main():
    torch.backends.cudnn.deterministic = True

    # ============ 结果目录 ============
    BASE_DIR = init_result_dir()
    ckpt_best = os.path.join(SAVE_DIR, "best.pt")  # 你原有目录也保留
    ckpt_last = os.path.join(SAVE_DIR, "last.pt")

    # ============ 初始化组件 ============
    policy = AttentionNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    baseline = ExponentialBaseline(beta=BETA)

    # 固定 100 张图（你已有）
    eval_envs = build_fixed_eval_envs(n_envs=200, tasks=GRAPH_TASKS, max_dist=MAX_DIST, seed=2025)

    # —— Resume / 继续训练 —— #
    start_epoch = 0
    best_val = -1e9
    if RESUME:
        load_path = RESUME_PATH if RESUME_PATH is not None else find_latest_ckpt(MODEL_DIR)
        if load_path is not None and os.path.exists(load_path):
            start_epoch, best_val = try_resume_from_ckpt(policy, optimizer, baseline, load_path, device=DEVICE)
        else:
            print(f"ℹ️ No checkpoint found to resume (looked at {RESUME_PATH or MODEL_DIR}). Start fresh.")

    # ============ 日志缓存（内存里） ============
    # 训练分布：每个 update 记一行
    train_rows = []  # {epoch, update, reward, loss}
    # 评测：每个 epoch 记一行
    eval_rows  = []  # {epoch, val_return, eval_time, train_time}

    try:
        for epoch in range(1, N_EPOCHS + 1):
            t0_train = time.time()
            loss_meter = []
            reward_meter = []

            # —— 每个 epoch 做 1000 次更新，每次基于 64 个独立图 —— #
            for u in range(1, UPDATES_PER_EPOCH + 1):
                mean_reward, loss_val = reinforce_update(policy, optimizer, baseline)
                loss_meter.append(loss_val)
                reward_meter.append(mean_reward)

                # 训练分布：逐 update 记录一行
                train_rows.append({
                    "epoch": epoch,
                    "update": u,
                    "reward": float(mean_reward),
                    "loss": float(loss_val),
                })

                # 控制台打印频率（可调）
                if u % 1 == 0:
                    print(f"update {u:4d}/{UPDATES_PER_EPOCH}, reward={mean_reward:.4f}  loss={loss_val:.4f}")

            train_dur = time.time() - t0_train

            # —— 评测（固定100图，贪心）——
            t0_eval = time.time()
            val_ret = evaluate_greedy_fixed(policy, eval_envs)
            eval_dur = time.time() - t0_eval

            # 评测日志：每个 epoch 一行
            eval_rows.append({
                "epoch": epoch,
                "val_return": float(val_ret),
                "eval_time": float(eval_dur),
                "train_time": float(train_dur),
            })

            # —— 控制台统计 ——
            print(
                f"[Epoch {epoch:03d}/{N_EPOCHS}] "
                f"updates={UPDATES_PER_EPOCH}  "
                f"train_loss(avg)={np.mean(loss_meter):.4f}  "
                f"train_reward(avg)={np.mean(reward_meter):.4f}  "
                f"val_return(greedy)={val_ret:.4f}  "
                f"baseline(ema)={baseline.value():.4f}  "
            )

            print(f"train_time={train_dur:.2f}s  eval_time={eval_dur:.2f}s")

            # —— 保存最好（按 val_ret 越大越好）——
            if val_ret > best_val:
                best_val = val_ret
                torch.save({
                    "model": policy.state_dict(),
                    "opt": optimizer.state_dict(),
                    "baseline": baseline.value(),
                    "epoch": epoch,
                    "best_val": best_val,
                }, ckpt_best)
                print(f"  ✅ Saved best to {ckpt_best} (val_return={best_val:.4f})")

            print(f"best_val={best_val:.4f}")

            # —— 每个 epoch 结束就落盘一次 Excel，抗中断 ——
            save_excels(BASE_DIR, train_rows, eval_rows)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (KeyboardInterrupt). Saving partial results...")

    finally:
        # —— 期末（或中断）保存 last 模型 ——
        torch.save({
            "model": policy.state_dict(),
            "opt": optimizer.state_dict(),
            "baseline": baseline.value(),
            "epoch": min(len(eval_rows), N_EPOCHS),
            "best_val": best_val,
        }, ckpt_last)
        print(f"📦 Last checkpoint saved to {ckpt_last}")

        # —— 确保把 Excel 落盘 ——
        eval_xlsx, train_xlsx = save_excels(BASE_DIR, train_rows, eval_rows)
        print(f"📑 Excel saved: \n  - {eval_xlsx}\n  - {train_xlsx}")

        # —— 画曲线 PNG ——
        plot_curves(BASE_DIR, train_rows, eval_rows)
        print(f"📈 Curves saved under: {BASE_DIR}")


if __name__ == "__main__":
    main()
