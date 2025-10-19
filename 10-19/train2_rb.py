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

import copy
import torch
from typing import List, Tuple

import os
import torch
from torch.nn import DataParallel



# ========== 可调超参 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAPH_TASKS = 100          # 每个图的任务点数量（你的 env 参数）
MAX_DIST = 3.0             # 每个图的最大里程
BATCH_SIZE = 2            # 每次 update 用 64 个独立图/episode
N_EPOCHS = 100             # 一共训练 100 个 epoch
UPDATES_PER_EPOCH = 10   # 每个 epoch 进行 1000 次反向传播（每次 64 图）

LR = 1e-4
GAMMA = 1
ENTROPY_COEF = 0.0         # 与 Kool 主干一致，通常设 0；需要探索可>0
MAX_GRAD_NORM = 2
BETA = 0.8                 # EMA baseline 平滑
VAL_EPISODES = 100         # 验证时的样本数量（可调）
SAVE_DIR = "checkpoints_kool_style"
os.makedirs(SAVE_DIR, exist_ok=True)


# ========== 工具：指数滑动基线 ==========

#========save model=====================
def get_inner_model(m):
    return m.module if isinstance(m, DataParallel) else m

def pack_baseline_state(baseline_or_rollout):
    # 兼容两类基线
    state = {}
    # EMA / ExponentialBaseline
    if hasattr(baseline_or_rollout, "value") and not hasattr(baseline_or_rollout, "baseline_policy"):
        state["type"] = "ema"
        state["ema_value"] = baseline_or_rollout.value()
    # RolloutBaseline
    elif hasattr(baseline_or_rollout, "baseline_policy"):
        state["type"] = "rollout"
        state["baseline_policy"] = baseline_or_rollout.baseline_policy.state_dict()
        state["mode"] = getattr(baseline_or_rollout, "mode", "greedy")
        state["n_samples"] = getattr(baseline_or_rollout, "n_samples", 1)
    else:
        state["type"] = "unknown"
    return state

def save_checkpoint(path, policy, optimizer, epoch, baseline_obj, extra=None):
    payload = {
        "model": get_inner_model(policy).state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "baseline_state": pack_baseline_state(baseline_obj),
        # 可选：为了可复现
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra:
        payload.update(extra)
    torch.save(payload, path)

#========base line=====================

class RolloutBaseline:
    """
    冻结一份 baseline 策略（结构同 policy），
    在同一张图上 rollout 得到基线回报 R_b，adv = R - R_b。
    """
    def __init__(self, policy_ctor, device, mode: str = "greedy", n_samples: int = 1):
        """
        policy_ctor: 一个可调用，返回 AttentionNet() 实例（与训练模型同结构）
        device:     设备
        mode:       "greedy" 或 "sample"（基线用贪心 or 采样）
        n_samples:  若 mode="sample"，可对同图 roll 多条平均
        """
        self.device = device
        self.mode = mode
        self.n_samples = n_samples
        self.baseline_policy = policy_ctor().to(device)
        self.baseline_policy.eval()  # 冻结评估模式

    @torch.no_grad()
    def sync(self, policy):
        """把当前训练 policy 的参数拷贝到 baseline（通常每 epoch 调一次）"""
        self.baseline_policy.load_state_dict(policy.state_dict())
        self.baseline_policy.eval()

    @torch.no_grad()
    def rollout_return(self, env) -> float:
        """
        在给定 env（已 reset 到目标 rewards/start_idx）的同一张图上，
        用 baseline_policy 跑出回报（贪心或多次采样平均）。
        """
        def run_once(e) -> float:
            done, ep_ret = False, 0.0
            while not done:
                points, agent_idx, rem_dist, valid_mask = e.observe()
                points_t = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)
                agent_t  = torch.tensor([agent_idx], dtype=torch.long, device=self.device)
                rem_t    = torch.tensor([[rem_dist]], dtype=torch.float32, device=self.device)
                mask_t   = torch.tensor(valid_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

                dist = self.baseline_policy(points_t, agent_t, rem_t, mask_t.bool())
                if self.mode == "greedy":
                    action = torch.argmax(dist.probs, dim=-1).item()
                else:
                    action = dist.sample().item()

                _, r, done = e.step(int(action))
                ep_ret += float(r)
            return ep_ret

        if self.mode == "greedy":
            # 贪心只需要一次
            return run_once(env)
        else:
            # 采样多次平均
            acc, K = 0.0, max(1, int(self.n_samples))
            for _ in range(K):
                # 为保证每次都在同一张图，需要每次 reset 回同一 rewards/start
                env.reset(env.world.rewards, env.world.start_depot_index)
                acc += run_once(env)
            return acc / K


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

# # ========== 采样一个 batch（64 条独立图）并计算一次 REINFORCE 损失 ==========
# def reinforce_update(policy: AttentionNet, optimizer, baseline: ExponentialBaseline) -> Tuple[float, float]:
#     """
#     进行一次参数更新（一次反向传播），基于 BATCH_SIZE=64 条独立图/episode。
#     返回:
#       mean_cost:  本次 batch 未折扣总回报的负值均值（当作“代价”，仅统计）
#       loss_val:   本次用于反向传播的 loss 标量（float）
#     """
#     policy.train()
#     seq_logps = []
#     returns = []
#     ep_returns = []
#
#     # 为加速，这里重复使用 64 个独立 env，每个 env 只跑一条轨迹
#     envs = [PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST) for _ in range(BATCH_SIZE)]
#
#     for env in envs:
#         seq_logp_sum, Rt, ep_ret = run_one_episode_collect_seq(policy, env)
#         seq_logps.append(seq_logp_sum)  # []
#         returns.append(Rt)              # []
#         ep_returns.append(ep_ret)
#
#     seq_logps_t = torch.stack(seq_logps)    # [B]
#     returns_t   = torch.stack(returns)      # [B]
#
#     # 基线（常数基线）：EMA(returns.mean)
#     b = baseline.eval_and_update(returns_t)  # 标量
#     adv = returns_t - b                      # [B]
#
#     # 可选：标准化 advantage（通常对稳定训练有益）
#     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
#
#     # REINFORCE： ((R - b) * logπ).mean()，我们最小化 -该期望
#     reinforce_loss = -(adv * seq_logps_t).mean()
#
#     # 可选：加熵正则（论文主干通常不用）
#     # 这里没有逐步 entropy，所以先不加；若你要加，在收集时把 step entropy 累加一起平均即可。
#     loss = reinforce_loss
#
#     optimizer.zero_grad()
#     loss.backward()
#     nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
#     optimizer.step()
#
#     mean_cost = float(np.mean(ep_returns))  # 仅统计显示（把“回报”取负，当作 cost）
#     return mean_cost, float(loss.item())

def reinforce_update_with_rollout_baseline(
    policy: AttentionNet,
    optimizer,
    rollout_bl: RolloutBaseline,
    batch_size: int,
    tasks: int,
    max_dist: float,
    gamma: float = 0.99,
) -> Tuple[float, float]:
    """
    用 Rollout Baseline 做一次参数更新（一次反向传播）。
    返回:
      mean_cost:  本次 batch 未折扣总回报的负值均值（仅统计）
      loss_val:   本次优化用的标量 loss
    """
    policy.train()
    seq_logps: List[torch.Tensor] = []
    returns:   List[torch.Tensor] = []
    ep_returns = []

    # 为了在同一张图上做 baseline，需要“保存奖励&起点”，
    # 这里每条样本单独构一个 env，先跑 policy 采样，再用 baseline_policy 重跑。
    envs = [PointsEnv(tasks_number=tasks, max_distance=max_dist) for _ in range(batch_size)]

    # === 先跑“训练策略”的一条采样轨迹，得到 logπ_sum 和 Rt，并记录图的 rewards/start ===
    saved_world = []  # list of dict(rewards, start_idx)
    for env in envs:
        # 让 env 随机生成一套奖励；如果你想控制随机种子，可在外层固定
        env.reset()  # 这会内部随机生成一套 rewards 和 start_idx（你的设定）

        # 保存“同一张图”的信息，以便 baseline 重放
        rewards_i = copy.deepcopy(env.world.rewards)
        start_i   = int(env.world.start_depot_index)
        saved_world.append({"rewards": rewards_i, "start_idx": start_i})

        # 用“训练中的 policy”采样一条轨迹，返回整条 logπ 之和 + 折扣回报
        seq_logp_sum, Rt, ep_ret = run_one_episode_collect_seq(policy, env)
        seq_logps.append(seq_logp_sum)  # []
        returns.append(Rt)              # []
        ep_returns.append(ep_ret)

    seq_logps_t = torch.stack(seq_logps)  # [B]
    returns_t   = torch.stack(returns)    # [B]

    # === 用 baseline_policy 在“同一张图”上 rollout 得到基线回报 R_b（逐样本） ===
    Rb_list = []
    for i in range(batch_size):
        # 为 baseline 重置到相同 rewards 和起点
        env = envs[i]
        env.reset(saved_world[i]["rewards"], saved_world[i]["start_idx"])
        Rb = rollout_bl.rollout_return(env)  # float
        Rb_list.append(Rb)
    Rb_t = torch.tensor(Rb_list, dtype=torch.float32, device=seq_logps_t.device)  # [B]

    # === advantage：逐样本 (R - R_b)，可再做标准化（通常更稳） ===
    adv = returns_t - Rb_t
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = adv.detach()  # baseline 不反传

    loss = -(adv * seq_logps_t).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    mean_cost = float(np.mean(ep_returns))  # 仅统计显示
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
            action = torch.argmax(dist.probs, dim=-1).item()

            _, r, done = env.step(int(action))
            ep_ret += float(r)

        rets.append(ep_ret)
    return float(np.mean(rets))


# ========== 主训练 ==========
def main():
    torch.backends.cudnn.deterministic = True

    policy = AttentionNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # —— 创建 Rollout Baseline：贪心（最常用），也可 mode="sample", n_samples=5 —— #
    rollout_bl = RolloutBaseline(
        policy_ctor=AttentionNet,   # 重要：传“类”或无参构造器
        device=DEVICE,
        mode="greedy",              # 或 "sample"
        n_samples=1                 # mode="sample" 时可>1
    )
    # 先同步一次
    rollout_bl.sync(policy)

    #100eval_envs
    eval_envs = build_fixed_eval_envs(n_envs=100, tasks=GRAPH_TASKS, max_dist=MAX_DIST, seed=2025)

    ckpt_best = os.path.join(SAVE_DIR, "best.pt")
    best_val = -1e9

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        loss_meter, cost_meter = [], []

        for u in range(1, UPDATES_PER_EPOCH + 1):
            mean_cost, loss_val = reinforce_update_with_rollout_baseline(
                policy, optimizer, rollout_bl,
                batch_size=BATCH_SIZE,
                tasks=GRAPH_TASKS,
                max_dist=MAX_DIST,
                gamma=GAMMA
            )
            loss_meter.append(loss_val)
            cost_meter.append(mean_cost)

            # 也可以每隔一定步数打印一次
            if u % 1 == 0:
                print(f"update{u:4d}/{UPDATES_PER_EPOCH}, reward={mean_cost}  loss={loss_val:.4f}")

        # 验证（贪心）
        t_dur = time.time() - t0

        t0_v = time.time()
        #val_ret = evaluate_greedy(policy, n_eval=VAL_EPISODES)
        val_ret = evaluate_greedy_fixed(policy, eval_envs)
        v_dur = time.time() - t0_v

        # —— 关键：epoch 末把 policy 权重快照到 baseline —— #
        rollout_bl.sync(policy)

        print(
            f"[Epoch {epoch:03d}/{N_EPOCHS}] "
              f"updates={UPDATES_PER_EPOCH}  "
              f"train_loss={np.mean(loss_meter):.4f}  "
              f"train_reward(neg_ret)={np.mean(cost_meter):.2f}  "
              f"val_return(greedy)={val_ret:.2f}  "
              )

        print(
              f"train_one_epoch_time={t_dur:.2f}"
              f"validate_one_epoch_time={v_dur:.2f}"
        )

        # 保存最好
        # —— 保存最好 —— #
        if val_ret > best_val:  # 若你的度量是“回报越大越好”
            best_val = val_ret
            save_checkpoint(ckpt_best, policy, optimizer, epoch, rollout_bl)  # 或 rollout_bl
            print(f"  ✅ Saved best to {ckpt_best} (val_return={best_val:.2f})")

        print(f"best_val={best_val:.2f}")

        # —— 期末再存一份 —— #
    ckpt_last = os.path.join(SAVE_DIR, "last.pt")
    save_checkpoint(ckpt_last, policy, optimizer, N_EPOCHS, rollout_bl)  # 或 rollout_bl
    print(f"Training done. Last saved to {ckpt_last}")
    print(f"best_val={best_val:.2f}")

    #     if val_ret > best_val:
    #         best_val = val_ret
    #         torch.save({"model": policy.state_dict(),
    #                     "opt": optimizer.state_dict(),
    #                     "baseline": rollout_bl.value(),
    #                     "epoch": epoch}, ckpt_best)
    #         print(f"  ✅ Saved best to {ckpt_best} (val_return={best_val:.2f})")
    #
    # # 期末再存一份
    # ckpt_last = os.path.join(SAVE_DIR, "last.pt")
    # torch.save({"model": policy.state_dict(),
    #             "opt": optimizer.state_dict(),
    #             "baseline": rollout_bl.value(),
    #             "epoch": N_EPOCHS}, ckpt_last)
    # print(f"Training done. Last saved to {ckpt_last}")

if __name__ == "__main__":
    main()
