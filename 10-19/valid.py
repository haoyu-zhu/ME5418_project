import torch
import random
import numpy as np
from gym_env import PointsEnv

GRAPH_TASKS = 100
MAX_DIST = 3

# ===== 固定评测集：构建一次，反复复用 =====
def build_fixed_eval_envs(n_envs=100, tasks=GRAPH_TASKS, max_dist=MAX_DIST, seed=1234):
    eval_envs = []
    for i in range(n_envs):
        # 通过设置全局随机种子，确保每个 env 生成的图可复现
        np.random.seed(seed + i)
        random.seed(seed + i)
        torch.manual_seed(seed + i)

        env = PointsEnv(tasks_number=tasks, max_distance=max_dist)
        # 立刻 reset 一次以冻结它的 world（坐标/奖励/起点）
        env.reset(env.world.rewards, env.world.start_depot_index)
        eval_envs.append(env)
    return eval_envs
