import torch
import numpy as np
from gym_env import PointsEnv
#from attention_net import AttentionNet
from network import AttentionNet
import time

# ======= 参数 =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#MODEL_PATH = "checkpoints/best_model_27.pt"     # 模型路径
MODEL_PATH = "checkpoints/10-18/best_model_25.pt"     # 模型路径
TASK_NUM = 100
MAX_DIST = 3.0
TEST_EPISODES = 100       # 控制测试次数：1=渲染模式，>1=多次评估模式（不渲染）
time_start = time.time()
# ======= 加载策略网络 =======
policy = AttentionNet().to(DEVICE)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy.eval()
print(f"✅ 模型已加载: {MODEL_PATH}")

# ======= 定义单次测试函数 =======
def run_one_episode(env, render=False):
    env.reset()
    reward_sum = 0.0
    done = False

    if render:
        env.render()

    while not done:
        points, agent_idx, remaining_distance, valid_mask = env.observe()

        points = torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, N, 3]
        agent_idx = torch.tensor([agent_idx], dtype=torch.int64, device=DEVICE)  # [1]
        remaining_distance = torch.tensor([[remaining_distance]], dtype=torch.float32, device=DEVICE)  # [1, 1]
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)  # [1, N]

        with torch.no_grad():
            dist = policy(points, agent_idx, remaining_distance, valid_mask)
            action = torch.argmax(dist.probs, dim=-1).item()

        # 防止过早回到仓库
        valid_actions = env.list_valid_actions()
        # while len(valid_actions) > 1 and action <= 3:
        #     action = np.random.choice(valid_actions)

        obs, reward, done = env.step(action)
        reward_sum += reward

        if render:
            env.render()

    return reward_sum, env.remaining_distance

# ======= 执行测试 =======w
env = PointsEnv()
if TEST_EPISODES == 1:
    reward_sum, remaining = run_one_episode(env, render=True)
    print(f"🏁 单次测试完成: 总奖励 = {reward_sum:.2f}, 剩余距离 = {remaining:.2f}")

else:
    rewards = []
    for i in range(TEST_EPISODES):
        reward_sum, remaining = run_one_episode(env, render=False)
        #print(reward_sum)
        rewards.append(reward_sum)
        print(f"[{i+1}/{TEST_EPISODES}] Reward = {reward_sum:.2f}, Remaining = {remaining:.2f}")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    time_end = time.time()
    print(f"\n✅ {TEST_EPISODES} 次测试完成: 平均奖励 = {avg_reward:.2f} ± {std_reward:.2f} 总耗时 = {time_end-time_start:.2f}")