import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class CliffWalkEnv:
    """悬崖行走环境"""
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start_pos = (3, 0)    # 起点位置
        self.goal_pos = (3, 11)    # 终点位置
        # 悬崖位置：第4行的第1-10列
        self.cliff_positions = [(3, i) for i in range(1, 11)]
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.state = self.start_pos
        return self.state
    
    def step(self, action):
        """执行动作并返回新状态、奖励、是否结束"""
        row, col = self.state
        
        # 动作映射: 0=上, 1=右, 2=下, 3=左
        if action == 0:    # 上
            row = max(0, row - 1)
        elif action == 1:  # 右
            col = min(self.cols - 1, col + 1)
        elif action == 2:  # 下
            row = min(self.rows - 1, row + 1)
        elif action == 3:  # 左
            col = max(0, col - 1)
        
        self.state = (row, col)
        
        # 奖励计算
        if self.state in self.cliff_positions:  # 掉入悬崖
            reward = -100
            self.state = self.start_pos  # 回到起点
            done = False
        elif self.state == self.goal_pos:  # 到达终点
            reward = 0
            done = True
        else:  # 普通移动
            reward = -1
            done = False
        
        return self.state, reward, done

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """ε-贪婪策略"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # 随机探索
    else:
        return np.argmax(Q[state])  # 选择最优动作

def sarsa_learning(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):

    n_actions = 4
    Q = np.zeros((env.rows, env.cols, n_actions))  # Q表初始化
    
    rewards_per_episode = []  # 记录每回合奖励
    steps_per_episode = []    # 记录每回合步数
    
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 执行动作
            next_state, reward, done = env.step(action)
            # 选择下一个动作
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, n_actions)
            
            # SARSA更新公式: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            current_q = Q[state][action]
            next_q = Q[next_state][next_action] if not done else 0
            Q[state][action] += alpha * (reward + gamma * next_q - current_q)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        
        # 每100回合打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"SARSA Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return Q, rewards_per_episode, steps_per_episode

def extract_optimal_policy(Q):
    policy = np.zeros((Q.shape[0], Q.shape[1]), dtype=int)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            policy[i, j] = np.argmax(Q[i, j])
    return policy

def save_sarsa_results(env, Q, rewards, steps, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    with open(f"{output_dir}/sarsa_results.txt", "w", encoding='utf-8') as f:
        f.write("SARSA算法实验结果\n")
        f.write("="*50 + "\n\n")
        
        f.write("环境参数:\n")
        f.write(f"网格大小: {env.rows} x {env.cols}\n")
        f.write(f"起点位置: {env.start_pos}\n")
        f.write(f"终点位置: {env.goal_pos}\n")
        f.write(f"悬崖位置: {env.cliff_positions}\n\n")
        
        f.write("训练参数:\n")
        f.write(f"训练回合数: {len(rewards)}\n")
        f.write(f"学习率(alpha): 0.1\n")
        f.write(f"折扣因子(gamma): 1.0\n")
        f.write(f"探索率(epsilon): 0.1\n\n")
        
        f.write("性能统计:\n")
        final_100_rewards = rewards[-100:]
        final_100_steps = steps[-100:]
        f.write(f"平均奖励(最后100回合): {np.mean(final_100_rewards):.2f}\n")
        f.write(f"奖励标准差: {np.std(final_100_rewards):.2f}\n")
        f.write(f"最佳奖励: {np.max(final_100_rewards)}\n")
        f.write(f"最差奖励: {np.min(final_100_rewards)}\n")
        f.write(f"平均步数: {np.mean(final_100_steps):.1f}\n\n")
        
        # 保存最优策略
        optimal_policy = extract_optimal_policy(Q)
        f.write("最优策略矩阵:\n")
        for i in range(optimal_policy.shape[0]):
            f.write(" ".join(str(x) for x in optimal_policy[i]) + "\n")
        f.write("\n(0=上, 1=右, 2=下, 3=左)\n\n")
        
        # 保存Q值统计
        f.write("Q值统计:\n")
        f.write(f"最大Q值: {np.max(Q):.4f}\n")
        f.write(f"最小Q值: {np.min(Q):.4f}\n")
        f.write(f"平均Q值: {np.mean(Q):.4f}\n")

def visualize_and_save_sarsa_results(env, Q, rewards, steps, output_dir):

    plt.figure(figsize=(15, 10))
    
    # 1. 学习曲线
    plt.subplot(2, 2, 1)
    window = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title('SARSA Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.grid(True, alpha=0.3)
    
    # 2. 步数变化
    plt.subplot(2, 2, 2)
    smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_steps, linewidth=2, color='orange')
    plt.title('Steps per Episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    state = env.reset()
    path = [state]
    
    for _ in range(100):  # 防止无限循环
        action = np.argmax(Q[state])
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    
    grid = np.zeros((env.rows, env.cols))
    for (r, c) in path:
        grid[r, c] = 1
    
    grid[env.start_pos] = 0.5  # 起点
    grid[env.goal_pos] = 0.8   # 终点
    for cliff in env.cliff_positions:
        grid[cliff] = 0.3      # 悬崖
    
    plt.imshow(grid, cmap='RdYlBu', interpolation='nearest')
    plt.title('SARSA Optimal Path', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # 4. Q值热力图
    plt.subplot(2, 2, 4)
    max_q_values = np.max(Q, axis=2)
    plt.imshow(max_q_values, cmap='hot', interpolation='nearest')
    plt.title('Max Q-values Distribution', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sarsa_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return path

if __name__ == "__main__":
    output_dir = "output/SARSA"
    
    env = CliffWalkEnv()
    Q_sarsa, rewards_sarsa, steps_sarsa = sarsa_learning(env, episodes=1000)
    
    save_sarsa_results(env, Q_sarsa, rewards_sarsa, steps_sarsa, output_dir)
    optimal_path = visualize_and_save_sarsa_results(env, Q_sarsa, rewards_sarsa, steps_sarsa, output_dir)

    final_avg_reward = np.mean(rewards_sarsa[-100:])
    final_avg_steps = np.mean(steps_sarsa[-100:])
    print(f"最终性能统计:")
    print(f"平均奖励(最后100回合): {final_avg_reward:.2f}")
    print(f"平均步数(最后100回合): {final_avg_steps:.1f}")