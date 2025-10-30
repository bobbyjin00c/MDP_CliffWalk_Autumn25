import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class CliffWalkEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start_pos = (3, 0)
        self.goal_pos = (3, 11)
        self.cliff_positions = [(3, i) for i in range(1, 11)]
        self.reset()
    
    def reset(self):
        self.state = self.start_pos
        return self.state
    
    def step(self, action):
        row, col = self.state
        
        if action == 0:    # 上
            row = max(0, row - 1)
        elif action == 1:  # 右
            col = min(self.cols - 1, col + 1)
        elif action == 2:  # 下
            row = min(self.rows - 1, row + 1)
        elif action == 3:  # 左
            col = max(0, col - 1)
        
        self.state = (row, col)
        
        if self.state in self.cliff_positions:
            reward = -100
            self.state = self.start_pos
            done = False
        elif self.state == self.goal_pos:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done

def epsilon_greedy_policy(Q, state, epsilon, n_actions):

    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

def q_learning(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):

    n_actions = 4
    Q = np.zeros((env.rows, env.cols, n_actions))
    
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 选择动作
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # Q-learning更新公式: Q(s,a) ← Q(s,a) + α[r + γmaxₐQ(s',a) - Q(s,a)]
            current_q = Q[state][action]
            max_next_q = np.max(Q[next_state]) if not done else 0
            Q[state][action] += alpha * (reward + gamma * max_next_q - current_q)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Q-learning Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return Q, rewards_per_episode, steps_per_episode

def extract_optimal_policy(Q):

    policy = np.zeros((Q.shape[0], Q.shape[1]), dtype=int)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            policy[i, j] = np.argmax(Q[i, j])
    return policy

def save_qlearning_results(env, Q, rewards, steps, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_dir}/qlearning_results.txt", "w", encoding='utf-8') as f:
        f.write("Q-learning算法实验结果\n")
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
        
        # 特别说明Q-learning的特点
        f.write("\n算法特点说明:\n")
        f.write("Q-learning是off-policy算法，在线性能可能较差\n")
        f.write("但能收敛到理论最优策略\n")

def visualize_and_save_qlearning_results(env, Q, rewards, steps, output_dir):

    plt.figure(figsize=(15, 10))
    
    # 1. 学习曲线
    plt.subplot(2, 2, 1)
    window = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_rewards, linewidth=2, color='red')
    plt.title('Q-learning Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward per Episode')
    plt.grid(True, alpha=0.3)
    
    # 2. 步数变化
    plt.subplot(2, 2, 2)
    smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_steps, linewidth=2, color='purple')
    plt.title('Steps per episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    
    # 3. 最优路径可视化
    plt.subplot(2, 2, 3)
    state = env.reset()
    path = [state]
    
    for _ in range(100):
        action = np.argmax(Q[state])
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    
    # 创建路径图
    grid = np.zeros((env.rows, env.cols))
    for (r, c) in path:
        grid[r, c] = 1
    
    # 标记特殊位置
    grid[env.start_pos] = 0.5  # 起点
    grid[env.goal_pos] = 0.8   # 终点
    for cliff in env.cliff_positions:
        grid[cliff] = 0.3      # 悬崖
    
    plt.imshow(grid, cmap='RdYlBu', interpolation='nearest')
    plt.title('Q-learning Optimal Path', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # 4. Q值热力图
    plt.subplot(2, 2, 4)
    max_q_values = np.max(Q, axis=2)
    plt.imshow(max_q_values, cmap='hot', interpolation='nearest')
    plt.title('Max Q-values Distribution', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qlearning_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return path

if __name__ == "__main__":
    output_dir = "output/Q-learning"
    
    env = CliffWalkEnv()
    Q_qlearning, rewards_qlearning, steps_qlearning = q_learning(env, episodes=1000)
    
    save_qlearning_results(env, Q_qlearning, rewards_qlearning, steps_qlearning, output_dir)
    optimal_path = visualize_and_save_qlearning_results(env, Q_qlearning, rewards_qlearning, steps_qlearning, output_dir)
    
    final_avg_reward = np.mean(rewards_qlearning[-100:])
    final_avg_steps = np.mean(steps_qlearning[-100:])
    print(f"最终性能统计:")
    print(f"平均奖励(最后100回合): {final_avg_reward:.2f}")
    print(f"平均步数(最后100回合): {final_avg_steps:.1f}")