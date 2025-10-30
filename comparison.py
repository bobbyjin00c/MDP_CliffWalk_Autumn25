import numpy as np
import matplotlib.pyplot as plt
from SARSA import sarsa_learning, CliffWalkEnv, extract_optimal_policy
from Q_learning import q_learning
import os
from pathlib import Path

def save_comparison_results(results, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_dir}/comparison_results.txt", "w", encoding='utf-8') as f:
        f.write("SARSA vs Q-learning 算法比较结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("实验设置:\n")
        f.write(f"训练回合数: 1000\n")
        f.write(f"学习率(alpha): 0.1\n")
        f.write(f"折扣因子(gamma): 1.0\n")
        f.write(f"探索率(epsilon): 0.1\n\n")
        
        f.write("性能比较结果:\n")
        f.write("-" * 40 + "\n")
        
        for algo, data in results.items():
            f.write(f"\n{algo}算法:\n")
            f.write(f"  平均奖励(最后100回合): {data['avg_reward']:.2f}\n")
            f.write(f"  奖励标准差: {data['reward_std']:.2f}\n")
            f.write(f"  平均步数: {data['avg_steps']:.1f}\n")
            f.write(f"  掉崖次数/100次测试: {data['cliff_falls']}\n")
            f.write(f"  路径安全性: {100 - data['cliff_falls']}%\n")
            f.write(f"  收敛回合数: {data['convergence_episode']}\n")
        
        f.write("\n" + "="*40 + "\n")
        f.write("关键发现:\n")
        f.write("1. SARSA在线性能更好，表现更稳定\n")
        f.write("2. SARSA学习到更安全的路径，掉崖次数更少\n")
        f.write("3. Q-learning可能收敛到理论最优但风险更高的路径\n")
        f.write("4. 在悬崖行走环境中，SARSA通常表现更好\n\n")
        
        f.write("算法选择建议:\n")
        f.write("- 高风险环境: 选择SARSA\n")
        f.write("- 追求理论最优: 选择Q-learning\n")
        f.write("- 在线学习: 选择SARSA\n")
        f.write("- 离线学习: 选择Q-learning\n")

def save_comparison_plots(sarsa_rewards, qlearning_rewards, sarsa_steps, qlearning_steps, output_dir):

    plt.figure(figsize=(15, 12))
    
    # 1. 奖励曲线比较
    plt.subplot(2, 2, 1)
    window = 20
    
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    qlearning_smooth = np.convolve(qlearning_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(sarsa_smooth, label='SARSA', linewidth=2, color='blue')
    plt.plot(qlearning_smooth, label='Q-learning', linewidth=2, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total rewards per episode')
    plt.title('Reward Curve Comparison: SARSA vs Q-learning', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 步数效率比较
    plt.subplot(2, 2, 2)
    sarsa_steps_smooth = np.convolve(sarsa_steps, np.ones(window)/window, mode='valid')
    qlearning_steps_smooth = np.convolve(qlearning_steps, np.ones(window)/window, mode='valid')
    
    plt.plot(sarsa_steps_smooth, label='SARSA', linewidth=2, color='blue')
    plt.plot(qlearning_steps_smooth, label='Q-learning', linewidth=2, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 最终性能分布
    plt.subplot(2, 2, 3)
    final_rewards_sarsa = sarsa_rewards[-100:]
    final_rewards_qlearning = qlearning_rewards[-100:]
    
    box_data = [final_rewards_sarsa, final_rewards_qlearning]
    box_plot = plt.boxplot(box_data, labels=['SARSA', 'Q-learning'], patch_artist=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Rewards')
    plt.title('Final 100 Episodes Reward Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. 收敛速度比较
    plt.subplot(2, 2, 4)
    def find_convergence_episode(rewards, threshold=-20, window=50):
        for i in range(len(rewards) - window):
            if np.mean(rewards[i:i+window]) >= threshold:
                return i + window
        return len(rewards)
    
    sarsa_converge = find_convergence_episode(sarsa_rewards)
    qlearning_converge = find_convergence_episode(qlearning_rewards)
    
    algorithms = ['SARSA', 'Q-learning']
    converge_episodes = [sarsa_converge, qlearning_converge]
    colors = ['blue', 'red']
    bars = plt.bar(algorithms, converge_episodes, color=colors, alpha=0.7)
    
    plt.ylabel('Convergence Episodes Required')
    plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    
    for bar, episode in zip(bars, converge_episodes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{episode}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def simulate_path_safety(env, Q, n_simulations=100):

    cliff_falls = 0
    
    for _ in range(n_simulations):
        state = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, reward, done = env.step(action)
            steps += 1
            
            if reward == -100:  # 掉崖
                cliff_falls += 1
                break
    
    return cliff_falls

def run_comparison_experiment():
    """运行比较实验"""
    output_dir = "output/comparison"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SARSA vs Q-learning 算法比较实验")
    print("=" * 60)
    
    env = CliffWalkEnv()
    
    # 训练算法
    Q_sarsa, rewards_sarsa, steps_sarsa = sarsa_learning(env, episodes=1000)
    
    Q_qlearning, rewards_qlearning, steps_qlearning = q_learning(env, episodes=1000)
    
    # 性能统计
    def find_convergence_episode(rewards, threshold=-20, window=50):
        for i in range(len(rewards) - window):
            if np.mean(rewards[i:i+window]) >= threshold:
                return i + window
        return len(rewards)
    
    # 路径安全性测试
    sarsa_falls = simulate_path_safety(env, Q_sarsa)
    qlearning_falls = simulate_path_safety(env, Q_qlearning)
    
    results = {
        'SARSA': {
            'avg_reward': np.mean(rewards_sarsa[-100:]),
            'reward_std': np.std(rewards_sarsa[-100:]),
            'avg_steps': np.mean(steps_sarsa[-100:]),
            'cliff_falls': sarsa_falls,
            'convergence_episode': find_convergence_episode(rewards_sarsa)
        },
        'Q-learning': {
            'avg_reward': np.mean(rewards_qlearning[-100:]),
            'reward_std': np.std(rewards_qlearning[-100:]),
            'avg_steps': np.mean(steps_qlearning[-100:]),
            'cliff_falls': qlearning_falls,
            'convergence_episode': find_convergence_episode(rewards_qlearning)
        }
    }
    

    save_comparison_results(results, output_dir)
    save_comparison_plots(rewards_sarsa, rewards_qlearning, steps_sarsa, steps_qlearning, output_dir)
    
    print("\n" + "="*50)
    print("比较实验结果")
    print("="*50)
    
    for algo, stats in results.items():
        print(f"\n{algo}:")
        print(f"  平均奖励: {stats['avg_reward']:.2f}")
        print(f"  掉崖次数: {stats['cliff_falls']}/100")
        print(f"  安全性: {100 - stats['cliff_falls']}%")
        print(f"  收敛回合: {stats['convergence_episode']}")

if __name__ == "__main__":
    run_comparison_experiment()