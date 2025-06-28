# visualizations/draft_strategy_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_draft_strategy_heatmap():
    """Show drafting patterns by position and round"""
    
    # Simulated data - you'd get this from analyzing actual draft logs
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    rounds = list(range(1, 17))
    
    # Agent strategy (2021 successful model)
    agent_strategy = np.array([
        [5, 15, 10, 5, 0, 0, 8, 3, 2, 1, 1, 0, 0, 0, 0, 0],  # QB
        [40, 35, 30, 25, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 1, 0],  # RB  
        [35, 30, 35, 40, 35, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2, 1],  # WR
        [10, 15, 20, 25, 30, 25, 20, 15, 10, 8, 6, 4, 2, 1, 0, 0],  # TE
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 15, 25, 40, 50, 30],  # K
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 15, 25, 35, 45, 40, 60]   # DST
    ])
    
    # Baseline strategy 
    baseline_strategy = np.array([
        [8, 12, 15, 8, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # QB
        [35, 40, 35, 30, 25, 20, 15, 12, 10, 8, 6, 4, 3, 2, 1, 1],  # RB
        [40, 35, 40, 45, 40, 35, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2],  # WR  
        [12, 10, 8, 15, 25, 35, 25, 15, 10, 8, 6, 4, 3, 2, 1, 0],  # TE
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 15, 30, 45, 55, 70],  # K
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 20, 40, 50, 45, 25]   # DST
    ])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Agent strategy heatmap
    sns.heatmap(agent_strategy, xticklabels=rounds, yticklabels=positions,
                annot=False, cmap='Blues', ax=ax1, cbar_kws={'label': 'Pick Frequency (%)'})
    ax1.set_title('RL Agent Strategy (2021)')
    ax1.set_xlabel('Draft Round')
    ax1.set_ylabel('Position')
    
    # Baseline strategy heatmap  
    sns.heatmap(baseline_strategy, xticklabels=rounds, yticklabels=positions,
                annot=False, cmap='Reds', ax=ax2, cbar_kws={'label': 'Pick Frequency (%)'})
    ax2.set_title('Heuristic Baseline Strategy')
    ax2.set_xlabel('Draft Round')
    ax2.set_ylabel('')
    
    # Difference (Agent - Baseline)
    difference = agent_strategy - baseline_strategy
    sns.heatmap(difference, xticklabels=rounds, yticklabels=positions,
                annot=False, cmap='RdBu_r', center=0, ax=ax3, 
                cbar_kws={'label': 'Difference (Agent - Baseline)'})
    ax3.set_title('Strategy Difference\n(Blue = Agent Higher, Red = Baseline Higher)')
    ax3.set_xlabel('Draft Round')
    ax3.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('results/draft_strategy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_position_timing_analysis():
    """Analyze K/DST timing patterns"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rounds = list(range(10, 17))  # Focus on late rounds
    
    # K/DST drafting percentages by round
    agent_k = [0, 5, 10, 15, 25, 40, 50]
    agent_dst = [2, 8, 15, 25, 35, 45, 40]
    baseline_k = [0, 0, 5, 15, 30, 45, 55] 
    baseline_dst = [0, 2, 8, 20, 40, 50, 45]
    
    # Kicker timing
    ax1.plot(rounds, agent_k, 'o-', linewidth=3, label='RL Agent', color='#2E86AB')
    ax1.plot(rounds, baseline_k, 's-', linewidth=3, label='Baseline', color='#A23B72')
    ax1.set_xlabel('Draft Round')
    ax1.set_ylabel('Pick Percentage (%)')
    ax1.set_title('Kicker Drafting Timing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # DST timing  
    ax2.plot(rounds, agent_dst, 'o-', linewidth=3, label='RL Agent', color='#2E86AB')
    ax2.plot(rounds, baseline_dst, 's-', linewidth=3, label='Baseline', color='#A23B72')
    ax2.set_xlabel('Draft Round')
    ax2.set_ylabel('Pick Percentage (%)')
    ax2.set_title('Defense/ST Drafting Timing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/position_timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_key_insights_summary():
    """Visual summary of key findings"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Generalization Challenge
    years = ['2021', '2022', '2023', '2024']
    improvements = [26.2, 22.6, 8.7, -16.8]
    colors = ['green', 'green', 'orange', 'red']
    
    bars1 = ax1.bar(years, improvements, color=colors, alpha=0.7)
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('1. Generalization Failure')
    ax1.axhline(y=0, color='black', linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    for bar, imp in zip(bars1, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                f'{imp:+.1f}%', ha='center', fontweight='bold')
    
    # 2. Training Data Bias
    train_years = ['2021', '2022', '2023']
    train_weights = [15, 35, 50]  # Training weights
    train_performance = [26.2, 22.6, 8.7]
    
    ax2_twin = ax2.twinx()
    bars2 = ax2.bar(train_years, train_weights, alpha=0.3, color='gray', 
                    label='Training Weight (%)')
    line2 = ax2_twin.plot(train_years, train_performance, 'ro-', linewidth=3, 
                         markersize=8, label='Performance (%)')
    
    ax2.set_ylabel('Training Weight (%)', color='gray')
    ax2_twin.set_ylabel('Performance Improvement (%)', color='red')
    ax2.set_title('2. Recency Bias ≠ Better Performance')
    ax2.grid(True, alpha=0.3)
    
    # 3. Meta Evolution Impact
    meta_factors = ['Rule\nChanges', 'Offensive\nSchemes', 'Player\nUsage', 'Draft\nStrategy']
    impact_scores = [3, 4, 5, 4]  # Arbitrary impact scores
    
    bars3 = ax3.bar(meta_factors, impact_scores, color='orange', alpha=0.7)
    ax3.set_ylabel('Impact Level (1-5)')
    ax3.set_title('3. Fantasy Meta Evolution Factors')
    ax3.grid(True, alpha=0.3)
    
    for bar, score in zip(bars3, impact_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., score + 0.1,
                f'{score}', ha='center', fontweight='bold')
    
    # 4. RL vs Domain Characteristics
    characteristics = ['Rule\nStability', 'Pattern\nPersistence', 'Signal/\nNoise', 'Historical\nRelevance']
    rl_needs = [5, 5, 4, 5]  # What RL needs (high scores)
    fantasy_reality = [2, 1, 2, 1]  # What fantasy provides (low scores)
    
    x = np.arange(len(characteristics))
    width = 0.35
    
    ax4.bar(x - width/2, rl_needs, width, label='RL Requirements', 
            color='#2E86AB', alpha=0.8)
    ax4.bar(x + width/2, fantasy_reality, width, label='Fantasy Reality', 
            color='#A23B72', alpha=0.8)
    
    ax4.set_ylabel('Level (1-5)')
    ax4.set_title('4. RL-Domain Mismatch')
    ax4.set_xticks(x)
    ax4.set_xticklabels(characteristics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/key_insights_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_draft_strategy_heatmap()
    create_position_timing_analysis()
    create_key_insights_summary()