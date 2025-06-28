# visualizations/performance_distribution.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def create_performance_distributions():
    """Show distribution of agent vs baseline scores"""
    
    # Simulated data based on your results (you'd use actual episode scores)
    np.random.seed(42)
    
    # 2021 (successful)
    agent_2021 = np.random.normal(2418, 120, 300)
    baseline_2021 = np.random.normal(1916, 80, 300)
    
    # 2024 (failed)
    agent_2024 = np.random.normal(1644, 150, 300)
    baseline_2024 = np.random.normal(1975, 85, 300)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2021 distributions
    ax1.hist(baseline_2021, bins=30, alpha=0.7, label='Baseline', color='#A23B72', density=True)
    ax1.hist(agent_2021, bins=30, alpha=0.7, label='RL Agent', color='#2E86AB', density=True)
    ax1.axvline(np.mean(baseline_2021), color='#A23B72', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(agent_2021), color='#2E86AB', linestyle='--', linewidth=2)
    ax1.set_xlabel('Fantasy Points')
    ax1.set_ylabel('Density')
    ax1.set_title('2021: Successful Generalization\n(Agent clearly dominates)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2024 distributions
    ax2.hist(baseline_2024, bins=30, alpha=0.7, label='Baseline', color='#A23B72', density=True)
    ax2.hist(agent_2024, bins=30, alpha=0.7, label='RL Agent', color='#2E86AB', density=True)
    ax2.axvline(np.mean(baseline_2024), color='#A23B72', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(agent_2024), color='#2E86AB', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fantasy Points')
    ax2.set_ylabel('Density')
    ax2.set_title('2024: Failed Generalization\n(Baseline dominates)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Win rate analysis
    years = ['2021', '2022', '2023', '2024']
    win_rates = [100, 100, 93, 0]  # Your actual win rates
    colors = ['green' if x > 50 else 'red' for x in win_rates]
    
    bars = ax3.bar(years, win_rates, color=colors, alpha=0.7)
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_xlabel('Year')
    ax3.set_title('Agent Win Rate vs Baseline')
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='Random Chance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    for bar, rate in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2., rate + 2,
                f'{rate}%', ha='center', fontweight='bold')
    
    # Statistical significance test
    years_data = ['2021', '2022', '2023', '2024']
    p_values = [0.001, 0.001, 0.02, 0.95]  # Simulated p-values
    significance = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.' 
                   for p in p_values]
    
    bars4 = ax4.bar(years_data, [-np.log10(p) for p in p_values], 
                    color=['green' if s != 'n.s.' else 'red' for s in significance], alpha=0.7)
    ax4.set_ylabel('-log₁₀(p-value)')
    ax4.set_xlabel('Year')
    ax4.set_title('Statistical Significance of Improvement')
    ax4.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.7, 
                label='p = 0.05 threshold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    for bar, sig in zip(bars4, significance):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                sig, ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_variance_analysis():
    """Analyze performance variance across different conditions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Draft position analysis
    draft_positions = list(range(1, 13))
    
    # Simulated performance by draft position (2021 vs 2024)
    performance_2021 = [28, 26, 24, 22, 25, 27, 26, 24, 23, 25, 27, 29]  # Relatively stable
    performance_2024 = [-15, -20, -18, -16, -14, -17, -19, -15, -16, -18, -17, -16]  # Consistently bad
    
    ax1.plot(draft_positions, performance_2021, 'o-', linewidth=3, 
             label='2021 (Success)', color='green', markersize=6)
    ax1.plot(draft_positions, performance_2024, 's-', linewidth=3, 
             label='2024 (Failure)', color='red', markersize=6)
    ax1.set_xlabel('Draft Position')
    ax1.set_ylabel('Improvement Over Baseline (%)')
    ax1.set_title('Performance by Draft Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Position value analysis
    positions = ['QB', 'RB', 'WR', 'TE', 'FLEX']
    value_captured_2021 = [85, 92, 88, 80, 90]  # % of available value captured
    value_captured_2024 = [45, 40, 42, 38, 41]  # Much worse
    
    x = np.arange(len(positions))
    width = 0.35
    
    ax2.bar(x - width/2, value_captured_2021, width, label='2021 Success', 
            color='green', alpha=0.7)
    ax2.bar(x + width/2, value_captured_2024, width, label='2024 Failure', 
            color='red', alpha=0.7)
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Value Captured (%)')
    ax2.set_title('Positional Value Capture')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_performance_distributions()
    create_variance_analysis()