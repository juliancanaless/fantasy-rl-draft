import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_generalization_chart():
    """Create the key chart showing generalization failure"""
    
    # Your actual results data
    years = ['2021', '2022', '2023', '2024']
    agent_scores = [2418, 2368, 2114, 1644]
    baseline_scores = [1916, 1931, 1944, 1975]
    improvements = [501, 437, 170, -331]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Absolute scores
    x = np.arange(len(years))
    width = 0.35
    
    ax1.bar(x - width/2, agent_scores, width, label='RL Agent', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, baseline_scores, width, label='Heuristic Baseline', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Fantasy Points')
    ax1.set_title('Agent vs Baseline Performance by Year')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add training/test annotations
    ax1.axvspan(-0.5, 2.5, alpha=0.1, color='green', label='Training Years')
    ax1.axvspan(2.5, 3.5, alpha=0.1, color='red', label='Test Year')
    ax1.text(1, 2500, 'Training Data', ha='center', fontweight='bold')
    ax1.text(3, 2500, 'Test Data', ha='center', fontweight='bold', color='red')
    
    # Right plot: Improvement over baseline
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax2.bar(years, improvements, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Points Above Baseline')
    ax2.set_title('Agent Improvement Over Baseline')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (20 if height > 0 else -40),
                f'{improvement:+.0f}', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    # Add generalization gap annotation
    ax2.annotate('Generalization Gap:\n700+ points', 
                xy=(3, -331), xytext=(2.2, -200),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/generalization_failure.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_progression():
    """Show how performance changed across training years"""
    
    years = ['2021', '2022', '2023']
    improvements = [26.2, 22.6, 8.7]  # Percentage improvements
    win_rates = [100, 100, 93]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Improvement percentage
    ax1.plot(years, improvements, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    ax1.set_ylabel('Improvement Over Baseline (%)')
    ax1.set_title('Training Performance Degradation Across Years')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 30)
    
    # Add trend line
    z = np.polyfit(range(len(years)), improvements, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(range(len(years))), "--", alpha=0.7, color='red', 
             label=f'Trend: {z[0]:.1f}% decline per year')
    ax1.legend()
    
    # Bottom: Win rate
    ax2.bar(years, win_rates, color='#A23B72', alpha=0.7)
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_xlabel('Training Year')
    ax2.set_ylim(80, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(win_rates):
        ax2.text(i, v + 1, f'{v}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/training_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_generalization_chart()
    create_training_progression()