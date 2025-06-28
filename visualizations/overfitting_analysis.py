# visualizations/overfitting_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_overfitting_comparison():
    """Compare single-year vs multi-year training approaches"""
    
    # Data for comparison
    approaches = ['2021\nSingle-Year', '2023\nSingle-Year', 'Multi-Year\n(2021-2023)']
    
    # Performance on training data
    training_performance = [42, -23, 15.8]  # % improvement on training year(s)
    
    # Performance on 2024 test data  
    test_performance = [np.nan, -23, -16.8]  # % improvement on 2024
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(approaches))
    width = 0.35
    
    # Training performance bars
    train_bars = ax.bar(x - width/2, training_performance, width, 
                       label='Training Performance', color='#2E86AB', alpha=0.8)
    
    # Test performance bars (excluding NaN for 2021 model)
    test_vals = [0 if np.isnan(val) else val for val in test_performance]
    test_bars = ax.bar(x + width/2, test_vals, width,
                      label='2024 Test Performance', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Training Approach')
    ax.set_ylabel('Improvement Over Baseline (%)')
    ax.set_title('Training vs Generalization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(training_performance, test_performance)):
        # Training labels
        ax.text(i - width/2, train_val + (2 if train_val > 0 else -4), 
               f'{train_val:+.0f}%', ha='center', fontweight='bold')
        
        # Test labels (skip NaN)
        if not np.isnan(test_val):
            ax.text(i + width/2, test_val + (2 if test_val > 0 else -4),
                   f'{test_val:+.1f}%', ha='center', fontweight='bold')
        else:
            ax.text(i + width/2, 5, 'Not Tested', ha='center', 
                   style='italic', color='gray')
    
    # Add insights
    ax.text(0.5, 35, 'Severe Overfitting', ha='center', fontweight='bold', 
           color='red', fontsize=12)
    ax.text(1.5, 15, 'Consistent Failure', ha='center', fontweight='bold', 
           color='orange', fontsize=12)
    ax.text(2.5, 10, 'Still Poor Generalization', ha='center', fontweight='bold', 
           color='darkred', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_domain_shift_visualization():
    """Visualize the fantasy football meta evolution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Baseline performance across years (showing meta evolution)
    years = [2021, 2022, 2023, 2024]
    baseline_scores = [1916, 1931, 1944, 1975]
    
    ax1.plot(years, baseline_scores, 'o-', linewidth=3, markersize=8, color='#A23B72')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Baseline Fantasy Points')
    ax1.set_title('Fantasy Football Meta Evolution\n(Rising Baseline Performance)')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(years, baseline_scores, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), "--", alpha=0.7, color='red',
             label=f'Trend: +{z[0]:.1f} points/year')
    ax1.legend()
    
    # Annotations
    ax1.annotate('Post-COVID\nPatterns', xy=(2021, 1916), xytext=(2021, 1870),
                arrowprops=dict(arrowstyle='->', color='blue'),
                ha='center', fontsize=10)
    
    ax1.annotate('Rule Changes\nNew Schemes', xy=(2022.5, 1937), xytext=(2022.5, 1890),
                arrowprops=dict(arrowstyle='->', color='orange'),
                ha='center', fontsize=10)
    
    ax1.annotate('Evolved Meta\nSmarter Drafting', xy=(2024, 1975), xytext=(2024, 2020),
                arrowprops=dict(arrowstyle='->', color='green'),
                ha='center', fontsize=10)
    
    # Right: Agent performance relative to evolving baseline
    agent_scores = [2418, 2368, 2114, 1644]
    relative_performance = [(agent - baseline) / baseline * 100 
                          for agent, baseline in zip(agent_scores, baseline_scores)]
    
    colors = ['green' if x > 0 else 'red' for x in relative_performance]
    bars = ax2.bar(years, relative_performance, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Agent Performance vs Baseline (%)')
    ax2.set_title('Agent Struggles as Meta Evolves')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, perf in zip(bars, relative_performance):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                f'{perf:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/domain_shift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_overfitting_comparison()
    create_domain_shift_visualization()