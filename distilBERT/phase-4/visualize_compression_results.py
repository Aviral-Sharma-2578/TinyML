#!/usr/bin/env python3
"""
Phase 4: Comprehensive Visualization and Analysis of Compression Results

This module analyzes and visualizes the performance of all models generated through
the compression pipeline (baseline → pruning → quantization) to provide insights
into the trade-offs between model size, accuracy, and inference speed.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
try:
    sns.set_palette("husl")
except:
    pass  # Use default palette if seaborn styling fails

class CompressionAnalyzer:
    """Analyzes and visualizes compression results across all phases"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / "outputs"
        self.phase4_dir = self.base_dir / "phase-4"
        self.phase4_dir.mkdir(exist_ok=True)
        
        # Load all metrics
        self.metrics = self._load_all_metrics()
        self.models_data = self._prepare_models_data()
        
    def _load_all_metrics(self) -> Dict[str, Any]:
        """Load metrics from all phases"""
        metrics = {}
        
        # Phase 1: Baseline
        baseline_path = self.outputs_dir / "phase-1" / "baseline_metrics.json"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                metrics['baseline'] = json.load(f)
        
        # Phase 2: Pruning
        pruning_path = self.outputs_dir / "phase-2" / "true_reduction_pruning_metrics.json"
        if pruning_path.exists():
            with open(pruning_path, 'r') as f:
                metrics['pruning'] = json.load(f)
        
        # Phase 3: Quantization (baseline)
        quant_baseline_path = self.outputs_dir / "phase-3" / "torchao_quantization_baseline_results.json"
        if quant_baseline_path.exists():
            with open(quant_baseline_path, 'r') as f:
                metrics['quantization_baseline'] = json.load(f)
        
        # Phase 3: Quantization (pruned)
        quant_pruned_path = self.outputs_dir / "phase-3" / "torchao_quantization_results.json"
        if quant_pruned_path.exists():
            with open(quant_pruned_path, 'r') as f:
                metrics['quantization_pruned'] = json.load(f)
        
        return metrics
    
    def _prepare_models_data(self) -> pd.DataFrame:
        """Prepare consolidated data for all models"""
        models = []
        
        # Baseline model
        if 'baseline' in self.metrics:
            baseline = self.metrics['baseline']
            models.append({
                'model_name': 'Baseline',
                'phase': 'Phase 1',
                'compression_type': 'None',
                'accuracy': baseline['accuracy'],
                'f1': baseline['f1'],
                'param_count': baseline['param_count'],
                'model_size_mb': baseline['model_size_mb'],
                'latency_sec': baseline['avg_latency_sec'],
                'memory_mb': baseline['memory_usage_mb'],
                'compression_ratio': 1.0,
                'size_reduction_percent': 0.0,
                'speed_improvement_percent': 0.0
            })
        
        # Pruned model (final step)
        if 'pruning' in self.metrics:
            pruning = self.metrics['pruning']
            models.append({
                'model_name': 'Pruned (3 steps)',
                'phase': 'Phase 2',
                'compression_type': 'Structured Pruning',
                'accuracy': pruning['final_accuracy'],
                'f1': pruning['final_f1'],
                'param_count': pruning['param_count_after'],
                'model_size_mb': pruning['model_size_mb_after'],
                'latency_sec': pruning['avg_latency_sec_final'],
                'memory_mb': pruning['memory_usage_mb_final'],
                'compression_ratio': pruning['compression_ratio'],
                'size_reduction_percent': pruning['reduction_percentage'],
                'speed_improvement_percent': ((baseline['avg_latency_sec'] - pruning['avg_latency_sec_final']) / baseline['avg_latency_sec']) * 100
            })
        
        # Quantized baseline
        if 'quantization_baseline' in self.metrics:
            quant_baseline = self.metrics['quantization_baseline']
            models.append({
                'model_name': 'Baseline + Int8',
                'phase': 'Phase 3',
                'compression_type': 'Int8 Quantization',
                'accuracy': quant_baseline['quantized_int8_wo']['accuracy'],
                'f1': quant_baseline['quantized_int8_wo']['f1'],
                'param_count': quant_baseline['quantized_int8_wo']['param_count'],
                'model_size_mb': quant_baseline['quantized_int8_wo']['model_size_mb'],
                'latency_sec': quant_baseline['quantized_int8_wo']['inference_time'],
                'memory_mb': baseline['memory_usage_mb'],  # Approximate
                'compression_ratio': baseline['model_size_mb'] / quant_baseline['quantized_int8_wo']['model_size_mb'],
                'size_reduction_percent': quant_baseline['improvements']['size_reduction_percent'],
                'speed_improvement_percent': quant_baseline['improvements']['speed_improvement_percent']
            })
        
        # Quantized pruned
        if 'quantization_pruned' in self.metrics:
            quant_pruned = self.metrics['quantization_pruned']
            models.append({
                'model_name': 'Pruned + Int8',
                'phase': 'Phase 3',
                'compression_type': 'Pruning + Int8',
                'accuracy': quant_pruned['quantized_int8_wo']['accuracy'],
                'f1': quant_pruned['quantized_int8_wo']['f1'],
                'param_count': quant_pruned['quantized_int8_wo']['param_count'],
                'model_size_mb': quant_pruned['quantized_int8_wo']['model_size_mb'],
                'latency_sec': quant_pruned['quantized_int8_wo']['inference_time'],
                'memory_mb': pruning['memory_usage_mb_final'],  # Approximate
                'compression_ratio': pruning['model_size_mb_after'] / quant_pruned['quantized_int8_wo']['model_size_mb'],
                'size_reduction_percent': quant_pruned['improvements']['size_reduction_percent'],
                'speed_improvement_percent': quant_pruned['improvements']['speed_improvement_percent']
            })
        
        return pd.DataFrame(models)
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        print("🎨 Creating comprehensive compression analysis dashboard...")
        
        # Create subplots with more vertical space
        fig = plt.figure(figsize=(20, 18))
        
        # 1. Model Size vs Accuracy (main trade-off)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_size_vs_accuracy(ax1)
        
        # 2. Parameter Count vs Model Size
        ax2 = plt.subplot(2, 3, 2)
        self._plot_params_vs_size(ax2)
        
        # 3. Compression Ratio Comparison
        ax3 = plt.subplot(2, 3, 3)
        self._plot_compression_ratios(ax3)
        
        # 4. Latency vs Accuracy
        ax4 = plt.subplot(2, 3, 4)
        self._plot_latency_vs_accuracy(ax4)
        
        # 5. Memory Usage Comparison
        ax5 = plt.subplot(2, 3, 5)
        self._plot_memory_comparison(ax5)
        
        # 6. Overall Performance Radar Chart
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        self._plot_radar_chart(ax6)
        
        # Adjust layout to prevent title overlap
        plt.subplots_adjust(
            left=0.08,    # Left margin
            right=0.95,   # Right margin  
            bottom=0.12,  # Bottom margin (increased to prevent overlap)
            top=0.95,     # Top margin
            wspace=0.3,   # Horizontal space between subplots
            hspace=0.4    # Vertical space between subplots (increased)
        )
        
        plt.savefig(self.phase4_dir / "compression_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed analysis
        self._save_detailed_analysis()
        
    def _plot_size_vs_accuracy(self, ax):
        """Plot model size vs accuracy trade-off"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (_, model) in enumerate(self.models_data.iterrows()):
            ax.scatter(model['model_size_mb'], model['accuracy'], 
                      s=200, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(model['model_name'], 
                       (model['model_size_mb'], model['accuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Size vs Accuracy Trade-off', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add compression annotations
        baseline_size = self.models_data.iloc[0]['model_size_mb']
        for _, model in self.models_data.iterrows():
            if model['model_name'] != 'Baseline':
                reduction = ((baseline_size - model['model_size_mb']) / baseline_size) * 100
                ax.annotate(f'{reduction:.1f}% smaller', 
                           (model['model_size_mb'], model['accuracy']),
                           xytext=(-20, -15), textcoords='offset points', fontsize=9,
                           color='red', fontweight='bold')
    
    def _plot_params_vs_size(self, ax):
        """Plot parameter count vs model size"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (_, model) in enumerate(self.models_data.iterrows()):
            ax.scatter(model['param_count'] / 1e6, model['model_size_mb'], 
                      s=200, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(model['model_name'], 
                       (model['param_count'] / 1e6, model['model_size_mb']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Count vs Model Size', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
    
    def _plot_compression_ratios(self, ax):
        """Plot compression ratios for different models"""
        models = self.models_data[self.models_data['model_name'] != 'Baseline']
        
        x = range(len(models))
        compression_ratios = models['compression_ratio'].values
        size_reductions = models['size_reduction_percent'].values
        
        bars = ax.bar(x, compression_ratios, color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, ratio, reduction) in enumerate(zip(bars, compression_ratios, size_reductions)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{ratio:.2f}x\n({reduction:.1f}% smaller)', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Variants', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Compression Ratios by Model', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\n') for name in models['model_name']], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_latency_vs_accuracy(self, ax):
        """Plot latency vs accuracy trade-off"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (_, model) in enumerate(self.models_data.iterrows()):
            ax.scatter(model['latency_sec'] * 1000, model['accuracy'], 
                      s=200, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(model['model_name'], 
                       (model['latency_sec'] * 1000, model['accuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Latency vs Accuracy Trade-off', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    def _plot_memory_comparison(self, ax):
        """Plot memory usage comparison"""
        models = self.models_data['model_name'].values
        memory_usage = self.models_data['memory_mb'].values
        
        bars = ax.bar(range(len(models)), memory_usage, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
        
        # Add value labels
        for i, (bar, mem) in enumerate(zip(bars, memory_usage)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{mem:.0f} MB', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Variants', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in models], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_radar_chart(self, ax):
        """Create a radar chart showing overall performance metrics"""
        # Normalize metrics to 0-1 scale for radar chart
        metrics_to_plot = ['accuracy', 'compression_ratio', 'speed_improvement_percent']
        
        # Normalize compression ratio (higher is better)
        max_compression = self.models_data['compression_ratio'].max()
        normalized_compression = self.models_data['compression_ratio'] / max_compression
        
        # Normalize speed improvement (convert negative to positive scale)
        speed_improvements = self.models_data['speed_improvement_percent'].values
        normalized_speed = (speed_improvements - speed_improvements.min()) / (speed_improvements.max() - speed_improvements.min())
        
        # Prepare data for radar chart
        categories = ['Accuracy', 'Compression', 'Speed']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (_, model) in enumerate(self.models_data.iterrows()):
            values = [
                model['accuracy'],  # Already 0-1
                normalized_compression.iloc[i],  # Normalized
                normalized_speed[i]  # Normalized
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model['model_name'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def _save_detailed_analysis(self):
        """Save detailed analysis results"""
        analysis = {
            'summary': self._generate_summary(),
            'compression_efficiency': self._analyze_compression_efficiency(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save as JSON
        with open(self.phase4_dir / "detailed_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save as CSV
        self.models_data.to_csv(self.phase4_dir / "models_comparison.csv", index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 COMPRESSION ANALYSIS SUMMARY")
        print("="*60)
        print(analysis['summary'])
        print("\n" + "="*60)
        print("💡 RECOMMENDATIONS")
        print("="*60)
        for rec in analysis['recommendations']:
            print(f"• {rec}")
    
    def _generate_summary(self) -> str:
        """Generate a text summary of the compression results"""
        baseline = self.models_data.iloc[0]
        
        summary = f"""
Compression Pipeline Results Summary:

📈 BASELINE MODEL:
   • Accuracy: {baseline['accuracy']:.4f}
   • Parameters: {baseline['param_count']:,}
   • Size: {baseline['model_size_mb']:.2f} MB
   • Latency: {baseline['latency_sec']*1000:.2f} ms

🔍 COMPRESSION RESULTS:
"""
        
        for _, model in self.models_data.iterrows():
            if model['model_name'] != 'Baseline':
                size_reduction = ((baseline['model_size_mb'] - model['model_size_mb']) / baseline['model_size_mb']) * 100
                param_reduction = ((baseline['param_count'] - model['param_count']) / baseline['param_count']) * 100
                accuracy_change = (model['accuracy'] - baseline['accuracy']) * 100
                
                summary += f"""
   • {model['model_name']}:
     - Size: {model['model_size_mb']:.2f} MB ({size_reduction:+.1f}%)
     - Parameters: {model['param_count']:,} ({param_reduction:+.1f}%)
     - Accuracy: {model['accuracy']:.4f} ({accuracy_change:+.2f}%)
     - Latency: {model['latency_sec']*1000:.2f} ms
"""
        
        return summary
    
    def _analyze_compression_efficiency(self) -> Dict[str, Any]:
        """Analyze the efficiency of different compression techniques"""
        baseline = self.models_data.iloc[0]
        
        efficiency = {}
        for _, model in self.models_data.iterrows():
            if model['model_name'] != 'Baseline':
                size_reduction = ((baseline['model_size_mb'] - model['model_size_mb']) / baseline['model_size_mb']) * 100
                param_reduction = ((baseline['param_count'] - model['param_count']) / baseline['param_count']) * 100
                accuracy_change = (model['accuracy'] - baseline['accuracy']) * 100
                
                # Efficiency score: (size_reduction + param_reduction) / abs(accuracy_change + 0.01)
                efficiency_score = (size_reduction + param_reduction) / (abs(accuracy_change) + 0.01)
                
                efficiency[model['model_name']] = {
                    'size_reduction_percent': size_reduction,
                    'param_reduction_percent': param_reduction,
                    'accuracy_change_percent': accuracy_change,
                    'efficiency_score': efficiency_score,
                    'compression_ratio': model['compression_ratio']
                }
        
        return efficiency
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Find best compression for different scenarios
        best_size_reduction = self.models_data.loc[self.models_data['size_reduction_percent'].idxmax()]
        best_accuracy = self.models_data.loc[self.models_data['accuracy'].idxmax()]
        best_compression_ratio = self.models_data.loc[self.models_data['compression_ratio'].idxmax()]
        
        recommendations.append(f"For maximum size reduction: Use {best_size_reduction['model_name']} ({best_size_reduction['size_reduction_percent']:.1f}% smaller)")
        recommendations.append(f"For best accuracy: Use {best_accuracy['model_name']} (accuracy: {best_accuracy['accuracy']:.4f})")
        recommendations.append(f"For best compression ratio: Use {best_compression_ratio['model_name']} ({best_compression_ratio['compression_ratio']:.2f}x)")
        
        # Check if quantization improved or hurt performance
        quant_models = self.models_data[self.models_data['compression_type'].str.contains('Int8')]
        for _, model in quant_models.iterrows():
            if 'Pruned' in model['model_name']:
                if model['accuracy'] > 0.90:  # Good accuracy threshold
                    recommendations.append(f"Pruned + Int8 quantization maintains good accuracy ({model['accuracy']:.4f}) with significant size reduction")
                else:
                    recommendations.append(f"Consider using only pruning without quantization to maintain accuracy")
        
        # Edge device recommendations
        recommendations.append("For edge devices with strict memory constraints, consider the Pruned + Int8 model")
        recommendations.append("For edge devices prioritizing accuracy, use the Pruned model without quantization")
        
        return recommendations

def main():
    """Main function to run the compression analysis"""
    print("🚀 Starting Phase 4: Compression Results Visualization")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CompressionAnalyzer()
    
    # Check if we have data to analyze
    if analyzer.models_data.empty:
        print("❌ No model data found. Please run phases 1-3 first.")
        return
    
    print(f"✅ Loaded data for {len(analyzer.models_data)} models:")
    for _, model in analyzer.models_data.iterrows():
        print(f"   • {model['model_name']} ({model['phase']})")
    
    # Create comprehensive dashboard
    analyzer.create_comprehensive_dashboard()
    
    print(f"\n🎉 Analysis complete! Results saved in: {analyzer.phase4_dir}")

if __name__ == "__main__":
    main()
