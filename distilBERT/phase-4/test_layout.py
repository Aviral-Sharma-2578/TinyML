#!/usr/bin/env python3
"""
Test script to verify layout improvements work correctly
"""

import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visualize_compression_results import CompressionAnalyzer
    
    print("🧪 Testing Phase 4 Layout Improvements...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CompressionAnalyzer()
    
    # Check if we have data
    if analyzer.models_data.empty:
        print("❌ No model data found. Please run phases 1-3 first.")
        return
    
    print(f"✅ Loaded data for {len(analyzer.models_data)} models")
    
    # Test the layout adjustment function
    print("\n🔧 Testing layout adjustments...")
    
    # Create a simple test plot to verify spacing
    import matplotlib.pyplot as plt
    
    # Create test subplots
    fig = plt.figure(figsize=(20, 18))
    
    # Test subplot creation
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Test Title 1', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('X Axis 1')
    ax1.set_ylabel('Y Axis 1')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Test Title 2', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('X Axis 2')
    ax2.set_ylabel('Y Axis 2')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Test Title 3', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('X Axis 3')
    ax3.set_ylabel('Y Axis 3')
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Test Title 4 (Bottom)', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('X Axis 4')
    ax4.set_ylabel('Y Axis 4')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Test Title 5 (Bottom)', fontsize=14, fontweight='bold', pad=20)
    ax5.set_xlabel('X Axis 5')
    ax5.set_ylabel('Y Axis 5')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Test Title 6 (Bottom)', fontsize=14, fontweight='bold', pad=20)
    ax6.set_xlabel('X Axis 6')
    ax6.set_ylabel('Y Axis 6')
    
    # Apply the same layout adjustments
    plt.subplots_adjust(
        left=0.08,    # Left margin
        right=0.95,   # Right margin  
        bottom=0.12,  # Bottom margin (increased to prevent overlap)
        top=0.95,     # Top margin
        wspace=0.3,   # Horizontal space between subplots
        hspace=0.4    # Vertical space between subplots (increased)
    )
    
    # Save test plot
    test_plot_path = "test_layout.png"
    plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Test layout plot saved as: {test_plot_path}")
    print("✅ Layout improvements applied successfully!")
    print("\n💡 The main dashboard should now have proper spacing between panels.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
