#!/usr/bin/env python3
"""
Test script to verify Phase 4 visualization works with actual data
"""

import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visualize_compression_results import CompressionAnalyzer
    
    print("🧪 Testing Phase 4 Visualization Module...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CompressionAnalyzer()
    
    # Check data loading
    print(f"📊 Loaded {len(analyzer.metrics)} metric files:")
    for key in analyzer.metrics.keys():
        print(f"   • {key}")
    
    # Check models data
    print(f"\n🔍 Prepared data for {len(analyzer.models_data)} models:")
    for _, model in analyzer.models_data.iterrows():
        print(f"   • {model['model_name']} ({model['phase']})")
        print(f"     - Size: {model['model_size_mb']:.2f} MB")
        print(f"     - Accuracy: {model['accuracy']:.4f}")
        print(f"     - Parameters: {model['param_count']:,}")
    
    # Test analysis functions
    print(f"\n📈 Testing analysis functions...")
    
    # Test summary generation
    summary = analyzer._generate_summary()
    print("✅ Summary generation: SUCCESS")
    
    # Test efficiency analysis
    efficiency = analyzer._analyze_compression_efficiency()
    print(f"✅ Efficiency analysis: SUCCESS ({len(efficiency)} models analyzed)")
    
    # Test recommendations
    recommendations = analyzer._generate_recommendations()
    print(f"✅ Recommendations: SUCCESS ({len(recommendations)} recommendations)")
    
    print("\n🎉 All tests passed! Phase 4 visualization module is working correctly.")
    print("\n💡 To run the full visualization:")
    print("   python3 visualize_compression_results.py")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
