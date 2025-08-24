#!/usr/bin/env python3
"""
Simple runner script for Phase 4 visualization
"""

import sys
import os

# Add the parent directory to path to import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visualize_compression_results import main
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error running visualization: {e}")
    import traceback
    traceback.print_exc()
