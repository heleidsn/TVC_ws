#!/usr/bin/env python3
"""
Launch script for Controller Comparison GUI
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller_comparison.controller_comparison_gui import main

if __name__ == "__main__":
    main()
