#!/usr/bin/env python3
"""
DBNN Enhanced Interface - Fixed early stopping and improved feature selection
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import traceback
import numpy as np
import json
import csv
import time
from typing import List, Dict, Any, Optional, Tuple

# Import the existing modules
try:
    from dbnn import DBNNCore, DBNNVisualizer, EnhancedDBNNInterface
    print("Successfully imported DBNN modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure dbnn.py is in the same directory")
    sys.exit(1)


def main():
    """Main function"""
    root = tk.Tk()
    app = EnhancedDBNNInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()
