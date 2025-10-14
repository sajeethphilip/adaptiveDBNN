#!/usr/bin/env python3
"""
Enhanced Command Line Interface for DBNN
"""

import sys
import os

# Add current directory to path to import dbnn module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dbnn import DBNNCommandLine
except ImportError as e:
    print(f"Error importing DBNN module: {e}")
    print("Make sure dbnn.py is in the same directory")
    sys.exit(1)

def main():
    """Main function"""
    try:
        cli = DBNNCommandLine()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    '''
# All previous commands still work exactly the same
python runDBNN_cmd.py --help
python runDBNN_cmd.py --train data.csv --target class --features f1 f2 f3
python runDBNN_cmd.py --model model.bin --predict new_data.csv
python runDBNN_cmd.py --interactive
    '''
    main()
