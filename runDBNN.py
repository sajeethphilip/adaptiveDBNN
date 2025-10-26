#!/usr/bin/env python3
"""
DBNN Enhanced Interface - Fixed early stopping and improved feature selection
WITH PRE-LOAD VALIDATION AND DEPENDENCY CHECKING
AND COMMAND-LINE INTERFACE FOR HEADLESS SYSTEMS
"""
# GUI detection at the top
GUI_AVAILABLE = False
import os
import sys
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except:
    print("GUI is switched off")
import traceback
import numpy as np
import json
import csv
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple


try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    # Test if GUI is actually available (not headless)
    if os.name == 'posix' and 'DISPLAY' not in os.environ:
        GUI_AVAILABLE = False
    else:
        test_root = tk.Tk()
        test_root.withdraw()
        test_root.destroy()
        GUI_AVAILABLE = True
except (ImportError, Exception):
    GUI_AVAILABLE = False

def validate_environment():
    """Enhanced environment validation for advanced features"""
    print("🔍 Validating environment and advanced dependencies...")

    # Basic checks (existing)
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return False

    if not os.path.exists('dbnn.py'):
        print("❌ dbnn.py not found in current directory")
        return False

    # Enhanced dependency checking
    required_packages = [
        ('numpy', 'np', 'Numerical computations'),
        ('pandas', 'pd', 'Data processing'),
        ('numba', 'numba', 'JIT compilation'),
        ('plotly', 'plotly', 'Visualization'),
    ]

    advanced_packages = [
        ('psutil', 'psutil', 'Memory optimization'),
        ('torch', 'torch', 'GPU acceleration'),
    ]

    print("\n📦 Checking required packages:")
    missing_packages = []
    for package, alias, purpose in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package:15} - {purpose}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package:15} - {purpose}")

    print("\n⚡ Checking advanced packages:")
    for package, alias, purpose in advanced_packages:
        try:
            __import__(package)
            print(f"   ✅ {package:15} - {purpose}")
        except ImportError:
            print(f"   ⚠️  {package:15} - {purpose} (optional)")

    if missing_packages:
        print(f"\n❌ Missing required packages: {missing_packages}")
        print("   Please install with: pip install " + " ".join(missing_packages))
        return False

    # Check for advanced features
    print("\n🔧 Checking advanced feature availability:")
    try:
        import numba
        print("   ✅ Numba JIT compilation available")
    except:
        print("   ⚠️  Numba not available (performance will be limited)")

    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   ✅ Memory monitoring: {memory.total/(1024**3):.1f}GB total")
    except:
        print("   ⚠️  Memory monitoring limited")

    return True

def preload_dbnn_modules():
    """Preload and validate DBNN modules with error handling"""
    print("\n🔧 Preloading DBNN modules...")

    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        # Import core components first
        from dbnn import DBNNCore, DBNNVisualizer, ClassEncoder, HybridDBNNCore
        print("✅ Core DBNN components loaded")

        # Test basic functionality
        print("🧪 Testing basic DBNN functionality...")

        # Test ClassEncoder
        encoder = ClassEncoder()
        test_labels = ['class_a', 'class_b', 'class_c']
        encoder.fit(test_labels)
        encoded = encoder.transform(test_labels)
        print(f"✅ ClassEncoder test passed: {len(encoded)} labels encoded")

        # Test core initialization
        config = {
            'resol': 50,  # Conservative resolution for testing
            'gain': 20.0,
            'margin': 0.2,
            'patience': 10
        }

        # Try HybridDBNNCore first (more robust)
        core = HybridDBNNCore(config)
        print("✅ HybridDBNNCore initialized successfully")

        # Test memory detection
        resources = core.detect_system_resources()
        print(f"💾 System resources detected: {resources['available_memory_gb']:.1f}GB RAM available")

        # Test visualizer
        visualizer = DBNNVisualizer()
        print("✅ DBNNVisualizer initialized successfully")

        return {
            'DBNNCore': DBNNCore,
            'DBNNVisualizer': DBNNVisualizer,
            'HybridDBNNCore': HybridDBNNCore,
            'ClassEncoder': ClassEncoder,
            'test_core': core,
            'test_visualizer': visualizer
        }

    except Exception as e:
        print(f"❌ DBNN module loading failed: {e}")
        print("📋 Detailed traceback:")
        traceback.print_exc()
        return None

def setup_environment():
    """Set up required directories and environment"""
    print("\n📁 Setting up environment...")

    required_dirs = ['Model', 'Visualisations', 'dbnn_cache']

    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")

    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        print(f"💾 Disk space: {free_gb}GB available")

        if free_gb < 1:
            print("⚠️  Warning: Low disk space (< 1GB)")
    except:
        print("⚠️  Could not check disk space")

def check_system_resources():
    """Check system resources and provide recommendations"""
    print("\n💻 System Resource Check:")

    try:
        import psutil

        # Memory check
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        print(f"   RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")

        if available_gb < 2:
            print("   ⚠️  Low RAM - Sparse mode will be used automatically")
        elif available_gb > 8:
            print("   ✅ Ample RAM - Can handle large datasets")

        # CPU check
        cpu_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"   CPU: {cpu_cores} physical cores, {logical_cores} logical cores")

    except ImportError:
        print("   ⚠️  psutil not available - using conservative memory estimates")

def create_sample_config():
    """Create a sample configuration file for new users"""
    config_path = "Model/sample_config.json"
    if not os.path.exists(config_path):
        sample_config = {
            "config": {
                "resol": 100,
                "gain": 20.0,
                "margin": 0.2,
                "patience": 10,
                "epochs": 100,
                "min_improvement": 0.1,
                "fst_gain": 1.0,
                "LoC": 0.65,
                "nLoC": 0.0,
                "nresol": 0,
                "skpchk": 0,
                "oneround": 100
            },
            "description": "Sample configuration for DBNN training",
            "recommended_for": "Medium-sized datasets (1,000-100,000 samples)",
            "notes": "Adjust 'resol' based on available memory and dataset size"
        }

        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"✅ Created sample configuration: {config_path}")

def run_headless_mode(args):
    """
    Run DBNN in headless mode with advanced features support
    """
    print("🚀 Starting DBNN in Headless Mode with Advanced Features")
    print("=" * 60)

    # Import required modules with enhanced features
    from dbnn import HybridDBNNCore, ClassEncoder, FeatureEncoder, UltraSparseDBNNCore
    import pandas as pd
    import numpy as np

    # Enhanced configuration with advanced features
    config = {
        'resol': args.resolution,
        'gain': args.gain,
        'margin': args.margin,
        'patience': args.patience,
        'epochs': args.epochs,
        'min_improvement': args.min_improvement,
        # Advanced memory management
        'max_memory_usage_gb': 0.6,  # Strict 60% limit
        'auto_batch_size': True,
        'memory_safety_factor': 0.8,
    }

    # Initialize core with advanced features
    core = HybridDBNNCore(config)

    def headless_log(message):
        print(f"[DBNN] {message}")

    core.set_log_callback(headless_log)

    # Memory monitoring
    headless_log("🔍 Initializing advanced memory management...")
    resources = core.detect_system_resources()
    headless_log(f"💾 System: {resources['available_memory_gb_60percent']:.1f}GB usable (60% limit)")

    if resources.get('has_gpu', False):
        headless_log(f"🎮 GPU: {resources.get('gpu_available_memory_gb_60percent', 0):.1f}GB usable")

    try:
        if args.mode == 'train':
            return _run_training_mode(args, core, headless_log)
        elif args.mode == 'predict':
            return _run_prediction_mode(args, core, headless_log)
        elif args.mode == 'test':
            return _run_test_mode(args, core, headless_log)
        elif args.mode == 'analyze':
            return _run_analyze_mode(args, core, headless_log)

    except Exception as e:
        headless_log(f"❌ Error in headless mode: {e}")
        traceback.print_exc()
        return False

def _run_training_mode(args, core, headless_log):
    """Enhanced training mode with advanced features"""
    headless_log(f"🎯 Training mode: {args.input_file}")

    # Auto-detect file format and configure advanced options
    use_csv = args.input_file.lower().endswith('.csv')

    # Configure advanced training options
    training_config = {
        'use_csv': use_csv,
        'target_column': args.target,
        'feature_columns': args.features,
        'enable_interactive_viz': False,  # Disable in headless mode
        'viz_capture_interval': 10,
        'enable_memory_optimization': True,
        'enable_auto_batching': True
    }

    # Enhanced training with memory optimization
    headless_log("🧠 Starting enhanced training with memory optimization...")

    success = core.train_with_memory_optimization(
        train_file=args.input_file,
        test_file=args.test_file,
        **training_config
    )

    if success:
        headless_log("✅ Training completed successfully!")

        # Show training results
        if hasattr(core, 'best_accuracy'):
            headless_log(f"📊 Best accuracy: {core.best_accuracy:.2f}% at round {getattr(core, 'best_round', 'N/A')}")

        # Show memory usage summary
        if hasattr(core, 'memory_monitor'):
            current_usage = core.memory_monitor.get_memory_usage()
            headless_log(f"💾 Peak memory usage: {current_usage:.2f}GB")

        # Auto-save model with enhanced metadata
        if args.auto_save:
            model_path = core.save_model_auto(
                model_dir="Model",
                data_filename=args.input_file,
                feature_columns=args.features,
                target_column=args.target
            )
            if model_path:
                headless_log(f"💾 Model auto-saved to: {model_path}")

                # Save training report
                _save_training_report(core, args, model_path)

        # Test if test file provided
        if args.test_file and os.path.exists(args.test_file):
            _run_validation_test(args.test_file, core, headless_log)

    else:
        headless_log("❌ Training failed!")

    return success

def _run_prediction_mode(args, core, headless_log):
    """Enhanced prediction mode with advanced features"""
    headless_log(f"🔮 Prediction mode: {args.input_file}")

    # Try to load model with enhanced error handling
    model_loaded = False
    if args.model_file and os.path.exists(args.model_file):
        headless_log(f"📂 Loading specified model: {args.model_file}")
        try:
            core.load_model(args.model_file)
            model_loaded = True
            headless_log("✅ Model loaded successfully")
        except Exception as e:
            headless_log(f"❌ Failed to load specified model: {e}")

    if not model_loaded and not core.is_trained:
        # Auto-detect latest model
        model_files = [f for f in os.listdir("Model") if f.endswith(('.bin', '.gz'))]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join("Model", x)))
            headless_log(f"🔄 Auto-loading latest model: {latest_model}")
            try:
                core.load_model(os.path.join("Model", latest_model))
                model_loaded = True
            except Exception as e:
                headless_log(f"❌ Auto-load failed: {e}")

    if not model_loaded:
        headless_log("❌ No trained model available.")
        return False

    # Enhanced prediction with memory optimization
    headless_log("🚀 Starting enhanced prediction...")

    try:
        # Load data with memory optimization
        features_batches, _, feature_columns_used, _ = core.load_data_optimized(
            args.input_file,
            target_column=None,  # Prediction mode
            feature_columns=args.features
        )

        if not features_batches:
            headless_log("❌ No prediction data loaded")
            return False

        headless_log(f"📊 Processing {sum(len(batch) for batch in features_batches)} samples")

        # Batch prediction with progress tracking
        all_predictions = []
        all_probabilities = []
        total_batches = len(features_batches)

        for batch_idx, features_batch in enumerate(features_batches):
            if batch_idx % 10 == 0:
                headless_log(f"🔍 Processing batch {batch_idx+1}/{total_batches}...")

            predictions, probabilities = core.predict_batch(features_batch)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

        # Decode predictions
        decoded_predictions = core.class_encoder.inverse_transform(all_predictions)

        # Enhanced output generation
        output_file = _generate_enhanced_output(
            args.input_file, decoded_predictions, all_probabilities,
            core, args.features, headless_log
        )

        return output_file is not None

    except Exception as e:
        headless_log(f"❌ Prediction error: {e}")
        return False

def _run_test_mode(args, core, headless_log):
    """Enhanced test mode with comprehensive reporting"""
    headless_log(f"🧪 Test mode: {args.input_file}")

    # Model loading (same as prediction mode)
    if not _load_model_for_inference(core, args, headless_log):
        return False

    # Enhanced testing with detailed analysis
    headless_log("🔍 Running comprehensive model evaluation...")

    try:
        features_batches, targets_batches, _, original_targets_batches = core.load_data_optimized(
            args.input_file,
            target_column=args.target,
            feature_columns=args.features
        )

        if not features_batches:
            headless_log("❌ No test data loaded")
            return False

        # Encode targets
        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = core.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        # Enhanced evaluation with confidence scores
        accuracy, correct_predictions, predictions = core.evaluate_hybrid(
            features_batches, encoded_targets_batches
        )

        total_samples = sum(len(batch) for batch in features_batches)

        # Generate comprehensive test report
        _generate_test_report(
            core, args, accuracy, correct_predictions, total_samples,
            predictions, encoded_targets_batches, headless_log
        )

        return True

    except Exception as e:
        headless_log(f"❌ Testing error: {e}")
        return False

def _run_analyze_mode(args, core, headless_log):
    """New analysis mode for dataset inspection"""
    headless_log(f"📊 Analysis mode: {args.input_file}")

    try:
        # Analyze dataset characteristics
        dataset_info = core._analyze_dataset(args.input_file, args.target, args.features)

        headless_log("📈 Dataset Analysis Report:")
        headless_log(f"   Samples: {dataset_info['total_samples']:,}")
        headless_log(f"   Features: {dataset_info['feature_count']}")
        headless_log(f"   File size: {dataset_info['file_size_bytes']/(1024**2):.1f} MB")
        headless_log(f"   Estimated classes: {dataset_info.get('num_classes_estimate', 'N/A')}")

        # Memory requirements estimation
        if dataset_info['feature_count'] > 0:
            dense_mem, sparse_mem = core._calculate_required_memory_hybrid(
                dataset_info['feature_count'],
                args.resolution,
                dataset_info.get('num_classes_estimate', 10)
            )

            headless_log("💾 Memory Requirements:")
            headless_log(f"   Dense mode: {dense_mem:.2f} GB")
            headless_log(f"   Sparse mode: {sparse_mem:.2f} GB")
            headless_log(f"   Recommended: {'SPARSE' if dense_mem > 1.0 else 'DENSE'} mode")

        return True

    except Exception as e:
        headless_log(f"❌ Analysis error: {e}")
        return False

def _load_model_for_inference(core, args, headless_log):
    """Enhanced model loading with better error handling"""
    if core.is_trained:
        return True

    model_loaded = False

    # Try specified model first
    if args.model_file and os.path.exists(args.model_file):
        try:
            core.load_model(args.model_file)
            model_loaded = True
            headless_log("✅ Model loaded successfully")
        except Exception as e:
            headless_log(f"❌ Failed to load specified model: {e}")

    # Auto-detect latest model
    if not model_loaded:
        model_files = [f for f in os.listdir("Model") if f.endswith(('.bin', '.gz'))]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join("Model", x)))
            try:
                core.load_model(os.path.join("Model", latest_model))
                model_loaded = True
                headless_log(f"✅ Auto-loaded: {latest_model}")
            except Exception as e:
                headless_log(f"❌ Auto-load failed: {e}")

    if not model_loaded:
        headless_log("❌ No trained model found. Please train a model first.")
        return False

    return True

def _generate_enhanced_output(input_file, predictions, probabilities, core, feature_columns, headless_log):
    """Generate enhanced output with comprehensive information"""
    try:
        # Load original data
        if input_file.lower().endswith('.csv'):
            original_data = pd.read_csv(input_file)
        else:
            # For DAT files, create basic structure
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        values = line.strip().split()
                        data.append(values)

            columns = feature_columns if feature_columns else [f'feature_{i+1}' for i in range(len(data[0]))]
            original_data = pd.DataFrame(data, columns=columns)

        # Create enhanced results
        results_df = original_data.copy()
        results_df['prediction'] = predictions
        results_df['prediction_confidence'] = [max(prob.values()) for prob in probabilities]

        # Add all class probabilities
        if probabilities and len(probabilities) > 0:
            all_class_names = list(core.class_encoder.encoded_to_class.values())
            for class_name in all_class_names:
                prob_values = [prob.get(class_name, 0.0) for prob in probabilities]
                results_df[f'prob_{class_name}'] = prob_values

        # Determine output file
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_predictions.csv"

        # Save with enhanced formatting
        results_df.to_csv(output_file, index=False)

        # Generate prediction summary
        _generate_prediction_summary(results_df, predictions, core, headless_log)

        headless_log(f"✅ Enhanced predictions saved to: {output_file}")
        headless_log(f"   Total predictions: {len(predictions)}")
        headless_log(f"   Output columns: {len(results_df.columns)}")

        return output_file

    except Exception as e:
        headless_log(f"❌ Output generation error: {e}")
        return None

def _generate_prediction_summary(results_df, predictions, core, headless_log):
    """Generate comprehensive prediction summary"""
    headless_log("\n📊 PREDICTION SUMMARY:")
    headless_log("=" * 50)

    # Basic statistics
    total_predictions = len(predictions)
    headless_log(f"Total predictions: {total_predictions}")

    # Prediction distribution
    prediction_counts = results_df['prediction'].value_counts()
    headless_log("\nPrediction distribution:")
    for pred, count in prediction_counts.items():
        percentage = (count / total_predictions) * 100
        headless_log(f"  {pred}: {count} ({percentage:.1f}%)")

    # Confidence statistics
    if 'prediction_confidence' in results_df.columns:
        avg_confidence = results_df['prediction_confidence'].mean()
        min_confidence = results_df['prediction_confidence'].min()
        max_confidence = results_df['prediction_confidence'].max()

        headless_log(f"\nConfidence statistics:")
        headless_log(f"  Average: {avg_confidence:.3f}")
        headless_log(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")

    headless_log("=" * 50)


def setup_command_line_parser():
    """
    Enhanced command line parser with advanced features
    """
    parser = argparse.ArgumentParser(
        description='DBNN - Advanced Neural Network with Memory Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Features:
  - Automatic memory optimization (60% limit)
  - Sparse mode for high-dimensional data
  - GPU acceleration when available
  - Feature encoding for categorical data
  - Comprehensive model analysis

Examples:
  python %(prog)s train --input data.csv --target income --features age education
  python %(prog)s predict --input new_data.csv --features age education
  python %(prog)s test --input test_data.csv --target income
  python %(prog)s analyze --input data.csv --target income
        """
    )

    # Main mode with new 'analyze' option
    parser.add_argument('mode', choices=['train', 'predict', 'test', 'analyze'],
                       help='Operation mode: train, predict, test, or analyze')

    # Input file (required for all modes)
    parser.add_argument('--input', '-i', dest='input_file', required=True,
                       help='Input data file (CSV or DAT format)')

    # Training specific
    parser.add_argument('--test', '-t', dest='test_file',
                       help='Test data file for evaluation during training')

    # Data configuration
    parser.add_argument('--target', '-tg', dest='target',
                       help='Target column name (for CSV files, required for train/test)')

    parser.add_argument('--features', '-f', dest='features', nargs='+',
                       help='Feature column names (for CSV files)')

    # Advanced model parameters
    advanced_group = parser.add_argument_group('Advanced Parameters')

    advanced_group.add_argument('--resolution', '-r', type=int, default=100,
                               help='Resolution parameter (default: 100)')

    advanced_group.add_argument('--gain', '-g', type=float, default=20.0,
                               help='Gain parameter (default: 20.0)')

    advanced_group.add_argument('--margin', '-m', type=float, default=0.2,
                               help='Margin parameter (default: 0.2)')

    advanced_group.add_argument('--patience', '-p', type=int, default=10,
                               help='Early stopping patience (default: 10)')

    advanced_group.add_argument('--epochs', '-e', type=int, default=100,
                               help='Maximum training epochs (default: 100)')

    advanced_group.add_argument('--min-improvement', '-mi', type=float, default=0.1,
                               help='Minimum improvement for early stopping (default: 0.1)')

    # Additional options
    advanced_group.add_argument('--auto-save', '-as', action='store_true',
                               help='Automatically save model after training')

    advanced_group.add_argument('--model', '-M', dest='model_file',
                               help='Specific model file to load for prediction/test')

    advanced_group.add_argument('--memory-limit', type=float, default=0.6,
                               help='Memory usage limit (0.0-1.0, default: 0.6)')

    advanced_group.add_argument('--enable-sparse', action='store_true',
                               help='Force sparse mode (for memory efficiency)')

    return parser

def main():
    """Main function with comprehensive pre-load validation and CLI support"""
    print("=" * 60)
    print("🚀 DBNN Enhanced Interface - Starting Pre-Load Validation")
    print("=" * 60)

    # Check if running in headless mode (command line arguments provided)
    if len(sys.argv) > 1:
        # Command line mode
        parser = setup_command_line_parser()
        args = parser.parse_args()

        # Validate environment for headless mode
        if not validate_environment():
            print("❌ Environment validation failed for headless mode!")
            sys.exit(1)

        # Setup environment
        setup_environment()

        # Run headless mode
        success = run_headless_mode(args)
        sys.exit(0 if success else 1)

    # GUI Mode - only run if GUI is available
    if not GUI_AVAILABLE:
        print("❌ GUI not available and no command line arguments provided.")
        print("💡 Usage examples:")
        print("   Training: python dbnn_enhanced_interface.py train --input data.csv --target class_column")
        print("   Prediction: python dbnn_enhanced_interface.py predict --input new_data.csv")
        print("   Testing: python dbnn_enhanced_interface.py test --input test_data.csv --target class_column")
        print("\nUse --help for full options")
        sys.exit(1)

    # Step 1: Validate environment for GUI mode
    if not validate_environment():
        print("\n❌ Environment validation failed!")
        print("Please fix the issues above and try again.")
        input("Press Enter to exit...")
        return

    # Step 2: Setup environment
    setup_environment()

    # Step 3: Check system resources
    check_system_resources()

    # Step 4: Preload DBNN modules
    modules = preload_dbnn_modules()
    if modules is None:
        print("\n❌ DBNN module loading failed!")
        print("Please check the errors above and ensure dbnn.py is compatible.")
        input("Press Enter to exit...")
        return

    # Step 5: Create sample files for new users
    create_sample_config()

    print("\n" + "=" * 60)
    print("✅ All pre-load checks passed! Starting GUI...")
    print("=" * 60)

    # Small delay to let user read the messages
    time.sleep(2)

    try:
        # Now import the GUI component
        from dbnn import EnhancedDBNNInterface

        # Start the GUI
        root = tk.Tk()
        app = EnhancedDBNNInterface(root)

        # Set window title with version info
        root.title("DBNN Enhanced Interface - Ready")

        print("🎯 GUI loaded successfully!")
        print("💡 Tips:")
        print("   - Use 'Analyze' button to examine your data file first")
        print("   - Start with default configuration for your first run")
        print("   - Enable 'Enhanced Visualization' for educational mode")
        print("   - Check the 'Configuration' tab for advanced settings")

        root.mainloop()

    except Exception as e:
        print(f"❌ GUI startup failed: {e}")
        print("📋 Detailed traceback:")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
