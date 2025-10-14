import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy
import glob
import time
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import imageio
from scipy.spatial import ConvexHull
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import dbnn

class DatasetConfig:
    """Dataset configuration handler"""

    @staticmethod
    def get_available_datasets():
        """Get list of available datasets from configuration files"""
        config_files = glob.glob("*.conf") + glob.glob("*.json")
        datasets = []
        for f in config_files:
            # Remove both .conf and .json extensions
            base_name = f.replace('.conf', '').replace('.json', '')
            if base_name not in datasets:  # Avoid duplicates
                datasets.append(base_name)
        return datasets

    @staticmethod
    def load_config(dataset_name):
        """Load configuration for a dataset - supports both .conf and .json"""
        # Try .json first, then .conf
        config_paths = [
            f"{dataset_name}.json",
            f"{dataset_name}.conf"
        ]

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
                    continue
        return {}

    @staticmethod
    def get_available_config_files():
        """Get all available configuration files with their types"""
        config_files = []
        # Look for JSON config files
        json_files = glob.glob("*.json")
        for f in json_files:
            # Skip the auto-saved config to avoid confusion
            if not f.endswith('_run_config.json') and not f.endswith('adaptive_dbnn_config.json'):
                config_files.append({'file': f, 'type': 'JSON'})

        # Look for CONF config files
        conf_files = glob.glob("*.conf")
        for f in conf_files:
            config_files.append({'file': f, 'type': 'CONF'})

        return config_files

class DataPreprocessor:
    """Comprehensive data preprocessing for DBNN"""

    def __init__(self, target_column: str = 'target', sentinel_value: float = -99999.0):
        self.target_column = target_column
        self.sentinel_value = sentinel_value
        self.feature_encoders = {}  # For encoding categorical features
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.missing_value_indicators = {}

    def preprocess_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess feature columns - handle mixed types, missing values, etc."""
        processed_features = []
        feature_names = []

        for col in X.columns:
            if col == self.target_column:
                continue

            feature_data = X[col].copy()

            # Handle missing values
            missing_mask = self._detect_missing_values(feature_data)

            # Convert to numeric, handling errors
            numeric_data = self._convert_to_numeric(feature_data, col)

            # Store missing value information
            self.missing_value_indicators[col] = {
                'missing_mask': missing_mask,
                'has_missing': np.any(missing_mask)
            }

            processed_features.append(numeric_data)
            feature_names.append(col)

        # Stack all features
        if processed_features:
            X_processed = np.column_stack(processed_features)
        else:
            X_processed = np.empty((len(X), 0))

        return X_processed, feature_names

    def _detect_missing_values(self, data: pd.Series) -> np.ndarray:
        """Detect various types of missing values"""
        # Standard missing values
        missing_mask = data.isna()

        # String representations of missing values
        if data.dtype == 'object':
            missing_strings = ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'NaN', 'nan', 'ERROR', 'error', 'MISSING', 'missing']
            missing_mask = missing_mask | data.isin(missing_strings)

        return missing_mask.values

    def _convert_to_numeric(self, data: pd.Series, col_name: str) -> np.ndarray:
        """Convert data to numeric, handling various data types"""
        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(data):
            numeric_data = data.values.astype(float)
            # Replace any remaining NaN with sentinel value
            numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
            return numeric_data

        # For categorical/string data
        if data.dtype == 'object':
            try:
                # Try direct conversion to numeric first
                numeric_data = pd.to_numeric(data, errors='coerce').values
                # Replace NaN with sentinel value
                numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
                return numeric_data
            except:
                # Use label encoding for categorical data
                if col_name not in self.feature_encoders:
                    self.feature_encoders[col_name] = LabelEncoder()

                # Handle missing values before encoding
                clean_data = data.fillna('MISSING')
                encoded_data = self.feature_encoders[col_name].fit_transform(clean_data)
                return encoded_data.astype(float)

        # Fallback: convert to string then label encode
        str_data = data.astype(str)
        if col_name not in self.feature_encoders:
            self.feature_encoders[col_name] = LabelEncoder()
        encoded_data = self.feature_encoders[col_name].fit_transform(str_data)
        return encoded_data.astype(float)

    def preprocess_target(self, y: pd.Series) -> np.ndarray:
        """Preprocess target column"""
        # Handle missing target values
        if y.isna().any():
            print(f"⚠️  Warning: Found {y.isna().sum()} missing target values. They will be removed.")
            # We'll handle this at the dataset level by removing these samples

        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y_processed = pd.to_numeric(y, errors='coerce')
                if y_processed.isna().any():
                    print(f"⚠️  Some target values couldn't be converted to numeric. Using label encoding.")
                    y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
                else:
                    y_processed = y_processed.values
            except:
                y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
        else:
            y_processed = y.values

        return y_processed.astype(int)

    def preprocess_dataset(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess entire dataset"""
        print("🔧 Preprocessing dataset...")

        # Separate features and target
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Create a copy to avoid modifying original data
        data_clean = data.copy()

        # Preprocess features
        X_processed, feature_names = self.preprocess_features(data_clean)

        # Preprocess target
        y_processed = self.preprocess_target(data_clean[self.target_column])

        # Remove samples with missing target values
        valid_mask = ~np.isnan(y_processed)
        if not np.all(valid_mask):
            removed_count = len(y_processed) - np.sum(valid_mask)
            print(f"⚠️  Removed {removed_count} samples with invalid target values")
            X_processed = X_processed[valid_mask]
            y_processed = y_processed[valid_mask]

        print(f"✅ Preprocessing complete: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        print(f"📊 Feature types: {len(feature_names)} numeric/categorical features")

        return X_processed, y_processed, feature_names

class DBNNVisualizer:
    """Visualization system for DBNN"""

    def __init__(self, model, output_dir='visualizations', enabled=True):
        self.model = model
        self.output_dir = output_dir
        self.enabled = enabled
        os.makedirs(output_dir, exist_ok=True)

    def create_visualizations(self, X, y, predictions=None):
        """Create various visualizations"""
        if not self.enabled:
            return

        print("Creating visualizations...")

        # Create some basic plots
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=20, alpha=0.7, color='blue')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.savefig(f'{self.output_dir}/class_distribution.png')
        plt.close()

class DBNNWrapper:
    """
    Wrapper for dbnn.py module that implements the exact adaptive learning requirements
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}

        # Initialize the core DBNN
        dbnn_config = {
            'resol': self.config.get('resol', 100),
            'gain': self.config.get('gain', 2.0),
            'margin': self.config.get('margin', 0.2),
            'patience': self.config.get('patience', 10),
            'epochs': self.config.get('max_epochs', 100),
            'min_improvement': self.config.get('min_improvement', 0.1)
        }
        self.core = dbnn.DBNNCore(dbnn_config)

        # Store architectural components separately for freezing
        self.architecture_frozen = False
        self.frozen_components = {}

        # Store data and preprocessing
        self.data = None
        self.target_column = self.config.get('target_column', 'target')
        self.preprocessor = DataPreprocessor(target_column=self.target_column)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Training state
        self.train_enabled = True
        self.max_epochs = self.config.get('max_epochs', 100)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)

        # Feature information
        self.feature_names = []
        self.initialized_with_full_data = False

    def load_data(self, file_path: str = None):
        """Load data from file with robust preprocessing"""
        if file_path is None:
            # Try to find dataset file - prioritize original data files
            possible_files = [
                f"{self.dataset_name}.csv",
                f"{self.dataset_name}.data",
                f"wine.data",
                f"wine.csv",
                "data.csv",
                "train.csv"
            ]
            for file in possible_files:
                if os.path.exists(file):
                    file_path = file
                    print(f"📁 Found data file: {file_path}")
                    break

        if file_path is None:
            # Try to find any CSV or DAT file in current directory
            csv_files = glob.glob("*.csv")
            dat_files = glob.glob("*.dat")
            all_files = csv_files + dat_files

            if all_files:
                file_path = all_files[0]
                print(f"📁 Auto-selected data file: {file_path}")
            else:
                raise ValueError("No data file found. Please provide a CSV or DAT file.")

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
            print(f"✅ Loaded CSV data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        else:
            # For .dat files, use simple loading
            print(f"📊 Loading DAT file: {file_path}")
            try:
                data = np.loadtxt(file_path)
                n_features = data.shape[1] - 1
                columns = [f'feature_{i}' for i in range(n_features)] + [self.target_column]
                self.data = pd.DataFrame(data, columns=columns)
                print(f"✅ Loaded DAT data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error loading DAT file: {e}")
                raise

        return self.data

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return self.preprocessor.preprocess_dataset(self.data)

    def initialize_with_full_data(self, X: np.ndarray, y: np.ndarray):
        """Step 1: Initialize DBNN architecture with full dataset"""
        print("🏗️ Initializing DBNN architecture with full dataset...")

        # Create temporary file with full data
        temp_file = f"temp_full_init_{int(time.time())}.csv"
        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        full_df = pd.DataFrame(X, columns=feature_cols)
        full_df[self.target_column] = y
        full_df.to_csv(temp_file, index=False)

        try:
            # First, manually initialize the DBNN core architecture
            self._initialize_dbnn_architecture(X, y, feature_cols)

            # Then train with full data to initialize architecture
            success = self._train_with_initialized_architecture(temp_file, feature_cols)

            if success:
                print("✅ DBNN architecture initialized with full dataset")
                self.initialized_with_full_data = True

                # Freeze the architecture
                self.freeze_architecture()
            else:
                print("❌ Failed to initialize DBNN architecture")

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _initialize_dbnn_architecture(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]):
        """Manually initialize DBNN architecture to avoid the dmyclass error"""
        print("🔧 Manually initializing DBNN architecture...")

        # Create temporary file for initialization
        temp_file = f"temp_init_{int(time.time())}.csv"
        init_df = pd.DataFrame(X, columns=feature_cols)
        init_df[self.target_column] = y
        init_df.to_csv(temp_file, index=False)

        try:
            # Load data to get feature information
            features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                temp_file, self.target_column, feature_cols
            )

            if not features_batches:
                raise ValueError("No data loaded for initialization")

            # Fit encoder first
            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            # Get encoded classes
            encoded_classes = self.core.class_encoder.get_encoded_classes()
            self.core.outnodes = len(encoded_classes)
            self.core.innodes = len(feature_cols)

            # Initialize arrays with proper dimensions
            resol = self.core.config.get('resol', 100)
            self.core.initialize_arrays(self.core.innodes, resol, self.core.outnodes)

            # Now set dmyclass values safely
            self.core.dmyclass[0] = self.core.config.get('margin', 0.2)
            for i, encoded_val in enumerate(encoded_classes, 1):
                if i < len(self.core.dmyclass):
                    self.core.dmyclass[i] = float(encoded_val)

            print(f"✅ Manual initialization complete: {self.core.innodes} inputs, {self.core.outnodes} outputs")

        except Exception as e:
            print(f"❌ Manual initialization failed: {e}")
            raise
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _train_with_initialized_architecture(self, train_file: str, feature_cols: List[str]):
        """Train with already initialized architecture (no re-initialization)"""
        try:
            # Load data without re-initializing
            features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                train_file, self.target_column, feature_cols
            )

            if not features_batches:
                return False

            # Encode targets using existing encoder
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training parameters
            resol = self.core.config.get('resol', 100)
            omax, omin = self._initialize_training_params(features_batches, encoded_targets_batches, resol)

            # Process training data for initial APF
            total_samples = sum(len(batch) for batch in features_batches)
            total_processed = 0

            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                processed = self._process_training_batch(features_batch, targets_batch)
                total_processed += processed
                if total_processed % 1000 == 0:
                    print(f"Processed {total_processed}/{total_samples} samples")

            # Training with early stopping
            gain = self.core.config.get('gain', 2.0)
            max_epochs = self.core.config.get('epochs', 100)
            patience = self.core.config.get('patience', 10)
            min_improvement = self.core.config.get('min_improvement', 0.1)

            print(f"Starting weight training with early stopping...")
            best_accuracy = 0.0
            best_round = 0
            patience_counter = 0

            for rnd in range(max_epochs + 1):
                if rnd == 0:
                    # Initial evaluation
                    current_accuracy, correct_predictions, _ = self._evaluate_model(features_batches, encoded_targets_batches)
                    print(f"Round {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")
                    best_accuracy = current_accuracy
                    continue

                # Training pass
                self._train_epoch(features_batches, encoded_targets_batches, gain)

                # Evaluation after training round
                current_accuracy, correct_predictions, _ = self._evaluate_model(features_batches, encoded_targets_batches)
                print(f"Round {rnd:3d}: Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")

                # Early stopping logic
                if current_accuracy > best_accuracy + min_improvement:
                    best_accuracy = current_accuracy
                    best_round = rnd
                    patience_counter = 0
                    print(f"  → New best accuracy! (Improved by {current_accuracy - best_accuracy:.2f}%)")
                else:
                    patience_counter += 1
                    print(f"  → No improvement (Patience: {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {rnd} rounds.")
                    print(f"Best accuracy {best_accuracy:.2f}% achieved at round {best_round}")
                    break

            self.core.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_training_params(self, features_batches, encoded_targets_batches, resol: int):
        """Initialize training parameters without re-initializing arrays"""
        # Find min/max values
        omax = -400.0
        omin = 400.0

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            for i in range(1, self.core.innodes + 1):
                batch_max = np.max(features_batch[:, i-1])
                batch_min = np.min(features_batch[:, i-1])
                if batch_max > self.core.max_val[i]:
                    self.core.max_val[i] = batch_max
                if batch_min < self.core.min_val[i]:
                    self.core.min_val[i] = batch_min

            # Update omax/omin from targets
            batch_omax = np.max(targets_batch)
            batch_omin = np.min(targets_batch)
            if batch_omax > omax:
                omax = batch_omax
            if batch_omin < omin:
                omin = batch_omin

        # Set resolutions
        for i in range(1, self.core.innodes + 1):
            self.core.resolution_arr[i] = resol
            for j in range(self.core.resolution_arr[i] + 1):
                self.core.binloc[i][j+1] = j * 1.0

        # Initialize network counts
        self.core.anti_wts.fill(1.0)
        for k in range(1, self.core.outnodes + 1):
            for i in range(1, self.core.innodes + 1):
                for j in range(self.core.resolution_arr[i] + 1):
                    for l in range(1, self.core.innodes + 1):
                        for m in range(self.core.resolution_arr[l] + 1):
                            self.core.anti_net[i, j, l, m, k] = 1

        return omax, omin

    def _process_training_batch(self, features_batch, targets_batch):
        """Process a single batch of training data"""
        batch_size = len(features_batch)
        processed_count = 0

        for sample_idx in range(batch_size):
            vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
            for i in range(1, self.core.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            # Use the core's processing function
            self.core.anti_net = dbnn.process_training_sample(
                vects, tmpv, self.core.anti_net, self.core.anti_wts, self.core.binloc,
                self.core.resolution_arr, self.core.dmyclass, self.core.min_val, self.core.max_val,
                self.core.innodes, self.core.outnodes
            )

            processed_count += 1

        return processed_count

    def _train_epoch(self, features_batches, encoded_targets_batches, gain: float):
        """Train for one epoch"""
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)

            for sample_idx in range(batch_size):
                vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
                for i in range(1, self.core.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                tmpv = targets_batch[sample_idx]

                # Compute probabilities
                classval = dbnn.compute_class_probabilities_numba(
                    vects, self.core.anti_net, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                    self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.core.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                # Update weights if wrong classification
                if abs(self.core.dmyclass[kmax] - tmpv) > self.core.dmyclass[0]:
                    self.core.anti_wts = dbnn.update_weights_numba(
                        vects, tmpv, classval, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                        self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes, gain
                    )

    def _evaluate_model(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy"""
        correct_predictions = 0
        total_samples = 0
        all_predictions = []

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)
            total_samples += batch_size

            for sample_idx in range(batch_size):
                vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
                for i in range(1, self.core.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                actual = targets_batch[sample_idx]

                # Compute class probabilities
                classval = dbnn.compute_class_probabilities_numba(
                    vects, self.core.anti_net, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                    self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.core.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                predicted = self.core.dmyclass[kmax]
                all_predictions.append(predicted)

                # Check if prediction is correct
                if abs(actual - predicted) <= self.core.dmyclass[0]:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, correct_predictions, all_predictions

    def train_with_data(self, X_train: np.ndarray, y_train: np.ndarray, reset_weights: bool = True):
        """Step 2: Train with given data (no train/test split)"""
        if not self.initialized_with_full_data:
            # Try to initialize if not already done
            print("⚠️  DBNN not initialized, attempting initialization...")
            self.initialize_with_full_data(X_train, y_train)
            if not self.initialized_with_full_data:
                raise ValueError("DBNN must be initialized with full data first")

        if reset_weights:
            print("🔄 Resetting weights for new training...")
            self._reset_weights()

        print(f"🎯 Training with {len(X_train)} samples...")

        # Create temporary file with training data
        temp_file = f"temp_train_{int(time.time())}.csv"
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[self.target_column] = y_train
        train_df.to_csv(temp_file, index=False)

        try:
            # Train using our custom training method that preserves architecture
            success = self._train_with_initialized_architecture(temp_file, feature_cols)

            if success:
                train_accuracy = self._compute_accuracy(X_train, y_train)
                print(f"✅ Training completed - Accuracy on training data: {train_accuracy:.4f}")
                return True
            else:
                print("❌ Training failed")
                return False

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _reset_weights(self):
        """Reset weights while preserving architecture"""
        # Instead of creating a new core, just reset the weights arrays
        if hasattr(self.core, 'anti_wts'):
            self.core.anti_wts.fill(1.0)  # Reset to uniform weights
            print("✅ Weights reset to uniform distribution")
        else:
            print("⚠️  Cannot reset weights - architecture not initialized")

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy on given data"""
        try:
            predictions = self.predict(X)
            # Ensure both arrays have the same data type for comparison
            predictions = predictions.astype(y.dtype)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        except Exception as e:
            print(f"❌ Accuracy computation error: {e}")
            return 0.0

    def predict(self, X: np.ndarray):
        """Predict classes for input data"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            # If not trained, return random predictions based on class distribution
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

        try:
            # Create temporary file for prediction
            temp_file = f"temp_predict_{int(time.time())}.csv"
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            predict_df = pd.DataFrame(X, columns=feature_cols)
            predict_df.to_csv(temp_file, index=False)

            # Load data for prediction
            features_batches, _, _, _ = self.core.load_data(
                temp_file,
                target_column=None,
                feature_columns=feature_cols
            )

            all_predictions = []

            for features_batch in features_batches:
                predictions, _ = self.core.predict_batch(features_batch)
                # Convert predictions to proper numeric type
                numeric_predictions = []
                for pred in predictions:
                    try:
                        numeric_predictions.append(float(pred))
                    except (ValueError, TypeError):
                        # If conversion fails, use the first class as fallback
                        numeric_predictions.append(1.0)
                all_predictions.extend(numeric_predictions)

            # Convert encoded predictions back to original labels
            decoded_predictions = []
            for pred in all_predictions:
                try:
                    # Try to decode using class encoder
                    if hasattr(self.core, 'class_encoder') and self.core.class_encoder.is_fitted:
                        decoded = self.core.class_encoder.inverse_transform([pred])[0]
                        decoded_predictions.append(decoded)
                    else:
                        # Fallback: use direct conversion
                        decoded_predictions.append(int(pred))
                except:
                    # Final fallback
                    decoded_predictions.append(1)

            # Convert to numpy array and ensure correct data type
            decoded_predictions = np.array(decoded_predictions, dtype=np.int64)

            # Ensure we have valid predictions
            if len(decoded_predictions) == 0:
                unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
                decoded_predictions = np.array([unique_classes[0]] * len(X))

            return decoded_predictions

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return random predictions
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _compute_batch_posterior(self, X: np.ndarray):
        """Compute posterior probabilities for a batch of samples"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            # Return uniform probabilities if not trained
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 3
            return np.ones((len(X), n_classes)) / n_classes

        try:
            temp_file = f"temp_posterior_{int(time.time())}.csv"
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            predict_df = pd.DataFrame(X, columns=feature_cols)
            predict_df.to_csv(temp_file, index=False)

            features_batches, _, _, _ = self.core.load_data(
                temp_file,
                target_column=None,
                feature_columns=feature_cols
            )

            all_probabilities = []

            for features_batch in features_batches:
                _, probabilities = self.core.predict_batch(features_batch)
                all_probabilities.extend(probabilities)

            # Convert probability dictionaries to numpy array
            n_classes = len(self.core.class_encoder.get_encoded_classes()) if hasattr(self.core.class_encoder, 'is_fitted') else 3

            # Ensure we have valid probabilities
            if not all_probabilities:
                return np.ones((len(X), n_classes)) / n_classes

            posteriors = np.zeros((len(all_probabilities), n_classes))

            for i, prob_dict in enumerate(all_probabilities):
                # If we don't have a proper probability dictionary, use uniform distribution
                if not prob_dict or not isinstance(prob_dict, dict):
                    posteriors[i] = np.ones(n_classes) / n_classes
                    continue

                # Extract probabilities in the correct order
                for j, class_val in enumerate(self.core.class_encoder.get_encoded_classes()):
                    class_name = self.core.class_encoder.encoded_to_class.get(class_val, f"Class_{j+1}")
                    posteriors[i, j] = prob_dict.get(class_name, 1.0/n_classes)

            return posteriors

        except Exception as e:
            print(f"❌ Posterior computation error: {e}")
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 3
            return np.ones((len(X), n_classes)) / n_classes
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def freeze_architecture(self):
        """Freeze architectural components"""
        self.architecture_frozen = True
        self.frozen_components = {
            'config': self.core.config.copy(),
            'feature_names': self.feature_names.copy() if hasattr(self, 'feature_names') else [],
            'target_column': self.target_column,
            'innodes': getattr(self.core, 'innodes', 0),
            'outnodes': getattr(self.core, 'outnodes', 0),
            'dmyclass': self.core.dmyclass.copy() if hasattr(self.core, 'dmyclass') else None,
        }
        print("✅ DBNN architecture frozen")

    def _save_best_weights(self):
        """Save current weights as best weights - not applicable for core DBNN"""
        pass

    def reset_weights(self):
        """Reset weights - for core DBNN, we need to retrain"""
        print("🔄 Weights reset requires retraining with core DBNN")

class AdaptiveDBNN:
    """Wrapper for DBNN that implements sophisticated adaptive learning with comprehensive analysis"""

    def __init__(self, dataset_name: str = None, config: Dict = None):
        # Handle dataset selection if not provided
        if dataset_name is None:
            dataset_name = self._select_dataset()

        self.dataset_name = dataset_name
        self.config = config or self._load_config(dataset_name)

        # Ensure config has required fields by using DatasetConfig
        if 'target_column' not in self.config:
            dataset_config = DatasetConfig.load_config(dataset_name)
            if dataset_config:
                self.config.update(dataset_config)

        # Enhanced adaptive learning configuration with proper defaults
        self.adaptive_config = self.config.get('adaptive_learning', {})
        # Set defaults for any missing parameters
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_margin_samples_per_class": 3,
            "margin_tolerance": 0.15,
            "kl_threshold": 0.1,
            "max_adaptive_rounds": 20,
            "patience": 10,
            "min_improvement": 0.001,
            "training_convergence_epochs": 50,
            "min_training_accuracy": 0.95,
            "min_samples_to_add_per_class": 5,
            "adaptive_margin_relaxation": 0.1,
            "max_divergence_samples_per_class": 5,
            "exhaust_all_failed": True,
            "min_failed_threshold": 10,
            "enable_kl_divergence": True,
            "max_samples_per_class_fallback": 2,
            "enable_3d_visualization": True,
            "3d_snapshot_interval": 10,
            "learning_rate": 1.0,
            "enable_acid_test": True,
            "min_training_percentage_for_stopping": 10.0,
            "max_training_percentage": 90.0,
            "margin_tolerance": 0.15,
            "kl_divergence_threshold": 0.1,
            "max_kl_samples_per_class": 5,
            "disable_sample_limit": False,
        }
        for key, default_value in default_config.items():
            if key not in self.adaptive_config:
                self.adaptive_config[key] = default_value

        self.stats_config = self.config.get('statistics', {
            'enable_confusion_matrix': True,
            'enable_progress_plots': True,
            'color_progress': 'green',
            'color_regression': 'red',
            'save_plots': True,
            'create_interactive_plots': True,
            'create_sample_analysis': True
        })

        # Visualization configuration
        self.viz_config = self.config.get('visualization_config', {
            'enabled': True,
            'output_dir': 'adaptive_visualizations',
            'create_animations': False,
            'create_reports': True,
            'create_3d_visualizations': True
        })

        # Initialize the base DBNN model using our wrapper
        self.model = DBNNWrapper(dataset_name, config=self.config)

        # Adaptive learning state
        self.training_indices = []
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_round = 0
        self.adaptive_round = 0
        self.patience_counter = 0

        # Statistics tracking
        self.round_stats = []
        self.previous_confusion = None
        self.start_time = datetime.now()
        self.adaptive_start_time = None
        self.device_type = self._get_device_type()

        # Store the full dataset for adaptive learning
        self.X_full = None
        self.y_full = None
        self.y_full_original = None
        self.original_data_shape = None

        # Track all selected samples for analysis
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

        # Initialize label encoder for adaptive learning
        self.label_encoder = LabelEncoder()

        # Initialize visualizers
        self.adaptive_visualizer = None
        self._initialize_visualizers()

        # Update config file with default settings if they don't exist
        self._update_config_file()

        # Show current settings
        self.show_adaptive_settings()

        # Add 3D visualization initialization
        self._initialize_3d_visualization()

    def _load_config(self, dataset_name: str) -> Dict:
        """Load configuration from file"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _select_dataset(self) -> str:
        """Select dataset from available configuration files or data files"""
        available_configs = DatasetConfig.get_available_config_files()

        # Also look for data files
        csv_files = glob.glob("*.csv")
        dat_files = glob.glob("*.dat")
        data_files = csv_files + dat_files

        if available_configs or data_files:
            print("📁 Available datasets and configuration files:")

            # Show configuration-based datasets
            if available_configs:
                print("\n🎯 Configuration files:")
                for i, config in enumerate(available_configs, 1):
                    base_name = config['file'].replace('.json', '').replace('.conf', '')
                    print(f"  {i}. {base_name} ({config['type']} configuration)")

            # Show data files
            if data_files:
                print("\n📊 Data files:")
                start_idx = len(available_configs) + 1
                for i, data_file in enumerate(data_files, start_idx):
                    print(f"  {i}. {data_file}")

            try:
                choice = input(f"\nSelect a dataset (1-{len(available_configs) + len(data_files)}): ").strip()
                choice_idx = int(choice) - 1

                if 0 <= choice_idx < len(available_configs):
                    selected_config = available_configs[choice_idx]
                    selected_dataset = selected_config['file'].replace('.json', '').replace('.conf', '')
                    print(f"🎯 Selected configuration: {selected_dataset} ({selected_config['type']})")
                    return selected_dataset
                elif len(available_configs) <= choice_idx < len(available_configs) + len(data_files):
                    data_file_idx = choice_idx - len(available_configs)
                    selected_file = data_files[data_file_idx]
                    dataset_name = selected_file.replace('.csv', '').replace('.dat', '')
                    print(f"📁 Selected data file: {selected_file}")
                    return dataset_name
                else:
                    print("❌ Invalid selection")
                    return input("Enter dataset name: ").strip()
            except ValueError:
                print("❌ Invalid input")
                return input("Enter dataset name: ").strip()
        else:
            print("❌ No configuration files or data files found.")
            print("   Looking for: *.json, *.conf, *.csv, *.dat")
            dataset_name = input("Enter dataset name: ").strip()
            if not dataset_name:
                dataset_name = "default_dataset"
            return dataset_name

    def _get_device_type(self) -> str:
        """Get the device type (CPU/GPU)"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown Device"

    def _initialize_visualizers(self):
        """Initialize visualization systems"""
        # Initialize adaptive visualizer
        if self.viz_config.get('enabled', True):
            try:
                self.adaptive_visualizer = DBNNVisualizer(
                    self.model,
                    output_dir=self.viz_config.get('output_dir', 'adaptive_visualizations'),
                    enabled=True
                )
                print("✓ Adaptive visualizer initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize adaptive visualizer: {e}")
                self.adaptive_visualizer = None

        # Create output directory
        os.makedirs(self.viz_config.get('output_dir', 'adaptive_visualizations'), exist_ok=True)

    def _update_config_file(self):
        """Update the dataset configuration file with adaptive learning settings"""
        config_path = f"{self.dataset_name}.conf"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            if 'adaptive_learning' not in config:
                config['adaptive_learning'] = {}

            config['adaptive_learning'].update(self.adaptive_config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"✅ Updated configuration file: {config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not update config file: {str(e)}")

    def show_adaptive_settings(self):
        """Display the current adaptive learning settings"""
        print("\n🔧 Advanced Adaptive Learning Settings:")
        print("=" * 60)
        for key, value in self.adaptive_config.items():
            if key in ['margin_tolerance', 'kl_divergence_threshold', 'max_kl_samples_per_class']:
                print(f"  {key:40}: {value} (KL Divergence)")
            elif key == 'disable_sample_limit':
                status = "DISABLED 🚫" if value else "ENABLED ✅"
                print(f"  {key:40}: {value} ({status})")
            else:
                print(f"  {key:40}: {value}")
        print(f"\n💻 Device: {self.device_type}")
        mode = "KL Divergence" if self.adaptive_config.get('enable_kl_divergence', False) else "Margin-Based"
        limit_status = "UNLIMITED" if self.adaptive_config.get('disable_sample_limit', False) else "LIMITED"
        print(f"🎯 Selection Mode: {mode} ({limit_status})")
        print()

    def _initialize_3d_visualization(self):
        """Initialize 3D visualization system"""
        self.visualization_output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        os.makedirs(f'{self.visualization_output_dir}/3d_animations', exist_ok=True)
        self.feature_grid_history = []
        self.epoch_timestamps = []

        print("🎨 3D Visualization system initialized")

    def _debug_predictions(self, y_remaining: np.ndarray, predictions: np.ndarray, posteriors: np.ndarray):
        """Debug method to understand prediction issues"""
        print(f"🔍 Debug - y_remaining unique: {np.unique(y_remaining)}")
        print(f"🔍 Debug - predictions unique: {np.unique(predictions) if len(predictions) > 0 else 'empty'}")
        print(f"🔍 Debug - predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
        print(f"🔍 Debug - posteriors shape: {posteriors.shape}")
        print(f"🔍 Debug - sample predictions: {predictions[:5] if len(predictions) > 5 else predictions}")
        print(f"🔍 Debug - sample posteriors: {posteriors[:2] if len(posteriors) > 2 else posteriors}")

    def prepare_full_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the full dataset for adaptive learning"""
        print("📊 Preparing full dataset...")

        # Load data using the model's method
        self.model.load_data()

        # Preprocess data using the enhanced preprocessor
        X, y, feature_names = self.model.preprocess_data()

        # Store original y for reference (before encoding)
        y_original = y.copy()

        # Store the full dataset
        self.X_full = X
        self.y_full = y
        self.y_full_original = y_original
        self.original_data_shape = X.shape

        print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Classes: {len(np.unique(y))} ({np.unique(y_original)})")
        print(f"🔧 Features: {feature_names}")

        return X, y, y_original

    def adaptive_learn(self, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning method following the exact requirements"""
        print("\n🚀 STARTING ADAPTIVE LEARNING")
        print("=" * 60)

        # Use provided data or prepare full data
        if X is None or y is None:
            print("📊 Preparing dataset...")
            X, y, y_original = self.prepare_full_data()
        else:
            y_original = y.copy()
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)

        # Store the full dataset
        self.X_full = X.copy()
        self.y_full = y.copy()
        self.y_full_original = y_original

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y_original)}")

        # STEP 1: Initialize DBNN architecture with full dataset
        self.model.initialize_with_full_data(X, y)

        # STEP 2: Select initial diverse training samples
        X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(X_train)} samples")
        print(f"📊 Remaining test set: {len(remaining_indices)} samples")

        # Initialize tracking variables
        self.best_accuracy = 0.0
        self.best_training_indices = initial_indices.copy()
        self.best_round = 0
        patience_counter = 0

        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']

        print(f"\n🔄 Starting adaptive learning for up to {max_rounds} rounds...")
        self.adaptive_start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num

            print(f"\n🎯 Round {round_num}/{max_rounds}")
            print("-" * 40)

            # STEP 2 (continued): Train with current training data (no split)
            print("🎯 Training with current training data...")
            success = self.model.train_with_data(X_train, y_train, reset_weights=True)

            if not success:
                print("❌ Training failed, stopping...")
                break

            # STEP 3: Run acid test on entire dataset
            print("🧪 Running acid test on entire dataset...")
            try:
                all_predictions = self.model.predict(X)
                # Ensure predictions and y have same data type
                all_predictions = all_predictions.astype(y.dtype)
                acid_test_accuracy = accuracy_score(y, all_predictions)
                print(f"📊 Acid test accuracy: {acid_test_accuracy:.4f}")

                # Check if all samples are correctly classified
                if acid_test_accuracy >= 0.999:  # 99.9% accuracy
                    print("🎉 All samples correctly classified!")
                    # Update best accuracy before breaking
                    if acid_test_accuracy > self.best_accuracy:
                        self.best_accuracy = acid_test_accuracy
                        self.best_training_indices = initial_indices.copy()
                        self.best_round = round_num
                    break
            except Exception as e:
                print(f"❌ Acid test failed: {e}")
                acid_test_accuracy = 0.0
            # STEP 3 (continued): Identify failed candidates in remaining data
            if not remaining_indices:
                print("💤 No more samples to add")
                break

            X_remaining = X[remaining_indices]
            y_remaining = y[remaining_indices]

            # Get predictions for remaining data
            remaining_predictions = self.model.predict(X_remaining)
            remaining_posteriors = self.model._compute_batch_posterior(X_remaining)

            # Debug predictions
            self._debug_predictions(y_remaining, remaining_predictions, remaining_posteriors)

            # Find misclassified samples
            misclassified_mask = remaining_predictions != y_remaining
            misclassified_indices = np.where(misclassified_mask)[0]

            if len(misclassified_indices) == 0:
                print("✅ No misclassified samples in remaining data!")
                # Update best accuracy before breaking
                if acid_test_accuracy > self.best_accuracy:
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                break

            print(f"📊 Found {len(misclassified_indices)} misclassified samples in remaining data")

            # STEP 3 (continued): Select most divergent failed candidates
            samples_to_add_indices = self._select_divergent_samples(
                X_remaining, y_remaining, remaining_predictions, remaining_posteriors,
                misclassified_indices, remaining_indices
            )

            if not samples_to_add_indices:
                print("💤 No divergent samples to add")
                break

            # Update training set
            initial_indices.extend(samples_to_add_indices)
            remaining_indices = [i for i in remaining_indices if i not in samples_to_add_indices]

            X_train = X[initial_indices]
            y_train = y[initial_indices]

            print(f"📈 Training set size: {len(X_train)}")
            print(f"📊 Remaining test set size: {len(remaining_indices)}")

            # Update best model based on acid test - FIXED: Always update when better
            if acid_test_accuracy > self.best_accuracy + self.adaptive_config['min_improvement']:
                self.best_accuracy = acid_test_accuracy
                self.best_training_indices = initial_indices.copy()
                self.best_round = round_num
                patience_counter = 0
                print(f"🏆 New best acid test accuracy: {acid_test_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"🔄 No improvement (Patience: {patience_counter}/{patience})")

            # Early stopping based on acid test
            if patience_counter >= patience:
                print(f"🛑 Patience exceeded: no improvement in acid test for {patience} rounds")
                break

        # Finalize with best configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")

        # Ensure we have valid best values
        if not hasattr(self, 'best_accuracy') or self.best_accuracy == 0.0:
            # Use final values if best wasn't set
            self.best_accuracy = acid_test_accuracy if 'acid_test_accuracy' in locals() else 0.0
            self.best_training_indices = initial_indices.copy()
            self.best_round = self.adaptive_round

        print(f"🏆 Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")

        # Use best configuration for final model
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[[i for i in range(len(X)) if i not in self.best_training_indices]]
        y_test_best = y[[i for i in range(len(X)) if i not in self.best_training_indices]]

        # Train final model with best configuration
        print("🔧 Training final model with best configuration...")
        self.model.train_with_data(X_train_best, y_train_best, reset_weights=True)

        # Final acid test
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)

        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")
        print(f"📈 Final training set size: {len(X_train_best)}")
        print(f"📊 Final remaining set size: {len(X_test_best)}")

        # Generate reports
        self._generate_adaptive_learning_report()

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _select_initial_training_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Select initial diverse training samples from each class"""
        initial_samples = self.adaptive_config['initial_samples_per_class']
        unique_classes = np.unique(y)

        initial_indices = []

        print("🎯 Selecting initial diverse training samples...")

        for class_id in unique_classes:
            class_indices = np.where(y == class_id)[0]

            if len(class_indices) > initial_samples:
                # Use k-means++ to select diverse samples
                class_data = X[class_indices]
                kmeans = KMeans(n_clusters=initial_samples, init='k-means++', n_init=1, random_state=42)
                kmeans.fit(class_data)

                # Find samples closest to cluster centers
                distances = kmeans.transform(class_data)
                closest_indices = np.argmin(distances, axis=0)
                selected_indices = class_indices[closest_indices]
            else:
                # Use all available samples
                selected_indices = class_indices

            initial_indices.extend(selected_indices)

        X_train = X[initial_indices]
        y_train = y[initial_indices]

        return X_train, y_train, initial_indices

    def _select_divergent_samples(self, X_remaining: np.ndarray, y_remaining: np.ndarray,
                                predictions: np.ndarray, posteriors: np.ndarray,
                                misclassified_indices: np.ndarray, remaining_indices: List[int]) -> List[int]:
        """Select most divergent failed candidates from each class"""
        samples_to_add = []
        unique_classes = np.unique(y_remaining)

        print("🔍 Selecting most divergent failed candidates...")

        # Group misclassified samples by true class
        class_samples = defaultdict(list)

        for idx_in_remaining in misclassified_indices:
            original_idx = remaining_indices[idx_in_remaining]
            true_class = y_remaining[idx_in_remaining]
            pred_class = predictions[idx_in_remaining]

            # Convert class labels to 0-based indices for array access
            true_class_idx_result = np.where(unique_classes == true_class)[0]
            pred_class_idx_result = np.where(unique_classes == pred_class)[0]

            # Check if we found valid indices
            if len(true_class_idx_result) == 0 or len(pred_class_idx_result) == 0:
                continue

            true_class_idx = true_class_idx_result[0]
            pred_class_idx = pred_class_idx_result[0]

            # Calculate margin (divergence)
            true_posterior = posteriors[idx_in_remaining, true_class_idx]
            pred_posterior = posteriors[idx_in_remaining, pred_class_idx]
            margin = pred_posterior - true_posterior

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_posterior': true_posterior,
                'pred_posterior': pred_posterior
            })

        # For each class, select most divergent samples
        max_samples = self.adaptive_config.get('max_margin_samples_per_class', 2)

        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                continue

            class_data = class_samples[class_id]

            # Sort by margin (most negative first - most divergent)
            class_data.sort(key=lambda x: x['margin'])

            # Select top divergent samples
            selected_for_class = class_data[:max_samples]

            for sample in selected_for_class:
                samples_to_add.append(sample['index'])

                # Track selection
                self.all_selected_samples[self._get_original_class_label(class_id)].append({
                    'index': sample['index'],
                    'margin': sample['margin'],
                    'selection_type': 'divergent',
                    'round': self.adaptive_round
                })

            if selected_for_class:
                print(f"   ✅ Class {class_id}: Selected {len(selected_for_class)} divergent samples")

        print(f"📥 Total divergent samples to add: {len(samples_to_add)}")
        return samples_to_add

    def _get_original_class_label(self, encoded_class: int) -> str:
        """Convert encoded class back to original label"""
        if hasattr(self.label_encoder, 'classes_'):
            try:
                return str(self.label_encoder.inverse_transform([encoded_class])[0])
            except:
                return str(encoded_class)
        return str(encoded_class)

    def _generate_adaptive_learning_report(self):
        """Generate comprehensive adaptive learning report"""
        print("\n📊 Generating Adaptive Learning Report...")

        # Ensure we have valid statistics
        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = 0.0
        if not hasattr(self, 'best_training_indices'):
            self.best_training_indices = []
        if not hasattr(self, 'best_round'):
            self.best_round = 0

        total_time = str(datetime.now() - self.adaptive_start_time) if hasattr(self, 'adaptive_start_time') and self.adaptive_start_time else "N/A"

        report = {
            'dataset': self.dataset_name,
            'total_samples': len(self.X_full) if hasattr(self, 'X_full') else 0,
            'final_training_size': len(self.best_training_indices),
            'final_remaining_size': (len(self.X_full) - len(self.best_training_indices)) if hasattr(self, 'X_full') else 0,
            'best_accuracy': float(self.best_accuracy),  # Ensure it's a float
            'best_round': self.best_round,
            'total_rounds': getattr(self, 'adaptive_round', 0),
            'total_time': total_time,
            'adaptive_config': self.adaptive_config,
            'round_statistics': getattr(self, 'round_stats', []),
            'selected_samples_by_class': {k: len(v) for k, v in self.all_selected_samples.items()}
        }

        # Save report
        report_path = f"{self.viz_config.get('output_dir', 'adaptive_visualizations')}/adaptive_learning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        print(f"✅ Report saved to: {report_path}")

        # Print summary with proper formatting
        print("\n📈 Adaptive Learning Summary:")
        print("=" * 50)
        print(f"   Dataset: {report['dataset']}")
        print(f"   Total samples: {report['total_samples']}")

        if report['total_samples'] > 0:
            training_percentage = (report['final_training_size'] / report['total_samples']) * 100
            print(f"   Final training set: {report['final_training_size']} ({training_percentage:.1f}%)")
        else:
            print(f"   Final training set: {report['final_training_size']}")

        print(f"   Best acid test accuracy: {report['best_accuracy']:.4f}")
        print(f"   Achieved in round: {report['best_round']}")
        print(f"   Total rounds: {report['total_rounds']}")
        print(f"   Total time: {report['total_time']}")
        print("=" * 50)


class AdaptiveDBNNGUI:
    """GUI interface for configuring and running Adaptive DBNN"""

    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive DBNN Configuration")
        self.root.geometry("1200x900")

        # Configuration storage
        self.config = {}
        self.config_file = "adaptive_dbnn_config.json"

        # Data storage
        self.data = None
        self.data_columns = []
        self.column_types = {}

        # Main variables
        self.dataset_name = tk.StringVar()
        self.data_file_path = tk.StringVar()
        self.target_column = tk.StringVar(value="target")
        self.output_dir = tk.StringVar(value="adaptive_visualizations")

        # Feature selection
        self.feature_vars = {}
        self.feature_frame = None

        self.setup_ui()
        self.load_saved_config()

    def setup_ui(self):
        """Setup the main GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Main Configuration Tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Dataset & Features")

        # Adaptive Learning Tab
        adaptive_frame = ttk.Frame(notebook)
        notebook.add(adaptive_frame, text="Adaptive Learning")

        # Advanced Tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")

        # Setup each tab
        self.setup_main_tab(main_frame)
        self.setup_adaptive_tab(adaptive_frame)
        self.setup_advanced_tab(advanced_frame)

        # Control buttons
        self.setup_control_buttons()

    def setup_main_tab(self, parent):
        """Setup main configuration tab with dataset and feature selection"""
        # Dataset selection
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding="10")
        dataset_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(dataset_frame, text="Dataset File:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(dataset_frame, textvariable=self.data_file_path, width=50).grid(row=0, column=1, sticky='w', pady=2, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_data_file).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="Load & Analyze", command=self.load_and_analyze_data).grid(row=0, column=3, padx=5)

        ttk.Label(dataset_frame, text="Dataset Name:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(dataset_frame, textvariable=self.dataset_name, width=30).grid(row=1, column=1, sticky='w', pady=2, padx=5)

        ttk.Label(dataset_frame, text="Target Column:").grid(row=2, column=0, sticky='w', pady=2)
        self.target_combo = ttk.Combobox(dataset_frame, textvariable=self.target_column, width=30, state="readonly")
        self.target_combo.grid(row=2, column=1, sticky='w', pady=2, padx=5)
        ttk.Button(dataset_frame, text="Auto-Detect Target", command=self.auto_detect_target).grid(row=2, column=2, padx=5)

        ttk.Label(dataset_frame, text="Output Directory:").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(dataset_frame, textvariable=self.output_dir, width=30).grid(row=3, column=1, sticky='w', pady=2, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_output_dir).grid(row=3, column=2, padx=5)

        # Data preview
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding="10")
        preview_frame.pack(fill='x', padx=5, pady=5)

        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=8, width=100)
        self.preview_text.pack(fill='both', expand=True)

        # Feature selection
        self.feature_frame = ttk.LabelFrame(parent, text="Feature Selection", padding="10")
        self.feature_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Feature selection controls will be populated after data load

        # DBNN Core Parameters
        core_frame = ttk.LabelFrame(parent, text="DBNN Core Parameters", padding="10")
        core_frame.pack(fill='x', padx=5, pady=5)

        # Create a grid for core parameters
        core_params = [
            ("Resolution:", "resolution", "100", "Higher = more precise, slower"),
            ("Gain:", "gain", "2.0", "Weight update intensity"),
            ("Margin:", "margin", "0.2", "Classification tolerance"),
            ("Max Epochs:", "max_epochs", "100", "Maximum training epochs"),
            ("Patience:", "patience", "10", "Early stopping rounds")
        ]

        for i, (label, attr, default, tooltip) in enumerate(core_params):
            ttk.Label(core_frame, text=label).grid(row=i, column=0, sticky='w', pady=2, padx=5)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            entry = ttk.Entry(core_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky='w', pady=2, padx=5)
            ttk.Label(core_frame, text=tooltip).grid(row=i, column=2, sticky='w', padx=5)
            self.create_tooltip(entry, tooltip)

    def setup_feature_selection(self):
        """Setup feature selection interface after data is loaded"""
        # Clear existing feature selection
        for widget in self.feature_frame.winfo_children():
            widget.destroy()

        if not self.data_columns:
            ttk.Label(self.feature_frame, text="No data loaded. Please load a dataset first.").pack(pady=10)
            return

        # Create header with selection controls
        header_frame = ttk.Frame(self.feature_frame)
        header_frame.pack(fill='x', pady=5)

        ttk.Label(header_frame, text="Select features for training:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)

        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side='right', padx=5)

        ttk.Button(button_frame, text="Select All", command=self.select_all_features).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_features).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Select Numeric", command=self.select_numeric_features).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Invert Selection", command=self.invert_feature_selection).pack(side='left', padx=2)

        # Create scrollable frame for feature checkboxes
        canvas = tk.Canvas(self.feature_frame, height=200)
        scrollbar = ttk.Scrollbar(self.feature_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create feature checkboxes in a grid
        self.feature_vars = {}
        num_columns = 4  # Number of columns for the grid
        features_per_column = (len(self.data_columns) + num_columns - 1) // num_columns

        for i, column in enumerate(self.data_columns):
            if column == self.target_column.get():
                continue  # Skip target column

            row = i % features_per_column
            col = i // features_per_column

            var = tk.BooleanVar(value=True)  # Default to selected
            self.feature_vars[column] = var

            # Determine column type and create appropriate display
            col_type = self.column_types.get(column, 'unknown')
            display_text = f"{column} ({col_type})"

            cb = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var)
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)

            # Color code based on type
            if col_type == 'numeric':
                cb.configure(style='Numeric.TCheckbutton')
            elif col_type == 'categorical':
                cb.configure(style='Categorical.TCheckbutton')

            self.create_tooltip(cb, f"Column: {column}\nType: {col_type}")

        # Add summary
        summary_frame = ttk.Frame(self.feature_frame)
        summary_frame.pack(fill='x', pady=5)

        self.feature_summary = tk.StringVar()
        ttk.Label(summary_frame, textvariable=self.feature_summary).pack(side='left')
        self.update_feature_summary()

    def analyze_column_types(self, df):
        """Analyze column types for recommendations"""
        column_types = {}

        for column in df.columns:
            # Skip if all null
            if df[column].isna().all():
                column_types[column] = 'all_null'
                continue

            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = 'numeric'
            else:
                # Check if it could be categorical
                unique_ratio = df[column].nunique() / len(df[column])
                if unique_ratio < 0.1:  # Low cardinality
                    column_types[column] = 'categorical_low'
                elif unique_ratio < 0.5:  # Medium cardinality
                    column_types[column] = 'categorical_medium'
                else:  # High cardinality
                    column_types[column] = 'categorical_high'

        return column_types

    def get_target_recommendations(self, df):
        """Get target column recommendations based on cardinality"""
        recommendations = []

        for column in df.columns:
            col_type = self.column_types.get(column, 'unknown')

            # Good candidates for classification targets
            if col_type in ['categorical_low', 'categorical_medium']:
                unique_count = df[column].nunique()
                if 2 <= unique_count <= 50:  # Reasonable number of classes
                    recommendations.append({
                        'column': column,
                        'type': 'classification',
                        'unique_count': unique_count,
                        'score': 100 - unique_count  # Lower unique count = better for classification
                    })

            # Good candidates for regression targets
            elif col_type == 'numeric':
                if df[column].nunique() > 20:  # High cardinality numeric
                    recommendations.append({
                        'column': column,
                        'type': 'regression',
                        'unique_count': df[column].nunique(),
                        'score': 50
                    })

        # Sort by score (higher is better)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

    def auto_detect_target(self):
        """Automatically detect and set the best target column"""
        if self.data is None:
            messagebox.showwarning("No Data", "Please load data first.")
            return

        recommendations = self.get_target_recommendations(self.data)

        if not recommendations:
            messagebox.showinfo("No Recommendations", "No suitable target columns found automatically.")
            return

        # Use the best recommendation
        best_target = recommendations[0]
        self.target_column.set(best_target['column'])

        # Update target combo
        self.target_combo.set(best_target['column'])

        messagebox.showinfo(
            "Target Auto-Detected",
            f"Selected '{best_target['column']}' as target.\n"
            f"Type: {best_target['type']}\n"
            f"Unique values: {best_target['unique_count']}"
        )

        # Refresh feature selection to exclude the new target
        self.setup_feature_selection()

    def browse_data_file(self):
        """Browse for data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.data_file_path.set(filename)
            # Extract dataset name from filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.dataset_name.set(base_name)

    def load_and_analyze_data(self):
        """Load and analyze the selected data file"""
        file_path = self.data_file_path.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid data file.")
            return

        try:
            # Load data based on file type
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:  # DAT file or other
                try:
                    self.data = pd.read_csv(file_path, delimiter=r'\s+')  # Space separated
                except:
                    self.data = pd.read_csv(file_path)  # Try with default parameters

            self.data_columns = self.data.columns.tolist()

            # Analyze column types
            self.column_types = self.analyze_column_types(self.data)

            # Update target column combobox
            self.target_combo['values'] = self.data_columns

            # Update data preview
            self.update_data_preview()

            # Setup feature selection
            self.setup_feature_selection()

            # Show target recommendations
            self.show_target_recommendations()

            messagebox.showinfo("Success", f"Data loaded successfully!\n{self.data.shape[0]} rows, {self.data.shape[1]} columns")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def update_data_preview(self):
        """Update the data preview text"""
        if self.data is None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "No data loaded.")
            return

        preview_info = f"Dataset: {self.data.shape[0]} rows × {self.data.shape[1]} columns\n"
        preview_info += f"Columns: {', '.join(self.data_columns)}\n\n"
        preview_info += "First 5 rows:\n"
        preview_info += self.data.head().to_string()

        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, preview_info)

    def show_target_recommendations(self):
        """Show target column recommendations"""
        if self.data is None:
            return

        recommendations = self.get_target_recommendations(self.data)

        if recommendations:
            rec_text = "Target Column Recommendations:\n"
            for i, rec in enumerate(recommendations[:5]):  # Top 5
                rec_text += f"{i+1}. {rec['column']} ({rec['type']}, {rec['unique_count']} unique values)\n"

            # Show in a message box
            messagebox.showinfo("Target Recommendations", rec_text)

    def select_all_features(self):
        """Select all features"""
        for var in self.feature_vars.values():
            var.set(True)
        self.update_feature_summary()

    def deselect_all_features(self):
        """Deselect all features"""
        for var in self.feature_vars.values():
            var.set(False)
        self.update_feature_summary()

    def select_numeric_features(self):
        """Select only numeric features"""
        for column, var in self.feature_vars.items():
            col_type = self.column_types.get(column, 'unknown')
            var.set(col_type == 'numeric')
        self.update_feature_summary()

    def invert_feature_selection(self):
        """Invert feature selection"""
        for var in self.feature_vars.values():
            var.set(not var.get())
        self.update_feature_summary()

    def update_feature_summary(self):
        """Update the feature selection summary"""
        if not self.feature_vars:
            self.feature_summary.set("No features available")
            return

        selected_count = sum(1 for var in self.feature_vars.values() if var.get())
        total_count = len(self.feature_vars)

        # Count by type
        numeric_count = 0
        categorical_count = 0
        selected_numeric = 0
        selected_categorical = 0

        for column, var in self.feature_vars.items():
            col_type = self.column_types.get(column, 'unknown')
            is_selected = var.get()

            if 'numeric' in col_type:
                numeric_count += 1
                if is_selected:
                    selected_numeric += 1
            elif 'categorical' in col_type:
                categorical_count += 1
                if is_selected:
                    selected_categorical += 1

        summary = f"Selected: {selected_count}/{total_count} features"
        if numeric_count > 0:
            summary += f" | Numeric: {selected_numeric}/{numeric_count}"
        if categorical_count > 0:
            summary += f" | Categorical: {selected_categorical}/{categorical_count}"

        self.feature_summary.set(summary)

    def get_selected_features(self):
        """Get list of selected feature columns"""
        return [col for col, var in self.feature_vars.items() if var.get()]

    def setup_adaptive_tab(self, parent):
        """Setup adaptive learning configuration tab"""
        # Adaptive Learning Parameters
        adaptive_frame = ttk.LabelFrame(parent, text="Adaptive Learning Parameters", padding="10")
        adaptive_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create scrollable frame
        canvas = tk.Canvas(adaptive_frame)
        scrollbar = ttk.Scrollbar(adaptive_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Adaptive parameters
        params = [
            ("Enable Adaptive Learning", "enable_adaptive", "True", "checkbox", "Master switch for adaptive learning"),
            ("Initial Samples Per Class", "initial_samples_per_class", "5", "entry", "Number of initial samples per class for training"),
            ("Max Margin Samples Per Class", "max_margin_samples_per_class", "3", "entry", "Maximum margin-based samples to add per class per round"),
            ("Margin Tolerance", "margin_tolerance", "0.15", "entry", "Tolerance for margin-based selection"),
            ("KL Divergence Threshold", "kl_divergence_threshold", "0.1", "entry", "Threshold for KL divergence-based selection"),
            ("Max Adaptive Rounds", "max_adaptive_rounds", "20", "entry", "Maximum number of adaptive learning rounds"),
            ("Min Improvement", "min_improvement", "0.001", "entry", "Minimum improvement for early stopping"),
            ("Training Convergence Epochs", "training_convergence_epochs", "50", "entry", "Epochs to wait for training convergence"),
            ("Min Training Accuracy", "min_training_accuracy", "0.95", "entry", "Minimum training accuracy threshold"),
            ("Min Samples To Add Per Class", "min_samples_to_add_per_class", "5", "entry", "Minimum samples to add per class"),
            ("Adaptive Margin Relaxation", "adaptive_margin_relaxation", "0.1", "entry", "Margin relaxation factor"),
            ("Max Divergence Samples Per Class", "max_divergence_samples_per_class", "5", "entry", "Maximum divergence samples per class"),
            ("Exhaust All Failed", "exhaust_all_failed", "True", "checkbox", "Whether to exhaust all failed samples"),
            ("Min Failed Threshold", "min_failed_threshold", "10", "entry", "Minimum failed samples threshold"),
            ("Enable KL Divergence", "enable_kl_divergence", "True", "checkbox", "Enable KL divergence-based sampling"),
            ("Max Samples Per Class Fallback", "max_samples_per_class_fallback", "2", "entry", "Fallback samples per class"),
            ("Enable 3D Visualization", "enable_3d_visualization", "True", "checkbox", "Enable 3D visualization"),
            ("3D Snapshot Interval", "3d_snapshot_interval", "10", "entry", "Interval for 3D snapshots"),
            ("Learning Rate", "learning_rate", "1.0", "entry", "Learning rate for weight updates"),
            ("Enable Acid Test", "enable_acid_test", "True", "checkbox", "Enable acid test on full dataset"),
            ("Min Training Percentage For Stopping", "min_training_percentage_for_stopping", "10.0", "entry", "Minimum training percentage for stopping"),
            ("Max Training Percentage", "max_training_percentage", "90.0", "entry", "Maximum training percentage"),
            ("Max KL Samples Per Class", "max_kl_samples_per_class", "5", "entry", "Maximum KL-based samples per class"),
            ("Disable Sample Limit", "disable_sample_limit", "False", "checkbox", "Disable sample limits (use with caution)"),
        ]

        self.adaptive_vars = {}

        for i, (label, key, default, widget_type, tooltip) in enumerate(params):
            ttk.Label(scrollable_frame, text=label).grid(row=i, column=0, sticky='w', pady=2, padx=5)

            if widget_type == "checkbox":
                var = tk.BooleanVar(value=(default.lower() == "true"))
                cb = ttk.Checkbutton(scrollable_frame, variable=var)
                cb.grid(row=i, column=1, sticky='w', pady=2)
                self.adaptive_vars[key] = var
            else:  # entry
                var = tk.StringVar(value=default)
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky='w', pady=2)
                self.adaptive_vars[key] = var

            # Tooltip
            widget = scrollable_frame.grid_slaves(row=i, column=1)[0]
            self.create_tooltip(widget, f"Parameter: {key}\nDefault: {default}\n\n{tooltip}")

    def setup_advanced_tab(self, parent):
        """Setup advanced configuration tab"""
        # Visualization Configuration
        viz_frame = ttk.LabelFrame(parent, text="Visualization Configuration", padding="10")
        viz_frame.pack(fill='x', padx=5, pady=5)

        self.enable_visualization = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Enable Visualization",
                       variable=self.enable_visualization).grid(row=0, column=0, sticky='w', pady=2)

        self.create_animations = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Create Animations",
                       variable=self.create_animations).grid(row=1, column=0, sticky='w', pady=2)

        self.create_reports = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Create Reports",
                       variable=self.create_reports).grid(row=2, column=0, sticky='w', pady=2)

        self.create_3d_visualizations = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Create 3D Visualizations",
                       variable=self.create_3d_visualizations).grid(row=3, column=0, sticky='w', pady=2)

        # Statistics Configuration
        stats_frame = ttk.LabelFrame(parent, text="Statistics Configuration", padding="10")
        stats_frame.pack(fill='x', padx=5, pady=5)

        self.enable_confusion_matrix = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Enable Confusion Matrix",
                       variable=self.enable_confusion_matrix).grid(row=0, column=0, sticky='w', pady=2)

        self.enable_progress_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Enable Progress Plots",
                       variable=self.enable_progress_plots).grid(row=1, column=0, sticky='w', pady=2)

        self.create_interactive_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Create Interactive Plots",
                       variable=self.create_interactive_plots).grid(row=2, column=0, sticky='w', pady=2)

        self.create_sample_analysis = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Create Sample Analysis",
                       variable=self.create_sample_analysis).grid(row=3, column=0, sticky='w', pady=2)

        # Color Configuration
        color_frame = ttk.LabelFrame(parent, text="Color Configuration", padding="10")
        color_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(color_frame, text="Progress Color:").grid(row=0, column=0, sticky='w', pady=2)
        self.color_progress = tk.StringVar(value="green")
        ttk.Entry(color_frame, textvariable=self.color_progress, width=15).grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(color_frame, text="Regression Color:").grid(row=1, column=0, sticky='w', pady=2)
        self.color_regression = tk.StringVar(value="red")
        ttk.Entry(color_frame, textvariable=self.color_regression, width=15).grid(row=1, column=1, sticky='w', pady=2)

    def setup_control_buttons(self):
        """Setup control buttons at the bottom"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Save Configuration",
                  command=self.save_configuration).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Load Configuration",
                  command=self.load_configuration).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Reset to Defaults",
                  command=self.reset_to_defaults).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Run Adaptive DBNN",
                  command=self.run_adaptive_dbnn, style="Accent.TButton").pack(side='right', padx=5)

        ttk.Button(button_frame, text="Close",
                  command=self.root.quit).pack(side='right', padx=5)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="yellow",
                            relief='solid', borderwidth=1, padding=5, wraplength=300)
            label.pack()
            widget.tooltip = tooltip

        def leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from GUI"""
        config = {
            'dataset_name': self.dataset_name.get(),
            'data_file': self.data_file_path.get(),
            'target_column': self.target_column.get(),
            'feature_columns': self.get_selected_features(),
            'output_dir': self.output_dir.get(),

            # Core DBNN parameters
            'resol': int(self.resolution.get()),
            'gain': float(self.gain.get()),
            'margin': float(self.margin.get()),
            'max_epochs': int(self.max_epochs.get()),
            'patience': int(self.patience.get()),

            # Adaptive learning parameters
            'adaptive_learning': {},
            'visualization_config': {},
            'statistics': {}
        }

        # Adaptive learning parameters
        for key, var in self.adaptive_vars.items():
            if isinstance(var, tk.BooleanVar):
                config['adaptive_learning'][key] = var.get()
            else:
                # Try to convert to appropriate type
                value = var.get()
                try:
                    if '.' in value:
                        config['adaptive_learning'][key] = float(value)
                    else:
                        config['adaptive_learning'][key] = int(value)
                except ValueError:
                    config['adaptive_learning'][key] = value

        # Visualization configuration
        config['visualization_config'] = {
            'enabled': self.enable_visualization.get(),
            'output_dir': self.output_dir.get(),
            'create_animations': self.create_animations.get(),
            'create_reports': self.create_reports.get(),
            'create_3d_visualizations': self.create_3d_visualizations.get()
        }

        # Statistics configuration
        config['statistics'] = {
            'enable_confusion_matrix': self.enable_confusion_matrix.get(),
            'enable_progress_plots': self.enable_progress_plots.get(),
            'create_interactive_plots': self.create_interactive_plots.get(),
            'create_sample_analysis': self.create_sample_analysis.get(),
            'color_progress': self.color_progress.get(),
            'color_regression': self.color_regression.get(),
            'save_plots': True
        }

        return config

    def save_configuration(self):
        """Save configuration to file"""
        config = self.get_configuration()

        # Validate required fields
        if not config['data_file'] or not os.path.exists(config['data_file']):
            messagebox.showerror("Error", "Please select a valid data file first.")
            return

        if not config['target_column']:
            messagebox.showerror("Error", "Please select a target column.")
            return

        selected_features = self.get_selected_features()
        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
                messagebox.showinfo("Success", f"Configuration saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def load_configuration(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                self.apply_configuration(config)
                messagebox.showinfo("Success", f"Configuration loaded from:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def load_saved_config(self):
        """Load automatically saved configuration if exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.apply_configuration(config)
            except:
                pass  # Ignore errors in auto-load

    def apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration to GUI elements"""
        # Main parameters
        self.dataset_name.set(config.get('dataset_name', ''))
        self.data_file_path.set(config.get('data_file', ''))
        self.target_column.set(config.get('target_column', 'target'))
        self.output_dir.set(config.get('output_dir', 'adaptive_visualizations'))

        # Load data if file exists
        if os.path.exists(self.data_file_path.get()):
            self.load_and_analyze_data()

        # Core parameters
        self.resolution.set(str(config.get('resol', 100)))
        self.gain.set(str(config.get('gain', 2.0)))
        self.margin.set(str(config.get('margin', 0.2)))
        self.max_epochs.set(str(config.get('max_epochs', 100)))
        self.patience.set(str(config.get('patience', 10)))

        # Adaptive learning parameters
        adaptive_config = config.get('adaptive_learning', {})
        for key, var in self.adaptive_vars.items():
            if key in adaptive_config:
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(adaptive_config[key]))
                else:
                    var.set(str(adaptive_config[key]))

        # Visualization configuration
        viz_config = config.get('visualization_config', {})
        self.enable_visualization.set(viz_config.get('enabled', True))
        self.create_animations.set(viz_config.get('create_animations', False))
        self.create_reports.set(viz_config.get('create_reports', True))
        self.create_3d_visualizations.set(viz_config.get('create_3d_visualizations', True))

        # Statistics configuration
        stats_config = config.get('statistics', {})
        self.enable_confusion_matrix.set(stats_config.get('enable_confusion_matrix', True))
        self.enable_progress_plots.set(stats_config.get('enable_progress_plots', True))
        self.create_interactive_plots.set(stats_config.get('create_interactive_plots', True))
        self.create_sample_analysis.set(stats_config.get('create_sample_analysis', True))
        self.color_progress.set(stats_config.get('color_progress', 'green'))
        self.color_regression.set(stats_config.get('color_regression', 'red'))

        # Feature selection (need to be applied after data load)
        if self.data is not None and 'feature_columns' in config:
            # This will be applied when feature selection is set up
            pass

    def reset_to_defaults(self):
        """Reset all values to defaults"""
        default_config = {
            'dataset_name': '',
            'data_file': '',
            'target_column': 'target',
            'output_dir': 'adaptive_visualizations',
            'resol': 100,
            'gain': 2.0,
            'margin': 0.2,
            'max_epochs': 100,
            'patience': 10,
            'adaptive_learning': {
                'enable_adaptive': True,
                'initial_samples_per_class': 5,
                'max_margin_samples_per_class': 3,
                'margin_tolerance': 0.15,
                'kl_divergence_threshold': 0.1,
                'max_adaptive_rounds': 20,
                'min_improvement': 0.001,
                'training_convergence_epochs': 50,
                'min_training_accuracy': 0.95,
                'min_samples_to_add_per_class': 5,
                'adaptive_margin_relaxation': 0.1,
                'max_divergence_samples_per_class': 5,
                'exhaust_all_failed': True,
                'min_failed_threshold': 10,
                'enable_kl_divergence': True,
                'max_samples_per_class_fallback': 2,
                'enable_3d_visualization': True,
                '3d_snapshot_interval': 10,
                'learning_rate': 1.0,
                'enable_acid_test': True,
                'min_training_percentage_for_stopping': 10.0,
                'max_training_percentage': 90.0,
                'max_kl_samples_per_class': 5,
                'disable_sample_limit': False,
            },
            'visualization_config': {
                'enabled': True,
                'output_dir': 'adaptive_visualizations',
                'create_animations': False,
                'create_reports': True,
                'create_3d_visualizations': True
            },
            'statistics': {
                'enable_confusion_matrix': True,
                'enable_progress_plots': True,
                'color_progress': 'green',
                'color_regression': 'red',
                'save_plots': True,
                'create_interactive_plots': True,
                'create_sample_analysis': True
            }
        }

        self.apply_configuration(default_config)

    def run_adaptive_dbnn(self):
        """Run adaptive DBNN with current configuration"""
        config = self.get_configuration()

        # Validate required fields
        if not config['data_file'] or not os.path.exists(config['data_file']):
            messagebox.showerror("Error", "Please select a valid data file first!")
            return

        if not config['target_column']:
            messagebox.showerror("Error", "Please select a target column!")
            return

        selected_features = self.get_selected_features()
        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature!")
            return

        if config['target_column'] in selected_features:
            messagebox.showerror("Error", f"Target column '{config['target_column']}' cannot be in feature columns!")
            return

        # Save configuration automatically
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except:
            pass  # Ignore auto-save errors

        # Save configuration for the adaptive DBNN to use - use JSON format
        config_filename = f"{config['dataset_name']}_config.json"  # Changed to .json
        try:
            with open(config_filename, 'w') as f:
                json.dump(config, f, indent=4)

            messagebox.showinfo(
                "Configuration Saved",
                f"Configuration saved to:\n{config_filename}\n\n"
                f"Close this window and run:\n"
                f"python adaptive_dbnn.py --config {config_filename}\n\n"
                f"Selected {len(selected_features)} features:\n"
                f"{', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save run configuration:\n{e}")


def launch_gui():
    """Launch the GUI interface"""
    root = tk.Tk()

    # Set theme if available
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")

        # Create custom styles for feature checkboxes
        style = ttk.Style()
        style.configure('Numeric.TCheckbutton', foreground='blue')
        style.configure('Categorical.TCheckbutton', foreground='green')
    except ImportError:
        pass  # Use default theme

    app = AdaptiveDBNNGUI(root)
    root.mainloop()


# Update the main function in adaptive_dbnn.py to handle feature columns
def main():
    """Main function to run adaptive DBNN"""
    import sys

    # Check for GUI flag
    if "--gui" in sys.argv or "-g" in sys.argv:
        launch_gui()
        return

    # Check for config file parameter
    config_file = None
    for i, arg in enumerate(sys.argv):
        if arg in ["--config", "-c"] and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break

    print("🎯 Adaptive DBNN System")
    print("=" * 50)

    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {config_file}")

            # Print configuration summary
            if 'feature_columns' in config:
                print(f"📊 Using {len(config['feature_columns'])} features: {config['feature_columns']}")
            if 'target_column' in config:
                print(f"🎯 Target column: {config['target_column']}")
            if 'dataset_name' in config:
                print(f"📁 Dataset: {config['dataset_name']}")

        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            config = {}
    else:
        # Show available configuration files if no specific config provided
        available_configs = DatasetConfig.get_available_config_files()
        if available_configs and not config_file:
            print("\n📋 Available configuration files:")
            for cfg in available_configs:
                print(f"   • {cfg['file']} ({cfg['type']})")
            print("\n💡 Use: python adaptive_dbnn.py --config <filename> to use a specific configuration")

    # Create adaptive DBNN
    adaptive_model = AdaptiveDBNN(config.get('dataset_name'), config)

    # If no config file, use interactive configuration
    if not config_file:
        # Show available configuration files as an option
        available_configs = DatasetConfig.get_available_config_files()
        if available_configs:
            print(f"\n🔄 Found {len(available_configs)} configuration files")
            print("   You can load one by entering its number below")

        configure = input("\nConfigure adaptive learning settings? (y/N/1-9 for config): ").strip().lower()

        if configure.isdigit():
            # User selected a configuration file by number
            choice_idx = int(configure) - 1
            if 0 <= choice_idx < len(available_configs):
                selected_config = available_configs[choice_idx]
                config_file = selected_config['file']
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print(f"✅ Loaded configuration from: {config_file}")
                    # Update the model with loaded config
                    adaptive_model = AdaptiveDBNN(config.get('dataset_name'), config)
                except Exception as e:
                    print(f"❌ Failed to load configuration: {e}")

        elif configure == 'y':
            # Simple configuration interface
            print("\n🎛️  Configuration Options:")
            print("1. Enable KL Divergence sampling")
            print("2. Disable sample limits")
            print("3. Change number of rounds")
            print("4. Keep current settings")

            choice = input("Select option (1-4): ").strip()
            if choice == '1':
                adaptive_model.adaptive_config['enable_kl_divergence'] = True
                print("✅ KL Divergence sampling enabled")
            elif choice == '2':
                adaptive_model.adaptive_config['disable_sample_limit'] = True
                print("✅ Sample limits disabled")
            elif choice == '3':
                try:
                    rounds = int(input("Enter number of rounds: "))
                    adaptive_model.adaptive_config['max_adaptive_rounds'] = rounds
                    print(f"✅ Max rounds set to {rounds}")
                except ValueError:
                    print("❌ Invalid number, using default")

    # Run adaptive learning
    print("\n🚀 Starting adaptive learning...")
    X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()

    print(f"\n✅ Adaptive learning completed!")
    print(f"📦 Final training set size: {len(X_train)}")
    print(f"📊 Final test set size: {len(X_test)}")
    print(f"🏆 Best accuracy achieved: {adaptive_model.best_accuracy:.4f}")


if __name__ == "__main__":
    main()
