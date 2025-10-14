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

# Import the DBNN implementation from dbnn.py
import dbnn

class DatasetConfig:
    """Dataset configuration handler"""

    @staticmethod
    def get_available_datasets():
        """Get list of available datasets from configuration files"""
        config_files = glob.glob("*.conf")
        datasets = [f.replace('.conf', '') for f in config_files]
        return datasets

    @staticmethod
    def load_config(dataset_name):
        """Load configuration for a dataset"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

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
        available_datasets = DatasetConfig.get_available_datasets()

        # Also look for data files
        csv_files = glob.glob("*.csv")
        dat_files = glob.glob("*.dat")
        data_files = csv_files + dat_files

        if available_datasets or data_files:
            print("📁 Available datasets and data files:")

            # Show configuration-based datasets
            if available_datasets:
                print("\n🎯 Configured datasets:")
                for i, dataset in enumerate(available_datasets, 1):
                    print(f"  {i}. {dataset} (configuration)")

            # Show data files
            if data_files:
                print("\n📊 Data files:")
                for i, data_file in enumerate(data_files, len(available_datasets) + 1):
                    print(f"  {i}. {data_file}")

            try:
                choice = input(f"\nSelect a dataset (1-{len(available_datasets) + len(data_files)}): ").strip()
                choice_idx = int(choice) - 1

                if 0 <= choice_idx < len(available_datasets):
                    selected_dataset = available_datasets[choice_idx]
                    print(f"🎯 Selected configured dataset: {selected_dataset}")
                    return selected_dataset
                elif len(available_datasets) <= choice_idx < len(available_datasets) + len(data_files):
                    data_file_idx = choice_idx - len(available_datasets)
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
            print("❌ No dataset configurations or data files found.")
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

def main():
    """Main function to run adaptive DBNN"""
    print("🎯 Adaptive DBNN System")
    print("=" * 50)

    # Create adaptive DBNN
    adaptive_model = AdaptiveDBNN()

    # Ask if user wants to configure settings
    configure = input("\nConfigure adaptive learning settings? (y/N): ").strip().lower()
    if configure == 'y':
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
