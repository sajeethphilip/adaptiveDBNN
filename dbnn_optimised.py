import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
import torch
import os
import pickle
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Assume no keyboard control by default. If you have X11 running and want to be interactive, set nokbd = False
nokbd = True  # Disabled keyboard control for minimal systems
if nokbd==False:
    print("Will attempt keyboard interaction...")
    if os.name == 'nt' or 'darwin' in os.uname()[0].lower():  # Windows or MacOS
        try:
            from pynput import keyboard
            nokbd = False
        except:
            print("Could not initialize keyboard control")
    else:
        # Check if X server is available on Linux
        def is_x11_available():
            if os.name != 'posix':  # Not Linux/Unix
                return True

            # Check if DISPLAY environment variable is set
            if 'DISPLAY' not in os.environ:
                return False

            if os.path.isdir('/usr/lib/X11/'):
                return True

        # Only try to import pynput if X11 is available
        if is_x11_available():
            try:
                from pynput import keyboard
                nokbd = False
            except:
                print("Could not initialize keyboard control despite X11 being available")
        else:
            print("Keyboard control using q key for skipping training is not supported without X11!")
else:
    print('Keyboard is disabled by default. To enable it please set nokbd=False')

class DatasetConfig:
    """Handle dataset configuration loading and validation"""

    DEFAULT_CONFIG = {
    "training_config": {
        "trials": 100,
        "cardinality_threshold": 0.9,
        "cardinality_tolerance": 4,
        "learning_rate": 0.1,
        "random_seed": 42,
        "epochs": 1000,
        "test_fraction": 0.2,
        "train": True,
        "train_only": False,
        "predict": True,
        "gen_samples": False
    },
    "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000,
        "update_strategy": 3,
        "model_type": "histogram",
        "histogram_bins": 64,
        "laplace_smoothing": 1.0,
        "histogram_method": "vectorized"
    },
        "visualization_config": {
            "enabled": False,
            "output_dir": "visualizations",
            "epoch_interval": 10,
            "max_frames": 50,
            "create_animations": True,
            "create_reports": True,
            "create_3d_visualizations": True,  # Enable 3D VR-style visuals
            "rotation_speed": 5,              #  Degrees per frame
            "elevation_oscillation": True     # Add camera movement
        }
}


    @staticmethod
    def _extract_training_columns(df: pd.DataFrame, target_column: str) -> Dict:
        """Extract training columns information from DataFrame"""
        feature_columns = [col for col in df.columns if col != target_column]

        # Generate dataset fingerprint
        import hashlib
        col_info = [(col, str(df[col].dtype)) for col in df.columns]
        col_info.sort()
        fingerprint_str = ''.join([f"{col}_{dtype}" for col, dtype in col_info])
        dataset_fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()

        return {
            "feature_columns": feature_columns,
            "target_column": target_column,
            "total_features": len(feature_columns),
            "dataset_fingerprint": dataset_fingerprint,
            "timestamp": datetime.now().isoformat(),
            "auto_created": True
        }

    @staticmethod
    def _ensure_training_columns_config(config: Dict, df: pd.DataFrame) -> Dict:
        """Ensure training_columns section exists in config"""
        if 'training_columns' not in config:
            target_col = config['target_column']
            config['training_columns'] = DatasetConfig._extract_training_columns(df, target_col)

        # Ensure model_config exists
        if 'model_config' not in config:
            config['model_config'] = {}

        # Set default model_filename if not present
        if 'model_filename' not in config['model_config']:
            # Use config filename as default model name
            config['model_config']['model_filename'] = f"Best_{os.path.splitext(os.path.basename(config.get('file_path', 'default')))[0]}"

        return config

    @staticmethod
    def _get_user_input(prompt: str, default_value: Any = None, validation_fn: callable = None) -> Any:
        """Helper method to get validated user input"""
        while True:
            if default_value is not None:
                user_input = input(f"{prompt} (default: {default_value}): ").strip()
                if not user_input:
                    return default_value
            else:
                user_input = input(f"{prompt}: ").strip()

            if validation_fn:
                try:
                    validated_value = validation_fn(user_input)
                    return validated_value
                except ValueError as e:
                    print(f"Invalid input: {e}")
            else:
                return user_input

    @staticmethod
    def _validate_boolean(value: str) -> bool:
        """Validate and convert string to boolean"""
        value = value.lower()
        if value in ('true', 't', 'yes', 'y', '1'):
            return True
        elif value in ('false', 'f', 'no', 'n', '0'):
            return False
        raise ValueError("Please enter 'yes' or 'no'")

    @staticmethod
    def _validate_int(value: str) -> int:
        """Validate and convert string to integer"""
        try:
            return int(value)
        except ValueError:
            raise ValueError("Please enter a valid integer")

    @staticmethod
    def _validate_float(value: str) -> float:
        """Validate and convert string to float"""
        try:
            return float(value)
        except ValueError:
            raise ValueError("Please enter a valid number")

    @staticmethod
    def _prompt_for_config(dataset_name: str) -> Dict:
        """Prompt user for configuration parameters"""
        print(f"\nConfiguration file for {dataset_name} not found or invalid.")
        print("Please provide the following configuration parameters:\n")

        config = {}

        # Get file path
        config['file_path'] = DatasetConfig._get_user_input(
            "Enter the path to your CSV file",
            f"{dataset_name}.csv"
        )

        # Get target column
        target_column = DatasetConfig._get_user_input(
            "Enter the name or index of the target column",
            "target"
        )
        # Convert to integer if possible
        try:
            config['target_column'] = int(target_column)
        except ValueError:
            config['target_column'] = target_column

        # Get separator
        config['separator'] = DatasetConfig._get_user_input(
            "Enter the CSV separator character",
            ","
        )

        # Get header information
        config['has_header'] = DatasetConfig._get_user_input(
            "Does the file have a header row? True/False)",
            True,
            DatasetConfig._validate_boolean
        )

        # Get training configuration
        print("\nTraining Configuration:")
        config['training_config'] = {
            'trials': DatasetConfig._get_user_input(
                "Enter number of epochs to wait for improvement",
                100,
                DatasetConfig._validate_int
            ),
            'cardinality_threshold': DatasetConfig._get_user_input(
                "Enter cardinality threshold for feature filtering",
                0.9,
                DatasetConfig._validate_float
            ),
            'cardinality_tolerance': DatasetConfig._get_user_input(
                "Enter cardinality tolerance (decimal precision)",
                4,
                DatasetConfig._validate_int
            ),
            'learning_rate': DatasetConfig._get_user_input(
                "Enter learning rate",
                0.1,
                DatasetConfig._validate_float
            ),
            'random_seed': DatasetConfig._get_user_input(
                "Enter random seed (or -1 for no seed)",
                42,
                lambda x: int(x) if int(x) >= 0 else None
            ),
            'epochs': DatasetConfig._get_user_input(
                "Enter maximum number of epochs",
                1000,
                DatasetConfig._validate_int
            ),
            'test_fraction': DatasetConfig._get_user_input(
                "Enter test set fraction",
                0.2,
                DatasetConfig._validate_float
            ),
            'train': DatasetConfig._get_user_input(
                "Enable training?( True/False)",
                True,
                DatasetConfig._validate_boolean
            ),
            'train_only': DatasetConfig._get_user_input(
                "Train only (no evaluation)? (True/False)",
                False,
                DatasetConfig._validate_boolean
            ),
            'predict': DatasetConfig._get_user_input(
                "Enable prediction/evaluation? (True/Falseo)",
                True,
                DatasetConfig._validate_boolean
            ),
            'gen_samples': DatasetConfig._get_user_input(
                "Enable sample generation? (True/False)",
                False,
                DatasetConfig._validate_boolean
            )
        }

        # Get likelihood configuration
        print("\nLikelihood Configuration:")
        config['likelihood_config'] = {
            'feature_group_size': DatasetConfig._get_user_input(
                "Enter the feature group size",
                2,
                DatasetConfig._validate_int
            ),
            'max_combinations': DatasetConfig._get_user_input(
                "Enter the maximum number of feature combinations",
                1000,
                DatasetConfig._validate_int
            ),
            'update_strategy': DatasetConfig._get_user_input(
                "Enter update strategy (1: batch average, 2: max error add to failed, 3: max error add/subtract, 4: max error subtract only)",
                3,
                lambda x: int(x) if int(x) in [1, 2, 3, 4] else ValueError("Must be 1, 2, 3, or 4")
            ),
            'model_type': DatasetConfig._get_user_input(
                "Enter model type (gaussian/histogram)",
                "gaussian",
                lambda x: x if x in ['gaussian', 'histogram'] else ValueError("Must be 'gaussian' or 'histogram'")
            ),
            'histogram_bins': DatasetConfig._get_user_input(
                "Enter number of bins for histogram model",
                64,
                DatasetConfig._validate_int
            ),
            'laplace_smoothing': DatasetConfig._get_user_input(
                "Enter Laplace smoothing parameter for histogram",
                1.0,
                DatasetConfig._validate_float
            ),
            'histogram_method': DatasetConfig._get_user_input(
                "Enter histogram method (traditional/vectorized)",
                "traditional",
                lambda x: x if x in ['traditional', 'vectorized'] else ValueError("Must be 'traditional' or 'vectorized'")
            )
        }
        # Get visualization configuration
        print("\nVisualization Configuration:")
        config['visualization_config'] = {
            'enabled': DatasetConfig._get_user_input(
                "Enable visualization? (yes/no) - WARNING: This may slow down training",
                "no",
                DatasetConfig._validate_boolean
            ),
            'output_dir': DatasetConfig._get_user_input(
                "Enter output directory for visualizations",
                "visualizations"
            ),
            'epoch_interval': DatasetConfig._get_user_input(
                "Enter epoch interval for animation frames",
                10,
                DatasetConfig._validate_int
            ),
            'max_frames': DatasetConfig._get_user_input(
                "Enter maximum number of animation frames",
                50,
                DatasetConfig._validate_int
            ),
            'create_animations': DatasetConfig._get_user_input(
                "Create animated GIFs? (True/False)",
               True,
                DatasetConfig._validate_boolean
            ),
            'create_reports': DatasetConfig._get_user_input(
                "Create comprehensive reports? (True/False)",
                True,
                DatasetConfig._validate_boolean
            ),
            'create_3d_visualizations': DatasetConfig._get_user_input(  # NEW
                "Create 3D VR-style visualizations? (True/False)",
                True,
                DatasetConfig._validate_boolean
            ),
            'rotation_speed': DatasetConfig._get_user_input(  # NEW
                "Enter rotation speed (degrees per frame)",
                5,
                DatasetConfig._validate_int
            )
        }

        return config

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Load configuration from file, ignoring comments starting with #"""
        config_path = f"{dataset_name}.conf"

        try:
            if os.path.exists(config_path):
                # Read existing configuration
                with open(config_path, 'r') as f:
                    # Skip lines starting with # and join remaining lines
                    config_str = ''.join(line for line in f if not line.strip().startswith('#'))

                    try:
                        config = json.load(StringIO(config_str))
                    except json.JSONDecodeError:
                        print(f"Error reading configuration file: {config_path}")
                        config = DatasetConfig._prompt_for_config(dataset_name)
            else:
                config = DatasetConfig._prompt_for_config(dataset_name)

            # Validate and ensure all required fields exist
            required_fields = ['file_path', 'target_column', 'separator', 'has_header']
            missing_fields = [field for field in required_fields if field not in config]

            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
                config = DatasetConfig._prompt_for_config(dataset_name)

            # Ensure training_config exists with defaults
            if 'training_config' not in config:
                config['training_config'] = DatasetConfig.DEFAULT_CONFIG['training_config']
            else:
                # Ensure all training_config fields exist
                default_training = DatasetConfig.DEFAULT_CONFIG['training_config']
                for key, default_value in default_training.items():
                    if key not in config['training_config']:
                        config['training_config'][key] = default_value

            # Ensure likelihood_config exists with defaults
            if 'likelihood_config' not in config:
                config['likelihood_config'] = DatasetConfig.DEFAULT_CONFIG['likelihood_config']
            else:
                # Ensure all likelihood_config fields exist
                default_likelihood = DatasetConfig.DEFAULT_CONFIG['likelihood_config']
                for key, default_value in default_likelihood.items():
                    if key not in config['likelihood_config']:
                        config['likelihood_config'][key] = default_value

            # NEW: Ensure visualization_config exists with defaults (even if not in file)
            if 'visualization_config' not in config:
                config['visualization_config'] = DatasetConfig.DEFAULT_CONFIG['visualization_config']
            else:
                # Ensure all visualization_config fields exist
                default_visualization = DatasetConfig.DEFAULT_CONFIG['visualization_config']
                for key, default_value in default_visualization.items():
                    if key not in config['visualization_config']:
                        config['visualization_config'][key] = default_value

            # Save the configuration (this will add the new visualization_config if it was missing)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                print(f"\nConfiguration saved to: {config_path}")

            return config

        except Exception as e:
            print(f"Error handling configuration: {str(e)}")
            return DatasetConfig._prompt_for_config(dataset_name)

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available dataset configurations"""
        # Look for .conf files in the current directory
        return [f.split('.')[0] for f in os.listdir()
                if f.endswith('.conf') and os.path.isfile(f)]

#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame columns based on commented features in config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with filtered columns
    """
    # If no column names in config, return original DataFrame
    if 'column_names' not in config:
        return df

    # Get column names from config
    column_names = config['column_names']

    # Create mapping of position to column name
    col_mapping = {i: name.strip() for i, name in enumerate(column_names)}

    # Identify commented features (starting with #)
    commented_features = {
        i: name.lstrip('#').strip()
        for i, name in col_mapping.items()
        if name.startswith('#')
    }

    # Get current DataFrame columns
    current_cols = df.columns.tolist()

    # Columns to drop (either by name or position)
    cols_to_drop = []

    for pos, name in commented_features.items():
        # Try to drop by name first
        if name in current_cols:
            cols_to_drop.append(name)
        # If name not found, try position
        elif pos < len(current_cols):
            cols_to_drop.append(current_cols[pos])

    # Drop identified columns
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped commented features: {cols_to_drop}")

    return df
#-----------------------------------------------------------------------------------------------------------

class GPUDBNN:
    """Memory-Optimized Deep Bayesian Neural Network with CPU-friendly operations"""

    def __init__(
        self,
        dataset_name: str,
        device: str = None,
        config: Dict = None
    ):
        # Set device based on availability
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        self.use_gpu = self.device.type == 'cuda'

        self.dataset_name = dataset_name.lower()

        # Load dataset configuration and data - MODIFIED SECTION
        if config is not None:
            # Use the provided config directly
            self.config = config
            print(f"✅ Using provided configuration for {self.dataset_name}")
        else:
            # Load config from file as before
            self.config = DatasetConfig.load_config(self.dataset_name)


        # Initialize pruning structures
        self.active_feature_mask = None
        self.original_feature_indices = None

        # Extract training configuration
        training_config = self.config.get('training_config', {})
        self.trials = training_config.get('trials', 100)
        self.cardinality_threshold = training_config.get('cardinality_threshold', 0.9)
        self.cardinality_tolerance = training_config.get('cardinality_tolerance', 4)
        self.learning_rate = training_config.get('learning_rate', 0.1)
        self.random_state = training_config.get('random_seed', 42)
        self.max_epochs = training_config.get('epochs', 1000)
        self.test_size = training_config.get('test_fraction', 0.2)
        self.train_enabled = training_config.get('train', True)
        self.train_only = training_config.get('train_only', False)
        self.predict_enabled = training_config.get('predict', True)
        self.gen_samples_enabled = training_config.get('gen_samples', False)

        # Extract likelihood configuration
        likelihood_config = self.config.get('likelihood_config', {})
        self.model_type = likelihood_config.get('model_type', 'histogram')
        self.histogram_bins = likelihood_config.get('histogram_bins', 64)
        self.laplace_smoothing = likelihood_config.get('laplace_smoothing', 1.0)
        self.histogram_method = likelihood_config.get('histogram_method', 'vectorized')

        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.target_column = self.config['target_column']

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None
        self.best_accuracy = 0.0

        # Histogram model components
        self.histograms = None
        self.feature_min = None
        self.feature_max = None
        self.bin_edges = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load data
        self.data = self._load_dataset()

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

        # Add tracking for initial weights and pruning
        self.initial_W = None  # Store initial weights for reference
        self.pruning_warmup_epochs = 3  # epochs before starting to prune
        self.pruning_threshold = 1e-6   # minimum change from initial value to avoid pruning
        self.pruning_aggressiveness = 0.1  # percentage of stagnant connections to prune per epoch

        # Extract visualization configuration
        visualization_config = self.config.get('visualization_config', {})
        self.visualization_enabled = visualization_config.get('enabled', False)
        self.visualization_output_dir = visualization_config.get('output_dir', 'visualizations')
        self.visualization_output_dir = os.path.join(
            visualization_config.get('output_dir', 'visualizations'),
            self.dataset_name
        )
        self.visualization_epoch_interval = visualization_config.get('epoch_interval', 10)
        self.visualization_max_frames = visualization_config.get('max_frames', 50)

        print(f"Using {self.model_type} model for likelihood estimation")
        if self.model_type == 'histogram':
            print(f"Histogram method: {self.histogram_method}")

        # Import and initialize visualizer (conditional import)
        try:
            from dbnn_visualizer import DBNNVisualizer
            self.visualizer = DBNNVisualizer(
                self,
                output_dir=self.visualization_output_dir,
                enabled=self.visualization_enabled
            )
            self.visualizer.configure(visualization_config)
        except ImportError:
            print("Warning: Visualization module not available. Continuing without visualization.")
            self.visualizer = None
            self.visualization_enabled = False

        print(f"Using {self.model_type} model for likelihood estimation")
        if self.model_type == 'histogram':
            print(f"Histogram method: {self.histogram_method}")
        if self.visualization_enabled:
            print("Visualization: ENABLED")
        else:
            print("Visualization: DISABLED (default for performance)")
#-------------------

    def _get_model_filename(self) -> str:
        """Get model filename from config or use default"""
        if ('model_config' in self.config and
            'model_filename' in self.config['model_config']):
            return self.config['model_config']['model_filename']
        else:
            # Fallback to original behavior
            return f"Best_{self.dataset_name}"

    def _get_weights_filename(self):
        """Get the filename for saving/loading weights using config model_filename"""
        model_filename = self._get_model_filename()
        return os.path.join('Model', f'{model_filename}_weights.json')

    def _get_encoders_filename(self):
        """Get encoders filename using config model_filename"""
        model_filename = self._get_model_filename()
        return os.path.join('Model', f'{model_filename}_encoders.pkl')

    def _validate_feature_consistency(self, df: pd.DataFrame) -> bool:
        """Validate that current data features match training columns from config"""
        if 'training_columns' not in self.config:
            return True  # No training columns info, skip validation

        training_cols = self.config['training_columns']
        current_features = [col for col in df.columns if col != self.target_column]
        expected_features = training_cols.get('feature_columns', [])

        if set(current_features) != set(expected_features):
            print("❌ Feature mismatch detected!")
            print(f"   Expected: {expected_features}")
            print(f"   Current:  {current_features}")
            return False

        # Validate dataset fingerprint
        current_fingerprint = self._get_dataset_fingerprint(df)
        saved_fingerprint = training_cols.get('dataset_fingerprint')

        if saved_fingerprint and current_fingerprint != saved_fingerprint:
            print("❌ Dataset fingerprint mismatch!")
            print(f"   This may indicate the dataset has changed since training")
            return False

        return True

    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from file or URL with enhanced configuration"""
        file_path = self.config['file_path']
        separator = self.config['separator']
        has_header = self.config['has_header']

        print(f"Loading dataset from: {file_path}")

        try:
            # Check if file exists locally
            if os.path.exists(file_path):
                if has_header:
                    df = pd.read_csv(file_path, sep=separator)
                else:
                    df = pd.read_csv(file_path, sep=separator, header=None)
            else:
                # Try to load from URL
                response = requests.get(file_path)
                response.raise_for_status()

                if has_header:
                    df = pd.read_csv(StringIO(response.text), sep=separator)
                else:
                    df = pd.read_csv(StringIO(response.text), sep=separator, header=None)

            # Filter features based on commented column names in config
            df = _filter_features_from_config(df, self.config)

            # Handle missing values
            df = self._handle_missing_values(df)

            # Remove high cardinality columns
            df = self._remove_high_cardinality_columns(df, self._calculate_cardinality_threshold())

            # NEW: Ensure training_columns config exists and validate consistency
            self.config = DatasetConfig._ensure_training_columns_config(self.config, df)

            # Validate feature consistency for prediction
            if not self.train_enabled and self.predict_enabled:
                if not self._validate_feature_consistency(df):
                    print("⚠️  Feature inconsistency detected. Prediction may be unreliable.")
                    # You might want to return here or ask for user confirmation

            print(f"Dataset loaded with shape: {df.shape}")
            print(f"Model filename: {self._get_model_filename()}")
            return df

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    # -----------Enable/Disable pruning-------------
    def set_pruning_enabled(self, enabled: bool):
        """Enable or disable pruning"""
        if enabled:
            self.pruning_warmup_epochs = 3  # Default value
        else:
            self.pruning_warmup_epochs = 0  # Disables pruning

    def _calculate_cardinality_threshold(self):
        """Calculate the cardinality threshold based on the number of distinct classes"""
        return self.cardinality_threshold


    def _get_dataset_fingerprint(self, df: pd.DataFrame) -> str:
        """Generate a fingerprint of the dataset to detect changes"""
        import hashlib

        # Create a fingerprint based on column names and data types
        col_info = [(col, str(df[col].dtype)) for col in df.columns if col != self.target_column]
        col_info.sort()  # Sort for consistent ordering

        fingerprint_str = ''.join([f"{col}_{dtype}" for col, dtype in col_info])
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN and missing values by replacing with sentinel value (-9999)
        These will be filtered out during processing rather than imputed.
        """
        df_clean = df.copy()

        # Use a distinctive sentinel value that's unlikely to occur in real data
        SENTINEL_VALUE = -9999

        for column in df_clean.columns:
            if df_clean[column].isna().any():
                count_nan = df_clean[column].isna().sum()
                print(f"Found {count_nan} NaN values in column '{column}', replacing with sentinel value {SENTINEL_VALUE}")
                df_clean[column] = df_clean[column].fillna(SENTINEL_VALUE)

        return df_clean

    def _filter_sentinel_samples(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out samples that contain sentinel values (-9999)
        Returns filtered X and y arrays
        """
        SENTINEL_VALUE = -9999

        # Find rows without sentinel values
        valid_mask = ~np.any(X == SENTINEL_VALUE, axis=1)

        if not np.any(valid_mask):
            raise ValueError("All samples contain sentinel values - no valid data to process")

        X_filtered = X[valid_mask]

        if y is not None:
            y_filtered = y[valid_mask]
            filtered_count = len(X) - len(X_filtered)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} samples containing sentinel values")
            return X_filtered, y_filtered

        return X_filtered, None

    def _to_tensor(self, data, dtype=torch.float32):
        """Convert data to tensor and move to device"""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype).to(self.device)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype).to(self.device)
        else:
            tensor = torch.tensor(data, dtype=dtype).to(self.device)
        return tensor

    def _to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        return tensor

    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Remove columns with unique values exceeding the threshold percentage
        Uses configuration cardinality_threshold and cardinality_tolerance

        Args:
            df: Input DataFrame
            threshold: Maximum allowed percentage of unique values (uses config if None)

        Returns:
            DataFrame with high cardinality columns removed
        """
        # Use configured threshold if not provided
        if threshold is None:
            threshold = self.cardinality_threshold

        df_filtered = df.copy()
        columns_to_drop = []

        # First apply cardinality tolerance (rounding)
        df_rounded = df_filtered.round(self.cardinality_tolerance)

        print(f"🔍 Applying cardinality filtering: threshold={threshold}, tolerance={self.cardinality_tolerance}")

        for column in df_rounded.columns:
            # Skip target column
            if column == self.target_column:
                continue

            # Calculate percentage of unique values after rounding
            unique_ratio = len(df_rounded[column].unique()) / len(df_rounded)

            # Debug information
            n_unique = len(df_rounded[column].unique())
            n_total = len(df_rounded)

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                print(f"   🗑️  Dropping column '{column}': {n_unique}/{n_total} unique values ({unique_ratio:.3f} > {threshold})")
            else:
                print(f"   ✅ Keeping column '{column}': {n_unique}/{n_total} unique values ({unique_ratio:.3f} <= {threshold})")

        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            print(f"🗑️  Dropped {len(columns_to_drop)} high cardinality columns: {columns_to_drop}")
        else:
            print("✅ No high cardinality columns found")

        return df_filtered

    def _initialize_pruning_structures(self, n_pairs: int):
        """Initialize pruning structures with the correct size"""
        self.active_feature_mask = np.ones(n_pairs, dtype=bool)
        self.original_feature_indices = np.arange(n_pairs)
        print(f"Initialized pruning structures for {n_pairs} feature pairs")

    def _generate_feature_combinations(self, n_features: int, group_size: int, max_combinations: int = None) -> np.ndarray:
        """
        Generate feature combinations of specified size

        Args:
            n_features: Total number of features
            group_size: Number of features in each group
            max_combinations: Optional maximum number of combinations to use

        Returns:
            Array containing feature combinations
        """
        # Generate all possible combinations
        all_combinations = list(combinations(range(n_features), group_size))

        # If max_combinations specified and less than total combinations,
        # randomly sample combinations
        if max_combinations and len(all_combinations) > max_combinations:
            import random
            random.seed(self.random_state)
            all_combinations = random.sample(all_combinations, max_combinations)

        return np.array(all_combinations)

    def _compute_pairwise_likelihood_parallel(self, dataset: np.ndarray, labels: np.ndarray, feature_dims: int):
        """Compute likelihood parameters, filtering out sentinel values"""
        # Filter out samples with sentinel values
        SENTINEL_VALUE = -9999
        valid_mask = ~np.any(dataset == SENTINEL_VALUE, axis=1)
        dataset = dataset[valid_mask]
        labels = labels[valid_mask]

        if len(dataset) == 0:
            raise ValueError("No valid data after filtering sentinel values")

        # Get likelihood configuration
        group_size = self.config.get('likelihood_config', {}).get('feature_group_size', 2)
        max_combinations = self.config.get('likelihood_config', {}).get('max_combinations', None)

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            group_size,
            max_combinations
        )

        unique_classes = np.unique(labels)

        # Initialize storage for likelihood parameters
        n_combinations = len(self.feature_pairs)
        n_classes = len(unique_classes)

        # Preallocate arrays
        means = np.zeros((n_classes, n_combinations, group_size))
        covs = np.zeros((n_classes, n_combinations, group_size, group_size))

        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]

            if len(class_data) == 0:
                print(f"Warning: No data for class {class_id}, skipping")
                continue

            # Extract all feature groups
            group_data = np.stack([
                class_data[:, self.feature_pairs[i]] for i in range(n_combinations)
            ], axis=1)

            # Compute means for all groups
            means[class_idx] = np.mean(group_data, axis=0)

            # Compute covariances for all groups
            centered_data = group_data - means[class_idx][np.newaxis, :, :]

            for i in range(n_combinations):
                # Use efficient matrix multiplication for covariance
                batch_cov = np.dot(
                    centered_data[:, i].T,
                    centered_data[:, i]
                ) / max(1, (len(class_data) - 1))  # Avoid division by zero
                covs[class_idx, i] = batch_cov

            # Add small diagonal term for numerical stability
            covs[class_idx] += np.eye(group_size) * 1e-6

        # Convert to tensors if using GPU
        if self.use_gpu:
            means = self._to_tensor(means)
            covs = self._to_tensor(covs)
            unique_classes = self._to_tensor(unique_classes)

        return {
            'means': means,
            'covs': covs,
            'classes': unique_classes
        }

    def _compute_likelihood_parameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Compute likelihood parameters based on selected model type"""
        if self.model_type == 'histogram':
            if self.histogram_method == 'vectorized':
                self._compute_histogram_likelihood_vectorized(X_train, y_train)
            else:
                self._compute_histogram_likelihood_traditional(X_train, y_train)
        else:  # gaussian
            self._compute_gaussian_likelihood(X_train, y_train)

    def _compute_gaussian_likelihood(self, X_train: np.ndarray, y_train: np.ndarray):
        """Compute Gaussian likelihood parameters"""
        print("Computing Gaussian likelihood parameters...")

        # Get likelihood configuration
        group_size = self.config.get('likelihood_config', {}).get('feature_group_size', 2)
        max_combinations = self.config.get('likelihood_config', {}).get('max_combinations', None)

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            X_train.shape[1],
            group_size,
            max_combinations
        )

        # Compute likelihood parameters
        self.likelihood_params = self._compute_pairwise_likelihood_parallel(
            X_train, y_train, X_train.shape[1]
        )

        # Precompute inverse covariance matrices for faster prediction
        if self.use_gpu:
            covs = self.likelihood_params['covs']
            n_classes, n_combinations, _, _ = covs.shape
            eye = torch.eye(group_size, device=self.device)

            # Initialize inv_covs with proper shape
            inv_covs = torch.zeros_like(covs)

            for class_idx in range(n_classes):
                for comb_idx in range(n_combinations):
                    inv_covs[class_idx, comb_idx] = torch.linalg.inv(
                        covs[class_idx, comb_idx] + eye * 1e-6
                    )

            self.likelihood_params['inv_covs'] = inv_covs
        else:
            covs = self.likelihood_params['covs']
            n_classes, n_combinations, _, _ = covs.shape
            inv_covs = np.zeros_like(covs)
            for class_idx in range(n_classes):
                for comb_idx in range(n_combinations):
                    inv_covs[class_idx, comb_idx] = np.linalg.inv(
                        covs[class_idx, comb_idx] + np.eye(group_size) * 1e-6
                    )
            self.likelihood_params['inv_covs'] = inv_covs

        # Initialize weights if not already loaded
        if self.current_W is None:
            n_classes = len(self.likelihood_params['classes'])
            n_combinations = len(self.feature_pairs)
            self.current_W = np.ones((n_classes, n_combinations)) / n_classes
            self.best_W = self.current_W.copy()
            self.initial_W = self.current_W.copy()

            # Initialize pruning structures
            self._initialize_pruning_structures(n_combinations)

        print(f"Computed Gaussian parameters for {len(self.feature_pairs)} feature combinations")

    def _compute_histogram_likelihood_traditional(self, X_train: np.ndarray, y_train: np.ndarray):
        """Traditional histogram computation (original method)"""
        print(f"Computing histogram likelihood parameters with {self.histogram_bins} bins (traditional method)...")

        n_features = X_train.shape[1]
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)

        # Store feature ranges
        self.feature_min = np.min(X_train, axis=0)
        self.feature_max = np.max(X_train, axis=0)

        # Initialize histograms
        self.histograms = np.zeros((n_features, self.histogram_bins, n_classes))
        self.bin_edges = np.zeros((n_features, self.histogram_bins + 1))

        # Compute histograms for each feature and class (traditional loop-based)
        for feature_idx in range(n_features):
            # Compute bin edges for this feature
            self.bin_edges[feature_idx] = np.linspace(
                self.feature_min[feature_idx],
                self.feature_max[feature_idx],
                self.histogram_bins + 1
            )

            for class_idx, class_id in enumerate(unique_classes):
                # Get data for this class and feature
                class_mask = (y_train == class_id)
                class_data = X_train[class_mask, feature_idx]

                if len(class_data) > 0:
                    # Traditional histogram computation
                    hist, _ = np.histogram(class_data, bins=self.bin_edges[feature_idx])

                    # Apply Laplace smoothing
                    hist = hist + self.laplace_smoothing
                    total = np.sum(hist)

                    # Store normalized probabilities
                    self.histograms[feature_idx, :, class_idx] = hist / total

        # Initialize weights
        if self.current_W is None:
            self.current_W = np.ones((n_classes, n_features, self.histogram_bins))
            self.best_W = self.current_W.copy()
            self.initial_W = self.current_W.copy()

        # Initialize dummy pruning structures for histogram model
        self.active_feature_mask = np.ones(1, dtype=bool)
        self.original_feature_indices = np.array([0])
        self.feature_pairs = np.array([[0]])

        print(f"Computed histogram parameters for {n_features} features (traditional method)")

    def _compute_histogram_likelihood_vectorized(self, X_train: np.ndarray, y_train: np.ndarray):
        """Vectorized histogram computation using advanced NumPy operations"""
        print(f"Computing histogram likelihood parameters with {self.histogram_bins} bins (vectorized method)...")

        n_samples, n_features = X_train.shape
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)

        # Store feature ranges
        self.feature_min = np.min(X_train, axis=0)
        self.feature_max = np.max(X_train, axis=0)

        # Initialize histograms
        self.histograms = np.zeros((n_features, self.histogram_bins, n_classes))
        self.bin_edges = np.zeros((n_features, self.histogram_bins + 1))

        # Precompute bin edges for all features at once
        for feature_idx in range(n_features):
            self.bin_edges[feature_idx] = np.linspace(
                self.feature_min[feature_idx],
                self.feature_max[feature_idx],
                self.histogram_bins + 1
            )

        # Vectorized histogram computation using digitize and bincount
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (y_train == class_id)
            class_data = X_train[class_mask]

            if len(class_data) > 0:
                # Vectorized bin assignment for all features
                bin_indices = np.zeros((len(class_data), n_features), dtype=int)

                for feature_idx in range(n_features):
                    bin_indices[:, feature_idx] = np.digitize(
                        class_data[:, feature_idx],
                        self.bin_edges[feature_idx]
                    ) - 1
                    bin_indices[:, feature_idx] = np.clip(bin_indices[:, feature_idx], 0, self.histogram_bins - 1)

                # Vectorized histogram accumulation using bincount
                for feature_idx in range(n_features):
                    hist = np.bincount(bin_indices[:, feature_idx], minlength=self.histogram_bins)
                    hist = hist + self.laplace_smoothing  # Laplace smoothing
                    self.histograms[feature_idx, :, class_idx] = hist / np.sum(hist)

        # Initialize weights
        if self.current_W is None:
            self.current_W = np.ones((n_classes, n_features, self.histogram_bins))
            self.best_W = self.current_W.copy()
            self.initial_W = self.current_W.copy()

        # Initialize dummy pruning structures for histogram model
        self.active_feature_mask = np.ones(1, dtype=bool)
        self.original_feature_indices = np.array([0])
        self.feature_pairs = np.array([[0]])

        print(f"Computed histogram parameters for {n_features} features (vectorized method)")

    def _compute_batch_posterior(self, features: np.ndarray, epsilon: float = 1e-10):
        """Compute posterior probabilities based on selected model"""
        if self.model_type == 'histogram':
            if self.histogram_method == 'vectorized':
                return self._compute_histogram_posterior_vectorized(features, epsilon)
            else:
                return self._compute_histogram_posterior_traditional(features, epsilon)
        else:
            return self._compute_gaussian_posterior(features, epsilon)

    def _compute_gaussian_posterior(self, features: np.ndarray, epsilon: float = 1e-10):
        """Compute posterior probabilities for Gaussian model"""
        SENTINEL_VALUE = -9999

        # Identify samples with sentinel values
        sentinel_mask = np.any(features == SENTINEL_VALUE, axis=1)
        valid_features = features[~sentinel_mask]

        if len(valid_features) == 0:
            # Return uniform probabilities for all invalid samples
            n_classes = len(self.likelihood_params['classes'])
            uniform_probs = np.ones((len(features), n_classes)) / n_classes
            return uniform_probs

        batch_size = len(valid_features)
        n_classes = len(self.likelihood_params['classes'])
        n_combinations = len(self.feature_pairs)
        group_size = self.feature_pairs.shape[1]

        # Convert to tensors if using GPU
        if self.use_gpu:
            features_tensor = self._to_tensor(valid_features)
            means = self.likelihood_params['means']
            inv_covs = self.likelihood_params['inv_covs']
            current_W = self._to_tensor(self.current_W)
            feature_pairs = self._to_tensor(self.feature_pairs).long()

            # Extract all groups at once - [batch_size, n_combinations, group_size]
            batch_groups = features_tensor[:, feature_pairs]

            # Vectorized centered calculation - [n_classes, batch_size, n_combinations, group_size]
            centered = batch_groups.unsqueeze(0) - means.unsqueeze(1)

            # CORRECTED GPU einsum with proper indices:
            # centered: [c, b, i, j] where c=classes, b=batch, i=combinations, j=group_size
            # inv_covs: [c, i, j, k] where c=classes, i=combinations, j=group_size, k=group_size
            # Result: [c, b, i, k] -> [c, b, i, j] after second einsum
            temp = torch.einsum('cbij,cijk->cbik', centered, inv_covs)
            quad_form = torch.einsum('cbik,cbik->cbi', temp, centered)

            # Vectorized log determinant calculation (precomputed during likelihood computation)
            log_det = torch.log(torch.linalg.det(self.likelihood_params['covs'])).unsqueeze(1)

            # Compute log likelihood for all classes, batches, and combinations
            pair_log_likelihood = -0.5 * (
                group_size * np.log(2 * np.pi) +
                log_det +
                quad_form
            )

            # Add prior weights and sum over combinations
            weighted_likelihood = pair_log_likelihood + torch.log(current_W.unsqueeze(1) + epsilon)
            log_likelihoods = torch.sum(weighted_likelihood, dim=2)  # Sum over combinations

            # Compute posteriors using log-sum-exp trick
            max_log_likelihood = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
            likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
            valid_posteriors = likelihoods / (torch.sum(likelihoods, dim=1, keepdim=True) + epsilon)

            # Convert back to numpy and transpose to get (batch_size, n_classes)
            valid_posteriors = self._to_numpy(valid_posteriors).T

        else:
            # CPU implementation with optimized loops
            # Extract all feature groups at once - [batch_size, n_combinations, group_size]
            batch_groups = valid_features[:, self.feature_pairs]

            # Initialize log likelihoods
            log_likelihoods = np.zeros((batch_size, n_classes))

            for class_idx in range(n_classes):
                # Get parameters for current class
                class_means = self.likelihood_params['means'][class_idx]
                class_inv_covs = self.likelihood_params['inv_covs'][class_idx]
                class_priors = self.current_W[class_idx]

                # Compute centered data for all groups - [batch_size, n_combinations, group_size]
                centered = batch_groups - class_means[np.newaxis, :, :]

                # Compute quadratic form for all samples and groups using optimized loops
                quad_form = np.zeros((batch_size, n_combinations))
                for i in range(n_combinations):
                    # For each combination, compute the quadratic form across all samples
                    temp = np.dot(centered[:, i, :], class_inv_covs[i])
                    quad_form[:, i] = np.sum(temp * centered[:, i, :], axis=1)

                # Compute log determinant (precomputed during likelihood computation)
                log_det = np.log(np.linalg.det(self.likelihood_params['covs'][class_idx]))

                # Compute log likelihood for all groups
                pair_log_likelihood = -0.5 * (
                    group_size * np.log(2 * np.pi) +
                    log_det[np.newaxis, :] +
                    quad_form
                )

                # Add prior weights
                weighted_likelihood = pair_log_likelihood + np.log(class_priors + epsilon)

                # Sum over groups for each sample
                log_likelihoods[:, class_idx] = np.sum(weighted_likelihood, axis=1)

            # Compute posteriors using log-sum-exp trick
            max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
            likelihoods = np.exp(log_likelihoods - max_log_likelihood)
            valid_posteriors = likelihoods / (np.sum(likelihoods, axis=1, keepdims=True) + epsilon)

        # Create full posteriors array with uniform probabilities for invalid samples
        n_classes = len(self.likelihood_params['classes'])
        full_posteriors = np.ones((len(features), n_classes)) / n_classes

        # Ensure shapes match before assignment
        if valid_posteriors.shape[0] == np.sum(~sentinel_mask) and valid_posteriors.shape[1] == n_classes:
            full_posteriors[~sentinel_mask] = valid_posteriors
        else:
            # Handle shape mismatch by transposing if needed
            if valid_posteriors.shape[0] == n_classes and valid_posteriors.shape[1] == np.sum(~sentinel_mask):
                full_posteriors[~sentinel_mask] = valid_posteriors.T
            else:
                # Fallback: use uniform probabilities for all samples
                print(f"Warning: Shape mismatch in posterior computation. Using uniform probabilities.")
                print(f"Expected: ({np.sum(~sentinel_mask)}, {n_classes}), Got: {valid_posteriors.shape}")
                full_posteriors = np.ones((len(features), n_classes)) / n_classes

        return full_posteriors

    def _compute_histogram_posterior_traditional(self, features: np.ndarray, epsilon: float = 1e-10):
        """Traditional histogram posterior computation (loop-based)"""
        SENTINEL_VALUE = -9999

        # Identify samples with sentinel values
        sentinel_mask = np.any(features == SENTINEL_VALUE, axis=1)
        valid_features = features[~sentinel_mask]

        if len(valid_features) == 0:
            n_classes = self.histograms.shape[2]
            uniform_probs = np.ones((len(features), n_classes)) / n_classes
            return uniform_probs

        n_samples = len(valid_features)
        n_features = valid_features.shape[1]
        n_classes = self.histograms.shape[2]

        # Initialize posteriors (traditional loop-based approach)
        posteriors = np.ones((n_samples, n_classes))

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                value = valid_features[sample_idx, feature_idx]

                # Find which bin this value falls into
                bin_idx = np.digitize(value, self.bin_edges[feature_idx]) - 1
                bin_idx = np.clip(bin_idx, 0, self.histogram_bins - 1)

                # Get probabilities for this bin across all classes
                bin_probs = self.histograms[feature_idx, bin_idx, :]

                # Apply weights and multiply into posterior
                weighted_probs = bin_probs * self.current_W[:, feature_idx, bin_idx]
                posteriors[sample_idx] *= weighted_probs

        # Normalize posteriors
        posteriors = posteriors / (np.sum(posteriors, axis=1, keepdims=True) + epsilon)

        # Handle samples with sentinel values
        full_posteriors = np.ones((len(features), n_classes)) / n_classes
        full_posteriors[~sentinel_mask] = posteriors

        return full_posteriors

    def _compute_histogram_posterior_vectorized(self, features: np.ndarray, epsilon: float = 1e-10):
        """Vectorized histogram posterior computation"""
        SENTINEL_VALUE = -9999

        # Identify samples with sentinel values
        sentinel_mask = np.any(features == SENTINEL_VALUE, axis=1)
        valid_features = features[~sentinel_mask]

        if len(valid_features) == 0:
            n_classes = self.histograms.shape[2]
            uniform_probs = np.ones((len(features), n_classes)) / n_classes
            return uniform_probs

        n_samples = len(valid_features)
        n_features = valid_features.shape[1]
        n_classes = self.histograms.shape[2]

        if self.use_gpu:
            return self._compute_histogram_posterior_gpu(valid_features, n_samples, n_features, n_classes, epsilon, sentinel_mask, features)
        else:
            return self._compute_histogram_posterior_cpu_vectorized(valid_features, n_samples, n_features, n_classes, epsilon, sentinel_mask, features)

    def _compute_histogram_posterior_cpu_vectorized(self, valid_features, n_samples, n_features, n_classes, epsilon, sentinel_mask, original_features):
        """CPU-optimized vectorized posterior computation"""

        # Vectorized bin assignment for all samples and features
        bin_indices = np.zeros((n_samples, n_features), dtype=int)

        for feature_idx in range(n_features):
            bin_indices[:, feature_idx] = np.digitize(
                valid_features[:, feature_idx],
                self.bin_edges[feature_idx]
            ) - 1
            bin_indices[:, feature_idx] = np.clip(bin_indices[:, feature_idx], 0, self.histogram_bins - 1)

        # Vectorized probability lookup using advanced indexing
        posteriors = np.ones((n_samples, n_classes))

        for class_idx in range(n_classes):
            class_posterior = np.ones(n_samples)

            for feature_idx in range(n_features):
                # Get probabilities for all samples at once using advanced indexing
                feature_probs = self.histograms[feature_idx, bin_indices[:, feature_idx], class_idx]
                # Apply weights
                weighted_probs = feature_probs * self.current_W[class_idx, feature_idx, bin_indices[:, feature_idx]]
                class_posterior *= weighted_probs

            posteriors[:, class_idx] = class_posterior

        # Normalize posteriors
        posteriors = posteriors / (np.sum(posteriors, axis=1, keepdims=True) + epsilon)

        # Handle samples with sentinel values
        full_posteriors = np.ones((len(original_features), n_classes)) / n_classes
        full_posteriors[~sentinel_mask] = posteriors

        return full_posteriors

    def _compute_histogram_posterior_gpu(self, valid_features, n_samples, n_features, n_classes, epsilon, sentinel_mask, original_features):
        """GPU-accelerated histogram posterior computation"""
        try:
            # Convert to tensors
            features_tensor = self._to_tensor(valid_features)
            histograms_tensor = self._to_tensor(self.histograms)
            current_W_tensor = self._to_tensor(self.current_W)
            bin_edges_tensor = self._to_tensor(self.bin_edges)

            # Vectorized bin assignment on GPU
            bin_indices = torch.zeros((n_samples, n_features), dtype=torch.long, device=self.device)

            for feature_idx in range(n_features):
                # Find bins using GPU-accelerated search
                edges = bin_edges_tensor[feature_idx]
                # Expand dimensions for broadcasting
                features_expanded = features_tensor[:, feature_idx].unsqueeze(1)
                edges_expanded = edges.unsqueeze(0)

                # Find where feature values fall between bin edges
                in_bin = (features_expanded >= edges_expanded[:, :-1]) & (features_expanded < edges_expanded[:, 1:])
                bin_idx = torch.argmax(in_bin.int(), dim=1)
                bin_indices[:, feature_idx] = bin_idx

            # GPU-accelerated probability computation
            posteriors = torch.ones((n_samples, n_classes), device=self.device)

            for class_idx in range(n_classes):
                class_posterior = torch.ones(n_samples, device=self.device)

                for feature_idx in range(n_features):
                    # Advanced indexing on GPU
                    feature_probs = histograms_tensor[feature_idx, bin_indices[:, feature_idx], class_idx]
                    weights = current_W_tensor[class_idx, feature_idx, bin_indices[:, feature_idx]]
                    weighted_probs = feature_probs * weights
                    class_posterior *= weighted_probs

                posteriors[:, class_idx] = class_posterior

            # Normalize on GPU
            posteriors = posteriors / (torch.sum(posteriors, dim=1, keepdim=True) + epsilon)

            # Convert back to numpy and handle sentinel values
            valid_posteriors = self._to_numpy(posteriors)
            full_posteriors = np.ones((len(original_features), n_classes)) / n_classes
            full_posteriors[~sentinel_mask] = valid_posteriors

            return full_posteriors

        except Exception as e:
            print(f"GPU computation failed, falling back to CPU: {e}")
            return self._compute_histogram_posterior_cpu_vectorized(valid_features, n_samples, n_features, n_classes, epsilon, sentinel_mask, original_features)

    # ============Ensure all data is on the same device =============
    def _ensure_numpy(self, data):
        """Convert data to numpy array if it's a tensor"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data
    # ============Ensure all data is on the same device =============

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Update priors based on selected model type"""
        if self.model_type == 'histogram':
            if self.histogram_method == 'vectorized':
                self._update_histogram_priors_vectorized(failed_cases, batch_size)
            else:
                self._update_histogram_priors_traditional(failed_cases, batch_size)
        else:
            self._update_gaussian_priors(failed_cases, batch_size)

    def _update_gaussian_priors(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Update priors ONLY for hypercubes that contributed to failures"""
        n_failed = len(failed_cases)

        # Get the update strategy from config
        update_strategy = self.config.get('likelihood_config', {}).get('update_strategy', 3)

        # Initialize data structures
        n_classes = len(self.likelihood_params['classes'])

        # Convert classes to numpy array if it's a tensor
        if self.use_gpu:
            classes_numpy = self._to_numpy(self.likelihood_params['classes'])
        else:
            classes_numpy = self.likelihood_params['classes']

        # Process failed cases in batches for efficiency
        for batch_start in range(0, n_failed, batch_size):
            batch_end = min(batch_start + batch_size, n_failed)
            batch_cases = failed_cases[batch_start:batch_end]

            # Extract batch data
            batch_features = np.array([case[0] for case in batch_cases])
            batch_true_classes = np.array([case[1] for case in batch_cases])
            batch_posteriors = np.array([case[2] for case in batch_cases])

            # Get predicted classes
            batch_predicted_classes = np.argmax(batch_posteriors, axis=1)

            for i in range(len(batch_cases)):
                features, true_class, posteriors = batch_cases[i]
                predicted_class = batch_predicted_classes[i]

                # Get the true class index
                true_class_idx = np.where(classes_numpy == true_class)[0][0]

                # Only update if it's a genuine misclassification
                if predicted_class == true_class_idx:
                    continue

                # For each feature pair, calculate its contribution to the error
                feature_contributions = self._calculate_feature_contributions(
                    features, true_class_idx, predicted_class
                )

                # Compute adjustment based on error magnitude
                true_prob = posteriors[true_class_idx]
                max_other_prob = np.max(np.delete(posteriors, true_class_idx))
                adjustment = self.learning_rate * (1 - true_prob / max_other_prob)
                adjustment = np.clip(adjustment, -0.5, 0.5)

                # Apply the selected strategy to relevant hypercubes only
                if update_strategy == 1:
                    # Strategy 1: Batch average update - update ALL hypercubes for the class
                    self.current_W[true_class_idx] *= (1 + adjustment)
                    self.current_W[predicted_class] *= (1 - adjustment)

                elif update_strategy == 2:
                    # Strategy 2: Maximum error applied to failed class hypercubes
                    self._update_strategy2(feature_contributions, true_class_idx, predicted_class, adjustment)

                elif update_strategy == 3:
                    # Strategy 3: Add to correct class, subtract from wrong class for relevant hypercubes
                    self._update_strategy3(feature_contributions, true_class_idx, predicted_class, adjustment)

                elif update_strategy == 4:
                    # Strategy 4: Subtract only from wrong class hypercubes
                    self._update_strategy4(feature_contributions, true_class_idx, predicted_class, adjustment)

            # Clip weights for stability after each batch
            self.current_W = np.clip(self.current_W, 1e-10, 10.0)

    def _update_histogram_priors_traditional(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Traditional weight updates for histogram model"""
        n_failed = len(failed_cases)
        update_strategy = self.config.get('likelihood_config', {}).get('update_strategy', 3)

        # Process failed cases in batches
        for batch_start in range(0, n_failed, batch_size):
            batch_end = min(batch_start + batch_size, n_failed)
            batch_cases = failed_cases[batch_start:batch_end]

            for features, true_class, posteriors in batch_cases:
                predicted_class = np.argmax(posteriors)

                if predicted_class == true_class:
                    continue

                # Calculate adjustment based on error magnitude
                true_prob = posteriors[true_class]
                predicted_prob = posteriors[predicted_class]
                adjustment = self.learning_rate * (1 - true_prob / (predicted_prob + 1e-10))
                adjustment = np.clip(adjustment, -0.5, 0.5)

                # Update weights for each feature (traditional loop)
                for feature_idx in range(len(features)):
                    value = features[feature_idx]

                    # Find bin for this feature value
                    bin_idx = np.digitize(value, self.bin_edges[feature_idx]) - 1
                    bin_idx = np.clip(bin_idx, 0, self.histogram_bins - 1)

                    # Apply update strategy
                    if update_strategy == 1:
                        # Update all bins
                        self.current_W[true_class, feature_idx, :] *= (1 + adjustment)
                        self.current_W[predicted_class, feature_idx, :] *= (1 - adjustment)
                    elif update_strategy == 2:
                        # Update only the specific bin
                        self.current_W[true_class, feature_idx, bin_idx] *= (1 + adjustment)
                        self.current_W[predicted_class, feature_idx, bin_idx] *= (1 - adjustment)
                    elif update_strategy == 3:
                        # Update specific bin with different rates
                        self.current_W[true_class, feature_idx, bin_idx] *= (1 + adjustment)
                        self.current_W[predicted_class, feature_idx, bin_idx] *= (1 - adjustment * 0.5)
                    elif update_strategy == 4:
                        # Only decrease wrong class
                        self.current_W[predicted_class, feature_idx, bin_idx] *= (1 - adjustment)

                # Clip weights for stability
                self.current_W = np.clip(self.current_W, 1e-10, 10.0)

    def _update_histogram_priors_vectorized(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Vectorized weight updates for histogram model"""
        n_failed = len(failed_cases)
        update_strategy = self.config.get('likelihood_config', {}).get('update_strategy', 3)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 32)) as executor:
            # Process batches in parallel
            futures = []
            for batch_start in range(0, n_failed, batch_size):
                batch_end = min(batch_start + batch_size, n_failed)
                batch_cases = failed_cases[batch_start:batch_end]

                future = executor.submit(self._process_batch_weight_updates_vectorized, batch_cases, update_strategy)
                futures.append(future)

            # Collect results
            for future in futures:
                future.result()

    def _process_batch_weight_updates_vectorized(self, batch_cases, update_strategy):
        """Process a batch of weight updates in vectorized manner"""
        batch_updates = []

        for features, true_class, posteriors in batch_cases:
            predicted_class = np.argmax(posteriors)

            if predicted_class == true_class:
                continue

            # Calculate adjustment
            true_prob = posteriors[true_class]
            predicted_prob = posteriors[predicted_class]
            adjustment = self.learning_rate * (1 - true_prob / (predicted_prob + 1e-10))
            adjustment = np.clip(adjustment, -0.5, 0.5)

            # Collect updates for batch processing
            for feature_idx in range(len(features)):
                value = features[feature_idx]
                bin_idx = np.digitize(value, self.bin_edges[feature_idx]) - 1
                bin_idx = np.clip(bin_idx, 0, self.histogram_bins - 1)

                batch_updates.append((true_class, predicted_class, feature_idx, bin_idx, adjustment, update_strategy))

        # Apply all updates at once (vectorized)
        if batch_updates:
            self._apply_batch_weight_updates_vectorized(batch_updates)

    def _apply_batch_weight_updates_vectorized(self, batch_updates):
        """Apply weight updates in a vectorized manner"""
        # Group updates by type for vectorized operations
        true_class_indices = []
        pred_class_indices = []
        true_adjustments = []
        pred_adjustments = []

        for true_class, predicted_class, feature_idx, bin_idx, adjustment, update_strategy in batch_updates:
            if update_strategy in [1, 2, 3]:
                true_class_indices.append((true_class, feature_idx, bin_idx))
                true_adjustments.append(adjustment)

            if update_strategy in [1, 2, 3, 4]:
                decay_adjustment = adjustment if update_strategy != 3 else adjustment * 0.5
                pred_class_indices.append((predicted_class, feature_idx, bin_idx))
                pred_adjustments.append(-decay_adjustment)

        # Vectorized weight updates
        if true_class_indices:
            classes, features, bins = zip(*true_class_indices)
            indices = (np.array(classes), np.array(features), np.array(bins))
            adjustments_array = 1 + np.array(true_adjustments)
            # Use advanced indexing to update only the required elements
            self.current_W[indices] *= adjustments_array

        if pred_class_indices:
            classes, features, bins = zip(*pred_class_indices)
            indices = (np.array(classes), np.array(features), np.array(bins))
            adjustments_array = 1 + np.array(pred_adjustments)
            self.current_W[indices] *= adjustments_array

        # Clip weights
        self.current_W = np.clip(self.current_W, 1e-10, 10.0)

    def _compute_batch_pair_log_likelihood_gpu(self, feature_groups, class_idx):
        """Compute log-likelihood for a batch of feature pairs and specific class on GPU"""
        group_size = feature_groups.shape[1]
        n_pairs = len(self.feature_pairs)

        # Get parameters for the class
        means = self.likelihood_params['means'][class_idx]
        inv_covs = self.likelihood_params['inv_covs'][class_idx]

        # Center the data
        centered = feature_groups - means

        # Compute quadratic form using vectorization
        temp = torch.einsum('bi,bij->bj', centered, inv_covs)
        quad_form = torch.einsum('bi,bi->b', temp, centered)

        # Compute log determinant
        log_det = torch.log(torch.linalg.det(self.likelihood_params['covs'][class_idx]))

        # Compute log likelihood
        log_likelihood = -0.5 * (
            group_size * np.log(2 * np.pi) +
            log_det +
            quad_form
        )

        return log_likelihood

    def _calculate_feature_contributions(self, features, true_class_idx, predicted_class_idx):
        """Calculate how much each feature pair contributed to the misclassification"""
        n_pairs = len(self.feature_pairs)
        contributions = np.zeros(n_pairs)

        # Extract the feature values for all pairs at once
        feature_groups = features[self.feature_pairs]  # Shape: [n_pairs, group_size]

        if self.use_gpu:
            # GPU-optimized batch calculation
            feature_groups_tensor = self._to_tensor(feature_groups)

            # Compute for true class
            true_log_likelihoods = self._compute_batch_pair_log_likelihood_gpu(
                feature_groups_tensor, true_class_idx
            )

            # Compute for predicted class
            pred_log_likelihoods = self._compute_batch_pair_log_likelihood_gpu(
                feature_groups_tensor, predicted_class_idx
            )

            contributions = self._to_numpy(pred_log_likelihoods - true_log_likelihoods)

        else:
            # CPU-optimized batch calculation
            for pair_idx in range(n_pairs):
                # Get log-likelihood for true class for this pair
                true_log_likelihood = self._compute_pair_log_likelihood(
                    feature_groups[pair_idx], true_class_idx, pair_idx
                )

                # Get log-likelihood for predicted class for this pair
                pred_log_likelihood = self._compute_pair_log_likelihood(
                    feature_groups[pair_idx], predicted_class_idx, pair_idx
                )

                # Contribution is the difference (how much this pair favored wrong class)
                contributions[pair_idx] = pred_log_likelihood - true_log_likelihood

        return contributions

    def _compute_pair_log_likelihood(self, feature_values, class_idx, pair_idx):
        """Compute log-likelihood for a specific feature pair and class"""
        group_size = len(feature_values)

        # Convert to appropriate format based on device
        if self.use_gpu:
            feature_values = self._to_tensor(feature_values)
            mean = self.likelihood_params['means'][class_idx, pair_idx]
            cov = self.likelihood_params['covs'][class_idx, pair_idx]
        else:
            mean = self.likelihood_params['means'][class_idx, pair_idx]
            cov = self.likelihood_params['covs'][class_idx, pair_idx]

        centered = feature_values - mean

        # Compute quadratic form
        try:
            if self.use_gpu:
                inv_cov = torch.linalg.inv(cov)
                quad_form = torch.dot(centered, torch.matmul(inv_cov, centered))
                log_det = torch.log(torch.linalg.det(cov))
                log_likelihood = -0.5 * (group_size * np.log(2 * np.pi) + log_det + quad_form)
                return self._to_numpy(log_likelihood)
            else:
                inv_cov = np.linalg.inv(cov)
                quad_form = np.dot(centered, np.dot(inv_cov, centered))
                log_det = np.log(np.linalg.det(cov))
                return -0.5 * (group_size * np.log(2 * np.pi) + log_det + quad_form)
        except:
            # Handle singular matrices
            return -np.inf

    def _update_strategy2(self, contributions, true_class_idx, predicted_class_idx, adjustment):
        """Strategy 2: Apply maximum error adjustment to hypercubes that contributed most to failure"""
        # Find the hypercube that contributed most to the error
        max_contribution_idx = np.argmax(contributions)

        # Only update the most problematic hypercube
        self.current_W[true_class_idx, max_contribution_idx] *= (1 + adjustment)
        self.current_W[predicted_class_idx, max_contribution_idx] *= (1 - adjustment)

        # Clip weights for stability
        self.current_W = np.clip(self.current_W, 1e-10, 10.0)

    def _update_strategy3(self, contributions, true_class_idx, predicted_class_idx, adjustment):
        """Strategy 3: Add to correct class, subtract from wrong class for relevant hypercubes"""
        # Update all hypercubes that contributed positively to the error
        for pair_idx in range(len(contributions)):
            if contributions[pair_idx] > 0:  # This hypercube favored the wrong prediction
                self.current_W[true_class_idx, pair_idx] *= (1 + adjustment)
                self.current_W[predicted_class_idx, pair_idx] *= (1 - adjustment)

        # Clip weights for stability
        self.current_W = np.clip(self.current_W, 1e-10, 10.0)

    def _update_strategy4(self, contributions, true_class_idx, predicted_class_idx, adjustment):
        """Strategy 4: Subtract only from wrong class hypercubes"""
        # Only reduce weights for hypercubes that contributed to wrong prediction
        for pair_idx in range(len(contributions)):
            if contributions[pair_idx] > 0:  # This hypercube favored the wrong prediction
                self.current_W[predicted_class_idx, pair_idx] *= (1 - adjustment)

        # Clip weights for stability
        self.current_W = np.clip(self.current_W, 1e-10, 10.0)


    def _prune_feature_hypercubes(self, min_contribution_threshold: float = 0.01):
        """Prune feature hypercubes that don't contribute significantly to predictions"""
        if self.best_W is None:
            return

        # Calculate average weight per feature pair
        avg_weights = np.mean(self.best_W, axis=0)

        # Identify feature pairs with low contribution
        low_contribution_mask = avg_weights < min_contribution_threshold

        if np.any(low_contribution_mask):
            # Get indices of feature pairs to keep
            keep_indices = np.where(~low_contribution_mask)[0]

            # Create new active mask
            new_active_mask = np.zeros(len(self.feature_pairs), dtype=bool)
            new_active_mask[keep_indices] = True

            # Apply pruning
            self._apply_pruning(new_active_mask)

            print(f"Pruned {np.sum(low_contribution_mask)} feature hypercubes with low contribution")

    def _apply_pruning(self, new_active_mask):
        """Apply the pruning to all model components using the new mask"""
        if not np.any(~new_active_mask):  # Nothing to prune
            return

        # Initialize pruning structures if they don't exist
        if not hasattr(self, 'original_feature_indices') or self.original_feature_indices is None:
            self._initialize_pruning_structures(len(self.feature_pairs))

        if not hasattr(self, 'active_feature_mask') or self.active_feature_mask is None:
            self.active_feature_mask = np.ones(len(self.feature_pairs), dtype=bool)

        # Ensure the mask size matches
        if len(new_active_mask) != len(self.original_feature_indices):
            print(f"Warning: Pruning mask size {len(new_active_mask)} doesn't match "
                  f"original indices size {len(self.original_feature_indices)}. "
                  "Reinitializing pruning structures.")
            self._initialize_pruning_structures(len(self.feature_pairs))
            new_active_mask = np.ones(len(self.feature_pairs), dtype=bool)  # Reset to no pruning

        # Update the active feature mask
        self.active_feature_mask = new_active_mask

        # Prune current weights
        self.current_W = self.current_W[:, new_active_mask]

        # Prune best weights if they exist
        if self.best_W is not None:
            self.best_W = self.best_W[:, new_active_mask]

        # Prune initial weights reference
        if self.initial_W is not None:
            self.initial_W = self.initial_W[:, new_active_mask]

        # Prune feature pairs
        self.feature_pairs = self.feature_pairs[new_active_mask]

        # Prune likelihood parameters
        if self.likelihood_params is not None:
            self.likelihood_params['means'] = self.likelihood_params['means'][:, new_active_mask, :]
            self.likelihood_params['covs'] = self.likelihood_params['covs'][:, new_active_mask, :, :]

        # Update original indices tracking
        self.original_feature_indices = self.original_feature_indices[new_active_mask]

        print(f"After pruning: {len(self.feature_pairs)} feature pairs remaining")

        # Save the updated weights with pruning info
        self._save_best_weights()

    def _load_best_weights(self):
        """Load the best weights from file including ALL histogram parameters"""
        weights_file = self._get_weights_filename()

        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)

            # Check if dataset has changed (compare fingerprints)
            current_fingerprint = self._get_dataset_fingerprint(self.data)
            saved_fingerprint = weights_dict.get('dataset_fingerprint')

            if saved_fingerprint and saved_fingerprint != current_fingerprint:
                print(f"Dataset has changed since last training. Reinitializing weights.")
                self.best_W = None
                return

            try:
                version = weights_dict.get('version', 1)

                if version >= 5:
                    # New comprehensive format
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = weights_array.astype(np.float32)
                    self.current_W = self.best_W.copy()

                    # Load model type information
                    self.model_type = weights_dict.get('model_type', 'histogram')
                    self.histogram_bins = weights_dict.get('histogram_bins', 64)
                    self.histogram_method = weights_dict.get('histogram_method', 'vectorized')

                    # NEW: Load histogram parameters if available
                    if self.model_type == 'histogram':
                        if 'histogram_params' in weights_dict:
                            hist_params = weights_dict['histogram_params']
                            self.histograms = np.array(hist_params['histograms'])
                            self.bin_edges = np.array(hist_params['bin_edges'])
                            self.feature_min = np.array(hist_params['feature_min'])
                            self.feature_max = np.array(hist_params['feature_max'])
                            print(f"✅ Loaded histogram parameters: {self.histograms.shape}")
                        else:
                            print("⚠️ No histogram parameters found in weights file")

                    # Load pruning information
                    pruning_info = weights_dict.get('pruning_info', {})
                    if pruning_info.get('active_feature_mask'):
                        self.active_feature_mask = np.array(pruning_info['active_feature_mask'], dtype=bool)
                    if pruning_info.get('original_feature_indices'):
                        self.original_feature_indices = np.array(pruning_info['original_feature_indices'])
                    if weights_dict.get('feature_pairs'):
                        self.feature_pairs = np.array(weights_dict['feature_pairs'])

                    # Load critical parameters for prediction
                    if 'label_encoder_classes' in weights_dict and weights_dict['label_encoder_classes']:
                        self.label_encoder.classes_ = np.array(weights_dict['label_encoder_classes'])
                        print(f"✅ Loaded label encoder with {len(self.label_encoder.classes_)} classes")

                    if 'scaler_mean' in weights_dict and weights_dict['scaler_mean']:
                        self.scaler.mean_ = np.array(weights_dict['scaler_mean'])
                        self.scaler.scale_ = np.array(weights_dict['scaler_scale'])
                        # Set other required attributes for sklearn scaler
                        self.scaler.var_ = self.scaler.scale_ ** 2
                        self.scaler.n_features_in_ = len(self.scaler.mean_)
                        self.scaler.n_samples_seen_ = len(self.data) if hasattr(self, 'data') else 1
                        print("✅ Loaded scaler parameters")

                    print(f"✅ Loaded {self.model_type} model with all parameters from {weights_file}")

                elif version == 4:
                    # Old format - try to load but warn about missing parameters
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = weights_array.astype(np.float32)
                    self.current_W = self.best_W.copy()
                    print(f"⚠️ Loaded weights from version 4 - histogram parameters may be missing")

                else:
                    # Very old format - reinitialize
                    print(f"⚠️ Old weights format detected - reinitializing model")
                    self.best_W = None
                    return

                # For loaded weights, we can't track initial values, so disable pruning
                self.initial_W = None
                self.pruning_warmup_epochs = 0

            except Exception as e:
                print(f"Warning: Could not load weights from {weights_file}: {str(e)}")
                print("Reinitializing weights...")
                self.best_W = None
                self.current_W = None

    def _save_best_weights(self):
        """Save the best weights to file with ALL required parameters including histogram data"""
        if self.best_W is not None:
            # Handle None values for pruning structures
            active_mask = None
            original_indices = None
            feature_pairs_list = None

            if hasattr(self, 'active_feature_mask') and self.active_feature_mask is not None:
                active_mask = self.active_feature_mask.tolist()

            if hasattr(self, 'original_feature_indices') and self.original_feature_indices is not None:
                original_indices = self.original_feature_indices.tolist()

            if hasattr(self, 'feature_pairs') and self.feature_pairs is not None:
                feature_pairs_list = self.feature_pairs.tolist()

            # NEW: Prepare histogram parameters for saving
            histogram_params = {}
            if self.model_type == 'histogram':
                if (hasattr(self, 'histograms') and self.histograms is not None and
                    hasattr(self, 'bin_edges') and self.bin_edges is not None and
                    hasattr(self, 'feature_min') and self.feature_min is not None and
                    hasattr(self, 'feature_max') and self.feature_max is not None):

                    histogram_params = {
                        'histograms': self.histograms.tolist(),
                        'bin_edges': self.bin_edges.tolist(),
                        'feature_min': self.feature_min.tolist(),
                        'feature_max': self.feature_max.tolist()
                    }

            # Save ALL critical parameters
            weights_dict = {
                'version': 6,  # Incremented version for histogram support
                'weights': self.best_W.tolist(),
                'shape': list(self.best_W.shape),
                'dataset_fingerprint': self._get_dataset_fingerprint(self.data),
                'model_type': self.model_type,
                'histogram_bins': self.histogram_bins if self.model_type == 'histogram' else None,
                'histogram_method': self.histogram_method if self.model_type == 'histogram' else None,
                'histogram_params': histogram_params,  # NEW
                'feature_pairs': feature_pairs_list,
                'pruning_info': {
                    'active_feature_mask': active_mask,
                    'original_feature_indices': original_indices,
                    'pruning_warmup_epochs': self.pruning_warmup_epochs,
                    'pruning_threshold': self.pruning_threshold,
                    'pruning_aggressiveness': self.pruning_aggressiveness
                },
                # Critical parameters for prediction
                'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
                'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
                'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
                'feature_names': list(self.data.columns[:-1]) if hasattr(self, 'data') and self.data is not None else [],
                'target_column': self.target_column,
                'target_column_name': self.data.columns[self.target_column] if isinstance(self.target_column, int) and hasattr(self, 'data') and self.data is not None else str(self.target_column),
                'n_features': self.data.shape[1] - 1 if hasattr(self, 'data') and self.data is not None else None,
                'n_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else None
            }

            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f, indent=4)

    def _load_categorical_encoders(self):
        """Load categorical encoders if available"""
        encoders_file = os.path.join('Model', f'{self.dataset_name}_encoders.pkl')

        if os.path.exists(encoders_file):
            try:
                with open(encoders_file, 'rb') as f:
                    self.categorical_encoders = pickle.load(f)
                print(f"Loaded categorical encoders from {encoders_file}")
            except Exception as e:
                print(f"Error loading encoders: {str(e)}")
                self.categorical_encoders = {}

    def _save_categorical_encoders(self):
        """Save categorical encoders to file"""
        encoders_file = os.path.join('Model', f'{self.dataset_name}_encoders.pkl')

        try:
            with open(encoders_file, 'wb') as f:
                pickle.dump(self.categorical_encoders, f)
            print(f"Saved categorical encoders to {encoders_file}")
        except Exception as e:
            print(f"Error saving encoders: {str(e)}")

    def _initialize_weights(self):
        """Initialize weights based on number of classes and feature combinations"""
        # This will be properly initialized after computing likelihood parameters
        self.current_W = None
        self.best_W = None

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder"""
        df_encoded = df.copy()

        for column in df_encoded.columns:
            if column == self.target_column:
                continue

            # Check if column is categorical (object type or low cardinality)
            if (df_encoded[column].dtype == 'object' or
                    df_encoded[column].nunique() / len(df_encoded) < 0.05):

                if column not in self.categorical_encoders:
                    self.categorical_encoders[column] = LabelEncoder()
                    self.categorical_encoders[column].fit(df_encoded[column])

                df_encoded[column] = self.categorical_encoders[column].transform(df_encoded[column])

        return df_encoded

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data"""
        # Encode categorical features
        df_encoded = self._encode_categorical_features(self.data)

        # Separate features and target
        if isinstance(self.target_column, int):
            X = df_encoded.drop(df_encoded.columns[self.target_column], axis=1).values
            y = df_encoded.iloc[:, self.target_column].values
        else:
            X = df_encoded.drop(self.target_column, axis=1).values
            y = df_encoded[self.target_column].values

        # Filter out samples with sentinel values
        X, y = self._filter_sentinel_samples(X, y)

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _prune_stagnant_connections(self, epoch: int):
        """Prune connections that haven't changed significantly from initial values"""
        # Early return if pruning is disabled or for histogram model
        if (self.pruning_warmup_epochs == 0 or
            self.model_type == 'histogram' or  # Skip pruning for histogram model
            epoch <= self.pruning_warmup_epochs or
            self.initial_W is None or
            self.current_W is None or
            self.initial_W.shape != self.current_W.shape):
            return

        # Calculate absolute change from initial weights
        weight_changes = np.abs(self.current_W - self.initial_W)

        # Find connections that haven't changed much
        stagnant_mask = weight_changes < self.pruning_threshold

        # Calculate how many to prune (only prune if we have enough connections)
        n_stagnant = np.sum(stagnant_mask)
        if n_stagnant == 0:
            return

        n_to_prune = int(n_stagnant * self.pruning_aggressiveness)

        # Ensure we don't prune all connections
        min_connections = max(5, int(self.current_W.shape[1] * 0.1))  # Keep at least 10% or 5 connections
        if (self.current_W.shape[1] - n_to_prune) < min_connections:
            n_to_prune = max(0, self.current_W.shape[1] - min_connections)

        if n_to_prune <= 0:
            return

        # Get indices of most stagnant connections
        flat_indices = np.argsort(weight_changes, axis=None)[:n_to_prune]
        row_indices, col_indices = np.unravel_index(flat_indices, weight_changes.shape)

        # Create a mask for connections to keep (invert the stagnant ones)
        keep_mask = np.ones(self.current_W.shape[1], dtype=bool)
        unique_cols_to_prune = np.unique(col_indices)
        keep_mask[unique_cols_to_prune] = False

        # Apply pruning
        self._apply_pruning(keep_mask)

        if epoch % 10 == 0:
            print(f"Pruned {len(unique_cols_to_prune)} stagnant feature pairs")

    def train(self):
        """Train the model"""
        if not self.train_enabled:
            print("Training is disabled in configuration")
            return

        print("Starting training...")

        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data()

        # Set up visualization
        if self.visualization_enabled and self.visualizer is not None:
            # Get class names from label encoder
            class_names = [str(cls) for cls in self.label_encoder.classes_]

            # Set data context for meaningful visualizations
            self.visualizer.set_data_context(
                X_train=X_train,
                y_train=y_train,
                feature_names=[f'Feature_{i}' for i in range(X_train.shape[1])],
                class_names=class_names
            )

            print("Visualization system configured with data context")

        # Compute likelihood parameters based on selected model
        self._compute_likelihood_parameters(X_train, y_train)

        # Convert to appropriate format based on device
        if self.use_gpu:
            X_train = self._to_tensor(X_train)
            y_train = self._to_tensor(y_train, torch.long)
            X_test = self._to_tensor(X_test)
            y_test = self._to_tensor(y_test, torch.long)

        # Training loop
        best_epoch = 0
        trials_without_improvement = 0
        training_errors = []
        test_errors = []
        training_accuracies = []
        test_accuracies = []

        # Keyboard control setup
        if not nokbd:
            self._setup_keyboard_control()

        # Use tqdm for progress bar
        from tqdm import trange
        progress_bar = trange(self.max_epochs, desc="Training")

        try:
            for epoch in progress_bar:
                # Prune stagnant connections (only for Gaussian model)
                if self.model_type == 'gaussian':
                    self._prune_stagnant_connections(epoch)

                # Compute posteriors for training set
                train_posteriors = self._compute_batch_posterior(
                    self._ensure_numpy(X_train) if self.use_gpu else X_train
                )

                # Compute training error and accuracy
                train_predictions = np.argmax(train_posteriors, axis=1)
                y_train_numpy = self._ensure_numpy(y_train) if self.use_gpu else y_train

                train_error = 1 - accuracy_score(y_train_numpy, train_predictions)
                train_accuracy = accuracy_score(y_train_numpy, train_predictions)

                training_errors.append(train_error)
                training_accuracies.append(train_accuracy)

                # Create visualizations - UPDATED METHOD CALLS
                if (self.visualization_enabled and self.visualizer is not None
                    and self.current_W is not None):

                    # Create interactive prior distribution visualization
                    if epoch % 10 == 0:  # Every 10 epochs for priors
                        self.visualizer.create_interactive_prior_distribution(epoch, self.current_W)

                    # Create feature space visualization less frequently
                    if epoch % 25 == 0:  # Every 25 epochs for feature space
                        self.visualizer.create_interactive_feature_space_3d(epoch)

                # Compute test error and accuracy if not train_only

                if not self.train_only=='yes':
                    test_posteriors = self._compute_batch_posterior(
                        self._ensure_numpy(X_test) if self.use_gpu else X_test
                    )
                    test_predictions = np.argmax(test_posteriors, axis=1)
                    y_test_numpy = self._ensure_numpy(y_test) if self.use_gpu else y_test

                    test_error = 1 - accuracy_score(y_test_numpy, test_predictions)
                    test_accuracy = accuracy_score(y_test_numpy, test_predictions)

                    test_errors.append(test_error)
                    test_accuracies.append(test_accuracy)

                    # Check for improvement
                    if test_error < self.best_error:
                        print(f"Test accuracy improved from {self.best_accuracy} to {test_accuracy}")
                        self.best_error = test_error
                        self.best_accuracy = test_accuracy
                        self.best_W = self.current_W.copy()
                        best_epoch = epoch
                        trials_without_improvement = 0
                        self._save_best_weights()
                    else:
                        trials_without_improvement += 1

                    # Early stopping
                    if trials_without_improvement >= self.trials:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    # Update progress bar
                    progress_bar.set_postfix({
                        'Train Acc': f'{train_accuracy:.4f}',
                        'Test Acc': f'{test_accuracy:.4f}',
                        'Best Acc': f'{self.best_accuracy:.4f}',
                        'No Improve': trials_without_improvement
                    })
                else:
                    print()
                    print('=='*60)
                    print("In train only mode")
                    print('=='*60)
                    # For train_only mode, just track training performance
                    if self.best_accuracy  < 1- train_accuracy:
                        self.best_error = 1- train_error
                        self.best_accuracy = 1- train_accuracy #Always use Test accuracy as guide to avoid over fitting
                        self.best_W = self.current_W.copy()
                        best_epoch = epoch
                        trials_without_improvement = 0
                        self._save_best_weights()
                    else:
                        trials_without_improvement += 1

                    # Early stopping
                    if trials_without_improvement >= self.trials:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    # Update progress bar
                    progress_bar.set_postfix({
                        'Train Acc': f'{train_accuracy:.4f}',
                        'Best Acc': f'{self.best_accuracy:.4f}',
                        'No Improve': trials_without_improvement
                    })

                # Identify misclassified samples
                misclassified_indices = np.where(train_predictions != y_train_numpy)[0]

                if len(misclassified_indices) > 0:
                    # Collect failed cases
                    failed_cases = []
                    batch_size = 1000 if self.histogram_method == 'vectorized' else 100
                    for idx in misclassified_indices[:batch_size]:
                        features = (X_train[idx].cpu().numpy() if self.use_gpu else X_train[idx])
                        true_class = (y_train[idx].cpu().item() if self.use_gpu else y_train[idx])
                        posteriors = train_posteriors[idx]
                        failed_cases.append((features, true_class, posteriors))

                    # Update priors based on failed cases
                    self._update_priors_parallel(failed_cases, batch_size=100)

                # Check for keyboard interrupt
                if not nokbd and self.skip_training:
                    print("Training interrupted by user")
                    break

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            if not nokbd:
                self._cleanup_keyboard_control()
            progress_bar.close()

        # Restore best weights
        self.current_W = self.best_W.copy()

        print(f"Training completed. Best epoch: {best_epoch}")
        print(f"Best Test error: {self.best_error:.4f}")
        print(f"Best  Test accuracy: {self.best_accuracy:.4f}")

        # Finalize all visualizations after training - UPDATED METHOD CALL
        if self.visualization_enabled and self.visualizer is not None:
            self.visualizer.finalize_visualizations(
                self.current_W, training_errors, training_accuracies
            )

        # Save categorical encoders
        self._save_categorical_encoders()

        # Plot training history if not in train_only mode
        if not self.train_only:
            self._plot_training_history(training_errors, test_errors, training_accuracies, test_accuracies)


    def _setup_keyboard_control(self):
        """Setup keyboard listener for training control"""
        self.skip_training = False

        def on_press(key):
            try:
                if key.char == 'q':
                    self.skip_training = True
                    return False  # Stop listener
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def _cleanup_keyboard_control(self):
        """Cleanup keyboard listener"""
        if hasattr(self, 'listener'):
            self.listener.stop()

    def _plot_training_history(self, training_errors, test_errors, training_accuracies, test_accuracies):
        """Plot training history"""
        plt.figure(figsize=(12, 5))

        # Plot errors
        plt.subplot(1, 2, 1)
        plt.plot(training_errors, label='Training Error')
        if test_errors:
            plt.plot(test_errors, label='Test Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training and Test Error')
        plt.legend()
        plt.grid(True)

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(training_accuracies, label='Training Accuracy')
        if test_accuracies:
            plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_training_history.png')
        plt.close()

    def _validate_prediction_ready(self) -> bool:
        """Validate that all required components for prediction are available"""
        checks = []

        # Check weights
        if self.best_W is None and self.current_W is None:
            checks.append("❌ No weights available")

        # Check label encoder
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            checks.append("❌ Label encoder not fitted")

        # Check scaler
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            checks.append("❌ Scaler not fitted")

        # Check likelihood parameters based on model type
        if self.model_type == 'histogram':
            if self.histograms is None:
                checks.append("❌ Histogram parameters not computed")
        else:  # gaussian
            if self.likelihood_params is None:
                checks.append("❌ Gaussian likelihood parameters not computed")

        if checks:
            print("Prediction readiness check failed:")
            for check in checks:
                print(f"  {check}")
            return False

        return True

    def _ensure_likelihood_parameters(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Ensure likelihood parameters are computed if missing"""
        if self.model_type == 'histogram':
            if self.histograms is None:
                print("Computing histogram parameters for prediction...")
                try:
                    self._compute_likelihood_parameters(X_train, y_train)
                    return True
                except Exception as e:
                    print(f"❌ Failed to compute histogram parameters: {e}")
                    return False
        else:  # gaussian
            if self.likelihood_params is None:
                print("Computing Gaussian likelihood parameters for prediction...")
                try:
                    self._compute_likelihood_parameters(X_train, y_train)
                    return True
                except Exception as e:
                    print(f"❌ Failed to compute Gaussian parameters: {e}")
                    return False
        return True

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """Make predictions on new data"""
        if not self.predict_enabled:
            print("Prediction is disabled in configuration")
            return None

        # Validate that we have all required components
        if not self._validate_prediction_ready():
            print("❌ Model not ready for prediction. Required components missing.")
            return None

        if X is None:
            # Use test data for evaluation - FIX: Prepare data first
            X_train, X_test, y_train, y_test = self._prepare_data()

            # Ensure likelihood parameters are computed
            if not self._ensure_likelihood_parameters(X_train, y_train):
                print("❌ Failed to compute likelihood parameters for prediction")
                return None

            if self.use_gpu:
                X_test_tensor = self._to_tensor(X_test)
                test_posteriors = self._compute_batch_posterior(
                    self._ensure_numpy(X_test_tensor) if self.use_gpu else X_test
                )
                test_predictions = np.argmax(test_posteriors, axis=1)
                y_test_numpy = self._ensure_numpy(y_test) if self.use_gpu else y_test
            else:
                test_posteriors = self._compute_batch_posterior(X_test)
                test_predictions = np.argmax(test_posteriors, axis=1)
                y_test_numpy = y_test

            # Calculate and print metrics
            accuracy = accuracy_score(y_test_numpy, test_predictions)
            print(f"Test Accuracy: {accuracy:.4f}")

            # Classification report - convert numeric class labels to strings
            target_names = [str(cls) for cls in self.label_encoder.classes_]
            print("\nClassification Report:")
            print(classification_report(y_test_numpy, test_predictions,
                                      target_names=target_names))

            # Confusion matrix
            cm = confusion_matrix(y_test_numpy, test_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names,
                       yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{self.dataset_name}_confusion_matrix.png')
            plt.close()

            return test_predictions

        else:
            # Predict on new data
            # Handle missing values
            X_clean = self._handle_missing_values(pd.DataFrame(X))

            # Encode categorical features if any
            for i, column in enumerate(X_clean.columns):
                if column in self.categorical_encoders:
                    X_clean[column] = self.categorical_encoders[column].transform(X_clean[column])

            # Scale features
            X_scaled = self.scaler.transform(X_clean.values)

            # Filter out samples with sentinel values
            X_scaled, _ = self._filter_sentinel_samples(X_scaled, None)

            # Compute posteriors
            posteriors = self._compute_batch_posterior(X_scaled)
            predictions = np.argmax(posteriors, axis=1)

            # Decode predictions
            decoded_predictions = self.label_encoder.inverse_transform(predictions)

            return decoded_predictions

    def generate_samples(self, n_samples: int = 100, class_id: int = None):
        """Generate synthetic samples"""
        if not self.gen_samples_enabled:
            print("Sample generation is disabled in configuration")
            return None

        print(f"Generating {n_samples} synthetic samples...")

        if self.model_type == 'histogram':
            return self._generate_histogram_samples(n_samples, class_id)
        else:
            return self._generate_gaussian_samples(n_samples, class_id)

    def _generate_gaussian_samples(self, n_samples: int = 100, class_id: int = None):
        """Generate synthetic samples using Gaussian model"""
        # Get likelihood parameters
        means = self.likelihood_params['means']
        covs = self.likelihood_params['covs']
        classes = self.likelihood_params['classes']

        if class_id is None:
            # Generate samples from all classes proportionally
            class_probs = np.sum(self.current_W, axis=1)
            class_probs = class_probs / np.sum(class_probs)
            class_ids = np.random.choice(len(classes), size=n_samples, p=class_probs)
        else:
            # Generate samples from specific class
            class_ids = np.full(n_samples, class_id)

        # Convert to numpy if using GPU
        if self.use_gpu:
            means = self._to_numpy(means)
            covs = self._to_numpy(covs)
            classes = self._to_numpy(classes)

        # Generate samples
        samples = []
        for class_id in class_ids:
            # Randomly select a feature pair based on weights
            pair_probs = self.current_W[class_id] / np.sum(self.current_W[class_id])
            pair_idx = np.random.choice(len(self.feature_pairs), p=pair_probs)

            # Generate sample from multivariate normal distribution
            mean = means[class_id, pair_idx]
            cov = covs[class_id, pair_idx]

            try:
                sample = np.random.multivariate_normal(mean, cov)
            except:
                # Handle singular covariance matrices
                sample = mean + np.random.normal(0, 0.1, size=len(mean))

            samples.append(sample)

        samples = np.array(samples)

        # Inverse transform scaling
        samples_original = self.scaler.inverse_transform(samples)

        return samples_original, class_ids

    def _generate_histogram_samples(self, n_samples: int = 100, class_id: int = None):
        """Generate synthetic samples using histogram model"""
        n_features = self.histograms.shape[0]
        n_classes = self.histograms.shape[2]

        if class_id is None:
            # Generate samples from all classes proportionally
            class_probs = np.sum(self.current_W, axis=(1, 2))
            class_probs = class_probs / np.sum(class_probs)
            class_ids = np.random.choice(n_classes, size=n_samples, p=class_probs)
        else:
            # Generate samples from specific class
            class_ids = np.full(n_samples, class_id)

        samples = []
        for class_id in class_ids:
            sample = []
            for feature_idx in range(n_features):
                # Select bin based on histogram probabilities
                bin_probs = self.histograms[feature_idx, :, class_id]
                bin_idx = np.random.choice(self.histogram_bins, p=bin_probs)

                # Generate value within the selected bin
                bin_start = self.bin_edges[feature_idx, bin_idx]
                bin_end = self.bin_edges[feature_idx, bin_idx + 1]
                value = np.random.uniform(bin_start, bin_end)
                sample.append(value)

            samples.append(sample)

        samples = np.array(samples)

        # Inverse transform scaling
        samples_original = self.scaler.inverse_transform(samples)

        return samples_original, class_ids


def main():
    """Main function to run the GPUDBNN model"""
    # Get available datasets
    available_datasets = DatasetConfig.get_available_datasets()

    if available_datasets:
        print("Available datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"{i}. {dataset}")

        # Let user select a dataset
        choice = input(f"\nSelect a dataset (1-{len(available_datasets)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_datasets):
                dataset_name = available_datasets[choice_idx]
            else:
                print("Invalid selection, using default dataset name")
                dataset_name = input("Enter dataset name: ").strip()
        except ValueError:
            print("Invalid input, using default dataset name")
            dataset_name = input("Enter dataset name: ").strip()
    else:
        print("No existing dataset configurations found.")
        dataset_name = input("Enter dataset name: ").strip()

    # Create and run the model
    model = GPUDBNN(dataset_name)
    print(f"Train only: {model.train_only}")
    print(f"Train enabled: {model.train_enabled}")
    print(f"Predict enabled: {model.predict_enabled}")
    # Train the model
    if model.train_enabled:
        model.train()

    # Make predictions
    if model.predict_enabled:
        model.predict()

    # Generate samples if enabled
    if model.gen_samples_enabled:
        samples, class_ids = model.generate_samples(n_samples=100)
        if samples is not None:
            print(f"Generated {len(samples)} synthetic samples")
            # Save samples to CSV
            samples_df = pd.DataFrame(samples)
            samples_df['class'] = class_ids
            samples_df.to_csv(f'{dataset_name}_synthetic_samples.csv', index=False)
            print(f"Samples saved to {dataset_name}_synthetic_samples.csv")

if __name__ == "__main__":
    main()
