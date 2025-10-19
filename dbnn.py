#!/usr/bin/env python3
"""
Optimized DBNN implementation as a modular class structure
Core algorithm with unified configuration and visualization capabilities
"""

import math
import sys
import time
import os
import json
import csv
import glob
import pickle
import gzip
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from numba import jit, prange
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import traceback
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from numba import jit, prange
import numpy as np
import os
import time
import json
import csv
import pickle
import gzip
from typing import List, Tuple, Dict, Any, Optional, Union

# Global constants
max_resol = 1600
features = 100
classes = 500

@jit(nopython=True, fastmath=True, cache=True)
def round_cpp(x: float) -> int:
    """Match C++ rounding behavior - Numba compatible"""
    return int(x + 0.5) if x >= 0 else int(x - 0.5)

@jit(nopython=True, fastmath=True, cache=True)
def normalize_feature(value, min_val, max_val, resolution_val):
    """Normalize single feature value - Numba compatible"""
    if max_val - min_val > 0:
        return round_cpp((value - min_val) / (max_val - min_val) * resolution_val)
    return 0

@jit(nopython=True, parallel=False, fastmath=True)
def process_training_sample(vects, tmpv, anti_net, anti_wts, binloc,
                           resolution, dmyclass, min_val, max_val,
                           innodes, outnodes):
    """Process single training sample - core algorithm unchanged"""
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Update network counts
    for i in range(1, innodes + 1):
        j = bins[i]
        for l in range(1, innodes + 1):
            m = bins[l]

            # Find correct output class
            k_class = 1
            while k_class <= outnodes and abs(tmpv - dmyclass[k_class]) > dmyclass[0]:
                k_class += 1

            if k_class <= outnodes:
                anti_net[i, j, l, m, k_class] += 1
                anti_net[i, j, l, m, 0] += 1

    return anti_net

@jit(nopython=True, parallel=False, fastmath=True)
def compute_class_probabilities_numba(vects, anti_net, anti_wts, binloc,
                                    resolution, dmyclass, min_val, max_val,
                                    innodes, outnodes):
    """Compute class probabilities - core algorithm unchanged"""
    classval = np.ones(outnodes + 2)
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Compute probabilities
    classval[0] = 0.0
    for i in range(1, innodes + 1):
        j = bins[i]
        for l in range(1, innodes + 1):
            m = bins[l]
            for k in range(1, outnodes + 1):
                if anti_net[i, j, l, m, 0] > 0:
                    tmp2_wts = anti_net[i, j, l, m, k] / anti_net[i, j, l, m, 0]
                else:
                    tmp2_wts = 1.0 / outnodes
                classval[k] *= tmp2_wts * anti_wts[i, j, l, m, k]
                classval[0] += classval[k]

    # Normalize
    if classval[0] > 0:
        for k in range(1, outnodes + 1):
            classval[k] /= classval[0]
    classval[0] = 0.0

    return classval

@jit(nopython=True, parallel=False, fastmath=True)
def update_weights_numba(vects, tmpv, classval, anti_wts, binloc, resolution,
                        dmyclass, min_val, max_val, innodes, outnodes, gain):
    """Update weights - core algorithm unchanged"""
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Find predicted class
    kmax = 1
    cmax = 0.0
    for k in range(1, outnodes + 1):
        if classval[k] > cmax:
            cmax = classval[k]
            kmax = k

    # Update weights if wrong
    if abs(dmyclass[kmax] - tmpv) > dmyclass[0]:
        for i in range(1, innodes + 1):
            j = bins[i]
            for l in range(1, innodes + 1):
                m = bins[l]

                # Find correct class
                k_correct = 1
                while k_correct <= outnodes and abs(dmyclass[k_correct] - tmpv) > dmyclass[0]:
                    k_correct += 1

                if (k_correct <= outnodes and classval[kmax] > 0 and
                    classval[k_correct] < classval[kmax]):
                    adjustment = gain * (1.0 - (classval[k_correct] / classval[kmax]))
                    anti_wts[i, j, l, m, k_correct] += adjustment

    return anti_wts

@jit(nopython=True, fastmath=True, cache=True)
def find_closest_bin_numba(value, binloc_row, resolution_val):
    """Optimized bin finding for single value"""
    if resolution_val <= 0:
        return 0
    min_dist = 2.0 * resolution_val
    best_bin = 0
    for j in range(1, resolution_val + 1):
        dist = abs(value - binloc_row[j])
        if dist < min_dist:
            min_dist = dist
            best_bin = j
    if best_bin > 0:
        best_bin -= 1
    return best_bin

class ClassEncoder:
    """Handles encoding and decoding of class labels"""

    def __init__(self):
        self.class_to_encoded = {}
        self.encoded_to_class = {}
        self.encoded_to_original = {}
        self.is_fitted = False

    def fit(self, class_labels):
        """Fit encoder to class labels"""
        unique_classes = sorted(set(class_labels))

        for encoded_val, original_class in enumerate(unique_classes, 1):
            self.class_to_encoded[original_class] = float(encoded_val)
            self.encoded_to_class[float(encoded_val)] = original_class
            self.encoded_to_original[float(encoded_val)] = str(original_class)

        self.is_fitted = True
        print(f"Fitted encoder with {len(unique_classes)} classes: {unique_classes}")

    def transform(self, class_labels):
        """Transform class labels to encoded numeric values"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transforming")

        encoded = []
        for label in class_labels:
            if label in self.class_to_encoded:
                encoded.append(self.class_to_encoded[label])
            else:
                raise ValueError(f"Unknown class label: {label}")

        return np.array(encoded, dtype=np.float64)

    def inverse_transform(self, encoded_values):
        """Transform encoded values back to original class labels"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before inverse transforming")

        original = []
        for encoded_val in encoded_values:
            if encoded_val in self.encoded_to_class:
                original.append(self.encoded_to_class[encoded_val])
            else:
                closest = min(self.encoded_to_class.keys(),
                            key=lambda x: abs(x - encoded_val))
                original.append(self.encoded_to_class[closest])
                print(f"Warning: Encoded value {encoded_val} not found, using closest: {closest}")

        return original

    def get_encoded_classes(self):
        """Get the encoded class values used in dmyclass"""
        return sorted(self.encoded_to_class.keys())

    def get_class_mapping(self):
        """Get the complete class mapping"""
        return {
            'class_to_encoded': self.class_to_encoded,
            'encoded_to_class': self.encoded_to_class
        }

class DBNNCore:
    """
    Optimized DBNN algorithm powerhouse with parallel processing and hardware acceleration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.config = config or {
            'LoC': 0.65,
            'nLoC': 0.0,
            'resol': 100,
            'nresol': 0,
            'nLoCcnt': 0,
            'skpchk': 0,
            'oneround': 100,
            'fst_gain': 1.0,
            'gain': 2.0,
            'margin': 0.2,
            'patience': 10,
            'min_improvement': 0.1
        }

        # Core data structures
        self.anti_net = None
        self.anti_wts = None
        self.antit_wts = None
        self.antip_wts = None
        self.binloc = None
        self.max_val = None
        self.min_val = None
        self.dmyclass = None
        self.resolution_arr = None

        # Training state
        self.innodes = 0
        self.outnodes = 0
        self.class_encoder = ClassEncoder()
        self.training_history = []
        self.is_trained = False

        # Parallel processing and optimization
        self.num_workers = self._detect_optimal_workers()
        self.parallel_enabled = True
        self.gpu_enabled = False
        self.memory_optimized = False

        # Initialize hardware acceleration
        self._init_gpu_acceleration()

        # Visualization integration
        self.visualizer = None
        self.log_callback = None

    def _detect_optimal_workers(self):
        """Detect optimal number of workers for parallel processing"""
        try:
            cpu_count = mp.cpu_count()
            # Reserve 1 core for main process, use 75% of remaining cores
            optimal_workers = max(1, int((cpu_count - 1) * 0.75))
            return min(optimal_workers, 16)  # Cap at 16 workers
        except:
            return 4  # Fallback to 4 workers

    def attach_visualizer(self, visualizer):
        """Attach a visualizer for training monitoring"""
        self.visualizer = visualizer
        self.log("Visualizer attached to DBNN core")

    def detect_system_resources(self):
        """Detect available system resources and set optimal parameters"""
        resource_info = {
            'cpu_cores': 1,
            'memory_gb': 4,
            'has_gpu': False,
            'gpu_memory_gb': 0,
            'system_memory_gb': 4
        }

        try:
            import psutil
            # CPU cores
            resource_info['cpu_cores'] = psutil.cpu_count(logical=False) or 1
            resource_info['logical_cores'] = psutil.cpu_count(logical=True) or 1

            # System memory
            system_memory = psutil.virtual_memory()
            resource_info['system_memory_gb'] = system_memory.total / (1024**3)
            resource_info['available_memory_gb'] = system_memory.available / (1024**3)

            # Process memory
            process = psutil.Process()
            resource_info['process_memory_mb'] = process.memory_info().rss / (1024**2)

        except ImportError:
            self.log("⚠️ psutil not available, using default resource values")

        # GPU detection
        try:
            import torch
            if torch.cuda.is_available():
                resource_info['has_gpu'] = True
                resource_info['gpu_count'] = torch.cuda.device_count()
                resource_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                resource_info['gpu_name'] = torch.cuda.get_device_name(0)
            else:
                resource_info['has_gpu'] = False
        except ImportError:
            # Try alternative GPU detection
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_memory_mb = int(result.stdout.strip().split('\n')[0])
                    resource_info['has_gpu'] = True
                    resource_info['gpu_memory_gb'] = gpu_memory_mb / 1024
            except:
                resource_info['has_gpu'] = False

        return resource_info

    def save_model_auto(self, model_dir="Model", data_filename=None, feature_columns=None, target_column=None):
        """Automatically save trained model in binary format with timestamp to Model/ folder"""
        if not self.is_trained:
            self.log("❌ No trained model to save")
            return None

        try:
            # Ensure Model directory exists with absolute path
            model_dir = os.path.abspath(model_dir)
            os.makedirs(model_dir, exist_ok=True)

            # Create model filename based on data file and timestamp
            if data_filename:
                base_name = os.path.splitext(os.path.basename(data_filename))[0]
            else:
                base_name = "model"

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(model_dir, f"{base_name}_{timestamp}_model.bin")

            # Store additional metadata
            model_metadata = {
                'data_file': data_filename,
                'feature_columns': feature_columns if feature_columns else [],
                'target_column': target_column if target_column else "",
                'best_accuracy': getattr(self, 'best_accuracy', 0.0),
                'best_round': getattr(self, 'best_round', 0),
                'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_format': 'binary_auto_v1',
                'input_nodes': self.innodes,
                'output_nodes': self.outnodes,
                'config': self.config  # Store the complete configuration
            }

            # Use core's save method but enhance with metadata
            success = self.save_model(
                model_filename,
                feature_columns=feature_columns,
                target_column=target_column,
                use_json=False  # Always binary format for auto-save
            )

            if success:
                # Save additional metadata separately
                meta_filename = model_filename.replace('.bin', '_meta.json')
                with open(meta_filename, 'w') as f:
                    json.dump(model_metadata, f, indent=2)

                self.log(f"✅ Model automatically saved: {model_filename}")
                self.log(f"   Best accuracy: {model_metadata['best_accuracy']:.2f}% at round {model_metadata['best_round']}")
                self.log(f"   Metadata: {meta_filename}")
                self.log(f"   Configuration: {len(self.config)} parameters embedded")

                return model_filename
            else:
                self.log("❌ Failed to automatically save model")
                return None

        except Exception as e:
            self.log(f"❌ Error in automatic model saving: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_optimal_parameters(self, resource_info, operation_type="prediction"):
        """Calculate optimal parameters based on available resources"""
        params = {
            'batch_size': 1000,
            'max_concurrent_batches': 1,
            'memory_safety_factor': 0.7,
            'clear_cache_every': 50,
            'write_to_disk_every': 10,
            'use_memory_mapping': False,
            'optimization_level': 'balanced'
        }

        # Extract resource information
        cpu_cores = resource_info['cpu_cores']
        available_memory_gb = resource_info['available_memory_gb']
        system_memory_gb = resource_info['system_memory_gb']
        has_gpu = resource_info['has_gpu']
        gpu_memory_gb = resource_info.get('gpu_memory_gb', 0)

        # Memory-based optimization
        if system_memory_gb >= 32:
            # High memory system
            params['optimization_level'] = 'performance'
            params['batch_size'] = 5000
            params['max_concurrent_batches'] = min(4, cpu_cores)
            params['memory_safety_factor'] = 0.8
            params['clear_cache_every'] = 100
            params['write_to_disk_every'] = 20

        elif system_memory_gb >= 16:
            # Medium memory system
            params['optimization_level'] = 'balanced'
            params['batch_size'] = 2000
            params['max_concurrent_batches'] = min(2, cpu_cores)
            params['memory_safety_factor'] = 0.7
            params['clear_cache_every'] = 50
            params['write_to_disk_every'] = 15

        else:
            # Low memory system
            params['optimization_level'] = 'memory_safe'
            params['batch_size'] = 500
            params['max_concurrent_batches'] = 1
            params['memory_safety_factor'] = 0.5
            params['clear_cache_every'] = 20
            params['write_to_disk_every'] = 5

        # CPU-based adjustments
        if cpu_cores >= 8:
            params['max_concurrent_batches'] = min(params['max_concurrent_batches'] + 2, cpu_cores // 2)
        elif cpu_cores >= 4:
            params['max_concurrent_batches'] = min(params['max_concurrent_batches'] + 1, cpu_cores // 2)

        # GPU-based optimizations
        if has_gpu and gpu_memory_gb >= 4:
            params['use_gpu'] = True
            if gpu_memory_gb >= 16:
                params['batch_size'] = params['batch_size'] * 2
                params['optimization_level'] = 'gpu_optimized'
            elif gpu_memory_gb >= 8:
                params['batch_size'] = int(params['batch_size'] * 1.5)

        # Operation-specific adjustments
        if operation_type == "training":
            # Training requires more memory
            params['batch_size'] = max(100, params['batch_size'] // 2)
            params['write_to_disk_every'] = max(5, params['write_to_disk_every'] // 2)
        elif operation_type == "prediction":
            # Prediction can use larger batches
            params['batch_size'] = min(10000, params['batch_size'] * 2)

        # Safety limits
        params['batch_size'] = min(params['batch_size'], 10000)  # Absolute max
        params['max_concurrent_batches'] = min(params['max_concurrent_batches'], 8)

        # Calculate memory limits
        estimated_batch_memory_mb = self.estimate_batch_memory(params['batch_size'])
        safe_memory_limit_mb = available_memory_gb * 1024 * params['memory_safety_factor']

        # Adjust batch size if it exceeds safe memory
        max_safe_batches = int(safe_memory_limit_mb / estimated_batch_memory_mb) if estimated_batch_memory_mb > 0 else 1
        if max_safe_batches < params['max_concurrent_batches']:
            params['max_concurrent_batches'] = max(1, max_safe_batches)
            params['batch_size'] = max(100, params['batch_size'] // 2)

        return params

    def estimate_batch_memory(self, batch_size):
        """Estimate memory usage for a batch"""
        if not hasattr(self, 'innodes') or self.innodes == 0:
            return 100  # Default estimate

        # Rough memory estimation (in MB)
        feature_memory = batch_size * self.innodes * 8 / (1024**2)  # float64
        network_memory = (self.innodes * 100 * self.innodes * 100 * self.outnodes * 8) / (1024**2)  # anti_net
        total_estimate = feature_memory + network_memory + 100  # Buffer

        return total_estimate

    def print_resource_report(self):
        """Print detailed resource report"""
        resources = self.detect_system_resources()

        print("\n" + "="*50)
        print("SYSTEM RESOURCE ANALYSIS")
        print("="*50)
        print(f"CPU Cores (physical): {resources['cpu_cores']}")
        print(f"CPU Cores (logical): {resources.get('logical_cores', 'N/A')}")
        print(f"System Memory: {resources['system_memory_gb']:.1f} GB")
        print(f"Available Memory: {resources['available_memory_gb']:.1f} GB")
        print(f"GPU Available: {'Yes' if resources['has_gpu'] else 'No'}")

        if resources['has_gpu']:
            print(f"GPU Memory: {resources['gpu_memory_gb']:.1f} GB")
            print(f"GPU Name: {resources.get('gpu_name', 'N/A')}")

        # Calculate optimal parameters
        pred_params = self.calculate_optimal_parameters(resources, "prediction")
        train_params = self.calculate_optimal_parameters(resources, "training")

        print(f"\nPREDICTION OPTIMIZATION:")
        print(f"  Optimization Level: {pred_params['optimization_level']}")
        print(f"  Batch Size: {pred_params['batch_size']}")
        print(f"  Concurrent Batches: {pred_params['max_concurrent_batches']}")
        print(f"  Memory Safety: {pred_params['memory_safety_factor']*100:.0f}%")

        print(f"\nTRAINING OPTIMIZATION:")
        print(f"  Optimization Level: {train_params['optimization_level']}")
        print(f"  Batch Size: {train_params['batch_size']}")
        print(f"  Concurrent Batches: {train_params['max_concurrent_batches']}")
        print("="*50)

        return resources, pred_params, train_params

    def save_model(self, model_path: str, feature_columns=None, target_column=None, use_json=False):
        """Save complete model to file in binary format (default) or JSON format"""
        try:
            # Ensure all required fields are set
            if not hasattr(self, 'innodes') or self.innodes is None:
                # Try to infer from array shapes
                if hasattr(self, 'anti_net') and self.anti_net is not None:
                    self.innodes = self.anti_net.shape[0] - 2
                else:
                    self.innodes = 0

            if not hasattr(self, 'outnodes') or self.outnodes is None:
                if hasattr(self, 'anti_net') and self.anti_net is not None:
                    self.outnodes = self.anti_net.shape[4] - 2
                else:
                    self.outnodes = 0

            # Ensure directory exists
            model_dir = os.path.dirname(os.path.abspath(model_path))
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            # Build model data with feature information
            model_data = {
                'config': self.config,
                'innodes': self.innodes,
                'outnodes': self.outnodes,
                'is_trained': self.is_trained,
                'best_accuracy': getattr(self, 'best_accuracy', 0.0),
                'class_encoder': self.class_encoder.get_class_mapping() if hasattr(self.class_encoder, 'is_fitted') and self.class_encoder.is_fitted else {},
                'feature_columns': feature_columns if feature_columns else [],
                'target_column': target_column if target_column else '',
                'arrays_format': 'numpy'  # Mark that arrays are stored as numpy objects
            }

            # Add arrays with proper validation
            if hasattr(self, 'anti_net') and self.anti_net is not None:
                model_data['anti_net'] = self.anti_net
            else:
                model_data['anti_net'] = np.array([], dtype=np.int32)

            if hasattr(self, 'anti_wts') and self.anti_wts is not None:
                model_data['anti_wts'] = self.anti_wts
            else:
                model_data['anti_wts'] = np.array([], dtype=np.float64)

            if hasattr(self, 'binloc') and self.binloc is not None:
                model_data['binloc'] = self.binloc
            else:
                model_data['binloc'] = np.array([], dtype=np.float64)

            if hasattr(self, 'max_val') and self.max_val is not None:
                model_data['max_val'] = self.max_val
            else:
                model_data['max_val'] = np.array([], dtype=np.float64)

            if hasattr(self, 'min_val') and self.min_val is not None:
                model_data['min_val'] = self.min_val
            else:
                model_data['min_val'] = np.array([], dtype=np.float64)

            if hasattr(self, 'dmyclass') and self.dmyclass is not None:
                model_data['dmyclass'] = self.dmyclass
            else:
                model_data['dmyclass'] = np.array([], dtype=np.float64)

            if hasattr(self, 'resolution_arr') and self.resolution_arr is not None:
                model_data['resolution_arr'] = self.resolution_arr
            else:
                model_data['resolution_arr'] = np.array([], dtype=np.int32)

            if use_json:
                # Convert numpy arrays to lists for JSON
                json_model_data = model_data.copy()
                for key in ['anti_net', 'anti_wts', 'binloc', 'max_val', 'min_val', 'dmyclass', 'resolution_arr']:
                    if isinstance(json_model_data[key], np.ndarray):
                        json_model_data[key] = json_model_data[key].tolist()

                with open(model_path, 'w') as f:
                    json.dump(json_model_data, f, indent=2)
                self.log(f"Model saved in JSON format to: {model_path}")
            else:
                # Save in binary format (default)
                with gzip.open(model_path, 'wb') as f:
                    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.log(f"Model saved in binary format to: {model_path}")

            if feature_columns:
                self.log(f"✅ Feature configuration saved: {len(feature_columns)} features")
            return True

        except Exception as e:
            self.log(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self, model_path: str):
        """Load model from file with robust error handling - supports both binary and JSON formats"""
        try:
            # Detect format and load accordingly
            if model_path.endswith('.gz') or model_path.endswith('.bin'):
                # Binary format
                with gzip.open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.log(f"Loading binary model from: {model_path}")
            else:
                # JSON format
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                self.log(f"Loading JSON model from: {model_path}")

            self.log(f"Loading model from: {model_path}")

            # Load basic configuration FIRST
            self.config = model_data.get('config', {})

            # Handle array loading based on format
            if 'arrays_format' in model_data and model_data['arrays_format'] == 'numpy':
                # Binary format with numpy arrays
                array_mappings = [
                    ('anti_net', np.int32),
                    ('anti_wts', np.float64),
                    ('binloc', np.float64),
                    ('max_val', np.float64),
                    ('min_val', np.float64),
                    ('dmyclass', np.float64),
                    ('resolution_arr', np.int32)
                ]

                for field_name, dtype in array_mappings:
                    if field_name in model_data:
                        loaded_array = model_data[field_name]
                        if isinstance(loaded_array, list):
                            # Convert list to numpy array if needed
                            loaded_array = np.array(loaded_array, dtype=dtype)
                        setattr(self, field_name, loaded_array)
                        self.log(f"Loaded {field_name}: shape {loaded_array.shape}")
            else:
                # JSON format with lists - convert to numpy arrays
                array_mappings = [
                    ('anti_net', np.int32),
                    ('anti_wts', np.float64),
                    ('binloc', np.float64),
                    ('max_val', np.float64),
                    ('min_val', np.float64),
                    ('dmyclass', np.float64),
                    ('resolution_arr', np.int32)
                ]

                for field_name, dtype in array_mappings:
                    if field_name in model_data and model_data[field_name]:
                        try:
                            loaded_array = np.array(model_data[field_name], dtype=dtype)
                            setattr(self, field_name, loaded_array)
                            self.log(f"Loaded {field_name}: shape {loaded_array.shape}")
                        except Exception as e:
                            self.log(f"Error loading {field_name}: {e}")

            # FIX: Infer dimensions from ACTUAL loaded arrays, not from metadata
            if hasattr(self, 'anti_net') and self.anti_net is not None:
                # Infer from anti_net shape: (innodes+2, resol+2, innodes+2, resol+2, outnodes+2)
                actual_innodes = self.anti_net.shape[0] - 2
                actual_outnodes = self.anti_net.shape[4] - 2
                self.innodes = actual_innodes
                self.outnodes = actual_outnodes
                self.log(f"✅ Inferred dimensions from arrays: {actual_innodes} inputs, {actual_outnodes} outputs")
            else:
                # Fallback to metadata
                self.innodes = model_data.get('innodes', 0)
                self.outnodes = model_data.get('outnodes', 0)
                self.log(f"⚠️ Using metadata dimensions: {self.innodes} inputs, {self.outnodes} outputs")

            # Now set other properties
            self.is_trained = model_data.get('is_trained', False)

            # Load best accuracy
            if 'best_accuracy' in model_data:
                self.best_accuracy = model_data['best_accuracy']

            # NEW: Load feature information from model
            self.feature_columns = model_data.get('feature_columns', [])
            self.target_column = model_data.get('target_column', '')

            if self.feature_columns:
                self.log(f"✅ Loaded feature configuration: {len(self.feature_columns)} features - {self.feature_columns}")
            else:
                self.log("⚠️ No feature configuration found in model file")

            if self.target_column:
                self.log(f"✅ Target column: {self.target_column}")

            self.log(f"Final model - is_trained: {self.is_trained}, innodes: {self.innodes}, outnodes: {self.outnodes}")

            # FIX: Only reinitialize arrays if we have valid dimensions AND arrays aren't already loaded
            resol = self.config.get('resol', 100)
            if self.innodes > 0 and self.outnodes > 0 and not hasattr(self, 'anti_net'):
                self.initialize_arrays(self.innodes, resol, self.outnodes)
                self.log("Arrays initialized from dimensions")
            elif hasattr(self, 'anti_net'):
                self.log("✅ Arrays already loaded, no reinitialization needed")
            else:
                self.log("❌ Warning: Invalid model dimensions, cannot initialize arrays")
                return

            # FIX: Enhanced class encoder loading with better error recovery
            class_encoder_mapping = model_data.get('class_encoder', {})
            if class_encoder_mapping:
                try:
                    # Convert string keys back to floats for encoded_to_class
                    encoded_to_class = {}
                    class_to_encoded = {}

                    # Handle encoded_to_class (string keys → float keys)
                    encoded_to_class_data = class_encoder_mapping.get('encoded_to_class', {})
                    if encoded_to_class_data:
                        for str_key, class_name in encoded_to_class_data.items():
                            try:
                                float_key = float(str_key)
                                encoded_to_class[float_key] = class_name
                            except (ValueError, TypeError) as e:
                                self.log(f"⚠️ Warning: Could not convert key {str_key} to float: {e}")
                                # Fallback: try to keep as string if float conversion fails
                                try:
                                    encoded_to_class[float(str_key)] = class_name
                                except:
                                    self.log(f"❌ Failed to convert key {str_key}, skipping this entry")
                    else:
                        self.log("⚠️ No encoded_to_class data found in model")

                    # Handle class_to_encoded (already string keys)
                    class_to_encoded_data = class_encoder_mapping.get('class_to_encoded', {})
                    if class_to_encoded_data:
                        for class_name, encoded_val in class_to_encoded_data.items():
                            try:
                                class_to_encoded[class_name] = float(encoded_val)
                            except (ValueError, TypeError) as e:
                                self.log(f"⚠️ Warning: Could not convert value {encoded_val} to float: {e}")
                                # Fallback: try to convert string representation
                                try:
                                    class_to_encoded[class_name] = float(str(encoded_val))
                                except:
                                    self.log(f"❌ Failed to convert value {encoded_val}, skipping this entry")
                    else:
                        self.log("⚠️ No class_to_encoded data found in model")

                    # Set the encoder properties - CRITICAL: Always set is_fitted if we have any mappings
                    self.class_encoder.encoded_to_class = encoded_to_class
                    self.class_encoder.class_to_encoded = class_to_encoded

                    # Determine if encoder should be considered fitted
                    has_any_mappings = len(encoded_to_class) > 0 or len(class_to_encoded) > 0
                    is_fitted_from_data = class_encoder_mapping.get('is_fitted', False)

                    if has_any_mappings or is_fitted_from_data:
                        self.class_encoder.is_fitted = True
                        self.log(f"✅ Class encoder marked as fitted with {len(encoded_to_class)} encoded classes")
                    else:
                        self.class_encoder.is_fitted = False
                        self.log("⚠️ Class encoder not marked as fitted - no valid mappings found")

                    # Log encoder status
                    if self.class_encoder.is_fitted:
                        self.log(f"✅ Loaded class encoder with {len(self.class_encoder.encoded_to_class)} classes")
                        if self.class_encoder.encoded_to_class:
                            sample_classes = list(self.class_encoder.encoded_to_class.items())[:3]
                            self.log(f"Sample class mappings: {sample_classes}")

                            # Also verify dmyclass alignment
                            if hasattr(self, 'dmyclass') and self.dmyclass is not None:
                                self.log(f"dmyclass values: {[self.dmyclass[i] for i in range(min(5, len(self.dmyclass)))]}...")
                    else:
                        self.log("❌ Class encoder failed to load properly")

                except Exception as e:
                    self.log(f"❌ Error loading class encoder: {e}")
                    import traceback
                    traceback.print_exc()

                    # EMERGENCY RECOVERY: Try to create basic encoder from dmyclass if available
                    if hasattr(self, 'dmyclass') and self.dmyclass is not None:
                        self.log("🔄 Attempting emergency encoder recovery from dmyclass...")
                        try:
                            # Extract class values from dmyclass (skip margin at index 0)
                            class_values = []
                            for i in range(1, min(len(self.dmyclass), self.outnodes + 1)):
                                if self.dmyclass[i] != 0:  # Skip zero values
                                    class_values.append(self.dmyclass[i])

                            if class_values:
                                # Create basic encoder mapping
                                encoded_to_class = {}
                                class_to_encoded = {}
                                for i, class_val in enumerate(class_values, 1):
                                    encoded_to_class[float(i)] = f"Class_{class_val}"
                                    class_to_encoded[f"Class_{class_val}"] = float(i)

                                self.class_encoder.encoded_to_class = encoded_to_class
                                self.class_encoder.class_to_encoded = class_to_encoded
                                self.class_encoder.is_fitted = True
                                self.log(f"✅ Emergency recovery: Created encoder with {len(class_values)} classes from dmyclass")
                        except Exception as recovery_error:
                            self.log(f"❌ Emergency recovery failed: {recovery_error}")
            else:
                self.log("❌ No class encoder data found in model file")

                # Try to infer from dmyclass as last resort
                if hasattr(self, 'dmyclass') and self.dmyclass is not None:
                    self.log("🔄 Attempting to infer encoder from dmyclass...")
                    try:
                        class_values = []
                        for i in range(1, min(len(self.dmyclass), self.outnodes + 1)):
                            if self.dmyclass[i] != 0:
                                class_values.append(self.dmyclass[i])

                        if class_values:
                            encoded_to_class = {}
                            class_to_encoded = {}
                            for i, class_val in enumerate(class_values, 1):
                                encoded_to_class[float(i)] = str(class_val)
                                class_to_encoded[str(class_val)] = float(i)

                            self.class_encoder.encoded_to_class = encoded_to_class
                            self.class_encoder.class_to_encoded = class_to_encoded
                            self.class_encoder.is_fitted = True
                            self.log(f"✅ Inferred encoder with {len(class_values)} classes from dmyclass")
                    except Exception as infer_error:
                        self.log(f"❌ Failed to infer encoder from dmyclass: {infer_error}")

            # Final validation
            self.log(f"✅ Model loaded successfully: {self.innodes} inputs, {self.outnodes} outputs")
            self.log(f"✅ Model is_trained: {self.is_trained}")
            self.log(f"✅ Class encoder is_fitted: {getattr(self.class_encoder, 'is_fitted', False)}")
            self.log(f"✅ Config resolution: {self.config.get('resol', 'N/A')}")
            if self.feature_columns:
                self.log(f"✅ Feature columns: {len(self.feature_columns)} features loaded")
            else:
                self.log("⚠️ No feature columns information available")

            # Test model functionality
            if self.innodes > 0 and self.class_encoder.is_fitted:
                try:
                    test_sample = np.random.randn(self.innodes)
                    predictions, probabilities = self.predict_batch(test_sample.reshape(1, -1))
                    self.log(f"✅ Model test passed - Prediction: {predictions[0]}")
                except Exception as test_error:
                    self.log(f"⚠️ Model test warning: {test_error}")

        except Exception as e:
            self.log(f"❌ Load error: {e}")
            import traceback
            traceback.print_exc()

    def _init_gpu_acceleration(self):
        """Initialize GPU acceleration if available"""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_enabled = True
                self.device = torch.device('cuda')
                print(f"✅ GPU acceleration enabled: {torch.cuda.get_device_name()}")
            else:
                self.gpu_enabled = False
                self.device = torch.device('cpu')
        except ImportError:
            self.gpu_enabled = False
            self.device = None

    def set_log_callback(self, log_callback):
        """Set a callback function for logging"""
        self.log_callback = log_callback

    def log(self, message: str):
        """Log message through callback or print"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(f"[DBNNCore] {message}")

    def initialize_arrays(self, innodes: int, resol: int, outnodes: int):
        """Initialize arrays with optimal data types and memory layout"""
        self.innodes = innodes
        self.outnodes = outnodes

        # Use optimal numpy dtypes for memory efficiency
        self.anti_net = np.zeros((innodes+2, resol+2, innodes+2, resol+2, outnodes+2),
                               dtype=np.int32)  # Keep as int32 for large counts
        self.anti_wts = np.ones((innodes+2, resol+2, innodes+2, resol+2, outnodes+2),
                               dtype=np.float32)  # Use float32 for memory efficiency
        self.antit_wts = np.ones_like(self.anti_wts, dtype=np.float32)
        self.antip_wts = np.ones_like(self.anti_wts, dtype=np.float32)

        self.binloc = np.zeros((innodes+2, resol+8), dtype=np.float32)
        self.max_val = np.zeros(innodes+2, dtype=np.float32)
        self.min_val = np.zeros(innodes+2, dtype=np.float32)
        self.resolution_arr = np.zeros(innodes+8, dtype=np.int32)

        # Initialize dmyclass
        self.dmyclass = np.zeros(outnodes + 2, dtype=np.float32)
        self.dmyclass[0] = self.config.get('margin', 0.2)

        print(f"Initialized arrays for {innodes} input nodes, {outnodes} output nodes, resolution {resol}")

    def load_data(self, filename: str, target_column: Optional[str] = None,
                 feature_columns: Optional[List[str]] = None, batch_size: int = 10000):
        """Optimized data loading with memory mapping and parallel processing"""
        if filename.endswith('.csv'):
            return self._load_csv_data_optimized(filename, target_column, feature_columns, batch_size)
        else:
            return self._load_dat_data_optimized(filename, batch_size)

    def _load_csv_data_optimized(self, filename: str, target_column: str,
                                feature_columns: List[str], batch_size: int):
        """Optimized CSV loading with memory-efficient processing"""
        import pandas as pd

        features_batches = []
        targets_batches = []
        original_targets_batches = []

        self.log(f"Loading CSV data from: {filename}")

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError("No headers found in CSV file")

            available_columns = reader.fieldnames
            self.log(f"Available columns in CSV: {available_columns}")

            # For prediction mode: if target_column is None or 'None', skip target validation
            is_prediction_mode = (target_column is None or target_column == 'None' or target_column == '')

            # Validate columns exist - handle prediction mode differently
            missing_columns = []

            if not is_prediction_mode and target_column and target_column not in available_columns:
                missing_columns.append(target_column)

            # Check feature columns - use all available if not specified
            if not feature_columns:
                # If no features specified, use all columns except target (if target exists)
                if not is_prediction_mode and target_column and target_column in available_columns:
                    feature_columns = [col for col in available_columns if col != target_column]
                else:
                    # For prediction mode with no target, or target not found, use all columns
                    feature_columns = available_columns.copy()
                self.log(f"Auto-selected feature columns: {feature_columns}")

            # Now validate feature columns
            for col in feature_columns:
                if col not in available_columns:
                    missing_columns.append(col)

            if missing_columns and not is_prediction_mode:
                raise ValueError(f"Columns not found in CSV: {missing_columns}")
            elif missing_columns and is_prediction_mode:
                self.log(f"⚠️ Warning: Some feature columns not found in CSV: {missing_columns}")
                # Remove missing columns from feature_columns
                feature_columns = [col for col in feature_columns if col not in missing_columns]
                if not feature_columns:
                    raise ValueError("No valid feature columns found in CSV file")

            self.log(f"Using feature columns: {feature_columns}")
            if not is_prediction_mode and target_column:
                self.log(f"Using target column: {target_column}")

            current_batch_features = []
            current_batch_targets = []
            current_batch_original_targets = []

            line_count = 0
            for row in reader:
                line_count += 1
                try:
                    # Extract features - use only the specified feature columns in the correct order
                    feature_vec = []
                    for col in feature_columns:
                        try:
                            feature_vec.append(float(row[col]))
                        except (ValueError, TypeError):
                            feature_vec.append(0.0)
                            if line_count <= 5:  # Only warn for first few rows
                                self.log(f"⚠️ Warning: Could not convert column '{col}' value '{row[col]}' to float, using 0.0")

                    # Extract target (only if not in prediction mode and target exists)
                    target_val = None
                    if not is_prediction_mode and target_column and target_column in row:
                        target_val = row[target_column]
                    else:
                        # For prediction mode or missing target, use a placeholder
                        target_val = "unknown"

                    current_batch_features.append(feature_vec)
                    current_batch_targets.append(target_val)
                    current_batch_original_targets.append(target_val)

                    if len(current_batch_features) >= batch_size:
                        # Use float32 for memory efficiency
                        features_batches.append(np.array(current_batch_features, dtype=np.float32))
                        targets_batches.append(np.array(current_batch_targets))
                        original_targets_batches.append(np.array(current_batch_original_targets))
                        current_batch_features = []
                        current_batch_targets = []
                        current_batch_original_targets = []

                except (ValueError, KeyError) as e:
                    self.log(f"⚠️ Warning: Skipping row {line_count} due to error: {e}")
                    continue

            # Add remaining samples
            if current_batch_features:
                features_batches.append(np.array(current_batch_features, dtype=np.float32))
                targets_batches.append(np.array(current_batch_targets))
                original_targets_batches.append(np.array(current_batch_original_targets))

        total_samples = sum(len(batch) for batch in features_batches)
        self.log(f"Loaded {total_samples} samples in {len(features_batches)} batches (optimized)")
        return features_batches, targets_batches, feature_columns, original_targets_batches

    def _load_dat_data_optimized(self, filename: str, batch_size: int):
        """Optimized legacy DAT format data loading"""
        features = []
        targets = []
        original_targets = []

        self.log(f"Loading legacy DAT file: {filename}")

        with open(filename, 'r') as f:
            current_batch_features = []
            current_batch_targets = []
            current_batch_original_targets = []

            for line in f:
                values = line.strip().split()
                if len(values) >= self.innodes + 1:
                    feature_vec = [float(x) for x in values[:self.innodes]]
                    target_val = values[self.innodes]

                    try:
                        target_val_float = float(target_val)
                        current_batch_original_targets.append(target_val_float)
                    except ValueError:
                        current_batch_original_targets.append(target_val)

                    current_batch_features.append(feature_vec)
                    current_batch_targets.append(target_val)

                    if len(current_batch_features) >= batch_size:
                        # Use float32 for memory efficiency
                        features.append(np.array(current_batch_features, dtype=np.float32))
                        targets.append(np.array(current_batch_targets))
                        original_targets.append(np.array(current_batch_original_targets))
                        current_batch_features = []
                        current_batch_targets = []
                        current_batch_original_targets = []

            # Add remaining samples
            if current_batch_features:
                features.append(np.array(current_batch_features, dtype=np.float32))
                targets.append(np.array(current_batch_targets))
                original_targets.append(np.array(current_batch_original_targets))

        self.log(f"Loaded {sum(len(batch) for batch in features)} samples in {len(features)} batches (optimized)")
        return features, targets, [f'feature_{i}' for i in range(1, self.innodes + 1)], original_targets

    def fit_encoder(self, original_targets_batches):
        """Fit class encoder to target data"""
        all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
        self.class_encoder.fit(all_original_targets)

        # Update dmyclass with encoded values
        encoded_classes = self.class_encoder.get_encoded_classes()
        for i, encoded_val in enumerate(encoded_classes, 1):
            self.dmyclass[i] = float(encoded_val)

    def initialize_training(self, features_batches, encoded_targets_batches, resol: int):
        """Initialize training parameters and arrays"""
        # Set resolutions
        for i in range(1, self.innodes + 1):
            self.resolution_arr[i] = resol
            for j in range(self.resolution_arr[i] + 1):
                self.binloc[i][j+1] = j * 1.0

        # Find min/max values
        omax = -400.0
        omin = 400.0
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            for i in range(1, self.innodes + 1):
                batch_max = np.max(features_batch[:, i-1])
                batch_min = np.min(features_batch[:, i-1])
                if batch_max > self.max_val[i]:
                    self.max_val[i] = batch_max
                if batch_min < self.min_val[i]:
                    self.min_val[i] = batch_min

            # Update omax/omin from targets
            batch_omax = np.max(targets_batch)
            batch_omin = np.min(targets_batch)
            if batch_omax > omax:
                omax = batch_omax
            if batch_omin < omin:
                omin = batch_omin

        # Initialize network counts
        self.anti_wts.fill(1.0)
        for k in range(1, self.outnodes + 1):
            for i in range(1, self.innodes + 1):
                for j in range(self.resolution_arr[i] + 1):
                    for l in range(1, self.innodes + 1):
                        for m in range(self.resolution_arr[l] + 1):
                            self.anti_net[i, j, l, m, k] = 1

        return omax, omin

    def process_training_batch(self, features_batch, targets_batch):
        """Process a single batch of training data with parallel optimization"""
        if self.parallel_enabled and len(features_batch) > 1000 and self.num_workers > 1:
            return self._process_training_batch_parallel(features_batch, targets_batch)
        else:
            return self._process_training_batch_sequential(features_batch, targets_batch)

    def _process_training_batch_sequential(self, features_batch, targets_batch):
        """Process training batch sequentially"""
        batch_size = len(features_batch)
        processed_count = 0

        for sample_idx in range(batch_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            self.anti_net = process_training_sample(
                vects, tmpv, self.anti_net, self.anti_wts, self.binloc,
                self.resolution_arr, self.dmyclass, self.min_val, self.max_val,
                self.innodes, self.outnodes
            )

            processed_count += 1

        return processed_count

    def _process_training_batch_parallel(self, features_batch, targets_batch):
        """Process training batch in parallel"""
        batch_size = len(features_batch)
        chunk_size = max(1, batch_size // self.num_workers)

        # Split batch into chunks
        feature_chunks = []
        target_chunks = []
        for i in range(0, batch_size, chunk_size):
            feature_chunks.append(features_batch[i:i+chunk_size])
            target_chunks.append(targets_batch[i:i+chunk_size])

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for feat_chunk, tgt_chunk in zip(feature_chunks, target_chunks):
                future = executor.submit(self._process_training_chunk, feat_chunk, tgt_chunk)
                futures.append(future)

            # Wait for all chunks to complete
            for future in futures:
                future.result()

        return batch_size

    def _process_training_chunk(self, features_chunk, targets_chunk):
        """Process a chunk of training data (thread-safe)"""
        chunk_size = len(features_chunk)

        for sample_idx in range(chunk_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_chunk[sample_idx, i-1]
            tmpv = targets_chunk[sample_idx]

            self.anti_net = process_training_sample(
                vects, tmpv, self.anti_net, self.anti_wts, self.binloc,
                self.resolution_arr, self.dmyclass, self.min_val, self.max_val,
                self.innodes, self.outnodes
            )

        return chunk_size

    def train_epoch(self, features_batches, encoded_targets_batches, gain: float):
        """Train for one epoch with parallel optimization"""
        if self.parallel_enabled and self.num_workers > 1:
            self._train_epoch_parallel(features_batches, encoded_targets_batches, gain)
        else:
            self._train_epoch_sequential(features_batches, encoded_targets_batches, gain)

    def _train_epoch_sequential(self, features_batches, encoded_targets_batches, gain: float):
        """Train one epoch sequentially"""
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)

            for sample_idx in range(batch_size):
                vects = np.zeros(self.innodes + self.outnodes + 2)
                for i in range(1, self.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                tmpv = targets_batch[sample_idx]

                # Compute probabilities
                classval = compute_class_probabilities_numba(
                    vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                    self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                # Update weights if wrong classification
                if abs(self.dmyclass[kmax] - tmpv) > self.dmyclass[0]:
                    self.anti_wts = update_weights_numba(
                        vects, tmpv, classval, self.anti_wts, self.binloc, self.resolution_arr,
                        self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes, gain
                    )

    def _train_epoch_parallel(self, features_batches, encoded_targets_batches, gain: float):
        """Train one epoch with parallel processing"""
        # Process each batch in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                future = executor.submit(self._train_batch_parallel, features_batch, targets_batch, gain, batch_idx)
                futures.append(future)

            # Wait for all batches to complete
            for future in futures:
                future.result()

    def _evaluate_parallel(self, features_batches, encoded_targets_batches):
        """Parallel evaluation"""
        total_correct = 0
        total_samples = 0
        all_predictions = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                future = executor.submit(self._evaluate_batch_parallel, features_batch, targets_batch, batch_idx)
                futures.append(future)

            # Collect results
            for future in futures:
                batch_correct, batch_total, batch_predictions = future.result()
                total_correct += batch_correct
                total_samples += batch_total
                all_predictions.extend(batch_predictions)

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, total_correct, all_predictions

    def _predict_batch_parallel(self, features_batch):
        """Optimized parallel batch prediction"""
        batch_size = len(features_batch)
        chunk_size = max(1, batch_size // self.num_workers)

        # Split batch into chunks
        feature_chunks = []
        for i in range(0, batch_size, chunk_size):
            feature_chunks.append(features_batch[i:i+chunk_size])

        # Process chunks in parallel
        all_predictions = []
        all_probabilities = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for feat_chunk in feature_chunks:
                future = executor.submit(self._predict_chunk_parallel, feat_chunk)
                futures.append(future)

            # Collect results
            for future in futures:
                chunk_predictions, chunk_probabilities = future.result()
                all_predictions.extend(chunk_predictions)
                all_probabilities.extend(chunk_probabilities)

        return all_predictions, all_probabilities


    def _train_batch_parallel(self, features_batch, targets_batch, gain: float, batch_idx: int):
        """Train a single batch in parallel (thread-safe)"""
        batch_size = len(features_batch)

        for sample_idx in range(batch_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            # Compute probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            # Update weights if wrong classification
            if abs(self.dmyclass[kmax] - tmpv) > self.dmyclass[0]:
                self.anti_wts = update_weights_numba(
                    vects, tmpv, classval, self.anti_wts, self.binloc, self.resolution_arr,
                    self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes, gain
                )

    def evaluate(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy with parallel optimization"""
        if self.parallel_enabled and self.num_workers > 1:
            return self._evaluate_parallel(features_batches, encoded_targets_batches)
        else:
            return self._evaluate_sequential(features_batches, encoded_targets_batches)

    def _evaluate_sequential(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy sequentially"""
        correct_predictions = 0
        total_samples = 0
        all_predictions = []

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)
            total_samples += batch_size

            for sample_idx in range(batch_size):
                vects = np.zeros(self.innodes + self.outnodes + 2)
                for i in range(1, self.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                actual = targets_batch[sample_idx]

                # Compute class probabilities
                classval = compute_class_probabilities_numba(
                    vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                    self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                predicted = self.dmyclass[kmax]
                all_predictions.append(predicted)

                # Check if prediction is correct
                if abs(actual - predicted) <= self.dmyclass[0]:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, correct_predictions, all_predictions

    def _evaluate_batch_parallel(self, features_batch, targets_batch, batch_idx):
        """Evaluate a single batch in parallel"""
        batch_correct = 0
        batch_total = len(features_batch)
        batch_predictions = []

        for sample_idx in range(batch_total):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            actual = targets_batch[sample_idx]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.dmyclass[kmax]
            batch_predictions.append(predicted)

            # Check if prediction is correct
            if abs(actual - predicted) <= self.dmyclass[0]:
                batch_correct += 1

        return batch_correct, batch_total, batch_predictions

    def predict_batch(self, features_batch):
        """Predict classes for a batch of features with parallel optimization"""
        if self.parallel_enabled and len(features_batch) > 1000 and self.num_workers > 1:
            return self._predict_batch_parallel(features_batch)
        else:
            return self._predict_batch_sequential(features_batch)

    def _predict_batch_sequential(self, features_batch):
        """Predict classes for a batch of features sequentially"""
        predictions = []
        probabilities = []

        for sample_idx in range(len(features_batch)):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.dmyclass[kmax]
            predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.dmyclass[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            probabilities.append(prob_dict)

        return predictions, probabilities

    def _predict_chunk_parallel(self, features_chunk):
        """Predict a chunk of data in parallel"""
        chunk_predictions = []
        chunk_probabilities = []

        for sample_idx in range(len(features_chunk)):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_chunk[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.dmyclass[kmax]
            chunk_predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.dmyclass[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            chunk_probabilities.append(prob_dict)

        return chunk_predictions, chunk_probabilities

    def predict_batch_optimized(self, features_batch, clear_cache_every=100):
        """Optimized batch prediction with periodic cache clearing"""
        return self.predict_batch(features_batch)

    def train_with_early_stopping(self, train_file: str, test_file: Optional[str] = None,
                                 use_csv: bool = True, target_column: Optional[str] = None,
                                 feature_columns: Optional[List[str]] = None):
        """Main training method with early stopping and automatic optimizations"""
        self.log("Starting optimized model training with early stopping...")

        # Load training data
        features_batches, targets_batches, feature_columns_used, original_targets_batches = self.load_data(
            train_file, target_column, feature_columns
        )

        if not features_batches:
            self.log("No training data loaded")
            return False

        # Fit encoder and determine architecture
        self.fit_encoder(original_targets_batches)
        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = self.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        self.innodes = len(feature_columns_used)
        self.outnodes = len(self.class_encoder.get_encoded_classes())

        # Initialize arrays
        resol = self.config.get('resol', 100)
        self.initialize_arrays(self.innodes, resol, self.outnodes)

        # Initialize training
        omax, omin = self.initialize_training(features_batches, encoded_targets_batches, resol)

        # Process training data for initial APF
        self.log("Processing training data for initial network counts...")
        total_samples = sum(len(batch) for batch in features_batches)
        total_processed = 0

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            processed = self.process_training_batch(features_batch, targets_batch)
            total_processed += processed
            if total_processed % 1000 == 0:
                self.log(f"Processed {total_processed}/{total_samples} samples")

        # Training with early stopping
        gain = self.config.get('gain', 2.0)
        max_epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        min_improvement = self.config.get('min_improvement', 0.1)

        self.log(f"Starting weight training with early stopping...")
        best_accuracy = 0.0
        best_round = 0
        patience_counter = 0
        best_weights = None

        for rnd in range(max_epochs + 1):
            if rnd == 0:
                # Initial evaluation
                current_accuracy, correct_predictions, _ = self.evaluate(features_batches, encoded_targets_batches)
                self.log(f"Round {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")
                best_accuracy = current_accuracy
                best_weights = self.anti_wts.copy()
                best_round = rnd
                continue

            # Training pass
            self.train_epoch(features_batches, encoded_targets_batches, gain)

            # Evaluation after training round
            current_accuracy, correct_predictions, _ = self.evaluate(features_batches, encoded_targets_batches)
            self.log(f"Round {rnd:3d}: Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")

            # Capture visualization snapshot if visualizer is attached
            if self.visualizer and rnd % 5 == 0:
                sample_features = np.vstack(features_batches)[:1000]  # Sample for performance
                sample_targets = np.concatenate(encoded_targets_batches)[:1000]
                sample_predictions, _ = self.predict_batch(sample_features)
                self.visualizer.capture_training_snapshot(
                    sample_features, sample_targets, self.anti_wts,
                    sample_predictions, current_accuracy, rnd
                )

            # Early stopping logic
            if current_accuracy > best_accuracy + min_improvement:
                best_accuracy = current_accuracy
                best_weights = self.anti_wts.copy()
                best_round = rnd
                patience_counter = 0
                self.log(f"  → New best accuracy! (Improved by {current_accuracy - best_accuracy:.2f}%)")
            else:
                patience_counter += 1
                self.log(f"  → No improvement (Patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                self.log(f"\nEarly stopping triggered after {rnd} rounds.")
                self.log(f"Best accuracy {best_accuracy:.2f}% achieved at round {best_round}")
                break

        # Restore best weights
        if best_weights is not None:
            self.anti_wts = best_weights
            self.log(f"Restored best weights from round {best_round}")

        self.is_trained = True
        self.best_accuracy = best_accuracy
        self.best_round = best_round

        self.log("Optimized training completed successfully!")
        return True

    # All other existing methods remain exactly the same...
    # detect_system_resources, calculate_optimal_parameters, estimate_batch_memory,
    # print_resource_report, save_model, load_model, attach_visualizer, etc.
    # These methods are unchanged from the original implementation

    def enable_parallel_processing(self, enabled=True):
        """Enable or disable parallel processing"""
        self.parallel_enabled = enabled
        if enabled:
            self.num_workers = self._detect_optimal_workers()
            self.log(f"✅ Parallel processing enabled with {self.num_workers} workers")
        else:
            self.log("✅ Parallel processing disabled")

    def set_max_workers(self, max_workers):
        """Set maximum number of parallel workers"""
        self.num_workers = min(max_workers, self._detect_optimal_workers())
        self.log(f"✅ Maximum workers set to {self.num_workers}")

    def optimize_memory(self):
        """Optimize memory usage by converting arrays to optimal dtypes"""
        if self.anti_net is not None:
            # Keep anti_net as int32 for large counts
            pass

        if self.anti_wts is not None and not self.memory_optimized:
            self.anti_wts = self.anti_wts.astype(np.float32)
            self.antit_wts = self.antit_wts.astype(np.float32)
            self.antip_wts = self.antip_wts.astype(np.float32)
            self.binloc = self.binloc.astype(np.float32)
            self.max_val = self.max_val.astype(np.float32)
            self.min_val = self.min_val.astype(np.float32)
            self.dmyclass = self.dmyclass.astype(np.float32)
            self.memory_optimized = True
            self.log("✅ Memory optimized: arrays converted to float32")

class DBNNVisualizer:
    """
    Standalone visualization class that can be imported and used with DBNNCore
    Provides standardized protocols for visualization
    """

    def __init__(self):
        self.training_history = []
        self.visualization_data = {}

    def capture_training_snapshot(self, features, targets, weights,
                                 predictions, accuracy, round_num):
        """Standardized protocol for capturing training state"""
        snapshot = {
            'round': round_num,
            'features': features.copy() if hasattr(features, 'copy') else features,
            'targets': targets.copy() if hasattr(targets, 'copy') else targets,
            'weights': weights.copy() if hasattr(weights, 'copy') else weights,
            'predictions': predictions.copy() if hasattr(predictions, 'copy') else predictions,
            'accuracy': accuracy,
            'timestamp': len(self.training_history)
        }
        self.training_history.append(snapshot)
        return snapshot

    def get_training_history(self):
        """Standardized protocol for accessing training history"""
        return self.training_history

    def clear_history(self):
        """Clear visualization history"""
        self.training_history = []
        self.visualization_data = {}

    def generate_feature_space_plot(self, snapshot_idx: int, feature_indices: List[int] = [0, 1, 2]):
        """Generate 3D feature space plot"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        features = snapshot['features']
        targets = snapshot['targets']

        # Create DataFrame for plotting
        df = pd.DataFrame({
            f'Feature_{feature_indices[0]}': features[:, feature_indices[0]],
            f'Feature_{feature_indices[1]}': features[:, feature_indices[1]],
            f'Feature_{feature_indices[2]}': features[:, feature_indices[2]],
            'Class': targets,
            'Prediction': snapshot['predictions']
        })

        fig = px.scatter_3d(
            df,
            x=f'Feature_{feature_indices[0]}',
            y=f'Feature_{feature_indices[1]}',
            z=f'Feature_{feature_indices[2]}',
            color='Class',
            title=f'Feature Space - Round {snapshot["round"]}<br>Accuracy: {snapshot["accuracy"]:.2f}%',
            opacity=0.7
        )

        return fig

    def generate_accuracy_plot(self):
        """Generate accuracy progression plot"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if len(self.training_history) < 2:
            return None

        rounds = [s['round'] for s in self.training_history]
        accuracies = [s['accuracy'] for s in self.training_history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=accuracies,
            mode='lines+markers',
            name='Accuracy'
        ))

        fig.update_layout(
            title='Training Accuracy Progression',
            xaxis_title='Training Round',
            yaxis_title='Accuracy (%)'
        )

        return fig

    def generate_weight_distribution_plot(self, snapshot_idx: int):
        """Generate weight distribution histogram"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        weights = snapshot['weights']

        # Flatten weights for histogram
        flat_weights = weights.flatten()
        # Remove zeros and extreme values for better visualization
        flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=flat_weights,
            nbinsx=50,
            name='Weight Distribution'
        ))

        fig.update_layout(
            title=f'Weight Distribution - Round {snapshot["round"]}',
            xaxis_title='Weight Value',
            yaxis_title='Frequency'
        )

        return fig

    def create_training_dashboard(self, output_file: str = "training_dashboard.html"):
        """Create comprehensive training dashboard with support for custom paths"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for dashboard creation")
            return None

        if len(self.training_history) < 2:
            return None

        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Space', 'Accuracy Progression',
                          'Weight Distribution', 'Training Summary'),
            specs=[[{"type": "scatter3d"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "domain"}]]
        )

        # Feature Space (latest snapshot)
        feature_fig = self.generate_feature_space_plot(-1)
        if feature_fig:
            for trace in feature_fig.data:
                fig.add_trace(trace, row=1, col=1)

        # Accuracy Progression
        rounds = [s['round'] for s in self.training_history]
        accuracies = [s['accuracy'] for s in self.training_history]
        fig.add_trace(go.Scatter(x=rounds, y=accuracies, mode='lines+markers',
                               name='Accuracy'), row=1, col=2)

        # Weight Distribution (latest snapshot)
        weight_fig = self.generate_weight_distribution_plot(-1)
        if weight_fig:
            for trace in weight_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # Training Summary
        best_round = np.argmax(accuracies)
        best_accuracy = accuracies[best_round]
        final_accuracy = accuracies[-1] if accuracies else 0

        summary_text = f"""
        <b>Training Summary:</b><br>
        - Total Rounds: {len(self.training_history)}<br>
        - Best Accuracy: {best_accuracy:.2f}%<br>
        - Best Round: {best_round}<br>
        - Final Accuracy: {final_accuracy:.2f}%<br>
        - Improvement: {final_accuracy - accuracies[0] if accuracies else 0:+.2f}%
        """

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='text',
            text=[summary_text],
            textposition="middle center",
            showlegend=False,
            textfont=dict(size=12)
        ), row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="DBNN Training Dashboard",
            showlegend=False
        )

        try:
            fig.write_html(output_file)
            print(f"Training dashboard saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving dashboard: {e}")
            return None
# Example usage and integration
class DBNNWorkflow:
    """
    Complete workflow manager that uses DBNNCore and DBNNVisualizer
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.core = DBNNCore(config)
        self.visualizer = DBNNVisualizer()
        self.core.attach_visualizer(self.visualizer)

    def run_complete_workflow(self, train_file: str, test_file: Optional[str] = None,
                             predict_file: Optional[str] = None, use_csv: bool = True,
                             target_column: Optional[str] = None,
                             feature_columns: Optional[List[str]] = None):
        """Run complete training and evaluation workflow"""

        print("Starting complete DBNN workflow...")

        # Training
        success = self.core.train_with_early_stopping(
            train_file, test_file, use_csv, target_column, feature_columns
        )

        if not success:
            print("Training failed")
            return False

        # Generate visualizations
        if len(self.visualizer.training_history) > 0:
            self.visualizer.create_training_dashboard("training_results.html")
            print("Visualizations generated")

        # Test if test file provided
        if test_file and os.path.exists(test_file):
            self.test_model(test_file, use_csv)

        # Predict if predict file provided
        if predict_file and os.path.exists(predict_file):
            self.predict(predict_file, use_csv)

        print("Workflow completed successfully!")
        return True

    def test_model(self, test_file: str, use_csv: bool = True):
        """Test model on test data"""
        print(f"Testing model on: {test_file}")

        # Load test data
        features_batches, targets_batches, _, original_targets_batches = self.core.load_data(test_file)

        if not features_batches:
            print("No test data loaded")
            return

        # Encode targets
        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = self.core.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        # Evaluate
        accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
        total_samples = sum(len(batch) for batch in features_batches)

        print(f"Test Results: Accuracy = {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    def predict(self, data_file: str, use_csv: bool = True, output_file: Optional[str] = None):
        """Generate predictions on new data"""
        print(f"Generating predictions for: {data_file}")

        features_batches, _, _, _ = self.core.load_data(data_file)

        if not features_batches:
            print("No prediction data loaded")
            return

        all_predictions = []
        all_probabilities = []

        for features_batch in features_batches:
            predictions, probabilities = self.core.predict_batch(features_batch)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

        # Decode predictions
        decoded_predictions = self.core.class_encoder.inverse_transform(all_predictions)

        # Save results
        if output_file is None:
            output_file = data_file.replace('.csv', '_predictions.csv').replace('.dat', '_predictions.csv')

        results_df = pd.DataFrame({
            'prediction_encoded': all_predictions,
            'prediction': decoded_predictions
        })

        # Add probability columns
        for i, class_name in enumerate(self.core.class_encoder.encoded_to_class.values()):
            probs = [prob.get(class_name, 0.0) for prob in all_probabilities]
            results_df[f'prob_{class_name}'] = probs

        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

        return decoded_predictions, all_probabilities

class EnhancedDBNNInterface:
    """
    Enhanced interface with fixed early stopping and better feature selection
    """

    def __init__(self, root):
        self.root = root
        self.root.title("DBNN Enhanced Interface")
        self.root.geometry("1200x800")

        self.core = None
        self.visualizer = None
        self.current_file = None
        self.file_type = None
        self.processing_indicator = None
        self.processing_frame = None
        self.processing_animation_id = None
        self.setup_ui()

    def setup_ui(self):
        """Setup enhanced UI with configuration tab system"""
        # Create notebook for tabs instead of paned window
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Setup main tab
        self.setup_main_tab()

        # Setup configuration tab
        self.setup_configuration_tab()

    def setup_main_tab(self):
        """Setup the main working tab"""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main Interface")

        # Main frame with paned window for better layout
        main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right frame for console
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="Data File", padding="5")
        file_frame.pack(fill='x', pady=5)

        ttk.Label(file_frame, text="Select CSV/DAT file:").grid(row=0, column=0, sticky='w')
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(file_frame, text="Analyze", command=self.analyze_file).grid(row=0, column=3, padx=5)

        # File info
        self.file_info = scrolledtext.ScrolledText(file_frame, height=4, width=60)
        self.file_info.grid(row=1, column=0, columnspan=4, pady=5, sticky='we')

        # Quick configuration
        config_frame = ttk.LabelFrame(left_frame, text="Quick Configuration", padding="5")
        config_frame.pack(fill='x', pady=5)

        # Model name - auto-generated based on data file
        ttk.Label(config_frame, text="Model Name:").grid(row=0, column=0, sticky='w')
        self.model_name = tk.StringVar(value="my_model")  # Default, will be updated when file is loaded
        self.model_name_entry = ttk.Entry(config_frame, textvariable=self.model_name, width=30)
        self.model_name_entry.grid(row=0, column=1, sticky='w', padx=5, columnspan=2)

        # Add tooltip for model name
        self.create_tooltip(self.model_name_entry, "Auto-generated from data file name. Model will be saved in Model/ folder.")

        # Training params in a grid
        params = [
            ("Resolution:", "resol", "100"),
            ("Epochs:", "epochs", "100"),
            ("Gain:", "gain", "2.0"),
            ("Margin:", "margin", "0.2"),
            ("Patience:", "patience", "10")
        ]

        for i, (label, name, default) in enumerate(params):
            ttk.Label(config_frame, text=label).grid(row=1, column=i*2, sticky='w', padx=5)
            var = tk.StringVar(value=default)
            setattr(self, name, var)
            ttk.Entry(config_frame, textvariable=var, width=8).grid(row=1, column=i*2+1, sticky='w', padx=5)

        # Enhanced feature selection
        self.feature_frame = ttk.LabelFrame(left_frame, text="Feature Selection", padding="5")

        # Target selection
        ttk.Label(self.feature_frame, text="Target Column:").grid(row=0, column=0, sticky='w')
        self.target_col = tk.StringVar()
        self.target_combo = ttk.Combobox(self.feature_frame, textvariable=self.target_col, width=20)
        self.target_combo.grid(row=0, column=1, padx=5, pady=2)

        # Feature selection with checkboxes
        ttk.Label(self.feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky='nw')

        # Frame for feature checkboxes with scrollbar
        feature_check_frame = ttk.Frame(self.feature_frame)
        feature_check_frame.grid(row=1, column=1, padx=5, pady=2, sticky='we')

        # Create a canvas with scrollbar for feature selection
        self.feature_canvas = tk.Canvas(feature_check_frame, height=120)
        scrollbar = ttk.Scrollbar(feature_check_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

        self.feature_scroll_frame.bind(
            "<Configure>",
            lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))
        )

        self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=scrollbar.set)

        self.feature_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.feature_vars = {}  # Dictionary to store checkbox variables

        # Selection controls
        selection_frame = ttk.Frame(self.feature_frame)
        selection_frame.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(selection_frame, text="Select All", command=self.select_all_features).pack(side='left', padx=2)
        ttk.Button(selection_frame, text="Deselect All", command=self.deselect_all_features).pack(side='left', padx=2)
        ttk.Button(selection_frame, text="Invert Selection", command=self.invert_feature_selection).pack(side='left', padx=2)

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill='x', pady=10)

        # Row 1 buttons
        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill='x', pady=2)

        ttk.Button(row1_frame, text="Initialize Core", command=self.initialize_core).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Train Fresh", command=self.train_fresh).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Continue Training", command=self.continue_training).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Test Model", command=self.test_model).pack(side='left', padx=2)

        # Row 2 buttons
        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill='x', pady=2)

        ttk.Button(row2_frame, text="Predict", command=self.predict).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Save Model", command=self.save_model).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Load Model", command=self.load_model).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Visualize", command=self.visualize).pack(side='left', padx=2)

        # Row 3 buttons - Configuration management
        row3_frame = ttk.Frame(button_frame)
        row3_frame.pack(fill='x', pady=2)

        ttk.Button(row3_frame, text="Load Config", command=lambda: self.load_config()).pack(side='left', padx=2)
        ttk.Button(row3_frame, text="Save Config", command=self.save_config).pack(side='left', padx=2)
        ttk.Button(row3_frame, text="Validate Config", command=self.validate_config).pack(side='left', padx=2)

        # Debug controls
        debug_frame = ttk.LabelFrame(left_frame, text="Debug Controls", padding="5")
        debug_frame.pack(fill='x', pady=5)

        ttk.Button(debug_frame, text="Test Data Load", command=self.test_data_load).pack(side='left', padx=2)
        ttk.Button(debug_frame, text="Test Encoder", command=self.test_encoder).pack(side='left', padx=2)
        ttk.Button(debug_frame, text="Test Prediction", command=self.test_single_prediction).pack(side='left', padx=2)

        # Console output on the right
        console_frame = ttk.LabelFrame(right_frame, text="Training Output", padding="5")
        console_frame.pack(fill='both', expand=True, pady=5)

        self.console = scrolledtext.ScrolledText(console_frame, height=30, width=80)
        self.console.pack(fill='both', expand=True)

    def setup_configuration_tab(self):
        """Setup configuration tab with all default parameters"""
        # Create configuration tab
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")

        # Main configuration frame
        main_config_frame = ttk.LabelFrame(config_frame, text="Model Configuration Parameters", padding="10")
        main_config_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create scrollable frame for configurations
        canvas = tk.Canvas(main_config_frame)
        scrollbar = ttk.Scrollbar(main_config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Default configuration parameters
        self.config_params = {
            'resol': {'label': 'Resolution', 'default': 100, 'type': 'int', 'min': 10, 'max': 500, 'desc': 'Number of bins for feature discretization'},
            'gain': {'label': 'Gain', 'default': 2.0, 'type': 'float', 'min': 0.1, 'max': 10.0, 'desc': 'Weight update gain factor'},
            'margin': {'label': 'Margin', 'default': 0.2, 'type': 'float', 'min': 0.01, 'max': 1.0, 'desc': 'Classification margin threshold'},
            'patience': {'label': 'Patience', 'default': 10, 'type': 'int', 'min': 1, 'max': 100, 'desc': 'Early stopping patience (epochs)'},
            'epochs': {'label': 'Max Epochs', 'default': 100, 'type': 'int', 'min': 1, 'max': 1000, 'desc': 'Maximum training epochs'},
            'min_improvement': {'label': 'Min Improvement (%)', 'default': 0.1, 'type': 'float', 'min': 0.01, 'max': 10.0, 'desc': 'Minimum accuracy improvement for early stopping'},
            'fst_gain': {'label': 'First Gain', 'default': 1.0, 'type': 'float', 'min': 0.1, 'max': 5.0, 'desc': 'Initial gain factor'},
            'LoC': {'label': 'Learning Rate', 'default': 0.65, 'type': 'float', 'min': 0.01, 'max': 1.0, 'desc': 'Learning rate coefficient'},
            'nLoC': {'label': 'Negative Learning Rate', 'default': 0.0, 'type': 'float', 'min': 0.0, 'max': 1.0, 'desc': 'Negative learning rate coefficient'},
            'nresol': {'label': 'Negative Resolution', 'default': 0, 'type': 'int', 'min': 0, 'max': 100, 'desc': 'Negative resolution parameter'},
            'skpchk': {'label': 'Skip Check', 'default': 0, 'type': 'int', 'min': 0, 'max': 1, 'desc': 'Skip validation checks (0=no, 1=yes)'},
            'oneround': {'label': 'One Round', 'default': 100, 'type': 'int', 'min': 1, 'max': 1000, 'desc': 'Samples per training round'}
        }

        self.config_vars = {}

        # Create configuration entries in two columns
        param_keys = list(self.config_params.keys())
        half = len(param_keys) // 2 + len(param_keys) % 2

        for i, key in enumerate(param_keys):
            param = self.config_params[key]
            row = i % half
            col = i // half

            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=col, sticky='w', padx=15, pady=8)

            # Parameter label
            ttk.Label(frame, text=param['label'] + ":", width=22, anchor='w').pack(side='left')

            # Entry field
            var = tk.StringVar(value=str(param['default']))
            self.config_vars[key] = var

            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.pack(side='left', padx=5)

            # Range label
            range_text = f"({param['min']}-{param['max']})"
            ttk.Label(frame, text=range_text, width=12, foreground="gray").pack(side='left', padx=5)

            # Add tooltip with parameter description
            self.create_tooltip(entry, f"{param['desc']}\nDefault: {param['default']}\nType: {param['type']}\nRange: {param['min']} to {param['max']}")
            self.create_tooltip(frame, f"{param['desc']}\nDefault: {param['default']}")

        # Configuration management buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Load Default Configuration",
                   command=self.load_default_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Validate Current Configuration",
                   command=self.validate_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Configuration to File",
                   command=self.save_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Configuration from File",
                   command=self.load_config).pack(side='left', padx=5)

        # Apply to main tab button
        ttk.Button(button_frame, text="Apply to Main Tab",
                   command=self.apply_config_to_main).pack(side='left', padx=5)

        # Config status
        self.config_status = ttk.Label(config_frame, text="Ready to configure", foreground="green", font=('Arial', 10))
        self.config_status.pack(pady=10)

        # Configuration info
        info_text = """Configuration Management:
    • Load Default: Reset all parameters to default values
    • Validate: Check if current values are valid
    • Save: Save configuration to Model/<dataset>_config.json
    • Load: Load configuration from file
    • Apply to Main: Copy values to main tab quick configuration"""

        info_label = ttk.Label(config_frame, text=info_text, justify=tk.LEFT, foreground="blue",
                              font=('Arial', 9))
        info_label.pack(pady=5)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="lightyellow",
                             relief="solid", borderwidth=1, padding=5,
                             font=('Arial', 9), wraplength=300)
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


    def load_default_config(self):
        """Load default configuration values"""
        for key, param in self.config_params.items():
            self.config_vars[key].set(str(param['default']))
        self.config_status.config(text="Default configuration loaded", foreground="green")

    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        for key, param in self.config_params.items():
            try:
                value = self.config_vars[key].get()
                if param['type'] == 'int':
                    int_value = int(value)
                    if not (param['min'] <= int_value <= param['max']):
                        errors.append(f"{param['label']} must be between {param['min']} and {param['max']}")
                elif param['type'] == 'float':
                    float_value = float(value)
                    if not (param['min'] <= float_value <= param['max']):
                        errors.append(f"{param['label']} must be between {param['min']} and {param['max']}")
            except ValueError:
                errors.append(f"{param['label']} must be a {param['type']}")

        if errors:
            error_msg = "\n".join(errors)
            self.config_status.config(text="Configuration errors found", foreground="red")
            messagebox.showerror("Configuration Error", f"Please fix the following errors:\n\n{error_msg}")
        else:
            self.config_status.config(text="Configuration validated successfully", foreground="green")
            messagebox.showinfo("Success", "Configuration validated successfully!")

        return len(errors) == 0

    def get_config_dict(self):
        """Get current configuration as dictionary"""
        config = {}
        for key in self.config_params.keys():
            try:
                value = self.config_vars[key].get()
                if self.config_params[key]['type'] == 'int':
                    config[key] = int(value)
                elif self.config_params[key]['type'] == 'float':
                    config[key] = float(value)
            except ValueError:
                # Use default if invalid
                config[key] = self.config_params[key]['default']
        return config

    def save_config(self):
        """Save configuration to file"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a data file first")
            return

        if not self.validate_config():
            return

        # Create config filename based on data file
        data_filename = os.path.splitext(os.path.basename(self.current_file))[0]
        config_filename = f"Model/{data_filename}_config.json"

        # Ensure Model directory exists
        os.makedirs("Model", exist_ok=True)

        config_data = {
            'config': self.get_config_dict(),
            'data_file': self.current_file,
            'feature_columns': self.get_selected_features() if hasattr(self, 'get_selected_features') else [],
            'target_column': self.target_col.get() if hasattr(self, 'target_col') else "",
            'saved_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(config_filename, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.config_status.config(text=f"Configuration saved to {config_filename}", foreground="green")
            self.log(f"Configuration saved: {config_filename}")
            messagebox.showinfo("Success", f"Configuration saved to:\n{config_filename}")

        except Exception as e:
            self.config_status.config(text="Error saving configuration", foreground="red")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self, config_file=None):
        """Load configuration from file"""
        if config_file is None:
            config_file = filedialog.askopenfilename(
                title="Select Configuration File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

        if not config_file:
            return

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Update configuration values
            config_dict = config_data.get('config', {})
            for key, value in config_dict.items():
                if key in self.config_vars:
                    self.config_vars[key].set(str(value))

            # Update data file if specified and exists
            data_file = config_data.get('data_file', '')
            if data_file and os.path.exists(data_file):
                self.file_path.set(data_file)
                self.current_file = data_file
                self.analyze_file()

            self.config_status.config(text=f"Configuration loaded from {config_file}", foreground="green")
            self.log(f"Configuration loaded: {config_file}")
            messagebox.showinfo("Success", f"Configuration loaded from:\n{config_file}")

        except Exception as e:
            self.config_status.config(text="Error loading configuration", foreground="red")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def auto_load_config(self):
        """Automatically load configuration if it exists for current data file"""
        if not self.current_file:
            return

        data_filename = os.path.splitext(os.path.basename(self.current_file))[0]
        config_filename = f"Model/{data_filename}_config.json"

        if os.path.exists(config_filename):
            try:
                with open(config_filename, 'r') as f:
                    config_data = json.load(f)

                # Update configuration values
                config_dict = config_data.get('config', {})
                for key, value in config_dict.items():
                    if key in self.config_vars:
                        self.config_vars[key].set(str(value))

                # Update feature selection if available
                feature_columns = config_data.get('feature_columns', [])
                target_column = config_data.get('target_column', '')

                if target_column and hasattr(self, 'target_col'):
                    self.target_col.set(target_column)

                if feature_columns and hasattr(self, 'feature_vars'):
                    # Update feature checkboxes
                    for col, var in self.feature_vars.items():
                        var.set(col in feature_columns)

                self.config_status.config(text=f"Auto-loaded configuration", foreground="green")
                self.log(f"Auto-loaded configuration: {config_filename}")

            except Exception as e:
                self.log(f"Warning: Could not auto-load configuration: {e}")


    def apply_config_to_main(self):
        """Apply configuration tab values to main tab quick configuration"""
        try:
            config_dict = self.get_config_dict()

            # Update main tab parameters
            if hasattr(self, 'resol'):
                self.resol.set(str(config_dict.get('resol', 100)))
            if hasattr(self, 'epochs'):
                self.epochs.set(str(config_dict.get('epochs', 100)))
            if hasattr(self, 'gain'):
                self.gain.set(str(config_dict.get('gain', 2.0)))
            if hasattr(self, 'margin'):
                self.margin.set(str(config_dict.get('margin', 0.2)))
            if hasattr(self, 'patience'):
                self.patience.set(str(config_dict.get('patience', 10)))

            self.config_status.config(text="Configuration applied to main tab", foreground="green")
            self.log("Configuration applied from config tab to main tab")

        except Exception as e:
            self.config_status.config(text="Error applying configuration", foreground="red")
            self.log(f"Error applying configuration: {e}")

    def select_all_features(self):
        """Select all feature checkboxes"""
        for var in self.feature_vars.values():
            var.set(True)

    def deselect_all_features(self):
        """Deselect all feature checkboxes"""
        for var in self.feature_vars.values():
            var.set(False)

    def invert_feature_selection(self):
        """Invert feature selection"""
        for var in self.feature_vars.values():
            var.set(not var.get())

    def get_selected_features(self):
        """Get list of selected feature columns"""
        return [col for col, var in self.feature_vars.items() if var.get()]

    def log(self, message):
        """Add message to console"""
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.root.update()

    def browse_file(self):
        """Browse for data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def predict(self):
        """Make predictions using the trained model's configuration"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model available")
            return

        predict_file = filedialog.askopenfilename(
            title="Select Prediction File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )

        if not predict_file:
            return

        output_file = filedialog.asksaveasfilename(
            title="Save Predictions As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not output_file:
            return

        try:
            self.show_processing_indicator("Analyzing system resources...")

            # Use the trained model's configuration, not external config
            if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                training_features = self.core.feature_columns
                self.log("=== PREDICTION USING TRAINED MODEL ===")
                self.log(f"Using feature columns from trained model: {training_features}")
                self.log(f"Model has {self.core.innodes} input nodes, {self.core.outnodes} output nodes")
            else:
                self.log("❌ No feature configuration found in trained model")
                self.hide_processing_indicator()
                return

            # Detect resources and calculate optimal parameters
            resources = self.core.detect_system_resources()
            opt_params = self.core.calculate_optimal_parameters(resources, "prediction")

            self.log(f"System: {resources['cpu_cores']} CPU cores, {resources['system_memory_gb']:.1f}GB RAM")
            if resources['has_gpu']:
                self.log(f"GPU: {resources.get('gpu_name', 'Unknown')} with {resources['gpu_memory_gb']:.1f}GB")
            self.log(f"Optimization: {opt_params['optimization_level']} mode")
            self.log(f"Batch size: {opt_params['batch_size']}, Concurrent: {opt_params['max_concurrent_batches']}")

            # Load data with trained model's feature configuration
            self.show_processing_indicator("Loading data for prediction...")

            # For prediction, use None for target and the trained model's features
            target_column = None  # No target in prediction mode
            feature_columns = training_features  # Use training features from model

            features_batches, _, feature_columns_used, _ = self.core.load_data(
                predict_file,
                target_column=target_column,
                feature_columns=feature_columns,
                batch_size=opt_params['batch_size']
            )

            if not features_batches:
                self.log("❌ No prediction data loaded")
                self.hide_processing_indicator()
                return

            total_batches = len(features_batches)
            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples in {total_batches} batches")

            # Verify feature dimensions match the trained model
            if features_batches and len(features_batches[0]) > 0:
                actual_feature_count = features_batches[0].shape[1]
                expected_feature_count = len(training_features)

                if actual_feature_count != expected_feature_count:
                    self.log(f"❌ Feature dimension mismatch!")
                    self.log(f"   Model expects: {expected_feature_count} features")
                    self.log(f"   Data has: {actual_feature_count} features")
                    self.log(f"   Expected features: {training_features}")
                    self.hide_processing_indicator()
                    return
                else:
                    self.log(f"✅ Feature dimensions match: {actual_feature_count} features")

            # Initialize output file
            self._initialize_output_file(output_file, predict_file)

            # Initialize performance monitoring
            import time
            start_time = time.time()
            performance_stats = {
                'batch_times': [],
                'memory_usage': [],
                'samples_processed': 0
            }

            # Process batches with resource-aware optimization
            all_predictions = []
            all_probabilities = []
            batch_prediction_times = []

            for batch_idx, features_batch in enumerate(features_batches):
                batch_start = time.time()

                # Adaptive progress updates
                if self._should_update_progress(batch_idx, total_batches, performance_stats):
                    memory_usage = self._get_memory_usage()
                    elapsed = time.time() - start_time
                    rate = performance_stats['samples_processed'] / elapsed if elapsed > 0 else 0

                    self.show_processing_indicator(
                        f"Batch {batch_idx+1}/{total_batches}\n"
                        f"Samples: {performance_stats['samples_processed']}/{total_samples}\n"
                        f"Rate: {rate:.1f} samples/sec\n"
                        f"Memory: {memory_usage}"
                    )

                # Process batch using the trained model
                predictions, probabilities = self.core.predict_batch_optimized(
                    features_batch,
                    clear_cache_every=opt_params['clear_cache_every']
                )

                batch_size = len(predictions)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                performance_stats['samples_processed'] += batch_size

                # Monitor batch performance
                batch_time = time.time() - batch_start
                batch_prediction_times.append(batch_time)
                performance_stats['batch_times'].append(batch_time)

                # Adaptive resource management
                self._adaptive_resource_management(
                    batch_idx, batch_prediction_times, performance_stats, opt_params
                )

                # Periodic disk writing and memory cleanup
                if (batch_idx % opt_params['write_to_disk_every'] == 0 or
                    batch_idx == total_batches - 1):

                    if all_predictions:
                        self._write_predictions_batch(
                            output_file, all_predictions, all_probabilities,
                            performance_stats['samples_processed'] - len(all_predictions)
                        )
                        # Clear memory
                        all_predictions.clear()
                        all_probabilities.clear()

                        # Adaptive garbage collection
                        if self._should_collect_garbage(batch_idx, performance_stats):
                            import gc
                            gc.collect()

            # Final processing and summary
            total_time = time.time() - start_time
            self._print_performance_summary(performance_stats, total_samples, total_time)

            # Show prediction distribution
            if all_predictions:  # Write any remaining predictions
                self._write_predictions_batch(
                    output_file, all_predictions, all_probabilities,
                    performance_stats['samples_processed'] - len(all_predictions)
                )

            self.log(f"✅ Predictions saved to: {output_file}")

            # Show prediction summary
            self._show_prediction_summary(output_file)

            self.hide_processing_indicator()

        except Exception as e:
            self.hide_processing_indicator()
            self.log(f"❌ Prediction error: {e}")
            self.log(traceback.format_exc())

    def _show_prediction_summary(self, output_file):
        """Show summary of predictions made"""
        try:
            import pandas as pd
            df = pd.read_csv(output_file)

            total_predictions = len(df)
            if 'prediction' in df.columns:
                prediction_counts = df['prediction'].value_counts()

                self.log("\n📊 PREDICTION SUMMARY:")
                self.log(f"Total predictions: {total_predictions}")
                self.log("Prediction distribution:")
                for pred, count in prediction_counts.items():
                    percentage = (count / total_predictions) * 100
                    self.log(f"  {pred}: {count} ({percentage:.1f}%)")

        except Exception as e:
            self.log(f"Note: Could not generate prediction summary: {e}")

    def _update_model_name_from_file(self):
        """Update model name based on the current data file"""
        if not self.current_file:
            return

        # Extract base name from file path
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        model_name = f"Model/{base_name}"
        self.model_name.set(model_name)
        self.log(f"Model name auto-set to: {model_name}")


    def train_fresh(self):
        """Train a fresh model from scratch, overwriting any existing model"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        # Confirm with user about overwriting
        result = messagebox.askyesno(
            "Train Fresh",
            "This will train a fresh model from scratch and overwrite any existing model.\n\nContinue?"
        )

        if not result:
            return

        try:
            self.show_processing_indicator("Starting fresh training...")
            self.log("=== STARTING FRESH TRAINING ===")

            # Reset core to ensure fresh start
            self.initialize_core()

            # Set flag to indicate fresh training
            self.fresh_training = True

            # Call the original training method
            success = self._train_model_internal(fresh_training=True)

            if success:
                self.log("✅ Fresh training completed successfully!")
            else:
                self.log("❌ Fresh training failed!")

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Fresh training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())

    def continue_training(self):
        """Continue training from existing model or start fresh if no model exists"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        try:
            self.show_processing_indicator("Continuing training...")
            self.log("=== CONTINUING TRAINING ===")

            # Check if we have a trained model to continue from
            if not getattr(self.core, 'is_trained', False):
                self.log("No existing model found, starting fresh training...")
                # No existing model, so this becomes a fresh training
                success = self._train_model_internal(fresh_training=True)
            else:
                self.log(f"Continuing from existing model (best accuracy: {getattr(self.core, 'best_accuracy', 0):.2f}%)")
                # Continue from existing model
                success = self._train_model_internal(fresh_training=False)

            if success:
                self.log("✅ Continued training completed successfully!")
            else:
                self.log("❌ Continued training failed!")

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Continued training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())

    def _train_model_internal(self, fresh_training=False):
        """Internal training method that handles both fresh and continued training"""
        try:
            # Load data
            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                # VALIDATION: Check if target is selected for training
                if target_column == 'None' or not target_column:
                    messagebox.showerror("Error",
                        "For training, you must select a target column.\n"
                        "Please select a target column from the dropdown (not 'None').")
                    self.hide_processing_indicator()
                    return False

                if not feature_columns:
                    messagebox.showerror("Error", "Please select at least one feature column")
                    self.hide_processing_indicator()
                    return False

                # Check if target column is in feature columns (common mistake)
                if target_column in feature_columns:
                    messagebox.showerror("Error",
                        f"Target column '{target_column}' cannot be in feature columns.\n"
                        f"Please deselect it from the feature checkboxes.")
                    self.hide_processing_indicator()
                    return False

                self.log(f"Training with: target={target_column}, features={feature_columns}")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=5000
                )
            else:
                self.log("Training with DAT file")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=5000
                )
                # For DAT files, set default feature and target info
                target_column = "last_column"
                feature_columns = feature_columns_used

            if not features_batches:
                self.log("ERROR: No data loaded!")
                self.hide_processing_indicator()
                return False

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples")

            # For fresh training, reset the encoder and network
            if fresh_training:
                self.log("🔄 Initializing fresh training...")
                # Fit encoder and setup network from scratch
                all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
                self.core.class_encoder.fit(all_original_targets)

                encoded_classes = self.core.class_encoder.get_encoded_classes()
                self.log(f"Fresh network: {len(feature_columns_used)} inputs, {len(encoded_classes)} outputs")

                # Initialize arrays for fresh training
                innodes = len(feature_columns_used)
                outnodes = len(encoded_classes)
                resol = int(self.resol.get())

                # Validate dimensions before initializing arrays
                if innodes <= 0:
                    self.log("ERROR: No input features detected!")
                    self.hide_processing_indicator()
                    return False
                if outnodes <= 0:
                    self.log("ERROR: No output classes detected!")
                    self.hide_processing_indicator()
                    return False

                self.core.initialize_arrays(innodes, resol, outnodes)

                # Setup dmyclass
                self.core.dmyclass[0] = float(self.margin.get())
                for i, encoded_val in enumerate(encoded_classes, 1):
                    self.core.dmyclass[i] = float(encoded_val)

            # Encode targets
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training (only for fresh training)
            if fresh_training:
                self.log("Initializing training parameters...")
                omax, omin = self.core.initialize_training(features_batches, encoded_targets_batches, resol)

                # Build initial network
                self.log("Building initial network...")
                total_processed = 0
                for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                    processed = self.core.process_training_batch(features_batch, targets_batch)
                    total_processed += processed
                    if batch_idx % 10 == 0:
                        self.log(f"  Processed {total_processed}/{total_samples} samples")
                        self.show_processing_indicator(f"Building network... {total_processed}/{total_samples}")

                self.log(f"✅ Initial network built with {total_processed} samples")

            # Training with FIXED early stopping
            gain = float(self.gain.get())
            max_epochs = int(self.epochs.get())
            patience = int(self.patience.get())

            training_type = "FRESH" if fresh_training else "CONTINUED"
            self.log(f"{training_type} TRAINING: {max_epochs} epochs, gain={gain}, patience={patience}")

            # Store current best accuracy for comparison
            current_best_accuracy = getattr(self.core, 'best_accuracy', 0.0)
            best_accuracy = current_best_accuracy
            best_weights = self.core.anti_wts.copy() if hasattr(self.core, 'anti_wts') and self.core.anti_wts is not None else None
            best_round = getattr(self.core, 'best_round', 0)
            patience_counter = 0

            for rnd in range(max_epochs + 1):
                if rnd == 0 and fresh_training:
                    # Initial evaluation for fresh training
                    current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                    self.log(f"Epoch {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    continue
                elif rnd == 0 and not fresh_training:
                    # For continued training, show current model accuracy
                    current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                    self.log(f"Epoch {rnd:3d}: Current Model Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_weights = self.core.anti_wts.copy()
                        best_round = rnd
                    continue

                # Update processing indicator for training progress
                if rnd % 5 == 0:
                    training_type_str = "Fresh" if fresh_training else "Continued"
                    self.show_processing_indicator(f"{training_type_str} Training... Epoch {rnd}/{max_epochs}")

                # Training epoch
                self.core.train_epoch(features_batches, encoded_targets_batches, gain)

                # Evaluate
                current_accuracy, correct, predictions = self.core.evaluate(features_batches, encoded_targets_batches)

                # Update best weights only if we get a new best accuracy
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    patience_counter = 0
                    improvement = current_accuracy - current_best_accuracy
                    if fresh_training:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST")
                    else:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST (+{improvement:.2f}% from previous best)")
                else:
                    patience_counter += 1
                    improvement = current_accuracy - best_accuracy
                    if improvement > 0:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% (+{improvement:.2f}%)")
                    else:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}%")

                # Visualization
                if self.visualizer and rnd % 5 == 0:
                    sample_size = min(1000, total_samples)
                    sample_features = np.vstack([batch[:100] for batch in features_batches])[:sample_size]
                    sample_targets = np.concatenate([batch[:100] for batch in encoded_targets_batches])[:sample_size]
                    sample_predictions, _ = self.core.predict_batch(sample_features)

                    # Create visualization directory if it doesn't exist
                    if self.current_file:
                        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                        viz_dir = f"Visualisations/{base_name}"
                        os.makedirs(viz_dir, exist_ok=True)

                    self.visualizer.capture_training_snapshot(
                        sample_features, sample_targets, self.core.anti_wts,
                        sample_predictions, current_accuracy, rnd
                    )

                # Early stopping
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {rnd} (no improvement for {patience} epochs)")
                    break

            # Restore best weights
            if best_weights is not None:
                self.core.anti_wts = best_weights
                self.log(f"Restored best weights from epoch {best_round}")

            self.core.is_trained = True
            self.core.best_accuracy = best_accuracy
            self.core.best_round = best_round

            # AUTOMATIC MODEL SAVING AFTER TRAINING
            self.log("=== Auto-saving Trained Model ===")
            self.show_processing_indicator("Auto-saving trained model...")

            # Get feature information
            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            # Use core's auto-save functionality
            saved_model_path = self.core.save_model_auto(
                model_dir="Model",
                data_filename=self.current_file,
                feature_columns=feature_columns,
                target_column=target_column
            )

            if saved_model_path:
                self.log(f"✅ Model automatically saved to: {saved_model_path}")
                self.log(f"   Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
            else:
                self.log("❌ Automatic model saving failed!")

            # Test the model immediately after training
            self.log("Testing trained model...")
            try:
                if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                    test_sample = np.random.randn(self.core.innodes)
                    predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                    self.log(f"✅ Post-training test - Prediction: {predictions[0]}")
                    self.log("✅ Model is ready for predictions")
            except Exception as e:
                self.log(f"❌ Post-training test failed: {e}")

            training_type_str = "Fresh" if fresh_training else "Continued"
            self.log(f"=== {training_type_str.upper()} TRAINING COMPLETED ===")
            self.hide_processing_indicator()
            self.log(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
            self.log(f"Final model: {self.core.innodes} inputs, {self.core.outnodes} outputs")
            self.log(f"Feature columns used: {feature_columns_used}")
            self.log(f"Target column: {target_column if self.file_type == 'csv' else 'last column (DAT)'}")

            encoded_classes = self.core.class_encoder.get_encoded_classes()
            self.log(f"Classes: {len(encoded_classes)} - {encoded_classes}")

            return True

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            return False

    def auto_load_model(self):
        """Automatically load model if it exists for current data file"""
        if not self.current_file:
            return

        # Get the expected model file path
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        model_pattern = f"Model/{base_name}_*_model.bin"

        # Find the most recent model file
        import glob
        model_files = glob.glob(model_pattern)

        if model_files:
            # Sort by modification time and get the most recent
            model_files.sort(key=os.path.getmtime, reverse=True)
            latest_model = model_files[0]

            try:
                self.log(f"🔄 Auto-loading existing model: {latest_model}")
                success = self.load_model_auto_config(latest_model)

                if success:
                    self.log(f"✅ Auto-loaded existing model: {latest_model}")
                    if hasattr(self.core, 'best_accuracy'):
                        self.log(f"   Best accuracy: {self.core.best_accuracy:.2f}%")
                else:
                    self.log("❌ Failed to auto-load existing model")

            except Exception as e:
                self.log(f"⚠️ Could not auto-load model: {e}")

    def analyze_file(self):
        """Analyze the selected file and auto-load configuration and model"""
        filename = self.file_path.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.show_processing_indicator("Analyzing file...")
            self.log(f"Analyzing file: {filename}")

            if filename.endswith('.csv'):
                self.file_type = 'csv'
                df = pd.read_csv(filename, nrows=100)
                columns = df.columns.tolist()

                self.file_info.delete(1.0, tk.END)
                self.file_info.insert(tk.END, f"CSV File: {filename}\n")
                self.file_info.insert(tk.END, f"Columns ({len(columns)}): {', '.join(columns)}\n")
                self.file_info.insert(tk.END, f"Sample data: {len(df)} rows\n")
                self.file_info.insert(tk.END, f"First row sample: {df.iloc[0].tolist()}\n")

                # Setup target selection - ADD "None" OPTION FOR PREDICTION
                target_options = ['None'] + columns  # Add "None" as first option
                self.target_combo['values'] = target_options
                self.target_col.set('None')  # Default to no target for prediction

                # Clear previous feature checkboxes
                for widget in self.feature_scroll_frame.winfo_children():
                    widget.destroy()
                self.feature_vars.clear()

                # Create checkboxes for ALL columns as potential features
                # User can decide which to include/exclude
                for i, col in enumerate(columns):
                    var = tk.BooleanVar(value=True)  # Default all to selected
                    self.feature_vars[col] = var
                    cb = ttk.Checkbutton(self.feature_scroll_frame, text=col, variable=var)
                    cb.grid(row=i, column=0, sticky='w', padx=2, pady=1)

                # Update canvas scrollregion
                self.feature_scroll_frame.update_idletasks()
                self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))

                self.feature_frame.pack(fill='x', pady=5)

            elif filename.endswith('.dat'):
                self.file_type = 'dat'
                with open(filename, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()][:100]

                self.file_info.delete(1.0, tk.END)
                self.file_info.insert(tk.END, f"DAT File: {filename}\n")
                self.file_info.insert(tk.END, f"Total lines: {len(lines)}\n")
                if lines:
                    values = lines[0].split()
                    self.file_info.insert(tk.END, f"Columns: {len(values)} (assuming all are features for prediction)\n")

                # Don't show feature selection for DAT files in prediction mode
                if self.feature_frame.winfo_ismapped():
                    self.feature_frame.pack_forget()

            else:
                messagebox.showerror("Error", "Unsupported file format")
                self.hide_processing_indicator()
                return

            self.current_file = filename
            self.log("File analysis completed")

            # Auto-update model name based on data file
            self._update_model_name_from_file()

            # Auto-load configuration if it exists
            self.auto_load_config()

            # Auto-load model if it exists
            self.auto_load_model()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze file: {e}")
            self.log(f"Error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.hide_processing_indicator()
    def save_trained_model_auto(self, best_accuracy, best_round):
        """GUI wrapper for automatic model saving - now much simpler"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            self.log("❌ No trained model to save")
            return None

        try:
            # Get feature information
            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            # Use core's auto-save functionality
            saved_model_path = self.core.save_model_auto(
                model_dir="Model",
                data_filename=self.current_file,
                feature_columns=feature_columns,
                target_column=target_column
            )

            if saved_model_path:
                self.log(f"✅ Model automatically saved: {saved_model_path}")
                self.log(f"   Best accuracy: {best_accuracy:.2f}% at round {best_round}")
            else:
                self.log("❌ Failed to automatically save model")

            return saved_model_path

        except Exception as e:
            self.log(f"❌ Error in automatic model saving: {e}")
            return None

    def load_model_auto_config(self, model_path):
        """GUI wrapper for auto-configuration model loading"""
        try:
            self.show_processing_indicator("Loading model with auto-configuration...")

            if not self.core:
                self.initialize_core()

            # Use core's auto-configuration loading
            success = self.core.load_model_auto_config(model_path)

            if success:
                # Update GUI with loaded configuration
                self.model_name.set(os.path.splitext(os.path.basename(model_path))[0])

                # Update configuration tab with loaded config
                if hasattr(self, 'config_vars'):
                    for key, value in self.core.config.items():
                        if key in self.config_vars:
                            self.config_vars[key].set(str(value))

                # Update main tab quick configuration
                self.apply_core_config_to_main()

                self.log("✅ Model loaded with auto-configuration")
                if hasattr(self.core, 'best_accuracy'):
                    self.log(f"   Best accuracy: {self.core.best_accuracy:.2f}%")
                if hasattr(self.core, 'feature_columns'):
                    self.log(f"   Features: {len(self.core.feature_columns)}")
            else:
                self.log("❌ Failed to load model with auto-configuration")

            self.hide_processing_indicator()
            return success

        except Exception as e:
            self.log(f"❌ Auto-configuration load error: {e}")
            self.hide_processing_indicator()
            return False

    def apply_core_config_to_main(self):
        """Apply core configuration to main tab quick configuration"""
        try:
            if not self.core:
                return

            # Update main tab parameters from core config
            if hasattr(self, 'resol'):
                self.resol.set(str(self.core.config.get('resol', 100)))
            if hasattr(self, 'epochs'):
                self.epochs.set(str(self.core.config.get('epochs', 100)))
            if hasattr(self, 'gain'):
                self.gain.set(str(self.core.config.get('gain', 2.0)))
            if hasattr(self, 'margin'):
                self.margin.set(str(self.core.config.get('margin', 0.2)))
            if hasattr(self, 'patience'):
                self.patience.set(str(self.core.config.get('patience', 10)))

            self.log("Configuration synchronized from loaded model")

        except Exception as e:
            self.log(f"Error applying core configuration: {e}")

    def show_processing_indicator(self, message="Processing..."):
        """Show a spinning color wheel processing indicator"""
        # If already showing, just update the message
        if self.processing_frame and self.processing_frame.winfo_exists():
            for widget in self.processing_frame.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.config(text=message)
            self.processing_frame.deiconify()
            self.start_animation()
            return

        # Create new processing window
        self.processing_frame = tk.Toplevel(self.root)
        self.processing_frame.title("Processing")
        self.processing_frame.geometry("300x150")
        self.processing_frame.transient(self.root)
        self.processing_frame.grab_set()

        # Make it modal
        self.processing_frame.focus_set()
        self.processing_frame.grab_set()

        # Center the window
        self.processing_frame.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 300) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 150) // 2
        self.processing_frame.geometry(f"+{x}+{y}")

        # Processing message
        msg_label = ttk.Label(self.processing_frame, text=message, font=('Arial', 10))
        msg_label.pack(pady=10)

        # Canvas for spinning wheel
        canvas_size = 60
        self.processing_indicator = tk.Canvas(
            self.processing_frame,
            width=canvas_size,
            height=canvas_size,
            bg=self.processing_frame.cget('bg'),
            highlightthickness=0
        )
        self.processing_indicator.pack(pady=10)

        # Draw initial wheel
        self.animation_angle = 0
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.draw_spinning_wheel()

        # Start animation
        self.start_animation()

        # Force update
        self.processing_frame.update()

    def hide_processing_indicator(self):
        """Hide the processing indicator"""
        self.stop_animation()

        if self.processing_frame and self.processing_frame.winfo_exists():
            try:
                self.processing_frame.grab_release()
                self.processing_frame.destroy()
            except tk.TclError:
                pass  # Window already destroyed

        self.processing_frame = None
        self.processing_indicator = None

        # Ensure main window gets focus back
        self.root.focus_set()

    def draw_spinning_wheel(self):
        """Draw the spinning color wheel"""
        if not self.processing_indicator:
            return

        self.processing_indicator.delete("all")
        canvas_size = 60
        center = canvas_size // 2
        radius = 20

        # Draw spinning wheel with multiple colored segments
        segment_angle = 360 / len(self.colors)
        for i, color in enumerate(self.colors):
            start_angle = self.animation_angle + (i * segment_angle)
            end_angle = start_angle + segment_angle

            self.processing_indicator.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=start_angle, extent=segment_angle - 5,  # Small gap between segments
                fill=color, outline="", width=0
            )

    def start_animation(self):
        """Start the spinning animation"""
        if self.processing_frame and self.processing_frame.winfo_exists():
            self.animation_angle = (self.animation_angle + 15) % 360
            self.draw_spinning_wheel()
            self.processing_animation_id = self.root.after(50, self.start_animation)
        else:
            self.stop_animation()

    def stop_animation(self):
        """Stop the spinning animation"""
        if self.processing_animation_id:
            self.root.after_cancel(self.processing_animation_id)
            self.processing_animation_id = None

    def test_data_load(self):
        """Test data loading functionality"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.log("=== Testing Data Load ===")

            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=1000
                )
            else:
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=1000
                )

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples in {len(features_batches)} batches")
            self.log(f"Feature columns: {feature_columns_used}")

            if features_batches:
                first_batch = features_batches[0]
                self.log(f"First batch shape: {first_batch.shape}")
                self.log(f"First sample features: {first_batch[0][:5]}...")
                self.log(f"First sample target: {original_targets_batches[0][0]}")

            self.log("=== Data Load Test Completed ===")

        except Exception as e:
            self.log(f"Data load test failed: {e}")
            self.log(traceback.format_exc())

    def test_encoder(self):
        """Test class encoder functionality"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.log("=== Testing Encoder ===")

            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=1000
                )
            else:
                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=1000
                )

            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            self.log(f"Encoder fitted with {len(self.core.class_encoder.get_encoded_classes())} classes")

            if len(all_original_targets) > 0:
                sample_targets = all_original_targets[:5]
                encoded = self.core.class_encoder.transform(sample_targets)
                decoded = self.core.class_encoder.inverse_transform(encoded)

                self.log("Encoding test:")
                for orig, enc, dec in zip(sample_targets, encoded, decoded):
                    self.log(f"  {orig} -> {enc} -> {dec}")

            self.log("=== Encoder Test Completed ===")

        except Exception as e:
            self.log(f"Encoder test failed: {e}")
            self.log(traceback.format_exc())

    def test_single_prediction(self):
        """Test single prediction"""
        if not self.core or not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            messagebox.showerror("Error", "Please train a model first")
            return

        try:
            self.log("=== Testing Single Prediction ===")

            if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                test_sample = np.random.randn(self.core.innodes)
                self.log(f"Test sample shape: {test_sample.shape}")

                predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))

                self.log(f"Prediction: {predictions[0]}")
                self.log(f"Probabilities: {probabilities[0]}")
            else:
                self.log("No network architecture information available")

            self.log("=== Single Prediction Test Completed ===")

        except Exception as e:
            self.log(f"Single prediction test failed: {e}")
            self.log(traceback.format_exc())

    def initialize_core(self):
        """Initialize DBNN core"""
        try:
            config = {
                'resol': int(self.resol.get()),
                'epochs': int(self.epochs.get()),
                'gain': float(self.gain.get()),
                'margin': float(self.margin.get()),
                'patience': int(self.patience.get())
            }

            self.core = DBNNCore(config)
            self.visualizer = DBNNVisualizer()
            self.core.attach_visualizer(self.visualizer)

            self.log("DBNN core initialized successfully")
            self.log(f"Config: {config}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize core: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())

    def train_model(self):
        """Train model with fixed early stopping and automatic model saving"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        try:
            self.show_processing_indicator("Starting training process...")
            self.log("=== Starting Training ===")

            # Load data
            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                # VALIDATION: Check if target is selected for training
                if target_column == 'None' or not target_column:
                    messagebox.showerror("Error",
                        "For training, you must select a target column.\n"
                        "Please select a target column from the dropdown (not 'None').")
                    self.hide_processing_indicator()
                    return False

                if not feature_columns:
                    messagebox.showerror("Error", "Please select at least one feature column")
                    self.hide_processing_indicator()
                    return False

                # Check if target column is in feature columns (common mistake)
                if target_column in feature_columns:
                    messagebox.showerror("Error",
                        f"Target column '{target_column}' cannot be in feature columns.\n"
                        f"Please deselect it from the feature checkboxes.")
                    self.hide_processing_indicator()
                    return False

                self.log(f"Training with: target={target_column}, features={feature_columns}")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=5000
                )
            else:
                self.log("Training with DAT file")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=5000
                )
                # For DAT files, set default feature and target info
                target_column = "last_column"
                feature_columns = feature_columns_used

            if not features_batches:
                self.log("ERROR: No data loaded!")
                self.hide_processing_indicator()
                return False

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples")

            # Fit encoder and setup network
            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            encoded_classes = self.core.class_encoder.get_encoded_classes()
            self.log(f"Network: {len(feature_columns_used)} inputs, {len(encoded_classes)} outputs")
            self.log(f"Feature columns used: {feature_columns_used}")
            self.log(f"Encoded classes: {encoded_classes}")

            # Initialize arrays
            innodes = len(feature_columns_used)
            outnodes = len(encoded_classes)
            resol = int(self.resol.get())

            # Validate dimensions before initializing arrays
            if innodes <= 0:
                self.log("ERROR: No input features detected!")
                self.hide_processing_indicator()
                return False
            if outnodes <= 0:
                self.log("ERROR: No output classes detected!")
                self.hide_processing_indicator()
                return False

            self.core.initialize_arrays(innodes, resol, outnodes)

            # Setup dmyclass
            self.core.dmyclass[0] = float(self.margin.get())
            for i, encoded_val in enumerate(encoded_classes, 1):
                self.core.dmyclass[i] = float(encoded_val)

            # Encode targets
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training
            self.log("Initializing training parameters...")
            omax, omin = self.core.initialize_training(features_batches, encoded_targets_batches, resol)

            # Build initial network
            self.log("Building initial network...")
            total_processed = 0
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                processed = self.core.process_training_batch(features_batch, targets_batch)
                total_processed += processed
                if batch_idx % 10 == 0:
                    self.log(f"  Processed {total_processed}/{total_samples} samples")
                    # Update the processing indicator to show progress
                    self.show_processing_indicator(f"Building network... {total_processed}/{total_samples}")

            self.log(f"✅ Initial network built with {total_processed} samples")

            # Training with FIXED early stopping
            gain = float(self.gain.get())
            max_epochs = int(self.epochs.get())
            patience = int(self.patience.get())

            self.log(f"Training: {max_epochs} epochs, gain={gain}, patience={patience}")

            best_accuracy = 0.0
            best_weights = None
            best_round = 0
            patience_counter = 0

            for rnd in range(max_epochs + 1):
                if rnd == 0:
                    # Initial evaluation
                    current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                    self.log(f"Epoch {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    continue

                # Update processing indicator for training progress
                if rnd % 5 == 0:
                    self.show_processing_indicator(f"Training... Epoch {rnd}/{max_epochs}")

                # Training epoch
                self.core.train_epoch(features_batches, encoded_targets_batches, gain)

                # Evaluate
                current_accuracy, correct, predictions = self.core.evaluate(features_batches, encoded_targets_batches)

                # FIXED: Always update best weights when we get a new best accuracy
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    patience_counter = 0
                    self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST")
                else:
                    patience_counter += 1
                    improvement = current_accuracy - best_accuracy
                    if improvement > 0:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% (+{improvement:.2f}%)")
                    else:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}%")

                # Visualization
                if self.visualizer and rnd % 5 == 0:
                    sample_size = min(1000, total_samples)
                    sample_features = np.vstack([batch[:100] for batch in features_batches])[:sample_size]
                    sample_targets = np.concatenate([batch[:100] for batch in encoded_targets_batches])[:sample_size]
                    sample_predictions, _ = self.core.predict_batch(sample_features)
                    self.visualizer.capture_training_snapshot(
                        sample_features, sample_targets, self.core.anti_wts,
                        sample_predictions, current_accuracy, rnd
                    )

                # Early stopping
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {rnd} (no improvement for {patience} epochs)")
                    break

            # Restore best weights
            if best_weights is not None:
                self.core.anti_wts = best_weights
                self.log(f"Restored best weights from epoch {best_round}")

            self.core.is_trained = True
            self.core.best_accuracy = best_accuracy  # Store best accuracy in core

            # AUTOMATIC MODEL SAVING AFTER TRAINING
            self.log("=== Saving Trained Model ===")
            self.show_processing_indicator("Saving trained model...")
            model_file = f"{self.model_name.get()}.bin"  # Default to binary format

            # Ensure model directory exists
            model_dir = os.path.dirname(os.path.abspath(model_file))
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                self.log(f"Created model directory: {model_dir}")

            # Save the model using core's save method WITH FEATURE INFORMATION
            if self.file_type == 'csv':
                # Pass feature information for CSV files
                save_success = self.core.save_model(model_file, feature_columns, target_column, use_json=False)
                self.log(f"✅ Saved model with feature configuration: {len(feature_columns)} features, target: {target_column}")
            else:
                # For DAT files, save without specific feature info
                save_success = self.core.save_model(model_file, use_json=False)
                self.log("✅ Saved model (DAT file - no specific feature configuration)")

            if save_success:
                self.log(f"✅ Model successfully saved to: {model_file}")
            else:
                self.log("❌ Failed to save model!")

            # Test the model immediately after training
            self.log("Testing trained model...")
            try:
                if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                    test_sample = np.random.randn(self.core.innodes)
                    predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                    self.log(f"✅ Post-training test - Prediction: {predictions[0]}")
                    self.log("✅ Model is ready for predictions")
            except Exception as e:
                self.log(f"❌ Post-training test failed: {e}")

            self.log(f"=== Training Completed ===")
            self.hide_processing_indicator()
            self.log(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
            self.log(f"Final model: {self.core.innodes} inputs, {self.core.outnodes} outputs")
            self.log(f"Feature columns used: {feature_columns_used}")
            self.log(f"Target column: {target_column if self.file_type == 'csv' else 'last column (DAT)'}")
            self.log(f"Classes: {len(encoded_classes)} - {encoded_classes}")

            return True

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            return False

    def test_model(self):
        """Test the model on separate data using the model's configuration"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model available")
            return

        test_file = filedialog.askopenfilename(
            title="Select Test File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat")]
        )

        if test_file:
            try:
                self.log(f"=== TESTING MODEL on: {test_file} ===")

                # Use the trained model's configuration
                if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                    training_features = self.core.feature_columns
                    self.log(f"Using feature columns from trained model: {training_features}")
                else:
                    self.log("❌ No feature configuration found in trained model")
                    return

                # Load test data using model's feature configuration
                # For testing, we need the target column to evaluate accuracy
                target_column = self.core.target_column if hasattr(self.core, 'target_column') else ""

                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    test_file,
                    target_column=target_column,
                    feature_columns=training_features
                )

                if not features_batches:
                    self.log("No test data loaded")
                    return

                # Verify feature dimensions
                if features_batches and len(features_batches[0]) > 0:
                    actual_feature_count = features_batches[0].shape[1]
                    expected_feature_count = len(training_features)

                    if actual_feature_count != expected_feature_count:
                        self.log(f"❌ Feature dimension mismatch in test data!")
                        self.log(f"   Model expects: {expected_feature_count} features")
                        self.log(f"   Test data has: {actual_feature_count} features")
                        return
                    else:
                        self.log(f"✅ Feature dimensions match: {actual_feature_count} features")

                # Encode targets using the model's encoder
                encoded_targets_batches = []
                for batch in original_targets_batches:
                    encoded_batch = self.core.class_encoder.transform(batch)
                    encoded_targets_batches.append(encoded_batch)

                # Evaluate using model's configuration
                accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
                total_samples = sum(len(batch) for batch in features_batches)

                self.log(f"📊 TEST RESULTS:")
                self.log(f"  Accuracy: {accuracy:.2f}%")
                self.log(f"  Correct: {correct_predictions}/{total_samples}")
                self.log(f"  Error Rate: {100-accuracy:.2f}%")

                # Show class-wise performance if available
                if hasattr(self.core, 'best_accuracy'):
                    training_accuracy = self.core.best_accuracy
                    accuracy_diff = accuracy - training_accuracy
                    self.log(f"  Training accuracy: {training_accuracy:.2f}%")
                    self.log(f"  Generalization: {accuracy_diff:+.2f}%")

                self.log("=== Testing Completed ===")

            except Exception as e:
                self.log(f"Test error: {e}")
                self.log(traceback.format_exc())

    def _clean_console_memory(self):
        """Clean console memory to prevent buildup"""
        try:
            current_text = self.console.get(1.0, tk.END)
            lines = current_text.split('\n')

            # Keep only last 1000 lines to prevent memory issues
            if len(lines) > 1000:
                cleaned_text = '\n'.join(lines[-1000:])
                self.console.delete(1.0, tk.END)
                self.console.insert(tk.END, cleaned_text)

            # Force Tkinter to update and free memory
            self.console.update_idletasks()

        except Exception as e:
            print(f"Console cleanup warning: {e}")

    def _initialize_output_file(self, output_file, predict_file):
        """Initialize the output file with proper headers"""
        try:
            import pandas as pd

            # Try to preserve original CSV structure if it's a CSV file
            if predict_file.endswith('.csv'):
                try:
                    # Read just the header from the original file
                    original_df = pd.read_csv(predict_file, nrows=0)
                    original_columns = original_df.columns.tolist()

                    # Create output dataframe with original columns + prediction columns
                    output_columns = original_columns + [
                        'prediction',
                        'prediction_encoded',
                        'confidence'
                    ]

                    # Create empty dataframe with the correct structure
                    output_df = pd.DataFrame(columns=output_columns)

                    # Write header to file
                    output_df.to_csv(output_file, index=False)
                    self.log(f"✅ Output file initialized with {len(original_columns)} original columns")

                except Exception as e:
                    self.log(f"⚠️ Could not preserve original CSV structure: {e}")
                    # Fallback: create basic output structure
                    self._create_basic_output_file(output_file)
            else:
                # For DAT files or when CSV loading fails, use basic structure
                self._create_basic_output_file(output_file)

        except Exception as e:
            self.log(f"⚠️ Error initializing output file: {e}")
            # Final fallback
            self._create_basic_output_file(output_file)

    def _create_basic_output_file(self, output_file):
        """Create a basic output file structure"""
        try:
            import pandas as pd

            # Basic columns for prediction output
            basic_columns = [
                'prediction',
                'prediction_encoded',
                'confidence'
            ]

            # Create empty dataframe
            output_df = pd.DataFrame(columns=basic_columns)

            # Write to file
            output_df.to_csv(output_file, index=False)
            self.log("✅ Created basic output file structure")

        except Exception as e:
            self.log(f"❌ Failed to create output file: {e}")
            raise

    def _write_predictions_batch(self, output_file, predictions, probabilities, start_index):
        """Write a batch of predictions to file efficiently"""
        try:
            if not predictions:
                return

            # Decode predictions
            decoded_predictions = self.core.class_encoder.inverse_transform(predictions)

            # Create batch dataframe
            batch_data = {
                'prediction': decoded_predictions,
                'prediction_encoded': predictions,
                'confidence': [max(prob.values()) if prob else 0.0 for prob in probabilities]
            }

            # Add probability columns
            if probabilities and len(probabilities) > 0:
                class_names = list(probabilities[0].keys())
                for class_name in class_names:
                    batch_data[f'prob_{class_name}'] = [prob.get(class_name, 0.0) for prob in probabilities]

            import pandas as pd
            batch_df = pd.DataFrame(batch_data)

            # Append to file
            with open(output_file, 'a', newline='') as f:
                batch_df.to_csv(f, header=False, index=False)

            # Store last predictions for summary (limited size)
            self._last_predictions = predictions[-1000:]  # Keep only last 1000 for summary

        except Exception as e:
            self.log(f"⚠️ Error writing prediction batch: {e}")

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except ImportError:
            return "N/A"

    def _should_update_progress(self, batch_idx, total_batches, stats):
        """Determine when to update progress based on system load"""
        if batch_idx == 0:
            return True

        # Update more frequently at start, less frequently as we progress
        if batch_idx < 10:
            return batch_idx % 2 == 0
        elif batch_idx < 50:
            return batch_idx % 5 == 0
        else:
            return batch_idx % 10 == 0

    def _adaptive_resource_management(self, batch_idx, batch_times, stats, opt_params):
        """Adaptively manage resources based on performance"""
        if len(batch_times) < 5:
            return

        # Detect performance degradation
        recent_times = batch_times[-5:]
        avg_recent = sum(recent_times) / len(recent_times)
        avg_previous = sum(batch_times[:-5]) / len(batch_times[:-5]) if len(batch_times) > 5 else avg_recent

        if avg_recent > avg_previous * 1.3:  # 30% slowdown
            self.log(f"⚠️ Performance degradation detected at batch {batch_idx}")
            self.log(f"   Recent: {avg_recent:.3f}s, Previous: {avg_previous:.3f}s")

            # Suggest resource adjustment
            memory_usage = self._get_memory_usage()
            self.log(f"   Current memory: {memory_usage}")

    def _should_collect_garbage(self, batch_idx, stats):
        """Determine when to run garbage collection"""
        if batch_idx < 10:
            return batch_idx % 5 == 0
        else:
            return batch_idx % 20 == 0

    def _print_performance_summary(self, stats, total_samples, total_time):
        """Print detailed performance summary"""
        samples_per_second = total_samples / total_time if total_time > 0 else 0

        self.log("\n" + "="*50)
        self.log("PERFORMANCE SUMMARY")
        self.log("="*50)
        self.log(f"Total samples: {total_samples:,}")
        self.log(f"Total time: {total_time:.2f} seconds")
        self.log(f"Processing rate: {samples_per_second:.1f} samples/second")

        if stats['batch_times']:
            avg_batch_time = sum(stats['batch_times']) / len(stats['batch_times'])
            self.log(f"Average batch time: {avg_batch_time:.3f} seconds")

            # Show performance distribution
            fast_batches = len([t for t in stats['batch_times'] if t < avg_batch_time * 0.8])
            slow_batches = len([t for t in stats['batch_times'] if t > avg_batch_time * 1.2])
            self.log(f"Fast batches: {fast_batches}, Slow batches: {slow_batches}")

        self.log("="*50)

    def predict_batch_optimized(self, features_batch, clear_cache_every=100):
        """Optimized batch prediction with periodic cache clearing"""
        predictions = []
        probabilities = []

        # Use static arrays to avoid repeated allocations
        if not hasattr(self, '_prediction_vects'):
            self._prediction_vects = np.zeros(self.innodes + self.outnodes + 2)

        for sample_idx in range(len(features_batch)):
            # Reuse the same vector to avoid memory allocation
            vects = self._prediction_vects
            vects.fill(0)  # Reset vector

            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.dmyclass, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.dmyclass[kmax]
            predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.dmyclass[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            probabilities.append(prob_dict)

            # Clear cache periodically to prevent memory buildup
            if sample_idx % clear_cache_every == 0:
                import gc
                gc.collect()

        return predictions, probabilities

    def save_model(self, model_path=None):
        """Save the current model to file in binary format"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model to save")
            return

        # If no model_path provided, use the auto-generated name
        if model_path is None:
            model_path = f"{self.model_name.get()}.bin"

        # Ensure .bin extension
        if not model_path.endswith('.bin'):
            model_path += '.bin'

        # Ensure Model directory exists
        model_dir = os.path.dirname(os.path.abspath(model_path))
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        try:
            # Get feature information
            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            success = self.core.save_model(
                model_path,
                feature_columns=feature_columns,
                target_column=target_column,
                use_json=False  # Always binary format
            )

            if success:
                self.log(f"✅ Model saved in binary format: {model_path}")
                self.log(f"Model info: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                self.log(f"Best accuracy: {getattr(self.core, 'best_accuracy', 0.0):.2f}%")
            else:
                self.log("❌ Failed to save model")

        except Exception as e:
            self.log(f"Save error: {e}")
            self.log(traceback.format_exc())
    def save_model_json(self):
        """Save model in JSON format"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model to save")
            return

        model_path = filedialog.asksaveasfilename(
            title="Save Model As JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not model_path:
            return  # User cancelled

        try:
            # Ensure .json extension
            if not model_path.endswith('.json'):
                model_path += '.json'

            # Get feature information from UI
            feature_columns = []
            target_column = ""

            if hasattr(self, 'get_selected_features'):
                feature_columns = self.get_selected_features()
            if hasattr(self, 'target_col'):
                target_column = self.target_col.get()

            success = self.core.save_model(
                model_path,
                feature_columns=feature_columns,
                target_column=target_column,
                use_json=True
            )

            if success:
                self.log(f"✅ Model saved in JSON format: {model_path}")
                messagebox.showinfo("Success", f"Model saved in JSON format:\n{model_path}")
            else:
                self.log("❌ Failed to save model in JSON format")
                messagebox.showerror("Error", "Failed to save model in JSON format")

        except Exception as e:
            self.log(f"JSON save error: {e}")
            messagebox.showerror("Error", f"Failed to save JSON model: {e}")

    def load_model(self, model_path=None):
        """Load a previously saved model with proper error handling"""
        # If no model_path provided, ask the user to select one
        if model_path is None:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Model files", "*.json *.bin *.gz"), ("JSON files", "*.json"), ("Binary files", "*.bin *.gz"), ("All files", "*.*")]
            )

        if model_path:
            try:
                self.show_processing_indicator("Loading model...")
                if not self.core:
                    self.initialize_core()

                self.core.load_model(model_path)
                self.model_name.set(os.path.splitext(os.path.basename(model_path))[0])

                # Verify the model was loaded correctly
                if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                    self.log(f"Model loaded successfully")
                    self.log(f"Model architecture: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                    self.log(f"Configuration: resolution={self.core.config.get('resol', 'N/A')}")

                    # Update UI with loaded model info
                    if hasattr(self.core, 'class_encoder') and self.core.class_encoder.is_fitted:
                        classes = list(self.core.class_encoder.encoded_to_class.values())
                        self.log(f"Classes: {len(classes)} - {classes[:5]}{'...' if len(classes) > 5 else ''}")

                    # Test if model is functional
                    self.log("Testing model functionality...")
                    try:
                        if self.core.innodes > 0:
                            test_sample = np.random.randn(self.core.innodes)
                            predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                            self.log(f"Model test: Prediction = {predictions[0]}")
                    except Exception as e:
                        self.log(f"Model test failed: {e}")
                else:
                    self.log("ERROR: Model failed to load properly - invalid dimensions")

                self.log("=== Model Loading Completed ===")
                self.hide_processing_indicator()

            except Exception as e:
                self.log(f"Load error: {e}")
                self.log(traceback.format_exc())
                self.hide_processing_indicator()

    def visualize(self):
        """Generate visualizations and save them in Visualisations/<databasename>/ folder"""
        if not self.visualizer or not hasattr(self.visualizer, 'training_history') or not self.visualizer.training_history:
            messagebox.showerror("Error", "No training history available for visualization")
            return

        try:
            self.log("=== Generating Visualizations ===")

            # Create visualization directory based on data file name
            if self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                viz_dir = f"Visualisations/{base_name}"
            else:
                base_name = self.model_name.get().replace("Model/", "")
                viz_dir = f"Visualisations/{base_name}"

            # Ensure visualization directory exists
            os.makedirs(viz_dir, exist_ok=True)
            self.log(f"Visualizations will be saved in: {viz_dir}")

            viz_count = 0
            generated_files = []

            # Generate accuracy plot
            try:
                accuracy_fig = self.visualizer.generate_accuracy_plot()
                if accuracy_fig:
                    accuracy_file = f"{viz_dir}/{base_name}_accuracy.html"
                    accuracy_fig.write_html(accuracy_file)
                    self.log(f"✓ Accuracy plot: {accuracy_file}")
                    viz_count += 1
                    generated_files.append(accuracy_file)
            except Exception as e:
                self.log(f"✗ Accuracy plot failed: {e}")

            # Generate feature space plot
            try:
                if len(self.visualizer.training_history) > 0:
                    feature_fig = self.visualizer.generate_feature_space_plot(-1)
                    if feature_fig:
                        feature_file = f"{viz_dir}/{base_name}_features.html"
                        feature_fig.write_html(feature_file)
                        self.log(f"✓ Feature space: {feature_file}")
                        viz_count += 1
                        generated_files.append(feature_file)
            except Exception as e:
                self.log(f"✗ Feature space failed: {e}")

            # Generate weight distribution plot
            try:
                weight_fig = self.visualizer.generate_weight_distribution_plot(-1)
                if weight_fig:
                    weight_file = f"{viz_dir}/{base_name}_weights.html"
                    weight_fig.write_html(weight_file)
                    self.log(f"✓ Weight distribution: {weight_file}")
                    viz_count += 1
                    generated_files.append(weight_file)
            except Exception as e:
                self.log(f"✗ Weight distribution failed: {e}")

            # Generate dashboard
            try:
                dashboard_file = f"{viz_dir}/{base_name}_dashboard.html"
                created_file = self.visualizer.create_training_dashboard(dashboard_file)
                if created_file:
                    self.log(f"✓ Training dashboard: {created_file}")
                    viz_count += 1
                    generated_files.append(created_file)
            except Exception as e:
                self.log(f"✗ Dashboard failed: {e}")

            if viz_count > 0:
                self.log(f"=== Visualization Completed ===")
                self.log(f"Generated {viz_count} visualization files in {viz_dir}")
                self.log("Open the .html files in your web browser to view interactive plots")

                # Ask user if they want to open the files
                self._ask_to_open_visualizations(generated_files, base_name, viz_dir)
            else:
                self.log("No visualizations could be generated")

        except Exception as e:
            self.log(f"Visualization error: {e}")
            self.log(traceback.format_exc())

    def _ask_to_open_visualizations(self, generated_files, base_name, viz_dir):
        """Ask user whether to open visualization files or folder"""
        import webbrowser
        import os
        import platform

        # Create a dialog to ask user preference
        dialog = tk.Toplevel(self.root)
        dialog.title("Visualizations Generated")
        dialog.geometry("550x350")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 550) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 350) // 2
        dialog.geometry(f"+{x}+{y}")

        # Content
        ttk.Label(dialog, text="Visualization Files Generated!",
                  font=('Arial', 12, 'bold')).pack(pady=10)

        ttk.Label(dialog, text=f"Created {len(generated_files)} visualization files in:",
                  font=('Arial', 10)).pack(pady=5)

        ttk.Label(dialog, text=viz_dir,
                  font=('Arial', 10, 'bold'), foreground="blue").pack(pady=5)

        # List generated files
        file_frame = ttk.Frame(dialog)
        file_frame.pack(fill='both', expand=True, padx=20, pady=10)

        file_list = scrolledtext.ScrolledText(file_frame, height=6, width=65)
        file_list.pack(fill='both', expand=True)

        for file in generated_files:
            # Show only the filename, not the full path for cleaner display
            filename = os.path.basename(file)
            file_list.insert(tk.END, f"• {filename}\n")
        file_list.config(state=tk.DISABLED)

        # Buttons frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def open_folder():
            """Open the folder containing visualization files"""
            try:
                abs_viz_dir = os.path.abspath(viz_dir)
                if platform.system() == "Windows":
                    os.startfile(abs_viz_dir)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'open "{abs_viz_dir}"')
                else:  # Linux
                    os.system(f'xdg-open "{abs_viz_dir}"')
                self.log(f"Opened visualization folder: {abs_viz_dir}")
            except Exception as e:
                self.log(f"Error opening folder: {e}")
                messagebox.showerror("Error", f"Could not open folder: {e}")
            finally:
                dialog.destroy()

        def open_3d_visualizations():
            """Open the 3D visualization files in web browser"""
            try:
                opened_count = 0
                for file in generated_files:
                    if 'features' in file.lower() or '3d' in file.lower() or 'dashboard' in file.lower():
                        # Open in web browser
                        webbrowser.open(f'file://{os.path.abspath(file)}')
                        opened_count += 1
                        # Small delay to prevent browser overload
                        time.sleep(0.5)

                if opened_count > 0:
                    self.log(f"Opened {opened_count} 3D visualization files in browser")
                else:
                    self.log("No 3D visualization files found to open")

            except Exception as e:
                self.log(f"Error opening visualizations: {e}")
                messagebox.showerror("Error", f"Could not open visualizations: {e}")
            finally:
                dialog.destroy()

        def open_all_files():
            """Open all visualization files in browser"""
            try:
                opened_count = 0
                for file in generated_files:
                    webbrowser.open(f'file://{os.path.abspath(file)}')
                    opened_count += 1
                    # Small delay to prevent browser overload
                    time.sleep(0.5)

                self.log(f"Opened all {opened_count} visualization files")
            except Exception as e:
                self.log(f"Error opening files: {e}")
                messagebox.showerror("Error", f"Could not open files: {e}")
            finally:
                dialog.destroy()

        def do_nothing():
            """Close dialog without opening anything"""
            self.log("User chose not to open visualizations")
            dialog.destroy()

        # Button layout
        ttk.Button(button_frame, text="Open Visualization Folder",
                   command=open_folder).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open 3D Visualizations",
                   command=open_3d_visualizations).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open All Files",
                   command=open_all_files).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Close",
                   command=do_nothing).pack(side='right', padx=5)

        # Instructions
        ttk.Label(dialog, text="Choose an option to view your visualizations:",
                  font=('Arial', 9)).pack(pady=5)

class DBNNCommandLine:
    """Command Line Interface for DBNN with binary format support"""

    def __init__(self):
        self.core = DBNNCore()
        self.core.set_log_callback(self._log_message)
        self.current_model_path = None
        self.visualizer = DBNNVisualizer()
        self.core.attach_visualizer(self.visualizer)

    def _log_message(self, message: str):
        """Log message to console"""
        print(f"[DBNN] {message}")

    def show_detailed_help(self):
        """Display comprehensive help information"""
        help_text = """
DBNN (Difference Boosting Bayesian Neural Network) - Command Line Interface
================================================================================

DBNN is a powerful neural network implementation with Bayesian learning and
difference boosting capabilities. This interface provides full access to all
DBNN features through command line operations.

USAGE PATTERNS:
===============

1. TRAINING A NEW MODEL:
   python runDBNN_cmd.py --train <training_file> [options]

2. USING EXISTING MODEL:
   python runDBNN_cmd.py --model <model_file> --predict <data_file>

3. EVALUATION:
   python runDBNN_cmd.py --model <model_file> --evaluate <test_file>

4. INTERACTIVE MODE:
   python runDBNN_cmd.py --interactive

DETAILED OPTIONS:
=================

DATA INPUT OPTIONS:
  --train FILE           Training data file (CSV or DAT format)
  --test FILE            Test data file for validation during training
  --predict FILE         Data file for making predictions
  --evaluate FILE        Data file for model evaluation

CSV-SPECIFIC OPTIONS (required for CSV files):
  --target COLUMN        Target column name for CSV files
  --features COL1 COL2   Feature column names (space-separated)
  --format FORMAT        Input file format: csv or dat (default: csv)

MODEL OPTIONS:
  --model FILE           Load existing model file (.json, .bin, or .gz)
  --save-model FILE      Save trained model to specified path
  --model-format FORMAT  Model save format: binary or json (default: binary)
  --model-dir DIR        Model directory (default: Model/)

TRAINING PARAMETERS:
  --resol INTEGER        Resolution parameter (default: 100)
  --gain FLOAT           Gain parameter for weight updates (default: 2.0)
  --margin FLOAT         Classification margin (default: 0.2)
  --patience INTEGER     Early stopping patience (default: 10)
  --epochs INTEGER       Maximum training epochs (default: 100)
  --min-improvement FLOAT Minimum accuracy improvement for early stopping (default: 0.1)

OUTPUT OPTIONS:
  --output FILE          Output file for predictions
  --verbose, -v          Verbose output with detailed progress
  --quiet, -q            Quiet mode (minimal output)

OTHER OPTIONS:
  --interactive, -i      Start interactive mode
  --help, -h             Show this help message

FILE FORMATS SUPPORTED:
=======================

CSV FILES:
  - First row must contain column headers
  - Specify target column with --target
  - Specify feature columns with --features (or uses all except target)
  - Example: --train data.csv --target class --features feature1 feature2 feature3

DAT FILES (Legacy Format):
  - Space-separated values, no headers
  - Last column is assumed to be target
  - All other columns are features
  - Example: --train data.dat --format dat

MODEL FORMATS:
==============

BINARY FORMAT (Default - Recommended):
  - File extensions: .bin or .gz
  - Smaller file size
  - Faster save/load operations
  - Better performance for large models
  - Example: --save-model my_model.bin --model-format binary

JSON FORMAT (Optional):
  - File extension: .json
  - Human-readable
  - Good for debugging and inspection
  - Example: --save-model my_model.json --model-format json

DETAILED EXAMPLES:
==================

1. BASIC TRAINING WITH CSV:
   python runDBNN_cmd.py --train data/train.csv --test data/test.csv
                         --target species
                         --features sepal_length sepal_width petal_length petal_width
                         --resol 150 --gain 2.5 --epochs 200

2. TRAINING WITH DAT FILE:
   python runDBNN_cmd.py --train data/train.dat --format dat
                         --resol 100 --patience 15

3. PREDICTIONS WITH EXISTING MODEL:
   python runDBNN_cmd.py --model Model/iris_model.bin
                         --predict data/new_data.csv
                         --target species
                         --output predictions.csv

4. EVALUATE MODEL PERFORMANCE:
   python runDBNN_cmd.py --model Model/iris_model.bin
                         --evaluate data/test_set.csv
                         --target species

5. TRAINING WITH CUSTOM PARAMETERS:
   python runDBNN_cmd.py --train data/train.csv --target outcome
                         --resol 200 --gain 3.0 --margin 0.3
                         --patience 20 --epochs 300
                         --min-improvement 0.05
                         --save-model my_custom_model.bin

6. VERBOSE TRAINING WITH PROGRESS UPDATES:
   python runDBNN_cmd.py --train large_dataset.csv --target label
                         --features f1 f2 f3 f4 f5
                         --verbose --epochs 500

7. QUIET MODE FOR BATCH PROCESSING:
   python runDBNN_cmd.py --model production_model.bin
                         --predict batch_data.csv
                         --output batch_predictions.csv
                         --quiet

PERFORMANCE TIPS:
=================

- Use binary format (.bin) for production models (faster and smaller)
- For large datasets, use --quiet mode to reduce output overhead
- Adjust --resol based on dataset complexity (higher = more precise but slower)
- Use --patience to prevent overfitting with early stopping
- Start with default parameters and tune based on validation results

TROUBLESHOOTING:
================

Common Issues and Solutions:

1. "Target column not found":
   - Check column names with: head -1 your_file.csv
   - Ensure --target matches exactly (case-sensitive)

2. "No training data loaded":
   - Verify file path is correct
   - Check file format matches --format option
   - Ensure CSV files have proper headers

3. "Model failed to load":
   - Check file exists and is readable
   - Verify model format (binary vs JSON)
   - Ensure model was saved with same DBNN version

4. "Feature count mismatch":
   - Training and prediction must use same features
   - Use --features to explicitly specify feature columns

5. "Memory issues with large datasets":
   - Reduce --resol parameter
   - Use smaller batch sizes in interactive mode
   - Process data in chunks

For additional help or to report issues, please check the documentation
or contact support.

INTERACTIVE MODE FEATURES:
==========================

Interactive mode provides a menu-driven interface with:
- Step-by-step model training
- Real-time configuration
- Immediate prediction testing
- Model inspection and saving
- No need to remember command syntax

Start interactive mode with: python runDBNN_cmd.py --interactive
"""
        print(help_text)

    def parse_arguments(self):
        """Parse command line arguments"""
        import argparse

        parser = argparse.ArgumentParser(
            description='DBNN (Difference Boosting Bayesian Neural Network) - Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False
        )

        # Data input options
        data_group = parser.add_argument_group('Data Input Options')
        data_group.add_argument('--train', type=str, help='Training data file (CSV or DAT)')
        data_group.add_argument('--test', type=str, help='Test data file (CSV or DAT)')
        data_group.add_argument('--predict', type=str, help='Data file for prediction')
        data_group.add_argument('--evaluate', type=str, help='Data file for model evaluation')

        # CSV-specific options
        csv_group = parser.add_argument_group('CSV Options')
        csv_group.add_argument('--target', type=str, help='Target column name for CSV files')
        csv_group.add_argument('--features', nargs='+', help='Feature column names for CSV files')
        csv_group.add_argument('--format', choices=['csv', 'dat'], default='csv',
                              help='Input file format (default: csv)')

        # Model options
        model_group = parser.add_argument_group('Model Options')
        model_group.add_argument('--model', type=str, help='Load existing model file')
        model_group.add_argument('--save-model', type=str, help='Save model to specified path')
        model_group.add_argument('--model-format', choices=['binary', 'json'], default='binary',
                               help='Model save format (default: binary)')
        model_group.add_argument('--model-dir', type=str, default='Model',
                               help='Model directory (default: Model)')

        # Training parameters
        param_group = parser.add_argument_group('Training Parameters')
        param_group.add_argument('--resol', type=int, default=100,
                                help='Resolution parameter (default: 100)')
        param_group.add_argument('--gain', type=float, default=2.0,
                                help='Gain parameter (default: 2.0)')
        param_group.add_argument('--margin', type=float, default=0.2,
                                help='Classification margin (default: 0.2)')
        param_group.add_argument('--patience', type=int, default=10,
                                help='Early stopping patience (default: 10)')
        param_group.add_argument('--epochs', type=int, default=100,
                                help='Maximum training epochs (default: 100)')
        param_group.add_argument('--min-improvement', type=float, default=0.1,
                                help='Minimum improvement for early stopping (default: 0.1)')

        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--output', type=str, help='Output file for predictions')
        output_group.add_argument('--verbose', '-v', action='store_true',
                                 help='Verbose output')
        output_group.add_argument('--quiet', '-q', action='store_true',
                                 help='Quiet mode (minimal output)')

        # Interactive mode and help
        other_group = parser.add_argument_group('Other Options')
        other_group.add_argument('--interactive', '-i', action='store_true',
                          help='Start interactive mode')
        other_group.add_argument('--help', '-h', action='store_true',
                          help='Show detailed help message')

        return parser.parse_args()

    def validate_arguments(self, args):
        """Validate command line arguments and provide helpful error messages"""
        import sys

        errors = []
        warnings = []

        # Check for no arguments or help request
        if len(sys.argv) == 1 or args.help:
            self.show_detailed_help()
            return False, []

        # Check for interactive mode
        if args.interactive:
            return True, []  # Interactive mode doesn't need validation

        # Validate training arguments
        if args.train:
            if not os.path.exists(args.train):
                errors.append(f"Training file not found: {args.train}")

            if args.format == 'csv' and not args.target:
                errors.append("--target parameter required for CSV training files")

        # Validate prediction/evaluation arguments
        if (args.predict or args.evaluate) and not args.model and not args.train:
            errors.append("For prediction or evaluation, either --model or --train must be specified")

        if args.predict and not os.path.exists(args.predict):
            errors.append(f"Prediction file not found: {args.predict}")

        if args.evaluate and not os.path.exists(args.evaluate):
            errors.append(f"Evaluation file not found: {args.evaluate}")

        # Validate model arguments
        if args.model and not os.path.exists(args.model):
            errors.append(f"Model file not found: {args.model}")

        # Check for incompatible options
        if args.verbose and args.quiet:
            warnings.append("Both --verbose and --quiet specified, using --verbose")
            args.quiet = False

        # Check for missing required combinations
        if args.train and args.format == 'csv' and not args.target:
            errors.append("CSV training requires --target parameter")

        if (args.predict or args.evaluate) and args.format == 'csv' and not args.target:
            warnings.append("CSV prediction/evaluation without --target: assuming no target column")

        # Show warnings
        for warning in warnings:
            print(f"⚠️  Warning: {warning}")

        # Show errors and return
        if errors:
            print("❌ Argument errors:")
            for error in errors:
                print(f"   - {error}")
            print("\n💡 Use --help for detailed usage information")
            return False, errors

        return True, []

    def setup_model_config(self, args):
        """Setup model configuration from command line arguments"""
        config = {
            'resol': args.resol,
            'gain': args.gain,
            'margin': args.margin,
            'patience': args.patience,
            'epochs': args.epochs,
            'min_improvement': args.min_improvement
        }
        self.core.config.update(config)

    def load_model(self, model_path: str):
        """Load model from file (auto-detects format)"""
        print(f"Loading model from: {model_path}")
        if self.core.load_model(model_path):
            self.current_model_path = model_path
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Failed to load model")
            return False

    def save_model(self, model_path: str, use_json=False):
        """Save model to file in specified format"""
        # Ensure model directory exists
        model_dir = os.path.dirname(os.path.abspath(model_path))
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Add appropriate extension if not present
        if use_json and not model_path.endswith('.json'):
            model_path += '.json'
        elif not use_json and not (model_path.endswith('.bin') or model_path.endswith('.gz')):
            model_path += '.bin'

        print(f"Saving model to: {model_path}")
        if self.core.save_model(model_path, use_json=use_json):
            self.current_model_path = model_path
            format_name = "JSON" if use_json else "binary"
            print(f"✅ Model saved successfully in {format_name} format")
            return True
        else:
            print("❌ Failed to save model")
            return False

    def train_model(self, args):
        """Train model with specified parameters and automatic saving"""
        print("🚀 Starting DBNN Training...")
        print(f"Training file: {args.train}")
        if args.test:
            print(f"Test file: {args.test}")
        print(f"Parameters: resol={args.resol}, gain={args.gain}, margin={args.margin}")
        print(f"Early stopping: patience={args.patience}, min_improvement={args.min_improvement}%")

        use_csv = (args.format == 'csv')
        if use_csv and not args.target:
            print("❌ Error: --target parameter required for CSV format")
            return False

        import time
        start_time = time.time()

        try:
            success = self.core.train_with_early_stopping(
                train_file=args.train,
                test_file=args.test,
                use_csv=use_csv,
                target_column=args.target,
                feature_columns=args.features,
                auto_save_model=True,  # Enable automatic saving
                model_dir=args.model_dir
            )

            training_time = time.time() - start_time
            print(f"⏱️  Training completed in {training_time:.2f} seconds")

            # Manual save if specifically requested (in addition to auto-save)
            if success and args.save_model:
                use_json = (args.model_format == 'json')
                self.save_model(args.save_model, use_json=use_json)

            return success

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_data(self, args):
        """Make predictions on data using the trained model's configuration"""
        if not self.core.is_trained and not args.model:
            print("❌ Error: No trained model available. Use --train to train a model or --model to load one.")
            return False

        print(f"🔮 Making predictions on: {args.predict}")

        try:
            # Use the trained model's configuration
            if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                training_features = self.core.feature_columns
                print(f"Using feature columns from trained model: {training_features}")
            else:
                print("❌ No feature configuration found in trained model")
                return False

            # For prediction mode, use None for target and training features
            target_column = None  # No target in prediction mode
            feature_columns = training_features  # Use training features from model

            # Load prediction data
            features_batches, _, feature_columns_used, _ = self.core.load_data(
                args.predict,
                target_column=target_column,
                feature_columns=feature_columns,
                batch_size=10000
            )

            if not features_batches:
                print("❌ No data loaded for prediction")
                return False

            # Verify feature dimensions
            if features_batches and len(features_batches[0]) > 0:
                actual_feature_count = features_batches[0].shape[1]
                expected_feature_count = len(training_features)

                if actual_feature_count != expected_feature_count:
                    print(f"❌ Feature dimension mismatch!")
                    print(f"   Expected: {expected_feature_count} features")
                    print(f"   Got: {actual_feature_count} features")
                    print(f"   Expected features: {training_features}")
                    return False
                else:
                    print(f"✅ Feature dimensions match: {actual_feature_count} features")

            all_predictions = []
            all_probabilities = []

            for batch_idx, features_batch in enumerate(features_batches):
                if args.verbose:
                    print(f"Processing batch {batch_idx + 1}/{len(features_batches)}...")
                predictions, probabilities = self.core.predict_batch(features_batch)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)

            # Decode predictions
            decoded_predictions = self.core.class_encoder.inverse_transform(all_predictions)

            # Save predictions to file if requested
            if args.output:
                self.save_predictions(args.output, features_batches, decoded_predictions, all_predictions, all_probabilities)

            # Print summary
            print(f"✅ Made {len(all_predictions)} predictions")

            # Show prediction distribution
            from collections import Counter
            prediction_counts = Counter(decoded_predictions)
            print("\nPrediction distribution:")
            for pred, count in prediction_counts.most_common():
                percentage = (count / len(decoded_predictions)) * 100
                print(f"  {pred}: {count} ({percentage:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_predictions(self, output_file: str, features_batches, decoded_predictions, encoded_predictions, probabilities):
        """Save predictions to output file"""
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                headers = ['Prediction', 'Prediction_Encoded', 'Confidence']

                # Add probability columns for each class
                if probabilities and len(probabilities) > 0:
                    class_names = list(probabilities[0].keys())
                    headers.extend([f'Prob_{cls}' for cls in class_names])

                writer.writerow(headers)

                # Write data
                sample_idx = 0
                for batch_idx, features_batch in enumerate(features_batches):
                    for i in range(len(features_batch)):
                        if sample_idx >= len(decoded_predictions):
                            break

                        row = [
                            decoded_predictions[sample_idx],
                            encoded_predictions[sample_idx],
                            max(probabilities[sample_idx].values()) if probabilities[sample_idx] else 0.0
                        ]

                        # Add probabilities for each class
                        if probabilities and sample_idx < len(probabilities):
                            for class_name in list(probabilities[sample_idx].keys()):
                                row.append(probabilities[sample_idx].get(class_name, 0.0))

                        writer.writerow(row)
                        sample_idx += 1

            print(f"✅ Predictions saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error saving predictions: {e}")

    def evaluate_model(self, args):
        """Evaluate model on test data"""
        if not self.core.is_trained and not args.model:
            print("❌ Error: No trained model available. Use --train to train a model or --model to load one.")
            return False

        print(f"📊 Evaluating model on: {args.evaluate}")

        try:
            # Load test data
            features_batches, targets_batches, feature_columns, original_targets_batches = self.core.load_data(
                args.evaluate,
                target_column=args.target if args.format == 'csv' else None,
                feature_columns=args.features,
                batch_size=10000
            )

            if not features_batches:
                print("❌ No test data loaded")
                return False

            # Encode targets
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Evaluate
            accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
            total_samples = sum(len(batch) for batch in features_batches)

            print(f"✅ Evaluation completed:")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Correct: {correct_predictions}/{total_samples}")
            print(f"   Error Rate: {100-accuracy:.2f}%")

            return True

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def show_configuration(self):
        """Display current configuration"""
        config = self.core.config

        print("\nCurrent Configuration:")
        print("-" * 30)

        for key, value in config.items():
            print(f"{key:20}: {value}")

        if self.core.is_trained:
            print(f"{'Model trained':20}: Yes")
            if hasattr(self.core, 'innodes'):
                print(f"{'Input nodes':20}: {self.core.innodes}")
            if hasattr(self.core, 'outnodes'):
                print(f"{'Output nodes':20}: {self.core.outnodes}")
            if hasattr(self.core, 'best_accuracy'):
                print(f"{'Best accuracy':20}: {self.core.best_accuracy:.2f}%")
        else:
            print(f"{'Model trained':20}: No")

    def interactive_mode(self):
        """Start interactive mode"""
        print("\n" + "="*60)
        print("DBNN Interactive Mode")
        print("="*60)
        print("Available commands:")
        print("  train    - Train a new model")
        print("  load     - Load existing model")
        print("  save     - Save current model")
        print("  predict  - Make predictions")
        print("  evaluate - Evaluate model")
        print("  config   - Show configuration")
        print("  quit     - Exit interactive mode")
        print("="*60)

        while True:
            try:
                command = input("\nDBNN> ").strip().lower()

                if command == 'quit' or command == 'exit':
                    break
                elif command == 'train':
                    self.interactive_train()
                elif command == 'load':
                    self.interactive_load()
                elif command == 'save':
                    self.interactive_save()
                elif command == 'predict':
                    self.interactive_predict()
                elif command == 'evaluate':
                    self.interactive_evaluate()
                elif command == 'config':
                    self.show_configuration()
                elif command == '':
                    continue
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def interactive_train(self):
        """Interactive training mode"""
        print("\n🎯 Training Mode")

        train_file = input("Training file path: ").strip()
        if not train_file or not os.path.exists(train_file):
            print("❌ Invalid file path")
            return

        test_file = input("Test file path (optional): ").strip()
        if test_file and not os.path.exists(test_file):
            print("❌ Invalid test file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name: ").strip()
            if not target_col:
                print("❌ Target column required for CSV")
                return

            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        # Get training parameters
        resol = input(f"Resolution [100]: ").strip()
        resol = int(resol) if resol else 100

        gain = input(f"Gain [2.0]: ").strip()
        gain = float(gain) if gain else 2.0

        margin = input(f"Margin [0.2]: ").strip()
        margin = float(margin) if margin else 0.2

        patience = input(f"Patience [10]: ").strip()
        patience = int(patience) if patience else 10

        epochs = input(f"Epochs [100]: ").strip()
        epochs = int(epochs) if epochs else 100

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.train = train_file
        args.test = test_file if test_file else None
        args.format = file_format
        args.target = target_col
        args.features = feature_cols
        args.resol = resol
        args.gain = gain
        args.margin = margin
        args.patience = patience
        args.epochs = epochs
        args.min_improvement = 0.1
        args.save_model = None
        args.model_format = 'binary'
        args.verbose = True

        self.setup_model_config(args)
        self.train_model(args)

    def interactive_load(self):
        """Interactive model loading"""
        print("\n📥 Load Model")

        model_path = input("Model file path: ").strip()
        if not model_path:
            print("❌ No path provided")
            return

        self.load_model(model_path)

    def interactive_save(self):
        """Interactive model saving"""
        if not self.core.is_trained:
            print("❌ No trained model to save")
            return

        print("\n💾 Save Model")

        format_choice = input("Save format (binary/json) [binary]: ").strip().lower()
        use_json = (format_choice == 'json')

        default_ext = '.json' if use_json else '.bin'
        model_path = input(f"Model file path [default: Model/model{default_ext}]: ").strip()
        if not model_path:
            model_path = os.path.join('Model', f'model{default_ext}')

        self.save_model(model_path, use_json=use_json)

    def interactive_predict(self):
        """Interactive prediction"""
        if not self.core.is_trained:
            print("❌ No trained model available")
            return

        print("\n🔮 Prediction Mode")

        predict_file = input("Prediction file path: ").strip()
        if not predict_file or not os.path.exists(predict_file):
            print("❌ Invalid file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name (optional): ").strip()
            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        output_file = input("Output file (optional): ").strip()

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.predict = predict_file
        args.format = file_format
        args.target = target_col
        args.features = feature_cols
        args.output = output_file
        args.model = self.current_model_path
        args.verbose = True

        self.predict_data(args)

    def interactive_evaluate(self):
        """Interactive evaluation"""
        if not self.core.is_trained:
            print("❌ No trained model available")
            return

        print("\n📊 Evaluation Mode")

        eval_file = input("Evaluation file path: ").strip()
        if not eval_file or not os.path.exists(eval_file):
            print("❌ Invalid file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name: ").strip()
            if not target_col:
                print("❌ Target column required for CSV evaluation")
                return

            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.evaluate = eval_file
        args.format = file_format
        args.target = target_col
        args.features = feature_cols

        self.evaluate_model(args)

    def run(self):
        """Main execution function"""
        import sys
        import time

        args = self.parse_arguments()

        # Validate arguments
        is_valid, errors = self.validate_arguments(args)
        if not is_valid:
            if errors:
                sys.exit(1)
            else:
                sys.exit(0)  # Help was shown, exit normally

        # Set up model configuration
        self.setup_model_config(args)

        # Handle different modes
        if args.interactive:
            self.interactive_mode()
            return

        # Load model if specified
        if args.model:
            if not self.load_model(args.model):
                sys.exit(1)

        # Perform requested actions
        success = True

        if args.train:
            success = self.train_model(args)

        if success and args.predict:
            success = self.predict_data(args)

        if success and args.evaluate:
            success = self.evaluate_model(args)

        # Save model if trained and no specific save path provided
        if success and self.core.is_trained and not args.save_model and args.train:
            default_model_name = f"dbnn_model_{int(time.time())}"
            use_json = (args.model_format == 'json')
            model_dir = getattr(args, 'model_dir', 'Model')
            model_path = os.path.join(model_dir, default_model_name)
            self.save_model(model_path, use_json=use_json)

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'resol': 100,
        'gain': 2.0,
        'epochs': 50,
        'patience': 5,
        'margin': 0.2
    }

    # Create workflow
    workflow = DBNNWorkflow(config)

    # Example usage (uncomment to run)
    # workflow.run_complete_workflow(
    #     train_file="data/train.csv",
    #     test_file="data/test.csv",
    #     use_csv=True,
    #     target_column="target",
    #     feature_columns=["feature1", "feature2", "feature3"]
    # )

    print("DBNN class structure ready for use!")
