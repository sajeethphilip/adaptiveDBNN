#!/usr/bin/env python3
"""
DBNN Feature Selector - Universal High-Dimensional CSV Processor
Supports any CSV format with automatic column detection and flexible configuration

 pip install cupy-cuda11x numba


"""

import numpy as np
import pandas as pd
import os
import time
import json
import gc
import argparse
import warnings
from typing import List, Dict, Tuple, Optional, Any
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
warnings.filterwarnings('ignore')

# Enhanced GPU detection
def setup_gpu():
    try:
        import cupy as cp
        from numba import cuda, jit
        # Test if GPU is actually available
        cuda.select_device(0)
        print("✅ GPU detected and initialized")
        return True
    except Exception as e:
        print(f"⚠️  GPU not available: {e}")
        return False

GPU_AVAILABLE = setup_gpu()

class ResourceManager:
    """Precise resource management with auto-detection"""

    def __init__(self, target_resolution=80, safety_margin=0.2, gpu_memory_gb=40.0):
        self.target_resolution = target_resolution
        self.safety_margin = safety_margin
        self.gpu_memory_gb = gpu_memory_gb
        self.usable_gpu_memory_gb = self.gpu_memory_gb * (1 - safety_margin)

    def calculate_max_features(self, num_classes: int, resolution: int = None) -> int:
        """Calculate maximum features for dense mode with safety margins"""
        if resolution is None:
            resolution = self.target_resolution

        # Memory calculation for 5D tensor
        bytes_per_element = 4  # float32
        elements_per_feature = (resolution + 2) ** 2 * (num_classes + 2)
        memory_per_feature_gb = elements_per_element * bytes_per_element / (1024**3)

        # Account for both anti_net and anti_wts
        total_memory_per_feature_gb = memory_per_feature_gb * 2

        max_features = int(self.usable_gpu_memory_gb / total_memory_per_feature_gb)

        print(f"🔧 Resource Calculation:")
        print(f"   GPU Memory: {self.gpu_memory_gb}GB, Usable: {self.usable_gpu_memory_gb:.1f}GB")
        print(f"   Resolution: {resolution}, Classes: {num_classes}")
        print(f"   Memory per feature: {total_memory_per_feature_gb:.4f}GB")
        print(f"   Max features (dense): {max_features}")

        return max(10, min(max_features, 10000))  # Increased practical limit

    def get_sampling_batch_size(self, total_features: int) -> int:
        """Calculate optimal batch size for 3-feature sampling"""
        if total_features <= 100:
            return min(50, total_features // 3)
        elif total_features <= 500:
            return 30
        elif total_features <= 1000:
            return 20
        else:
            return 15  # Conservative for 10,000+ features

class FeatureSampler:
    """GPU-accelerated 3-feature sampling and orthogonality analysis"""

    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.gpu_available = GPU_AVAILABLE

    def sample_feature_triplets(self, features: np.ndarray, targets: np.ndarray,
                               num_samples: int = 1000) -> List[Tuple[int, int, int]]:
        """Generate stratified 3-feature triplets covering all features"""
        num_features = features.shape[1]
        triplets = []

        # Ensure each feature appears in multiple triplets
        features_per_triplet = 3
        total_triplets_needed = num_samples

        for _ in range(total_triplets_needed):
            # Stratified sampling: prefer under-represented features
            if len(triplets) < num_features:
                # Initial phase: ensure each feature appears at least once
                base_feature = len(triplets) % num_features
                other_features = np.random.choice(
                    [i for i in range(num_features) if i != base_feature],
                    2, replace=False
                )
                triplet = (base_feature, other_features[0], other_features[1])
            else:
                # Random sampling with coverage bias
                triplet = tuple(np.random.choice(num_features, 3, replace=False))

            triplets.append(triplet)

        print(f"✅ Generated {len(triplets)} feature triplets")
        return triplets

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def compute_2d_histogram_cpu(features_2d: np.ndarray, targets: np.ndarray,
                               resolution: int) -> np.ndarray:
        """Compute 2D histograms for each class (CPU fallback)"""
        num_classes = len(np.unique(targets))
        histograms = np.zeros((resolution, resolution, num_classes), dtype=np.int32)

        # Find data ranges
        x_min, x_max = np.min(features_2d[:, 0]), np.max(features_2d[:, 0])
        y_min, y_max = np.min(features_2d[:, 1]), np.max(features_2d[:, 1])

        # Avoid division by zero
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0

        for i in range(len(features_2d)):
            x_val = features_2d[i, 0]
            y_val = features_2d[i, 1]
            target_class = int(targets[i])

            # Normalize to histogram coordinates
            x_bin = int(((x_val - x_min) / x_range) * (resolution - 1))
            y_bin = int(((y_val - y_min) / y_range) * (resolution - 1))

            # Clamp to valid range
            x_bin = max(0, min(resolution - 1, x_bin))
            y_bin = max(0, min(resolution - 1, y_bin))

            histograms[x_bin, y_bin, target_class] += 1

        return histograms

    def compute_orthogonality_score(self, histograms: np.ndarray) -> float:
        """Calculate orthogonality score from 2D histograms"""
        num_classes = histograms.shape[2]

        if num_classes < 2:
            return 0.0

        # Convert to probabilities
        class_probs = []
        for class_idx in range(num_classes):
            class_hist = histograms[:, :, class_idx]
            total = np.sum(class_hist)
            if total > 0:
                class_probs.append(class_hist / total)
            else:
                class_probs.append(np.zeros_like(class_hist))

        # Calculate pairwise Jensen-Shannon divergences
        js_divergences = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # Jensen-Shannon divergence approximation
                mean_dist = (class_probs[i] + class_probs[j]) / 2

                # Avoid log(0)
                eps = 1e-10
                p = class_probs[i] + eps
                q = class_probs[j] + eps
                m = mean_dist + eps

                kl_pm = np.sum(p * np.log(p / m))
                kl_qm = np.sum(q * np.log(q / m))

                js_div = (kl_pm + kl_qm) / 2
                js_divergences.append(js_div)

        if js_divergences:
            return float(np.mean(js_divergences))
        else:
            return 0.0

    def analyze_triplet(self, features: np.ndarray, targets: np.ndarray,
                       triplet: Tuple[int, int, int], resolution: int = 50) -> Dict[str, Any]:
        """Analyze a single 3-feature triplet"""
        feature_a, feature_b, feature_c = triplet

        results = {
            'triplet': triplet,
            'pair_scores': {},
            'feature_scores': {feature_a: 0.0, feature_b: 0.0, feature_c: 0.0}
        }

        # Test all three features as common axis
        for common_axis in [feature_a, feature_b, feature_c]:
            # Get the other two features
            other_features = [f for f in triplet if f != common_axis]

            if len(other_features) != 2:
                continue

            # Extract 2D feature data
            feature_pair_data = features[:, other_features]

            # Compute 2D histogram
            histograms = self.compute_2d_histogram_cpu(feature_pair_data, targets, resolution)

            # Calculate orthogonality score
            ortho_score = self.compute_orthogonality_score(histograms)

            # Store results
            pair_key = tuple(sorted(other_features))
            results['pair_scores'][pair_key] = ortho_score

            # Update feature scores (common axis gets credit for the pair's performance)
            results['feature_scores'][common_axis] += ortho_score

        return results

class ScoringEngine:
    """Composite feature scoring based on multiple criteria"""

    def __init__(self):
        self.individual_scores = {}
        self.pairwise_scores = {}
        self.versatility_scores = {}

    def compute_individual_discrimination(self, features: np.ndarray,
                                        targets: np.ndarray) -> Dict[int, float]:
        """Calculate individual feature discrimination power"""
        scores = {}
        num_features = features.shape[1]

        for feature_idx in range(num_features):
            feature_data = features[:, feature_idx]

            # Simple variance-based score
            variance_score = np.var(feature_data)

            # Class separation score (simplified)
            unique_classes = np.unique(targets)
            if len(unique_classes) > 1:
                class_means = [np.mean(feature_data[targets == cls]) for cls in unique_classes]
                separation_score = np.var(class_means) / (variance_score + 1e-10)
            else:
                separation_score = 0.0

            # Composite score
            scores[feature_idx] = separation_score * variance_score

        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def aggregate_triplet_results(self, all_triplet_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all triplet analyses"""
        feature_scores = {}
        pair_scores = {}
        feature_appearances = {}

        # Aggregate scores across all triplets
        for result in all_triplet_results:
            # Feature scores
            for feature_idx, score in result['feature_scores'].items():
                feature_scores[feature_idx] = feature_scores.get(feature_idx, 0.0) + score
                feature_appearances[feature_idx] = feature_appearances.get(feature_idx, 0) + 1

            # Pair scores
            for pair, score in result['pair_scores'].items():
                pair_scores[pair] = pair_scores.get(pair, 0.0) + score

        # Normalize feature scores by appearances
        for feature_idx in feature_scores:
            if feature_appearances[feature_idx] > 0:
                feature_scores[feature_idx] /= feature_appearances[feature_idx]

        return {
            'feature_scores': feature_scores,
            'pair_scores': pair_scores,
            'feature_appearances': feature_appearances
        }

    def compute_composite_scores(self, individual_scores: Dict[int, float],
                               triplet_aggregates: Dict[str, Any]) -> Dict[int, float]:
        """Compute final composite feature scores"""
        feature_scores = {}
        triplet_feature_scores = triplet_aggregates['feature_scores']
        feature_appearances = triplet_aggregates['feature_appearances']

        all_features = set(individual_scores.keys()) | set(triplet_feature_scores.keys())

        for feature_idx in all_features:
            individual = individual_scores.get(feature_idx, 0.0)
            triplet = triplet_feature_scores.get(feature_idx, 0.0)
            appearances = feature_appearances.get(feature_idx, 0)

            # Versatility score (how often feature appears in useful contexts)
            versatility = min(1.0, appearances / 10.0)  # Normalize by expected appearances

            # Composite score (weights can be tuned)
            composite_score = (
                0.4 * individual +      # Individual discrimination
                0.5 * triplet +         # Contextual performance
                0.1 * versatility       # Versatility across contexts
            )

            feature_scores[feature_idx] = composite_score

        return feature_scores

class PortfolioBuilder:
    """Optimal feature portfolio construction within resource constraints"""

    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def build_feature_portfolio(self, feature_scores: Dict[int, float],
                              pair_scores: Dict[Tuple[int, int], float],
                              max_features: int) -> Dict[str, Any]:
        """Construct optimal feature portfolio"""
        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top features
        selected_features = [feat_idx for feat_idx, score in sorted_features[:max_features]]
        selected_scores = {feat_idx: score for feat_idx, score in sorted_features[:max_features]}

        # Phase 2 candidates (next best features)
        candidate_cutoff = min(len(sorted_features), max_features * 2)
        phase2_candidates = [feat_idx for feat_idx, score in sorted_features[max_features:candidate_cutoff]]

        # Analyze portfolio quality
        portfolio_score = np.mean([score for _, score in sorted_features[:max_features]])

        # Find best performing pairs in selected portfolio
        portfolio_pairs = {}
        for pair, score in pair_scores.items():
            if pair[0] in selected_features and pair[1] in selected_features:
                portfolio_pairs[pair] = score

        return {
            'selected_features': selected_features,
            'selected_scores': selected_scores,
            'phase2_candidates': phase2_candidates,
            'portfolio_score': portfolio_score,
            'portfolio_pairs': portfolio_pairs,
            'total_features_considered': len(feature_scores)
        }

class DBNNFeatureSelector:
    """Universal feature selection for any high-dimensional CSV"""

    def __init__(self, target_resolution=80, safety_margin=0.2, gpu_memory_gb=40.0):
        self.resource_manager = ResourceManager(target_resolution, safety_margin, gpu_memory_gb)
        self.sampler = FeatureSampler(self.resource_manager)
        self.scoring_engine = ScoringEngine()
        self.portfolio_builder = PortfolioBuilder(self.resource_manager)

        self.results = {}


    def run_feature_selection(self, num_triplet_samples: int = 1000,
                            target_feature_count: Optional[int] = None,
                            resolution: int = 60) -> Dict[str, Any]:
        """Main feature selection pipeline"""
        print("🚀 Starting DBNN Feature Selection Pipeline")
        start_time = time.time()

        # Step 1: Calculate resource constraints
        if target_feature_count is None:
            target_feature_count = self.resource_manager.calculate_max_features(self.num_classes, resolution)

        print(f"🎯 Target: Select {target_feature_count} best features from {self.num_features}")

        # Step 2: Individual feature scoring
        print("📊 Step 1: Computing individual feature discrimination...")
        individual_scores = self.scoring_engine.compute_individual_discrimination(
            self.features, self.targets
        )

        # Step 3: 3-feature sampling and analysis
        print("🔄 Step 2: Performing 3-feature orthogonality analysis...")
        triplets = self.sampler.sample_feature_triplets(
            self.features, self.targets, num_triplet_samples
        )

        triplet_results = []
        for i, triplet in enumerate(triplets):
            if i % 100 == 0:
                print(f"   Analyzed {i}/{len(triplets)} triplets...")

            result = self.sampler.analyze_triplet(self.features, self.targets, triplet, resolution)
            triplet_results.append(result)

        # Step 4: Aggregate and score features
        print("📈 Step 3: Aggregating results and computing composite scores...")
        triplet_aggregates = self.scoring_engine.aggregate_triplet_results(triplet_results)
        composite_scores = self.scoring_engine.compute_composite_scores(
            individual_scores, triplet_aggregates
        )

        # Step 5: Build optimal portfolio
        print("🏗️  Step 4: Building optimal feature portfolio...")
        portfolio = self.portfolio_builder.build_feature_portfolio(
            composite_scores, triplet_aggregates['pair_scores'], target_feature_count
        )

        # Compile final results
        execution_time = time.time() - start_time

        self.results = {
            'portfolio': portfolio,
            'composite_scores': composite_scores,
            'individual_scores': individual_scores,
            'triplet_aggregates': triplet_aggregates,
            'execution_time': execution_time,
            'num_triplets_analyzed': len(triplet_results),
            'resource_info': {
                'target_feature_count': target_feature_count,
                'max_possible_features': self.resource_manager.calculate_max_features(self.num_classes, resolution),
                'gpu_available': GPU_AVAILABLE,
                'resolution': resolution
            }
        }

        self._print_summary()
        return self.results

    def _print_summary(self) -> None:
        """Print comprehensive results summary"""
        portfolio = self.results['portfolio']

        print("\n" + "="*60)
        print("🎉 FEATURE SELECTION COMPLETED")
        print("="*60)
        print(f"⏱️  Execution Time: {self.results['execution_time']:.2f} seconds")
        print(f"🔢 Features: {self.num_features} → {len(portfolio['selected_features'])}")
        print(f"📊 Portfolio Score: {portfolio['portfolio_score']:.4f}")
        print(f"🔄 Triplets Analyzed: {self.results['num_triplets_analyzed']}")
        print(f"🎯 Phase 2 Candidates: {len(portfolio['phase2_candidates'])}")

        print("\n🏆 TOP 10 SELECTED FEATURES:")
        print("-" * 40)
        top_features = sorted(portfolio['selected_scores'].items(),
                            key=lambda x: x[1], reverse=True)[:10]

        for feature_idx, score in top_features:
            feature_name = self.feature_names[feature_idx]
            print(f"  {feature_name:20} | Score: {score:.4f}")

    def export_selected_features(self, output_path: str = None,
                               export_original_data: bool = False,
                               original_csv_path: str = None) -> str:
        """Export selected features with dataset-based naming and target column included"""
        if not self.results:
            raise ValueError("No feature selection results available. Run selection first.")

        portfolio = self.results['portfolio']
        selected_indices = portfolio['selected_features']
        selected_names = [self.feature_names[i] for i in selected_indices]

        # Generate output path based on dataset name if not provided
        if output_path is None:
            if hasattr(self, 'dataset_name'):
                base_name = self.dataset_name
            else:
                base_name = "selected_features"
            output_path = f"{base_name}_SelectedFeatures.csv"

        # Create the output DataFrame with target column included
        if export_original_data and original_csv_path:
            # Export original data with only selected features + target
            original_df = pd.read_csv(original_csv_path)
            # Ensure target column is included
            columns_to_export = selected_names + [self.target_name]
            # Filter to only existing columns
            columns_to_export = [col for col in columns_to_export if col in original_df.columns]
            selected_df = original_df[columns_to_export]
        else:
            # Create new DataFrame with selected features + target
            selected_features = self.features[:, selected_indices]
            selected_df = pd.DataFrame(selected_features, columns=selected_names)
            selected_df[self.target_name] = self.targets

        # Save to CSV
        selected_df.to_csv(output_path, index=False)
        print(f"💾 Selected features + target saved to: {output_path}")

        return output_path

    def load_csv_data(self, csv_path: str, target_column: str = None,
                     feature_columns: List[str] = None, sample_size: int = None,
                     delimiter: str = ',') -> None:
        """Load data from any CSV file with dataset name extraction"""
        print(f"📁 Loading data from: {csv_path}")

        # Extract dataset name from file path
        self.dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"📊 Dataset: {self.dataset_name}")

        # Load CSV
        df = pd.read_csv(csv_path, delimiter=delimiter)

        # [Rest of the loading logic remains the same...]
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"📊 Using {sample_size} samples from {len(df)} total")

        # Auto-detect target column if not specified
        if target_column is None:
            # Try common target column names
            common_targets = ['target', 'class', 'label', 'y', 'output', 'response']
            for col in common_targets:
                if col in df.columns:
                    target_column = col
                    break

            # If still not found, use last column
            if target_column is None:
                target_column = df.columns[-1]
                print(f"⚠️  No target column specified, using last column: {target_column}")

        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")

        # Auto-select feature columns if not specified
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
            print(f"🔍 Using {len(feature_columns)} feature columns (all except target)")

        # Validate feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")

        # Extract data
        self.targets = df[target_column].values
        self.features = df[feature_columns].values
        self.feature_names = feature_columns
        self.target_name = target_column

        self.num_samples, self.num_features = self.features.shape
        self.num_classes = len(np.unique(self.targets))

        print(f"✅ Loaded data: {self.num_samples} samples, {self.num_features} features, {self.num_classes} classes")
        print(f"   Target: {self.target_name}, Features: {len(feature_columns)}")

    def get_selection_report(self) -> str:
        """Generate detailed selection report"""
        if not self.results:
            return "No selection results available."

        portfolio = self.results['portfolio']

        report = [
            "DBNN Feature Selection Report",
            "=" * 50,
            f"Original features: {self.num_features}",
            f"Selected features: {len(portfolio['selected_features'])}",
            f"Portfolio score: {portfolio['portfolio_score']:.4f}",
            f"Execution time: {self.results['execution_time']:.2f}s",
            f"GPU accelerated: {GPU_AVAILABLE}",
            f"Target column: {self.target_name}",
            "",
            "Top features by score:"
        ]

        top_features = sorted(portfolio['selected_scores'].items(),
                            key=lambda x: x[1], reverse=True)[:20]

        for i, (feature_idx, score) in enumerate(top_features, 1):
            feature_name = self.feature_names[feature_idx]
            report.append(f"{i:2d}. {feature_name:20} : {score:.4f}")

        return "\n".join(report)

def main():
    """Universal CSV feature selector with comprehensive examples"""
    parser = argparse.ArgumentParser(
        description='DBNN Universal Feature Selector - Process any high-dimensional CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:

═══════════════════════════════════════════════════════════════════════════════
BASIC USAGE (Auto-detection):
───────────────────────────────────────────────────────────────────────────────
  # Process any CSV with automatic column detection
  python dbnn_feature_selector.py --input data.csv

  # Auto-detects: target column, all feature columns, optimal parameters
  # Output: data_SelectedFeatures.csv, data_SelectionMetadata.json

═══════════════════════════════════════════════════════════════════════════════
SPECIFIC DATASET TYPES:
───────────────────────────────────────────────────────────────────────────────
  # Genomics data (high-dimensional)
  python dbnn_feature_selector.py --input genomics_data.csv --target "phenotype" --output-features 200 --triplets 2000

  # Financial data with specific features
  python dbnn_feature_selector.py --input stock_data.csv --features "open high low volume" --target "trend" --samples 10000

  # Medical data with conservative settings
  python dbnn_feature_selector.py --input patient_data.csv --target "diagnosis" --safety-margin 0.3 --resolution 80

  # Image features (like MNIST)
  python dbnn_feature_selector.py --input image_features.csv --target "label" --output-features 100

═══════════════════════════════════════════════════════════════════════════════
RESOURCE MANAGEMENT:
───────────────────────────────────────────────────────────────────────────────
  # Large dataset with sampling
  python dbnn_feature_selector.py --input huge_dataset.csv --samples 5000 --triplets 1500

  # Conservative memory usage
  python dbnn_feature_selector.py --input data.csv --safety-margin 0.3 --gpu-memory 32

  # High-resolution analysis
  python dbnn_feature_selector.py --input data.csv --resolution 100 --triplets 2000

═══════════════════════════════════════════════════════════════════════════════
FILE FORMATS:
───────────────────────────────────────────────────────────────────────────────
  # Tab-delimited files
  python dbnn_feature_selector.py --input data.tsv --delimiter tab

  # Semicolon-delimited files
  python dbnn_feature_selector.py --input data_european.csv --delimiter semicolon

  # Space-delimited files
  python dbnn_feature_selector.py --input data.txt --delimiter space

═══════════════════════════════════════════════════════════════════════════════
ADVANCED USAGE:
───────────────────────────────────────────────────────────────────────────────
  # Export original data structure with selected features
  python dbnn_feature_selector.py --input original_data.csv --export-original

  # Force specific number of output features
  python dbnn_feature_selector.py --input data.csv --output-features 50

  # Comprehensive analysis with all parameters
  python dbnn_feature_selector.py --input research_data.csv --target "outcome" --samples 15000 --triplets 2500 --resolution 90 --output-features 120 --safety-margin 0.25

═══════════════════════════════════════════════════════════════════════════════
QUICK START (Most Common Use Cases):
───────────────────────────────────────────────────────────────────────────────
  # Quick analysis on standard dataset
  python dbnn_feature_selector.py -i my_data.csv

  # High-dimensional data with many features
  python dbnn_feature_selector.py -i high_dim_data.csv -o 150 -r 70 -n 10000

  # Quick test run on large dataset
  python dbnn_feature_selector.py -i big_data.csv -n 2000 -t 500
        '''
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file path (required)')

    # Data configuration
    parser.add_argument('--target', '-t',
                       help='Target column name (auto-detected: target/class/label/y/output/response/last column)')
    parser.add_argument('--features', '-f', nargs='+',
                       help='Specific feature columns to use (all except target if not specified)')
    parser.add_argument('--samples', '-n', type=int,
                       help='Number of samples to use (all if not specified)')
    parser.add_argument('--delimiter', '-d', default='comma',
                       choices=['comma', 'tab', 'semicolon', 'space'],
                       help='CSV delimiter (default: comma)')

    # Algorithm parameters
    parser.add_argument('--triplets', '-trip', type=int, default=1000,
                       help='Number of triplets to sample (default: 1000)')
    parser.add_argument('--output-features', '-o', type=int,
                       help='Number of features to select (auto-calculated based on GPU memory if not specified)')
    parser.add_argument('--resolution', '-r', type=int, default=60,
                       help='Histogram resolution (default: 60, range: 20-150)')

    # Resource management
    parser.add_argument('--safety-margin', type=float, default=0.2,
                       help='GPU memory safety margin 0.1-0.5 (default: 0.2)')
    parser.add_argument('--gpu-memory', type=float, default=40.0,
                       help='Available GPU memory in GB (default: 40.0 for A100)')

    # Output options
    parser.add_argument('--output', default='selected_features',
                       help='Output file base name (default: auto-generated from input filename)')
    parser.add_argument('--export-original', action='store_true',
                       help='Export original data structure with selected features')

    args = parser.parse_args()

    # Map delimiter
    delimiter_map = {
        'comma': ',',
        'tab': '\t',
        'semicolon': ';',
        'space': ' '
    }
    delimiter = delimiter_map[args.delimiter]

    print("🚀 DBNN Universal Feature Selector")
    print("=" * 60)

    try:
        # Show configuration
        print(f"📋 Configuration:")
        print(f"   Input: {args.input}")
        print(f"   Target: {args.target or 'Auto-detect'}")
        print(f"   Features: {len(args.features) if args.features else 'All except target'}")
        print(f"   Samples: {args.samples or 'All available'}")
        print(f"   Triplets: {args.triplets}")
        print(f"   Output features: {args.output_features or 'Auto-calculated'}")
        print(f"   Resolution: {args.resolution}")
        print(f"   GPU Memory: {args.gpu_memory}GB, Safety: {args.safety_margin*100}%")
        print()

        # Initialize selector
        selector = DBNNFeatureSelector(
            target_resolution=args.resolution,
            safety_margin=args.safety_margin,
            gpu_memory_gb=args.gpu_memory
        )

        # Load data
        selector.load_csv_data(
            csv_path=args.input,
            target_column=args.target,
            feature_columns=args.features,
            sample_size=args.samples,
            delimiter=delimiter
        )

        # Run feature selection
        results = selector.run_feature_selection(
            num_triplet_samples=args.triplets,
            target_feature_count=args.output_features,
            resolution=args.resolution
        )

        # Export results with dataset-based naming
        if args.output == 'selected_features':  # Use auto-generated name
            csv_output = f"{selector.dataset_name}_SelectedFeatures.csv"
            json_output = f"{selector.dataset_name}_SelectionMetadata.json"
        else:
            csv_output = f"{args.output}_SelectedFeatures.csv"
            json_output = f"{args.output}_SelectionMetadata.json"

        # Export selected features WITH target column
        final_csv_path = selector.export_selected_features(
            csv_output,
            export_original_data=args.export_original,
            original_csv_path=args.input
        )
        selector.export_selected_features(json_output)

        # Print final summary
        print("\n" + "="*60)
        print("🎊 PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Original dataset: {selector.num_features} features")
        print(f"🎯 Selected features: {len(results['portfolio']['selected_features'])}")
        print(f"⏱️  Total time: {results['execution_time']:.2f}s")
        print(f"📈 Portfolio score: {results['portfolio']['portfolio_score']:.4f}")

        print(f"\n📁 OUTPUT FILES:")
        print(f"   ✅ {final_csv_path} - Selected features + target (ready for DBNN training)")
        print(f"   ✅ {json_output} - Selection metadata and scores")

        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Use {final_csv_path} for DBNN model training")
        print(f"   2. Review {json_output} for feature importance analysis")
        print(f"   3. Consider Phase 2 for borderline feature evaluation")

        # Show sample of selected features
        portfolio = results['portfolio']
        top_features = sorted(portfolio['selected_scores'].items(),
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🏆 TOP 5 FEATURES:")
        for i, (feature_idx, score) in enumerate(top_features, 1):
            feature_name = selector.feature_names[feature_idx]
            print(f"   {i}. {feature_name:25} (Score: {score:.4f})")

        print(f"\n✅ Feature selection completed successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n💡 TROUBLESHOOTING:")
        print(f"   • Check if file exists: {args.input}")
        print(f"   • Verify column names match your CSV")
        print(f"   • Try --delimiter tab for TSV files")
        print(f"   • Reduce --samples if memory issues occur")
        raise

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
