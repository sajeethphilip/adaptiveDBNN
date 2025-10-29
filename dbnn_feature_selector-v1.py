#!/usr/bin/env python3
"""
DBNN Feature Selector - Complete Standalone Module
Ultra-optimized feature selection for DBNN orthogonality optimization
Designed for A100 40GB with 1000+ → 100 features reduction
"""

import numpy as np
import pandas as pd
import os
import time
import json
import gc
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
# Try to import GPU modules
try:
    import cupy as cp
    from numba import cuda, jit
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU not available, using CPU fallback")



class ResourceManager:
    """Precise resource management for A100 40GB optimization"""

    def __init__(self, target_resolution=80, safety_margin=0.2):
        self.target_resolution = target_resolution
        self.safety_margin = safety_margin
        self.gpu_memory_gb = 40.0  # A100 40GB
        self.usable_gpu_memory_gb = self.gpu_memory_gb * (1 - safety_margin)

    def calculate_max_features(self, num_classes: int, resolution: int = None) -> int:
        """Calculate maximum features for dense mode with safety margins"""
        if resolution is None:
            resolution = self.target_resolution

        # Memory calculation for 5D tensor: (features+2)^2 * (resolution+2)^2 * (classes+2) * 4 bytes
        bytes_per_element = 4  # float32
        elements_per_feature = (resolution + 2) ** 2 * (num_classes + 2)
        memory_per_feature_gb = elements_per_feature * bytes_per_element / (1024**3)

        # Account for both anti_net and anti_wts
        total_memory_per_feature_gb = memory_per_feature_gb * 2

        max_features = int(self.usable_gpu_memory_gb / total_memory_per_feature_gb)

        print(f"🔧 Resource Calculation:")
        print(f"   GPU Memory: {self.gpu_memory_gb}GB, Usable: {self.usable_gpu_memory_gb:.1f}GB")
        print(f"   Resolution: {resolution}, Classes: {num_classes}")
        print(f"   Memory per feature: {total_memory_per_feature_gb:.4f}GB")
        print(f"   Max features (dense): {max_features}")

        return max(10, min(max_features, 200))  # Practical limits

    def get_sampling_batch_size(self, total_features: int) -> int:
        """Calculate optimal batch size for 3-feature sampling"""
        if total_features <= 100:
            return min(50, total_features // 3)
        elif total_features <= 500:
            return 30
        else:
            return 20  # Conservative for 1000+ features

    def check_memory_safety(self, current_usage_gb: float) -> bool:
        """Check if memory usage is within safe limits"""
        return current_usage_gb <= self.usable_gpu_memory_gb

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

    def analyze_triplets_parallel(self, triplets: List[Tuple], max_workers: int = 8):
        """Parallel triplet analysis"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for triplet in triplets:
                future = executor.submit(self.analyze_triplet_wrapper,
                                       self.features, self.targets, triplet)
                futures.append(future)

            results = []
            for i, future in enumerate(futures):
                if i % 100 == 0:
                    print(f"   Completed {i}/{len(triplets)} triplets...")
                results.append(future.result())
            return results

    @staticmethod
    def analyze_triplet_wrapper(features, targets, triplet):
        """Wrapper for parallel processing"""
        sampler = FeatureSampler(ResourceManager())  # Lightweight
        return sampler.analyze_triplet(features, targets, triplet)

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
    """Main feature selection module - Complete standalone implementation"""

    def __init__(self, target_resolution=80, safety_margin=0.2):
        self.resource_manager = ResourceManager(target_resolution, safety_margin)
        self.sampler = FeatureSampler(self.resource_manager)
        self.scoring_engine = ScoringEngine()
        self.portfolio_builder = PortfolioBuilder(self.resource_manager)

        self.results = {}

    def load_data(self, features: np.ndarray, targets: np.ndarray,
                 feature_names: Optional[List[str]] = None) -> None:
        """Load and validate input data"""
        self.features = features
        self.targets = targets
        self.num_samples, self.num_features = features.shape
        self.num_classes = len(np.unique(targets))

        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.num_features)]
        else:
            self.feature_names = feature_names

        print(f"✅ Loaded data: {self.num_samples} samples, {self.num_features} features, {self.num_classes} classes")

    def run_feature_selection(self, num_triplet_samples: int = 500,
                            target_feature_count: Optional[int] = None) -> Dict[str, Any]:
        """Main feature selection pipeline"""
        print("🚀 Starting DBNN Feature Selection Pipeline")
        start_time = time.time()

        # Step 1: Calculate resource constraints
        if target_feature_count is None:
            target_feature_count = self.resource_manager.calculate_max_features(self.num_classes)

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

            result = self.sampler.analyze_triplet(self.features, self.targets, triplet)
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
                'max_possible_features': self.resource_manager.calculate_max_features(self.num_classes),
                'gpu_available': GPU_AVAILABLE
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

    def export_selected_features(self, output_path: str,
                               original_data: Optional[pd.DataFrame] = None) -> None:
        """Export selected features to file"""
        if not self.results:
            raise ValueError("No feature selection results available. Run selection first.")

        portfolio = self.results['portfolio']
        selected_indices = portfolio['selected_features']

        # Create output data
        if original_data is not None:
            # If original DataFrame provided, select columns
            output_data = original_data.iloc[:, selected_indices]
        else:
            # If only features array, create new DataFrame
            selected_features = self.features[:, selected_indices]
            selected_names = [self.feature_names[i] for i in selected_indices]
            output_data = pd.DataFrame(selected_features, columns=selected_names)

        # Export based on file extension
        if output_path.endswith('.csv'):
            output_data.to_csv(output_path, index=False)
            print(f"💾 Selected features saved to: {output_path}")
        elif output_path.endswith('.json'):
            # Save selection metadata
            metadata = {
                'selected_feature_indices': selected_indices,
                'selected_feature_names': [self.feature_names[i] for i in selected_indices],
                'feature_scores': {self.feature_names[i]: score
                                 for i, score in portfolio['selected_scores'].items()},
                'portfolio_score': portfolio['portfolio_score'],
                'execution_time': self.results['execution_time']
            }
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Selection metadata saved to: {output_path}")
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

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
            "",
            "Top features by score:"
        ]

        top_features = sorted(portfolio['selected_scores'].items(),
                            key=lambda x: x[1], reverse=True)[:20]

        for i, (feature_idx, score) in enumerate(top_features, 1):
            feature_name = self.feature_names[feature_idx]
            report.append(f"{i:2d}. {feature_name:20} : {score:.4f}")

        return "\n".join(report)

# =============================================================================
# MNIST TEST FUNCTION
# =============================================================================

def test_mnist_feature_selection(csv_file_path: str, sample_size: int = 5000):
    """
    Test feature selection on MNIST extracted features
    """
    print("🧪 Testing DBNN Feature Selector on MNIST Data")
    print("=" * 60)

    # Load MNIST data
    print("📁 Loading MNIST features...")
    df = pd.read_csv(csv_file_path)

    # Sample data if too large for quick testing
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"📊 Using {sample_size} samples for quick testing")

    # Extract features and targets
    targets = df['Class'].values
    features = df.drop('Class', axis=1).values
    feature_names = [f'pixel_{i}' for i in range(features.shape[1])]

    print(f"📈 Dataset: {features.shape[0]} samples, {features.shape[1]} features, {len(np.unique(targets))} classes")

    # Initialize selector with A100-optimized settings
    selector = DBNNFeatureSelector(
        target_resolution=60,  # Lower resolution for speed
        safety_margin=0.3      # Conservative memory usage
    )

    # Load data
    selector.load_data(features, targets, feature_names)

    # Run feature selection
    print("\n🚀 Starting feature selection on MNIST data...")
    results = selector.run_feature_selection(
        num_triplet_samples=1000,    # Start with 1000 triplets
        target_feature_count=100     # Target 100 features
    )

    # Export results
    selector.export_selected_features("mnist_selected_features.csv")
    selector.export_selected_features("mnist_selection_metadata.json")

    # Print detailed report
    print("\n" + selector.get_selection_report())

    return results

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def create_test_data(num_samples=1000, num_features=100, num_classes=5):
    """Create synthetic test data for evaluation"""
    np.random.seed(42)

    # Create features with varying discrimination power
    features = np.random.randn(num_samples, num_features)
    targets = np.random.randint(0, num_classes, num_samples)

    # Make some features actually predictive
    for i in range(min(20, num_features)):
        features[:, i] += targets * 0.5  # Add class-dependent signal

    feature_names = [f'Test_Feature_{i:03d}' for i in range(num_features)]

    return features, targets, feature_names

def main():
    """Example usage - choose between test data and MNIST"""
    import argparse

    parser = argparse.ArgumentParser(description='DBNN Feature Selector')
    parser.add_argument('--mnist', type=str, help='Path to MNIST CSV file')
    parser.add_argument('--samples', type=int, default=5000, help='Sample size for MNIST')

    args = parser.parse_args()

    if args.mnist:
        # Test on MNIST data
        test_mnist_feature_selection(args.mnist, args.samples)
    else:
        # Test on synthetic data
        print("🧪 DBNN Feature Selector - Synthetic Data Test")

        features, targets, feature_names = create_test_data(
            num_samples=2000, num_features=200, num_classes=5
        )

        selector = DBNNFeatureSelector(target_resolution=80, safety_margin=0.2)
        selector.load_data(features, targets, feature_names)

        results = selector.run_feature_selection(
            num_triplet_samples=300,
            target_feature_count=50
        )

        selector.export_selected_features("selected_features.csv")
        selector.export_selected_features("selection_metadata.json")

        print("\n✅ Feature selection pipeline completed successfully!")

if __name__ == "__main__":
    main()
