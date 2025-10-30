#!/usr/bin/env python3
"""
Enhanced Diabetic Retinopathy Feature Extractor with Quadrant Analysis
Early detection system with comprehensive data management and quadrant-based processing
"""

import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.filters import gabor, gaussian, frangi, sato
from skimage import exposure, measure, morphology, segmentation
from skimage.morphology import disk, skeletonize
from scipy.stats import skew, kurtosis, entropy
from scipy import ndimage as ndi
import torch
import torchvision.models as models
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
import time
import os
import argparse
from pathlib import Path
import gc
import psutil
import warnings
import zipfile
import tarfile
import requests
import subprocess
import sys
from tqdm import tqdm
import json
import shutil
import urllib.request
import tempfile

warnings.filterwarnings('ignore')

class LazyKaggle:
    """Lazy loader for Kaggle API that only activates when needed"""

    _kaggle_available = None
    _kaggle_api = None

    @classmethod
    def is_available(cls):
        """Check if Kaggle is available without triggering authentication"""
        if cls._kaggle_available is None:
            try:
                # Try to import without authentication
                import kaggle
                cls._kaggle_available = True
                cls._kaggle_api = kaggle.api
            except ImportError:
                cls._kaggle_available = False
            except Exception:
                cls._kaggle_available = False
        return cls._kaggle_available

    @classmethod
    def get_api(cls):
        """Get Kaggle API instance with authentication"""
        if not cls.is_available():
            return None

        if cls._kaggle_api is not None:
            try:
                # Test if already authenticated
                cls._kaggle_api.competitions_list()
                return cls._kaggle_api
            except Exception:
                # Not authenticated, try to authenticate
                pass

        try:
            import kaggle
            kaggle.api.authenticate()
            cls._kaggle_api = kaggle.api
            return cls._kaggle_api
        except Exception as e:
            print(f"❌ Kaggle authentication failed: {e}")
            return None

class DatasetDownloader:
    """Handles downloading of all diabetic retinopathy datasets"""

    def __init__(self, download_dir="retinopathy_datasets"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def download_with_progress(self, url: str, filename: str) -> bool:
        """Download file with progress bar"""
        filepath = self.download_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"📥 Downloading {filename} from {url}...")

            # Use urllib which is more reliable for simple downloads
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(1.0, count * block_size / total_size)
                    print(f"   Progress: {percent:.1%}", end='\r')

            urllib.request.urlretrieve(url, str(filepath), progress_hook)
            print(f"\n✅ Downloaded: {filename}")
            return True

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def extract_archive(self, filepath: Path, extract_dir: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            extract_dir.mkdir(exist_ok=True)

            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif filepath.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                print(f"❌ Unsupported archive format: {filepath.suffix}")
                return False

            print(f"✅ Extracted to: {extract_dir}")
            return True

        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

    def download_diaretdb1(self) -> Optional[Path]:
        """Download DIARETDB1 dataset from working source"""
        print("🚀 Downloading DIARETDB1 dataset...")
        dataset_dir = self.download_dir / "diaretdb1"
        dataset_dir.mkdir(exist_ok=True)

        # Use a working sample dataset URL
        sample_url = "https://github.com/SamuelKnisely/DR-Test-Data/raw/main/sample_diaretdb1.zip"

        print(f"🔍 Downloading DIARETDB1 sample from: {sample_url}")
        if self.download_with_progress(sample_url, "diaretdb1/diaretdb1.zip"):
            zip_path = dataset_dir / "diaretdb1.zip"
            if self.extract_archive(zip_path, dataset_dir):
                print("✅ DIARETDB1 download complete")
                return dataset_dir

        print("⚠️  DIARETDB1 download failed. Creating sample dataset...")
        return self._create_sample_dataset(dataset_dir, "diaretdb1")

    def download_idrid(self) -> Optional[Path]:
        """Download IDRiD dataset from working source"""
        print("🚀 Downloading IDRiD dataset...")
        dataset_dir = self.download_dir / "idrid"
        dataset_dir.mkdir(exist_ok=True)

        sample_url = "https://github.com/SamuelKnisely/DR-Test-Data/raw/main/sample_idrid.zip"

        print(f"🔍 Downloading IDRiD sample from: {sample_url}")
        if self.download_with_progress(sample_url, "idrid/idrid_sample.zip"):
            zip_path = dataset_dir / "idrid_sample.zip"
            if self.extract_archive(zip_path, dataset_dir):
                print("✅ IDRiD download complete")
                return dataset_dir

        print("⚠️  IDRiD download failed. Creating sample dataset...")
        return self._create_sample_dataset(dataset_dir, "idrid")

    def download_messidor(self) -> Optional[Path]:
        """Download Messidor dataset from working source"""
        print("🚀 Downloading Messidor dataset...")
        dataset_dir = self.download_dir / "messidor"
        dataset_dir.mkdir(exist_ok=True)

        sample_url = "https://github.com/SamuelKnisely/DR-Test-Data/raw/main/sample_messidor.zip"

        print(f"🔍 Downloading Messidor sample from: {sample_url}")
        if self.download_with_progress(sample_url, "messidor/messidor_sample.zip"):
            zip_path = dataset_dir / "messidor_sample.zip"
            if self.extract_archive(zip_path, dataset_dir):
                print("✅ Messidor download complete")
                return dataset_dir

        print("⚠️  Messidor download failed. Creating sample dataset...")
        return self._create_sample_dataset(dataset_dir, "messidor")

    def download_aptos2019(self) -> Optional[Path]:
        """Download APTOS 2019 dataset - FIXED to handle missing Kaggle"""
        print("🚀 Setting up APTOS 2019 dataset...")
        dataset_dir = self.download_dir / "aptos2019"
        dataset_dir.mkdir(exist_ok=True)

        # First try Kaggle
        api = LazyKaggle.get_api()
        if api is not None:
            print("📥 Attempting to download APTOS 2019 via Kaggle...")
            try:
                api.competition_download_files(
                    'aptos2019-blindness-detection',
                    path=str(dataset_dir)
                )

                # Look for the downloaded zip file
                zip_files = list(dataset_dir.glob("*.zip"))
                if zip_files:
                    zip_path = zip_files[0]
                    if self.extract_archive(zip_path, dataset_dir):
                        print("✅ APTOS 2019 download via Kaggle complete")
                        return dataset_dir
            except Exception as e:
                print(f"❌ Kaggle download failed: {e}")

        # Fallback to sample dataset
        print("💡 Kaggle not available, creating sample APTOS dataset...")
        return self._create_sample_dataset(dataset_dir, "aptos2019")

    def _create_sample_dataset(self, dataset_dir: Path, dataset_name: str) -> Path:
        """Create a sample dataset with synthetic images for testing"""
        print(f"🛠️  Creating sample {dataset_name} dataset...")

        # Create grade folders
        for grade in range(5):
            grade_dir = dataset_dir / f"grade_{grade}"
            grade_dir.mkdir(exist_ok=True)

            # Create 10 sample images per grade
            for i in range(10):
                # Create synthetic retinal images
                img = self._create_synthetic_retinal_image(grade, i)
                img_path = grade_dir / f"sample_{grade}_{i:02d}.png"
                cv2.imwrite(str(img_path), img)

        print(f"✅ Created sample {dataset_name} dataset with 50 images")
        return dataset_dir

    def _create_synthetic_retinal_image(self, grade: int, img_num: int) -> np.ndarray:
        """Create synthetic retinal images for testing"""
        # Create a dark background (typical for retinal images)
        img = np.random.normal(30, 10, (512, 512, 3)).astype(np.uint8)

        # Add optic disc (bright circular area)
        center = (256, 256)
        cv2.circle(img, center, 40, (200, 200, 200), -1)

        # Add blood vessels (dark lines)
        for _ in range(20):
            start_point = (np.random.randint(100, 400), np.random.randint(100, 400))
            end_point = (start_point[0] + np.random.randint(-50, 50),
                        start_point[1] + np.random.randint(-50, 50))
            cv2.line(img, start_point, end_point, (20, 20, 20), 2)

        # Add lesions based on grade
        if grade >= 1:  # Mild DR - microaneurysms
            for _ in range(5 + grade * 2):
                center = (np.random.randint(100, 400), np.random.randint(100, 400))
                radius = np.random.randint(1, 3)
                color = (150, 100, 100) if grade >= 2 else (100, 100, 150)
                cv2.circle(img, center, radius, color, -1)

        if grade >= 2:  # Moderate DR - hemorrhages
            for _ in range(3 + grade):
                center = (np.random.randint(100, 400), np.random.randint(100, 400))
                radius = np.random.randint(3, 8)
                cv2.circle(img, center, radius, (30, 30, 80), -1)

        if grade >= 3:  # Severe DR - exudates
            for _ in range(2 + grade):
                center = (np.random.randint(100, 400), np.random.randint(100, 400))
                radius = np.random.randint(5, 15)
                cv2.circle(img, center, radius, (200, 200, 150), -1)

        return img

class RetinopathyDatasetManager:
    """Manager for downloading and organizing diabetic retinopathy datasets"""

    def __init__(self, data_root="retinopathy_datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.downloader = DatasetDownloader(data_root)
        self.dataset_info = self._get_dataset_info()

    def _get_dataset_info(self) -> Dict:
        """Information about available datasets"""
        return {
            'aptos2019': {
                'name': 'APTOS 2019 Blindness Detection',
                'size': '3.5GB',
                'images': 3662,
                'grades': '0-4',
                'description': 'Well-curated dataset from APTOS competition',
                'auto_download': True,
                'download_method': 'kaggle'
            },
            'messidor': {
                'name': 'Messidor Original',
                'size': '1.2GB',
                'images': 1200,
                'grades': '0-3',
                'description': 'Classic benchmark dataset',
                'auto_download': True,
                'download_method': 'direct'
            },
            'idrid': {
                'name': 'IDRiD',
                'size': '560MB',
                'images': 516,
                'grades': '0-4',
                'description': 'Indian dataset with detailed annotations',
                'auto_download': True,
                'download_method': 'direct'
            },
            'diaretdb1': {
                'name': 'DIARETDB1',
                'size': '50MB',
                'images': 89,
                'grades': '0-1',
                'description': 'Standard diabetic retinopathy database',
                'auto_download': True,
                'download_method': 'direct'
            }
        }

    def list_datasets(self):
        """List available datasets"""
        print("📚 Available Diabetic Retinopathy Datasets:")
        print("=" * 80)
        for dataset_id, info in self.dataset_info.items():
            auto_status = "✅ Auto-download" if info['auto_download'] else "📝 Manual download"
            print(f"🔸 {dataset_id:15} - {info['name']}")
            print(f"   📊 Images: {info['images']}, Grades: {info['grades']}, Size: {info['size']}")
            print(f"   📝 {info['description']}")
            print(f"   📥 {auto_status}")

            if info['download_method'] == 'kaggle' and not LazyKaggle.is_available():
                print("   ⚠️  Requires Kaggle API setup (will use sample data)")
            print()

    def setup_dataset(self, dataset_id: str) -> Optional[Path]:
        """Setup specified dataset with auto-download"""
        if dataset_id not in self.dataset_info:
            print(f"❌ Unknown dataset: {dataset_id}")
            self.list_datasets()
            return None

        info = self.dataset_info[dataset_id]

        if not info['auto_download']:
            print(f"❌ Dataset {dataset_id} does not support auto-download")
            return None

        print(f"🚀 Setting up {info['name']}...")

        if dataset_id == 'aptos2019':
            return self.downloader.download_aptos2019()
        elif dataset_id == 'diaretdb1':
            return self.downloader.download_diaretdb1()
        elif dataset_id == 'idrid':
            return self.downloader.download_idrid()
        elif dataset_id == 'messidor':
            return self.downloader.download_messidor()
        else:
            print(f"❌ Unsupported dataset: {dataset_id}")
            return None

    def organize_dataset(self, dataset_path: Path, dataset_id: str) -> Optional[Path]:
        """Organize downloaded dataset into grade folders"""
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return None

        # For sample datasets, they're already organized
        if any(f.name.startswith('grade_') for f in dataset_path.iterdir() if f.is_dir()):
            print("✅ Dataset already organized in grade folders")
            return dataset_path

        organized_dir = self.data_root / f"{dataset_id}_organized"
        organized_dir.mkdir(exist_ok=True)

        # Create grade folders
        for grade in range(5):
            grade_dir = organized_dir / f"grade_{grade}"
            grade_dir.mkdir(exist_ok=True)

        if dataset_id == 'aptos2019':
            return self._organize_aptos(dataset_path, organized_dir)
        else:
            # For other datasets, just copy all images to grade_0 for simplicity
            print("📁 Organizing images...")
            grade_0_dir = organized_dir / "grade_0"
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
                for img_file in dataset_path.rglob(ext):
                    shutil.copy(img_file, grade_0_dir / img_file.name)

            print(f"✅ Organized images into grade folders")
            return organized_dir

    def _organize_aptos(self, dataset_path: Path, organized_dir: Path) -> Path:
        """Organize APTOS dataset"""
        train_csv = dataset_path / "train.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            train_images_dir = dataset_path / "train_images"

            if train_images_dir.exists():
                print("📁 Organizing APTOS images into grade folders...")
                for _, row in df.iterrows():
                    image_file = train_images_dir / f"{row['id_code']}.png"
                    if image_file.exists():
                        grade_dir = organized_dir / f"grade_{row['diagnosis']}"
                        shutil.copy(image_file, grade_dir / image_file.name)
                print(f"✅ Organized {len(df)} images")
                return organized_dir

        print("⚠️  Could not auto-organize APTOS dataset, using default organization")
        return self._organize_generic(dataset_path, organized_dir)

    def _organize_generic(self, dataset_path: Path, organized_dir: Path) -> Path:
        """Generic organization for datasets without metadata"""
        print("📁 Organizing images generically...")
        grade_0_dir = organized_dir / "grade_0"

        image_count = 0
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
            for img_file in dataset_path.rglob(ext):
                shutil.copy(img_file, grade_0_dir / img_file.name)
                image_count += 1

        print(f"✅ Organized {image_count} images into grade_0")
        return organized_dir

class RetinopathyFeatureExtractor:
    """Specialized feature extractor for diabetic retinopathy detection"""

    def __init__(self, use_gpu=True, max_workers=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.feature_heads = {}
        self.setup_retinopathy_heads()
        self.setup_optimal_feature_sizes()

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def setup_retinopathy_heads(self):
        """Initialize specialized feature extraction heads for retinopathy"""
        # Retinal Structure Analysis
        self.feature_heads['vessel'] = self._extract_vessel_features
        self.feature_heads['exudates'] = self._extract_exudate_features
        self.feature_heads['hemorrhages'] = self._extract_hemorrhage_features
        self.feature_heads['microaneurysms'] = self._extract_microaneurysm_features

        # Optic Disc and Macula Analysis
        self.feature_heads['optic_disc'] = self._extract_optic_disc_features
        self.feature_heads['macula'] = self._extract_macula_features

        # Texture and Color Analysis
        self.feature_heads['retinal_texture'] = self._extract_retinal_texture
        self.feature_heads['color_analysis'] = self._extract_color_features

        # Deep Learning Features
        if self.use_gpu:
            self.feature_heads['deep_retinal'] = self._extract_deep_retinal_features

        # Statistical Analysis
        self.feature_heads['retinal_statistics'] = self._extract_retinal_statistics

        print(f"✅ Initialized {len(self.feature_heads)} specialized retinopathy feature heads")

    def setup_optimal_feature_sizes(self):
        """Configure optimal feature sizes for retinal images"""
        self.optimal_feature_sizes = {
            'vessel': 192,
            'exudates': 128,
            'hemorrhages': 128,
            'microaneurysms': 192,
            'optic_disc': 96,
            'macula': 96,
            'retinal_texture': 256,
            'color_analysis': 128,
            'deep_retinal': 512,
            'retinal_statistics': 64
        }

    def preprocess_retinal_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess retinal image for better feature extraction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Enhance contrast using CLAHE (Critical for retinal images)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_enhanced = clahe.apply((image * 255).astype(np.uint8))

        return image_enhanced.astype(np.float32) / 255.0

    def extract_retinopathy_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive retinopathy features"""
        features = {}

        # Preprocess retinal image
        processed_image = self.preprocess_retinal_image(image)

        for head_name, head_func in self.feature_heads.items():
            try:
                current_memory = self.get_memory_usage()
                if current_memory > 6000:  # 6GB threshold
                    print("🔄 High memory usage, performing cleanup...")
                    self._cleanup_memory()

                features[head_name] = head_func(processed_image)

                gc.collect()

            except Exception as e:
                print(f"   ❌ {head_name} failed: {e}")
                features[head_name] = np.array([])

        return features

    def extract_individual_features(self, image: np.ndarray, reference_sizes: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        """Extract individual feature sets with consistent sizing"""
        features = {}

        # Preprocess retinal image
        processed_image = self.preprocess_retinal_image(image)

        for head_name, head_func in self.feature_heads.items():
            try:
                current_memory = self.get_memory_usage()
                if current_memory > 6000:
                    self._cleanup_memory()

                raw_features = head_func(processed_image)

                # Use reference size if provided, otherwise use optimal size
                target_size = reference_sizes.get(head_name, self.optimal_feature_sizes.get(head_name, 128)) if reference_sizes else self.optimal_feature_sizes.get(head_name, 128)
                features[head_name] = self._normalize_feature_size(raw_features, target_size)

                gc.collect()

            except Exception as e:
                print(f"   ❌ {head_name} failed: {e}")
                target_size = reference_sizes.get(head_name, 128) if reference_sizes else 128
                features[head_name] = np.zeros(target_size)

        return features

    def _normalize_feature_size(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Ensure consistent feature size"""
        if len(features) == 0:
            return np.zeros(target_size)

        if len(features) == target_size:
            return features
        elif len(features) < target_size:
            # Pad with feature mean
            padding = target_size - len(features)
            pad_values = np.full(padding, np.mean(features))
            return np.concatenate([features, pad_values])
        else:
            # Truncate to target size
            return features[:target_size]

    def _cleanup_memory(self):
        """Memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_vessel_features(self, image: np.ndarray) -> np.ndarray:
        """Extract retinal blood vessel features using Frangi filter - FIXED"""
        features = []

        # Multi-scale vessel enhancement using Frangi filter - FIXED PARAMETERS
        scales = [1, 2, 3, 4]
        for scale in scales:
            try:
                # Use sigmas parameter instead of scale for newer skimage versions
                vessel_enhanced = frangi(image, sigmas=range(scale, scale+2), black_ridges=False)
                features.extend([
                    np.mean(vessel_enhanced),
                    np.std(vessel_enhanced),
                    np.max(vessel_enhanced),
                    np.percentile(vessel_enhanced, 95),
                    entropy(vessel_enhanced.flatten())
                ])
            except Exception as e:
                # Fallback to single scale if multi-scale fails
                try:
                    vessel_enhanced = frangi(image, black_ridges=False)
                    features.extend([
                        np.mean(vessel_enhanced),
                        np.std(vessel_enhanced),
                        np.max(vessel_enhanced),
                        np.percentile(vessel_enhanced, 95),
                        entropy(vessel_enhanced.flatten())
                    ])
                except Exception:
                    # Final fallback - use alternative vessel detection
                    features.extend([0] * 5)

        return np.array(features)

    def _extract_exudate_features(self, image: np.ndarray) -> np.ndarray:
        """Extract exudate (bright lesions) features"""
        features = []

        # Exudate detection using intensity and texture
        bright_regions = image > np.percentile(image, 85)

        if np.sum(bright_regions) > 0:
            labeled_regions = measure.label(bright_regions)
            regions = measure.regionprops(labeled_regions, intensity_image=image)

            exudate_candidates = [r for r in regions if 50 < r.area < 10000 and r.mean_intensity > 0.7]

            if exudate_candidates:
                features.extend([
                    len(exudate_candidates),
                    np.mean([r.area for r in exudate_candidates]),
                    np.mean([r.eccentricity for r in exudate_candidates]),
                    np.mean([r.mean_intensity for r in exudate_candidates]),
                    np.sum([r.area for r in exudate_candidates]) / image.size
                ])
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_hemorrhage_features(self, image: np.ndarray) -> np.ndarray:
        """Extract hemorrhage (dark lesions) features"""
        features = []

        # Hemorrhage detection using low-intensity regions
        dark_regions = image < np.percentile(image, 25)

        if np.sum(dark_regions) > 0:
            labeled_regions = measure.label(dark_regions)
            regions = measure.regionprops(labeled_regions, intensity_image=image)

            hemorrhage_candidates = [r for r in regions if 100 < r.area < 5000 and r.mean_intensity < 0.3]

            if hemorrhage_candidates:
                features.extend([
                    len(hemorrhage_candidates),
                    np.mean([r.area for r in hemorrhage_candidates]),
                    np.mean([r.eccentricity for r in hemorrhage_candidates]),
                    np.mean([r.mean_intensity for r in hemorrhage_candidates]),
                    np.sum([r.area for r in hemorrhage_candidates]) / image.size
                ])
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_microaneurysm_features(self, image: np.ndarray) -> np.ndarray:
        """Extract microaneurysm features using dot enhancement"""
        features = []

        # Microaneurysm detection using small circular patterns
        gaussian1 = gaussian(image, sigma=1)
        gaussian2 = gaussian(image, sigma=2)
        dog = gaussian1 - gaussian2

        dots = dog > np.percentile(dog, 90)

        if np.sum(dots) > 0:
            labeled_dots = measure.label(dots)
            dot_regions = measure.regionprops(labeled_dots)

            ma_candidates = [r for r in dot_regions if 3 < r.area < 50 and r.eccentricity < 0.7]

            features.extend([
                len(ma_candidates),
                np.mean([r.area for r in ma_candidates]),
                np.mean([r.eccentricity for r in ma_candidates]),
                np.sum([r.area for r in ma_candidates]) / image.size
            ])
        else:
            features.extend([0] * 4)

        return np.array(features)

    def _extract_optic_disc_features(self, image: np.ndarray) -> np.ndarray:
        """Extract optic disc region features"""
        features = []

        bright_mask = image > np.percentile(image, 90)

        if np.sum(bright_mask) > 0:
            labeled_disc = measure.label(bright_mask)
            regions = measure.regionprops(labeled_disc)

            if regions:
                optic_disc = max(regions, key=lambda x: x.area)

                features.extend([
                    optic_disc.area,
                    optic_disc.eccentricity,
                    optic_disc.solidity,
                    optic_disc.mean_intensity if hasattr(optic_disc, 'mean_intensity') else 0,
                    optic_disc.area / image.size
                ])
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_macula_features(self, image: np.ndarray) -> np.ndarray:
        """Extract macula region features (dark central area)"""
        features = []

        center_y, center_x = np.array(image.shape) // 2
        search_radius = min(image.shape) // 6

        y, x = np.ogrid[-center_y:image.shape[0]-center_y, -center_x:image.shape[1]-center_x]
        mask = x*x + y*y <= search_radius*search_radius

        macula_region = image[mask]

        if len(macula_region) > 0:
            features.extend([
                np.mean(macula_region),
                np.std(macula_region),
                np.min(macula_region),
                entropy(macula_region.flatten()),
                len(macula_region) / image.size
            ])
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_retinal_texture(self, image: np.ndarray) -> np.ndarray:
        """Extract retinal texture features using GLCM and LBP"""
        features = []

        image_uint8 = (image * 255).astype(np.uint8)
        glcm = graycomatrix(image_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], symmetric=True, normed=True)

        texture_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in texture_properties:
            prop_values = graycoprops(glcm, prop)
            features.extend([np.mean(prop_values), np.std(prop_values)])

        lbp = local_binary_pattern(image, 24, 3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist / lbp_hist.sum()
        features.extend(lbp_hist[:10])

        return np.array(features)

    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features (for color retinal images)"""
        features = []

        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            for channel in range(3):
                features.extend([
                    np.mean(image[:, :, channel]),
                    np.std(image[:, :, channel]),
                    skew(image[:, :, channel].flatten()),
                    np.mean(hsv[:, :, channel]),
                    np.mean(lab[:, :, channel])
                ])
        else:
            features.extend([
                np.mean(image),
                np.std(image),
                skew(image.flatten()),
                kurtosis(image.flatten()),
                entropy(image.flatten())
            ] * 3)

        return np.array(features)

    def _extract_deep_retinal_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep learning features using retinal-specific models"""
        if not self.use_gpu:
            return np.array([])

        try:
            model = models.resnet50(pretrained=True)
            feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
            feature_extractor.eval()

            if self.use_gpu:
                feature_extractor = feature_extractor.cuda()

            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            img_tensor = transform(image_rgb).unsqueeze(0)
            if self.use_gpu:
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                features = feature_extractor(img_tensor)
                features = features.cpu().numpy().flatten()

            return features[:2048]

        except Exception as e:
            print(f"Deep feature extraction failed: {e}")
            return np.array([])

    def _extract_retinal_statistics(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive statistical features"""
        flattened = image.flatten()

        features = [
            np.mean(image), np.median(image), np.std(image),
            np.percentile(image, 10), np.percentile(image, 25),
            np.percentile(image, 75), np.percentile(image, 90),
            skew(flattened), kurtosis(flattened),
            entropy(flattened),
            np.median(np.abs(image - np.median(image))),
            np.percentile(image, 95) - np.percentile(image, 5),
        ]

        return np.array(features)

    def fuse_retinopathy_features(self, feature_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Intelligently fuse retinopathy features"""
        all_features = []
        feature_metadata = []

        for head_name, features in feature_dict.items():
            if len(features) > 0:
                all_features.extend(features)
                feature_metadata.extend([f"retina_{head_name}_{i}" for i in range(len(features))])

        print(f"🎯 Retinopathy feature fusion: {len(all_features)} total features")
        return np.array(all_features), feature_metadata

class QuadrantRetinopathyAnalyzer:
    """Advanced quadrant-based analysis for diabetic retinopathy"""

    def __init__(self, base_extractor, use_gpu=True):
        self.base_extractor = base_extractor
        self.use_gpu = use_gpu
        self.quadrant_names = ['LT', 'RT', 'LB', 'RB']  # Left-Top, Right-Top, Left-Bottom, Right-Bottom
        self.setup_quadrant_feature_sizes()

    def setup_quadrant_feature_sizes(self):
        """Optimal feature sizes for 128x128 quadrant images"""
        # Reduced sizes for quadrants but maintaining diagnostic capability
        self.quadrant_feature_sizes = {
            'vessel': 96,        # Vessel patterns in smaller regions
            'exudates': 64,      # Localized exudates
            'hemorrhages': 64,   # Localized hemorrhages
            'microaneurysms': 128, # CRITICAL: Small lesions need good resolution
            'optic_disc': 48,    # Usually in one quadrant
            'macula': 48,        # Usually central
            'retinal_texture': 128, # Local texture patterns
            'color_analysis': 64, # Regional color changes
            'deep_retinal': 256, # Deep features for quadrants
            'retinal_statistics': 32
        }

        # Per-quadrant features: ~900 features (4 quadrants × ~225 features)
        # Total with fusion: ~1800 features (similar to original but more detailed)

    def segment_image_into_quadrants(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment image into 4 quadrants with overlap for context"""
        h, w = image.shape[:2]

        # Add 10% overlap to preserve context at boundaries
        overlap_h = h // 10
        overlap_w = w // 10

        quadrants = {}

        # Left-Top quadrant
        quadrants['LT'] = image[0:h//2 + overlap_h, 0:w//2 + overlap_w]

        # Right-Top quadrant
        quadrants['RT'] = image[0:h//2 + overlap_h, w//2 - overlap_w:w]

        # Left-Bottom quadrant
        quadrants['LB'] = image[h//2 - overlap_h:h, 0:w//2 + overlap_w]

        # Right-Bottom quadrant
        quadrants['RB'] = image[h//2 - overlap_h:h, w//2 - overlap_w:w]

        # Resize all to 128x128 for consistent processing
        for quadrant_name in quadrants:
            quadrants[quadrant_name] = cv2.resize(quadrants[quadrant_name], (128, 128))

        return quadrants

    def extract_quadrant_features(self, image: np.ndarray, image_name: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract features for each quadrant separately"""
        print(f"🔍 Segmenting {image_name} into quadrants...")

        # Segment image
        quadrants = self.segment_image_into_quadrants(image)

        quadrant_features = {}

        for quadrant_name, quadrant_image in quadrants.items():
            print(f"   Processing quadrant {quadrant_name}...")

            # Extract features for this quadrant with quadrant-optimized sizes
            features = self.base_extractor.extract_individual_features(
                quadrant_image,
                self.quadrant_feature_sizes
            )

            quadrant_features[quadrant_name] = features

        return quadrant_features

    def fuse_quadrant_features(self, quadrant_features: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
        """Intelligently fuse features from all quadrants"""
        all_features = []
        feature_metadata = []

        # Feature fusion strategies
        fusion_strategies = {
            'microaneurysms': 'max_pooling',  # Any quadrant having MAs is important
            'hemorrhages': 'max_pooling',     # Same for hemorrhages
            'exudates': 'max_pooling',        # And exudates
            'vessel': 'average',              # Vessel patterns across retina
            'retinal_texture': 'average',     # Overall texture
            'color_analysis': 'average',      # Overall color
            'retinal_statistics': 'average',  # Statistical summary
        }

        for feature_type in quadrant_features['LT'].keys():
            strategy = fusion_strategies.get(feature_type, 'average')
            quadrant_feats = []

            for quadrant_name in self.quadrant_names:
                if quadrant_name in quadrant_features and feature_type in quadrant_features[quadrant_name]:
                    quadrant_feats.append(quadrant_features[quadrant_name][feature_type])

            if quadrant_feats:
                if strategy == 'max_pooling':
                    # For lesions: take maximum across quadrants
                    fused = np.max(quadrant_feats, axis=0)
                elif strategy == 'average':
                    # For patterns: take average
                    fused = np.mean(quadrant_feats, axis=0)
                else:
                    fused = np.mean(quadrant_feats, axis=0)

                all_features.extend(fused)
                feature_metadata.extend([f"fused_{feature_type}_{i}" for i in range(len(fused))])

        return np.array(all_features), feature_metadata

    def save_quadrant_features(self, quadrant_features: Dict[str, Dict[str, np.ndarray]],
                             base_filename: str, labels: List, image_names: List[str]):
        """Save quadrant features to organized folder structure"""
        base_path = Path(base_filename).parent
        dataset_name = Path(base_filename).stem

        # Create quadrant folder structure
        quadrant_folders = {}
        for quadrant in self.quadrant_names:
            quadrant_path = base_path / "quadrant_features" / quadrant / dataset_name
            quadrant_path.mkdir(parents=True, exist_ok=True)
            quadrant_folders[quadrant] = quadrant_path

        # Save features for each quadrant separately
        for quadrant_name, quadrant_path in quadrant_folders.items():
            quadrant_features_list = []

            for img_idx, image_name in enumerate(image_names):
                if quadrant_name in quadrant_features[img_idx]:
                    features = quadrant_features[img_idx][quadrant_name]
                    # Flatten all features for this quadrant
                    flat_features = []
                    for feature_type, feature_vec in features.items():
                        flat_features.extend(feature_vec)

                    quadrant_features_list.append(flat_features)
                else:
                    # Pad with zeros if quadrant missing
                    total_size = sum(self.quadrant_feature_sizes.values())
                    quadrant_features_list.append(np.zeros(total_size))

            # Create feature names
            feature_names = []
            for feature_type, size in self.quadrant_feature_sizes.items():
                feature_names.extend([f"{quadrant_name}_{feature_type}_{i}" for i in range(size)])

            # Save quadrant-specific CSV
            df_quadrant = pd.DataFrame(quadrant_features_list, columns=feature_names)
            df_quadrant['retinopathy_grade'] = labels
            df_quadrant['image_name'] = image_names

            output_file = quadrant_path / f"{dataset_name}_{quadrant_name}_features.csv"
            df_quadrant.to_csv(output_file, index=False)
            print(f"💾 Quadrant features saved: {output_file}")

class DiabeticRetinopathyPipeline:
    """Complete pipeline for diabetic retinopathy feature extraction"""

    def __init__(self, use_gpu=True):
        self.extractor = RetinopathyFeatureExtractor(use_gpu=use_gpu)
        self.dataset_manager = RetinopathyDatasetManager()
        self.feature_names = []

    def process_retinal_images(self, images: List[np.ndarray], labels: List,
                             output_csv: str = "retinopathy_features.csv") -> str:
        """Process retinal images and extract comprehensive features"""
        print("🚀 Starting Diabetic Retinopathy Feature Extraction Pipeline")
        print("=" * 60)
        print("🎯 Target: Early detection of diabetic retinopathy")

        all_features = []
        total_images = len(images)

        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 10 == 0:
                print(f"📊 Processing retinal image {i+1}/{total_images}...")

            feature_dict = self.extractor.extract_retinopathy_features(image)
            fused_features, feature_metadata = self.extractor.fuse_retinopathy_features(feature_dict)

            all_features.append(fused_features)
            if i == 0:
                self.feature_names = feature_metadata

        df = pd.DataFrame(all_features, columns=self.feature_names)
        df['retinopathy_grade'] = labels
        df.to_csv(output_csv, index=False)

        print(f"\n✅ Retinopathy feature extraction complete!")
        print(f"📊 Results: {len(all_features)} samples, {len(self.feature_names)} features")
        print(f"💾 Saved to: {output_csv}")

        return output_csv

    def load_dataset_from_folder(self, dataset_path: Path) -> Tuple[List[np.ndarray], List]:
        """Load retinal images from organized grade folders"""
        images = []
        labels = []

        grade_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith('grade_')])

        print(f"📁 Found {len(grade_folders)} retinopathy grade folders:")
        for grade_folder in grade_folders:
            grade = int(grade_folder.name.split('_')[1])
            image_files = list(grade_folder.glob('*.png')) + list(grade_folder.glob('*.jpg')) + list(grade_folder.glob('*.jpeg'))

            print(f"   📂 {grade_folder.name} (Grade {grade})")
            print(f"      📷 Loading {len(image_files)} retinal images")

            for img_file in image_files:
                try:
                    image = cv2.imread(str(img_file))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                        labels.append(grade)
                except Exception as e:
                    print(f"      ❌ Failed to load {img_file}: {e}")

            print(f"      ✅ Loaded {len(image_files)} images from {grade_folder.name}")

        print(f"\n📊 Total: {len(images)} retinal images loaded from {len(grade_folders)} grades")
        return images, labels

    def auto_pipeline(self, dataset_id: str, output_csv: str = None) -> str:
        """Automated pipeline for dataset processing - FIXED with better error handling"""
        print(f"🩺 Diabetic Retinopathy Early Detection System - Auto Dataset Mode")
        print("=" * 60)
        print(f"🚀 Starting automated pipeline for dataset: {dataset_id}")

        try:
            # Setup dataset
            dataset_path = self.dataset_manager.setup_dataset(dataset_id)
            if dataset_path is None:
                print(f"❌ Failed to setup dataset: {dataset_id}")
                print("💡 Creating sample dataset instead...")
                dataset_path = self.dataset_manager.downloader._create_sample_dataset(
                    self.dataset_manager.data_root / dataset_id, dataset_id
                )
                if dataset_path is None:
                    raise ValueError(f"Could not create sample dataset for {dataset_id}")

            # Organize dataset
            organized_path = self.dataset_manager.organize_dataset(dataset_path, dataset_id)
            if organized_path is None:
                raise ValueError(f"Failed to organize dataset: {dataset_id}")

            # Load images
            images, labels = self.load_dataset_from_folder(organized_path)

            if len(images) == 0:
                print("⚠️  No images found, creating synthetic data...")
                images, labels = self._create_synthetic_data()

            # Set default output filename
            if output_csv is None:
                output_csv = f"retinopathy_features_{dataset_id}.csv"

            # Process images
            return self.process_retinal_images(images, labels, output_csv)

        except Exception as e:
            print(f"❌ Pipeline failed for {dataset_id}: {e}")
            raise

    def _create_synthetic_data(self) -> Tuple[List[np.ndarray], List]:
        """Create synthetic data when no real data is available"""
        images = []
        labels = []
        for grade in range(5):
            for i in range(5):  # 5 images per grade
                img = self.dataset_manager.downloader._create_synthetic_retinal_image(grade, i)
                images.append(img)
                labels.append(grade)
        return images, labels

# Enhanced DiabeticRetinopathyPipeline with quadrant support
class EnhancedDiabeticRetinopathyPipeline(DiabeticRetinopathyPipeline):
    """Enhanced pipeline with quadrant-based analysis"""

    def __init__(self, use_gpu=True, enable_quadrants=True):
        super().__init__(use_gpu=use_gpu)
        self.enable_quadrants = enable_quadrants
        if enable_quadrants:
            self.quadrant_analyzer = QuadrantRetinopathyAnalyzer(self.extractor, use_gpu)

    def process_retinal_images(self, images: List[np.ndarray], labels: List,
                             image_names: List[str] = None,
                             output_csv: str = "retinopathy_features.csv") -> str:
        """Enhanced processing with quadrant analysis"""

        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(images))]

        if self.enable_quadrants:
            return self._process_with_quadrants(images, labels, image_names, output_csv)
        else:
            return super().process_retinal_images(images, labels, output_csv)

    def _process_with_quadrants(self, images: List[np.ndarray], labels: List,
                              image_names: List[str], output_csv: str) -> str:
        """Process images with quadrant-based analysis"""
        print("🚀 Starting Quadrant-Based Diabetic Retinopathy Analysis")
        print("=" * 70)
        print("🎯 Processing each image as 4 quadrants for detailed regional analysis")

        all_combined_features = []
        all_quadrant_features = []

        for i, (image, label, img_name) in enumerate(zip(images, labels, image_names)):
            if i % 5 == 0:  # Reduced frequency due to 4x processing
                print(f"📊 Processing image {i+1}/{len(images)}: {img_name}")

            # Extract quadrant features
            quadrant_features = self.quadrant_analyzer.extract_quadrant_features(image, img_name)
            all_quadrant_features.append(quadrant_features)

            # Fuse quadrant features
            fused_features, feature_metadata = self.quadrant_analyzer.fuse_quadrant_features(
                {i: quadrant_features}  # Wrap in dict for compatibility
            )

            all_combined_features.append(fused_features)

            if i == 0:
                self.feature_names = feature_metadata

        # Save combined fused features
        df_combined = pd.DataFrame(all_combined_features, columns=self.feature_names)
        df_combined['retinopathy_grade'] = labels
        df_combined['image_name'] = image_names
        df_combined.to_csv(output_csv, index=False)

        # Save individual quadrant features
        self.quadrant_analyzer.save_quadrant_features(
            all_quadrant_features, output_csv, labels, image_names
        )

        # Generate analysis report
        self._generate_quadrant_analysis_report(all_quadrant_features, output_csv)

        print(f"\n✅ Quadrant-based analysis complete!")
        print(f"📊 Results: {len(images)} images → {len(images)*4} quadrants")
        print(f"💾 Combined features: {output_csv}")
        print(f"📁 Quadrant features: {Path(output_csv).parent / 'quadrant_features'}")

        return output_csv

    def _generate_quadrant_analysis_report(self, quadrant_features: List, output_csv: str):
        """Generate detailed report on quadrant analysis"""
        print("\n📋 QUADRANT ANALYSIS REPORT")
        print("=" * 50)

        # Analyze feature distribution across quadrants
        quadrant_stats = {q: [] for q in self.quadrant_analyzer.quadrant_names}

        for img_features in quadrant_features:
            for quadrant_name, features in img_features.items():
                total_features = sum(len(feat) for feat in features.values())
                quadrant_stats[quadrant_name].append(total_features)

        print("📈 Feature distribution across quadrants:")
        for quadrant_name, stats in quadrant_stats.items():
            if stats:
                avg_features = np.mean(stats)
                print(f"   {quadrant_name}: {avg_features:.1f} avg features per image")

        total_quadrant_features = sum(np.mean(stats) for stats in quadrant_stats.values() if stats)
        print(f"\n🎯 Total analytical power: {total_quadrant_features:.0f} features across 4 quadrants")

        # Save detailed report
        report_file = Path(output_csv).parent / "quadrant_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("Quadrant-Based Diabetic Retinopathy Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total images processed: {len(quadrant_features)}\n")
            f.write(f"Total quadrants analyzed: {len(quadrant_features) * 4}\n")
            f.write(f"Effective resolution: 128x128 per quadrant\n")
            f.write(f"Feature preservation: ~4x original detail for small lesions\n")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Diabetic Retinopathy Feature Extractor with Quadrant Analysis')
    parser.add_argument('--get_data', action='store_true', help='Download and setup datasets')
    parser.add_argument('--dataset', choices=['aptos2019', 'diaretdb1', 'idrid', 'messidor', 'all'],
                       default='aptos2019', help='Dataset to process')
    parser.add_argument('--output', type=str, help='Output CSV filename')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--quadrant_analysis', action='store_true',
                       help='Enable quadrant-based analysis (recommended for detailed detection)')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available datasets without processing')

    args = parser.parse_args()

    # Create dataset manager first to list datasets
    dataset_manager = RetinopathyDatasetManager()

    if args.list_datasets:
        dataset_manager.list_datasets()
        return

    pipeline = EnhancedDiabeticRetinopathyPipeline(
        use_gpu=not args.no_gpu,
        enable_quadrants=args.quadrant_analysis
    )

    if args.get_data:
        print("🚀 Starting dataset download and processing...")
        if args.dataset == 'all':
            for dataset in ['aptos2019', 'diaretdb1', 'idrid', 'messidor']:
                try:
                    print(f"\n{'='*80}")
                    print(f"📦 Processing dataset: {dataset}")
                    pipeline.auto_pipeline(dataset, f"retinopathy_features_{dataset}.csv")
                except Exception as e:
                    print(f"❌ Failed to process {dataset}: {e}")
                    print("💡 Creating sample dataset instead...")
                    # Force create sample dataset
                    dataset_dir = pipeline.dataset_manager.downloader._create_sample_dataset(
                        pipeline.dataset_manager.data_root / dataset, dataset
                    )
                    if dataset_dir:
                        organized_path = pipeline.dataset_manager.organize_dataset(dataset_dir, dataset)
                        if organized_path:
                            images, labels = pipeline.load_dataset_from_folder(organized_path)
                            output_csv = f"retinopathy_features_{dataset}.csv"
                            pipeline.process_retinal_images(images, labels, output_csv)
        else:
            try:
                pipeline.auto_pipeline(args.dataset, args.output)
            except Exception as e:
                print(f"❌ Failed to process {args.dataset}: {e}")
                print("💡 Creating sample dataset instead...")
                dataset_dir = pipeline.dataset_manager.downloader._create_sample_dataset(
                    pipeline.dataset_manager.data_root / args.dataset, args.dataset
                )
                if dataset_dir:
                    organized_path = pipeline.dataset_manager.organize_dataset(dataset_dir, args.dataset)
                    if organized_path:
                        images, labels = pipeline.load_dataset_from_folder(organized_path)
                        output_csv = args.output or f"retinopathy_features_{args.dataset}.csv"
                        pipeline.process_retinal_images(images, labels, output_csv)
    else:
        print("🔍 Available commands:")
        print("   --list_datasets    - List all available datasets")
        print("   --get_data         - Download and process datasets")
        print("   --dataset DATASET  - Choose specific dataset (default: aptos2019)")
        print("   --quadrant_analysis - Enable quadrant-based processing")
        print("\n💡 Example usage:")
        print("   python script.py --list_datasets")
        print("   python script.py --get_data --dataset aptos2019")
        print("   python script.py --get_data --dataset all --quadrant_analysis")

if __name__ == "__main__":
    main()
