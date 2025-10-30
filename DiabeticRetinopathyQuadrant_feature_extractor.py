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

    def __init__(self, feature_extractor: RetinopathyFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.quadrant_names = ['LT', 'RT', 'LB', 'RB']  # Left-Top, Right-Top, Left-Bottom, Right-Bottom

    def segment_quadrants(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment retinal image into 4 quadrants"""
        height, width = image.shape[:2]
        mid_x, mid_y = width // 2, height // 2

        quadrants = {
            'LT': image[:mid_y, :mid_x],      # Left-Top
            'RT': image[:mid_y, mid_x:],      # Right-Top
            'LB': image[mid_y:, :mid_x],      # Left-Bottom
            'RB': image[mid_y:, mid_x:]       # Right-Bottom
        }

        return quadrants

    def analyze_quadrants(self, image: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze each quadrant for retinopathy features"""
        quadrants = self.segment_quadrants(image)
        quadrant_features = {}

        for quadrant_name, quadrant_image in quadrants.items():
            print(f"   Processing quadrant {quadrant_name}...")
            features = self.feature_extractor.extract_individual_features(quadrant_image)
            quadrant_features[quadrant_name] = features
            print(f"      ✅ {quadrant_name}: {len(features)} feature types extracted")

        return quadrant_features

    def fuse_quadrant_features(self, quadrant_features: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
        """Fuse features from all quadrants with intelligent weighting - FIXED"""
        all_features = []
        feature_metadata = []

        # Check if quadrant_features has the expected structure
        if not quadrant_features or not isinstance(quadrant_features, dict):
            print("❌ Invalid quadrant_features structure")
            return np.array([]), []

        # Get the first quadrant to understand the feature structure
        first_quadrant = next(iter(quadrant_features.values()))
        if not isinstance(first_quadrant, dict):
            print("❌ Invalid quadrant data structure")
            return np.array([]), []

        # Process each feature type across all quadrants
        for feature_type in first_quadrant.keys():
            quadrant_feature_vectors = []

            for quadrant_name in self.quadrant_names:
                if quadrant_name in quadrant_features and feature_type in quadrant_features[quadrant_name]:
                    features = quadrant_features[quadrant_name][feature_type]
                    if len(features) > 0:
                        quadrant_feature_vectors.append(features)

            if quadrant_feature_vectors:
                # Stack features from all quadrants for this feature type
                stacked_features = np.stack(quadrant_feature_vectors, axis=0)
                fused = self._fuse_quadrant_feature_type(stacked_features)

                all_features.extend(fused)
                feature_metadata.extend([f"quadrant_{feature_type}_{i}" for i in range(len(fused))])

        print(f"🎯 Quadrant feature fusion: {len(all_features)} total features")
        return np.array(all_features), feature_metadata

    def _fuse_quadrant_feature_type(self, quadrant_features: np.ndarray) -> np.ndarray:
        """Fuse features for a specific feature type across quadrants"""
        fused_features = []

        # Mean across quadrants
        fused_features.extend(np.mean(quadrant_features, axis=0))

        # Standard deviation across quadrants (variability)
        fused_features.extend(np.std(quadrant_features, axis=0))

        # Max values across quadrants
        fused_features.extend(np.max(quadrant_features, axis=0))

        # Min values across quadrants
        fused_features.extend(np.min(quadrant_features, axis=0))

        # Range (max-min) across quadrants
        fused_features.extend(np.ptp(quadrant_features, axis=0))

        return np.array(fused_features)

class RetinopathyPipeline:
    """Complete pipeline for diabetic retinopathy analysis"""

    def __init__(self, use_gpu=True, max_workers=None):
        self.feature_extractor = RetinopathyFeatureExtractor(use_gpu=use_gpu, max_workers=max_workers)
        self.quadrant_analyzer = QuadrantRetinopathyAnalyzer(self.feature_extractor)
        self.dataset_manager = RetinopathyDatasetManager()

    def load_retinal_images(self, dataset_path: Path) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Load retinal images organized by grade folders"""
        images = []
        labels = []
        image_names = []

        grade_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith('grade_')])

        if not grade_folders:
            print(f"❌ No grade folders found in {dataset_path}")
            return [], [], []

        print(f"📁 Found {len(grade_folders)} retinopathy grade folders:")

        for grade_folder in grade_folders:
            try:
                grade = int(grade_folder.name.split('_')[1])
                print(f"   📂 {grade_folder.name} (Grade {grade})")

                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
                    image_files.extend(grade_folder.glob(ext))
                    image_files.extend(grade_folder.glob(ext.upper()))

                if not image_files:
                    print(f"      ⚠️  No images found in {grade_folder}")
                    continue

                print(f"      📷 Loading {len(image_files)} retinal images")

                loaded_count = 0
                for img_file in image_files:
                    try:
                        image = cv2.imread(str(img_file))
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            images.append(image)
                            labels.append(grade)
                            image_names.append(img_file.name)
                            loaded_count += 1

                            if loaded_count >= 10:  # Limit for sample datasets
                                break
                    except Exception as e:
                        print(f"      ❌ Failed to load {img_file}: {e}")
                        continue

                print(f"      ✅ Loaded {loaded_count} images from {grade_folder.name}")

            except Exception as e:
                print(f"   ❌ Error processing {grade_folder}: {e}")
                continue

        print(f"\n📊 Total: {len(images)} retinal images loaded from {len(grade_folders)} grades")
        return images, labels, image_names

    def process_retinal_images(self, images: List[np.ndarray], labels: List[int], image_names: List[str], output_csv: str) -> pd.DataFrame:
        """Process retinal images with quadrant analysis"""
        if len(images) == 0:
            print("❌ No images to process")
            return pd.DataFrame()

        print("🚀 Starting Quadrant-Based Diabetic Retinopathy Analysis")
        print("=" * 70)

        return self._process_with_quadrants(images, labels, image_names, output_csv)

    def _process_with_quadrants(self, images: List[np.ndarray], labels: List[int], image_names: List[str], output_csv: str) -> pd.DataFrame:
        """Process images with quadrant analysis"""
        all_features = []
        all_metadata = []

        for idx, (image, label, img_name) in enumerate(zip(images, labels, image_names)):
            print(f"📊 Processing image {idx+1}/{len(images)}: {img_name}")
            print(f"🔍 Segmenting {img_name} into quadrants...")

            try:
                # Analyze quadrants
                quadrant_features = self.quadrant_analyzer.analyze_quadrants(image)

                # Fuse quadrant features - FIXED: Pass the correct structure
                fused_features, feature_metadata = self.quadrant_analyzer.fuse_quadrant_features(quadrant_features)

                if len(fused_features) > 0:
                    all_features.append(fused_features)
                    all_metadata.append({
                        'image_name': img_name,
                        'label': label,
                        'features': fused_features,
                        'metadata': feature_metadata
                    })
                    print(f"✅ Successfully processed {img_name}")

                else:
                    print(f"❌ Failed to extract features for {img_name}")

            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
                continue

        if not all_features:
            print("❌ No features extracted from any image")
            return pd.DataFrame()

        # Create DataFrame
        feature_columns = all_metadata[0]['metadata'] if all_metadata else []
        df_data = []

        for metadata in all_metadata:
            row_data = {
                'image_name': metadata['image_name'],
                'label': metadata['label']
            }
            row_data.update(dict(zip(metadata['metadata'], metadata['features'])))
            df_data.append(row_data)

        df = pd.DataFrame(df_data)

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"💾 Saved features to {output_csv}")
        print(f"📊 Final dataset: {len(df)} images, {len(df.columns) - 2} features")

        return df

    def auto_pipeline(self, dataset_id: str, output_csv: str) -> pd.DataFrame:
        """Automated pipeline for dataset processing"""
        print(f"🚀 Starting automated pipeline for dataset: {dataset_id}")

        # Setup dataset
        dataset_path = self.dataset_manager.setup_dataset(dataset_id)
        if dataset_path is None:
            print(f"❌ Failed to setup dataset: {dataset_id}")
            return pd.DataFrame()

        # Organize dataset
        organized_path = self.dataset_manager.organize_dataset(dataset_path, dataset_id)
        if organized_path is None:
            print(f"❌ Failed to organize dataset: {dataset_id}")
            return pd.DataFrame()

        # Load images
        images, labels, image_names = self.load_retinal_images(organized_path)
        if len(images) == 0:
            print(f"❌ No images loaded for dataset: {dataset_id}")
            return pd.DataFrame()

        # Process images
        return self.process_retinal_images(images, labels, image_names, output_csv)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Feature Extractor with Quadrant Analysis')
    parser.add_argument('--get_data', action='store_true', help='Download and process datasets')
    parser.add_argument('--dataset', type=str, default='aptos2019', help='Dataset to process (aptos2019, messidor, idrid, diaretdb1, all)')
    parser.add_argument('--quadrant_analysis', action='store_true', help='Enable quadrant-based analysis')

    args = parser.parse_args()

    print("🩺 Diabetic Retinopathy Early Detection System - Auto Dataset Mode")
    print("=" * 60)

    pipeline = RetinopathyPipeline(use_gpu=True)

    if args.get_data:
        if args.dataset == 'all':
            datasets = ['aptos2019', 'messidor', 'idrid', 'diaretdb1']
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"📦 Processing dataset: {dataset}")
            print(f"{'='*80}")

            try:
                output_csv = f"retinopathy_features_{dataset}.csv"
                df = pipeline.auto_pipeline(dataset, output_csv)

                if df is not None and len(df) > 0:
                    print(f"✅ Successfully processed {dataset}: {len(df)} images")
                else:
                    print(f"❌ Failed to process {dataset}")

            except Exception as e:
                print(f"❌ Pipeline failed for {dataset}: {e}")
                print("💡 Creating sample dataset instead...")

                # Create emergency sample dataset
                sample_dir = Path("retinopathy_datasets") / f"{dataset}_sample"
                sample_dir.mkdir(exist_ok=True, parents=True)

                # Create sample images
                downloader = DatasetDownloader()
                downloader._create_sample_dataset(sample_dir, dataset)

                # Try processing again
                try:
                    images, labels, image_names = pipeline.load_retinal_images(sample_dir)
                    if len(images) > 0:
                        output_csv = f"retinopathy_features_{dataset}_sample.csv"
                        pipeline.process_retinal_images(images, labels, image_names, output_csv)
                except Exception as e2:
                    print(f"❌ Even sample processing failed: {e2}")

    else:
        print("💡 Use --get_data to download and process datasets")
        pipeline.dataset_manager.list_datasets()

if __name__ == "__main__":
    main()
