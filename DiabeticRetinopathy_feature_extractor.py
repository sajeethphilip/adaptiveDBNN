#!/usr/bin/env python3
"""
Multi-Head Ensemble Feature Extractor for Diabetic Retinopathy Detection
Early detection system for life-saving diagnosis through comprehensive feature analysis
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
warnings.filterwarnings('ignore')

class RetinopathyFeatureExtractor:
    """Specialized feature extractor for diabetic retinopathy detection"""

    def __init__(self, use_gpu=True, max_workers=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.feature_heads = {}
        self.setup_retinopathy_heads()

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

        initial_memory = self.get_memory_usage()
        print(f"🧠 Initial memory: {initial_memory:.1f} MB")

        # Preprocess retinal image
        processed_image = self.preprocess_retinal_image(image)

        for head_name, head_func in self.feature_heads.items():
            try:
                current_memory = self.get_memory_usage()
                if current_memory > 6000:  # 6GB threshold
                    print("🔄 High memory usage, performing cleanup...")
                    self._cleanup_memory()

                print(f"🔍 Extracting {head_name} features...")
                features[head_name] = head_func(processed_image)
                print(f"   ✅ {head_name}: {len(features[head_name])} features")

                gc.collect()

            except Exception as e:
                print(f"   ❌ {head_name} failed: {e}")
                features[head_name] = np.array([])

        return features

    def _cleanup_memory(self):
        """Memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_vessel_features(self, image: np.ndarray) -> np.ndarray:
        """Extract retinal blood vessel features using Frangi filter"""
        features = []

        # Multi-scale vessel enhancement using Frangi filter
        scales = [1, 2, 3, 4]
        for scale in scales:
            vessel_enhanced = frangi(image, scale=scale, black_ridges=False)
            features.extend([
                np.mean(vessel_enhanced),
                np.std(vessel_enhanced),
                np.max(vessel_enhanced),
                np.percentile(vessel_enhanced, 95),  # Highlight strong vessel responses
                entropy(vessel_enhanced.flatten())
            ])

        # Vessel segmentation using morphological operations
        binary_vessels = self._segment_vessels(image)
        if np.sum(binary_vessels) > 0:
            vessel_props = measure.regionprops(measure.label(binary_vessels))
            if vessel_props:
                total_vessel_area = sum([prop.area for prop in vessel_props])
                features.extend([
                    total_vessel_area / image.size,  # Vessel density
                    len(vessel_props),  # Number of vessel segments
                    np.mean([prop.eccentricity for prop in vessel_props]),
                    np.mean([prop.solidity for prop in vessel_props])
                ])

        return np.array(features)

    def _segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """Segment retinal blood vessels"""
        # Green channel extraction (best for vessel contrast)
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image

        # Morphological reconstruction for vessel enhancement
        selem = disk(1)
        top_hat = morphology.white_tophat(green_channel, selem)

        # Thresholding
        threshold = np.percentile(top_hat, 95)
        vessels = top_hat > threshold

        return vessels.astype(np.uint8)

    def _extract_exudate_features(self, image: np.ndarray) -> np.ndarray:
        """Extract exudate (bright lesions) features"""
        features = []

        # Exudate detection using intensity and texture
        bright_regions = image > np.percentile(image, 85)

        if np.sum(bright_regions) > 0:
            # Morphological features of bright regions
            labeled_regions = measure.label(bright_regions)
            regions = measure.regionprops(labeled_regions, intensity_image=image)

            # Filter potential exudates by size and intensity
            exudate_candidates = [r for r in regions if 50 < r.area < 10000 and r.mean_intensity > 0.7]

            if exudate_candidates:
                features.extend([
                    len(exudate_candidates),  # Number of exudates
                    np.mean([r.area for r in exudate_candidates]),
                    np.mean([r.eccentricity for r in exudate_candidates]),
                    np.mean([r.mean_intensity for r in exudate_candidates]),
                    np.sum([r.area for r in exudate_candidates]) / image.size  # Exudate density
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

            # Filter potential hemorrhages
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
        # Use difference of Gaussians to enhance small dots
        gaussian1 = gaussian(image, sigma=1)
        gaussian2 = gaussian(image, sigma=2)
        dog = gaussian1 - gaussian2

        # Enhanced dot detection
        dots = dog > np.percentile(dog, 90)

        if np.sum(dots) > 0:
            labeled_dots = measure.label(dots)
            dot_regions = measure.regionprops(labeled_dots)

            # Filter microaneurysm candidates (small, round)
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

        # Optic disc typically appears as bright circular region
        bright_mask = image > np.percentile(image, 90)

        if np.sum(bright_mask) > 0:
            labeled_disc = measure.label(bright_mask)
            regions = measure.regionprops(labeled_disc)

            if regions:
                # Assume largest bright region is optic disc
                optic_disc = max(regions, key=lambda x: x.area)

                features.extend([
                    optic_disc.area,
                    optic_disc.eccentricity,
                    optic_disc.solidity,
                    optic_disc.mean_intensity if hasattr(optic_disc, 'mean_intensity') else 0,
                    optic_disc.area / image.size  # Relative size
                ])
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_macula_features(self, image: np.ndarray) -> np.ndarray:
        """Extract macula region features (dark central area)"""
        features = []

        # Macula typically appears as darker region near center
        center_y, center_x = np.array(image.shape) // 2
        search_radius = min(image.shape) // 6

        # Create circular ROI around center
        y, x = np.ogrid[-center_y:image.shape[0]-center_y, -center_x:image.shape[1]-center_x]
        mask = x*x + y*y <= search_radius*search_radius

        macula_region = image[mask]

        if len(macula_region) > 0:
            features.extend([
                np.mean(macula_region),
                np.std(macula_region),
                np.min(macula_region),
                entropy(macula_region.flatten()),
                len(macula_region) / image.size  # Relative area
            ])
        else:
            features.extend([0] * 5)

        return np.array(features)

    def _extract_retinal_texture(self, image: np.ndarray) -> np.ndarray:
        """Extract retinal texture features using GLCM and LBP"""
        features = []

        # GLCM texture features
        image_uint8 = (image * 255).astype(np.uint8)
        glcm = graycomatrix(image_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], symmetric=True, normed=True)

        texture_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in texture_properties:
            prop_values = graycoprops(glcm, prop)
            features.extend([np.mean(prop_values), np.std(prop_values)])

        # LBP texture features
        lbp = local_binary_pattern(image, 24, 3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist / lbp_hist.sum()
        features.extend(lbp_hist[:10])  # First 10 bins

        return np.array(features)

    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features (for color retinal images)"""
        features = []

        if len(image.shape) == 3:  # Color image
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Statistical features for each channel
            for channel in range(3):
                features.extend([
                    np.mean(image[:, :, channel]),
                    np.std(image[:, :, channel]),
                    skew(image[:, :, channel].flatten()),
                    np.mean(hsv[:, :, channel]),
                    np.mean(lab[:, :, channel])
                ])
        else:  # Grayscale image
            features.extend([
                np.mean(image),
                np.std(image),
                skew(image.flatten()),
                kurtosis(image.flatten()),
                entropy(image.flatten())
            ] * 3)  # Repeat for compatibility

        return np.array(features)

    def _extract_deep_retinal_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep learning features using retinal-specific models"""
        if not self.use_gpu:
            return np.array([])

        try:
            # Use models pre-trained on medical images if available, otherwise ImageNet
            model = models.resnet50(pretrained=True)
            feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
            feature_extractor.eval()

            if self.use_gpu:
                feature_extractor = feature_extractor.cuda()

            # Prepare image (handle both grayscale and color)
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

            return features[:2048]  # Reduce dimensionality

        except Exception as e:
            print(f"Deep feature extraction failed: {e}")
            return np.array([])

    def _extract_retinal_statistics(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive statistical features"""
        flattened = image.flatten()

        features = [
            # Basic statistics
            np.mean(image), np.median(image), np.std(image),
            # Intensity distribution
            np.percentile(image, 10), np.percentile(image, 25),
            np.percentile(image, 75), np.percentile(image, 90),
            # Shape statistics
            skew(flattened), kurtosis(flattened),
            # Information content
            entropy(flattened),
            # Robust measures
            np.median(np.abs(image - np.median(image))),  # MAD
            np.percentile(image, 95) - np.percentile(image, 5),  # IQR-like
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

class DiabeticRetinopathyPipeline:
    """Complete pipeline for diabetic retinopathy feature extraction"""

    def __init__(self, use_gpu=True):
        self.extractor = RetinopathyFeatureExtractor(use_gpu=use_gpu)
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
            print(f"\n📊 Processing retinal image {i+1}/{total_images}...")

            # Extract comprehensive retinopathy features
            feature_dict = self.extractor.extract_retinopathy_features(image)

            # Fuse features
            fused_features, feature_metadata = self.extractor.fuse_retinopathy_features(feature_dict)

            all_features.append(fused_features)
            if i == 0:
                self.feature_names = feature_metadata

            # Progress reporting
            if (i + 1) % 5 == 0 or (i + 1) == total_images:
                print(f"🎯 Progress: {i+1}/{total_images} retinal images processed")

        # Create DataFrame
        df = pd.DataFrame(all_features, columns=self.feature_names)
        df['retinopathy_grade'] = labels  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative

        # Save for model training
        df.to_csv(output_csv, index=False)

        print(f"\n✅ Retinopathy feature extraction complete!")
        print(f"📊 Results: {len(all_features)} samples, {len(self.feature_names)} features")
        print(f"💾 Saved to: {output_csv}")

        return output_csv

def load_retinal_images(folder_path: str, max_images_per_class=None) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load retinal images from structured folder
    Expected structure: folder/grade_0/, folder/grade_1/, etc.
    """
    images = []
    labels = []
    filenames = []

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    # Look for grade subfolders
    grade_folders = [f for f in folder_path.iterdir() if f.is_dir() and 'grade' in f.name.lower()]

    if not grade_folders:
        # Alternative: look for any subfolders
        grade_folders = [f for f in folder_path.iterdir() if f.is_dir()]

    print(f"📁 Found {len(grade_folders)} retinopathy grade folders:")

    for grade_folder in grade_folders:
        grade_name = grade_folder.name
        print(f"   📂 {grade_name}")

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.png'}
        image_files = [f for f in grade_folder.iterdir()
                      if f.suffix.lower() in valid_extensions and f.is_file()]

        if max_images_per_class:
            image_files = image_files[:max_images_per_class]

        print(f"      📷 Loading {len(image_files)} retinal images")

        loaded_count = 0
        for image_file in image_files:
            try:
                # Load image (keep color information for retinal analysis)
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"      ⚠️  Warning: Could not read {image_file}")
                    continue

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Normalize
                image = image.astype(np.float32) / 255.0

                images.append(image)
                labels.append(grade_name)
                filenames.append(image_file.name)
                loaded_count += 1

            except Exception as e:
                print(f"      ❌ Error loading {image_file}: {e}")

        print(f"      ✅ Loaded {loaded_count} images from {grade_name}")

    print(f"\n📊 Total: {len(images)} retinal images loaded from {len(grade_folders)} grades")
    return images, labels, filenames

def main():
    """Main function for diabetic retinopathy feature extraction"""
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Feature Extractor')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to folder containing retinopathy grade subfolders')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum images per class')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage')

    args = parser.parse_args()

    print("🩺 Diabetic Retinopathy Early Detection System")
    print("=" * 50)
    print("🎯 Mission: Early detection through comprehensive feature analysis")
    print(f"🔧 GPU Usage: {'DISABLED' if args.no_gpu else 'ENABLED (if available)'}")

    try:
        images, labels, filenames = load_retinal_images(args.images, args.max_images)
    except Exception as e:
        print(f"❌ Error loading retinal images: {e}")
        return

    if len(images) == 0:
        print("❌ No retinal images found to process")
        return

    # Set output filename
    if args.output is None:
        folder_name = Path(args.images).name
        output_csv = f"retinopathy_{folder_name}_features.csv"
    else:
        output_csv = args.output

    # Initialize pipeline
    pipeline = DiabeticRetinopathyPipeline(use_gpu=not args.no_gpu)

    # Process images
    print(f"\n🚀 Starting feature extraction for {len(images)} retinal images...")
    start_time = time.time()

    try:
        feature_csv = pipeline.process_retinal_images(images, labels, output_csv)

        # Add filename information
        df = pd.read_csv(feature_csv)
        df['filename'] = filenames
        df.to_csv(feature_csv, index=False)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n✅ RETINOPATHY FEATURE EXTRACTION COMPLETE!")
        print(f"💾 Saved to: {feature_csv}")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📊 Retinal images processed: {len(images)}")
        print(f"🔧 Features per image: {len(pipeline.feature_names)}")

        # Show feature breakdown
        print(f"\n📈 FEATURE BREAKDOWN BY TYPE:")
        feature_types = {}
        for name in pipeline.feature_names:
            head = name.split('_')[1]  # retina_<head>_<index>
            feature_types[head] = feature_types.get(head, 0) + 1

        for head, count in sorted(feature_types.items()):
            print(f"   {head:20}: {count:4} features")

        # Show grade distribution
        print(f"\n📊 RETINOPATHY GRADE DISTRIBUTION:")
        grade_counts = pd.Series(labels).value_counts()
        for grade, count in grade_counts.items():
            percentage = (count / len(images)) * 100
            print(f"   {grade:20}: {count:4} images ({percentage:.1f}%)")

        print(f"\n💡 Next steps:")
        print(f"   1. Train machine learning model on extracted features")
        print(f"   2. Validate on independent test set")
        print(f"   3. Deploy for early detection screening")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print("💡 Try using --max_images to reduce dataset size")

if __name__ == "__main__":
    main()
