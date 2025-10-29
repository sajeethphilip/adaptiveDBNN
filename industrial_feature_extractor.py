#!/usr/bin/env python3
"""
Multi-Head Ensemble Feature Extractor for Industrial Defect Detection
Parallel feature extraction with intelligent fusion for DBNN optimization
"""

import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.filters import gabor, gaussian
from skimage import exposure, measure
from scipy.stats import skew, kurtosis, entropy
from scipy import ndimage as ndi
import torch
import torchvision.models as models
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
import time
import os
import argparse
from pathlib import Path

class MultiHeadFeatureExtractor:
    """Ensemble feature extractor with parallel processing heads"""

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.feature_heads = {}
        self.setup_feature_heads()

    def setup_feature_heads(self):
        """Initialize all feature extraction heads"""
        # Texture Analysis Head
        self.feature_heads['gabor'] = self._extract_gabor_features
        self.feature_heads['lbp'] = self._extract_multi_radius_lbp
        self.feature_heads['glcm'] = self._extract_glcm_features

        # Structural Analysis Head
        self.feature_heads['hog'] = self._extract_multi_scale_hog
        self.feature_heads['morphology'] = self._extract_morphological_features

        # Frequency Domain Head
        self.feature_heads['fourier'] = self._extract_fourier_features
        self.feature_heads['wavelet'] = self._extract_wavelet_features

        # Deep Learning Head
        if self.use_gpu:
            self.feature_heads['deep'] = self._extract_deep_features

        # Statistical Head
        self.feature_heads['statistical'] = self._extract_statistical_features

        # Edge/Corner Detection Head
        self.feature_heads['edge'] = self._extract_edge_features

        print(f"✅ Initialized {len(self.feature_heads)} feature extraction heads")

    def extract_features_parallel(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features in parallel using multiple heads"""
        features = {}

        with ThreadPoolExecutor(max_workers=min(8, len(self.feature_heads))) as executor:
            # Submit all feature extraction tasks
            future_to_head = {
                executor.submit(head_func, image): head_name
                for head_name, head_func in self.feature_heads.items()
            }

            # Collect results as they complete
            for future in future_to_head:
                head_name = future_to_head[future]
                try:
                    features[head_name] = future.result(timeout=30)  # 30s timeout per head
                    print(f"   ✅ {head_name}: {len(features[head_name])} features")
                except Exception as e:
                    print(f"   ❌ {head_name} failed: {e}")
                    features[head_name] = np.array([])

        return features

    def _extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Gabor filter bank for texture analysis"""
        features = []
        frequencies = [0.1, 0.3, 0.5, 0.7]
        orientations = 8

        for freq in frequencies:
            for theta in range(orientations):
                real, imag = gabor(image, frequency=freq, theta=theta*np.pi/orientations)
                # Multiple statistics per filter response
                features.extend([
                    np.mean(real), np.std(real), np.var(real),
                    np.mean(imag), np.std(imag), np.var(imag),
                    np.max(real), np.min(real), np.median(real)
                ])

        return np.array(features)

    def _extract_multi_radius_lbp(self, image: np.ndarray) -> np.ndarray:
        """Multi-radius Local Binary Patterns"""
        features = []
        radii = [1, 2, 3, 4]
        points = [8, 16, 24]

        for radius in radii:
            for n_points in points:
                lbp = local_binary_pattern(image, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
                # Normalize and add multiple histogram statistics
                hist_norm = hist / (hist.sum() + 1e-8)
                features.extend(hist_norm)
                features.extend([np.mean(lbp), np.std(lbp), entropy(hist_norm)])

        return np.array(features)

    def _extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """Gray-Level Co-occurrence Matrix features"""
        # Quantize image to 8-bit for GLCM
        image_uint8 = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)

        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(image_uint8, distances=distances, angles=angles,
                           symmetric=True, normed=True)

        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                     'correlation', 'ASM']
        features = []

        for prop in properties:
            prop_values = graycoprops(glcm, prop).ravel()
            features.extend(prop_values)
            # Add statistics for each property
            features.extend([np.mean(prop_values), np.std(prop_values), np.var(prop_values)])

        return np.array(features)

    def _extract_multi_scale_hog(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale Histogram of Oriented Gradients"""
        features = []
        scales = [(64, 64), (128, 128), (256, 256)]
        cell_sizes = [(8, 8), (16, 16)]

        for scale in scales:
            resized = cv2.resize(image, scale)
            for cell_size in cell_sizes:
                hog_features = hog(resized, orientations=9, pixels_per_cell=cell_size,
                                 cells_per_block=(2, 2), block_norm='L2-Hys',
                                 feature_vector=True)
                features.extend(hog_features)

        return np.array(features)

    def _extract_morphological_features(self, image: np.ndarray) -> np.ndarray:
        """Morphological and shape-based features"""
        features = []

        # Threshold for binary operations
        thresh = np.percentile(image, 75)
        binary = (image > thresh).astype(np.uint8)

        # Connected components analysis
        labels = measure.label(binary)
        regions = measure.regionprops(labels, intensity_image=image)

        if regions:
            largest_region = max(regions, key=lambda x: x.area)
            # Shape descriptors
            features.extend([
                largest_region.area,
                largest_region.perimeter,
                largest_region.eccentricity,
                largest_region.solidity,
                largest_region.extent,
                largest_region.major_axis_length,
                largest_region.minor_axis_length
            ])
        else:
            features.extend([0] * 7)

        # Morphological gradients
        gradient = ndi.morphological_gradient(image, size=(3, 3))
        features.extend([np.mean(gradient), np.std(gradient), np.max(gradient)])

        return np.array(features)

    def _extract_fourier_features(self, image: np.ndarray) -> np.ndarray:
        """Frequency domain features"""
        # 2D Fourier Transform
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Radial and angular profiles
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.indices(magnitude_spectrum.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Radial distribution
        radial_profile = ndi.mean(magnitude_spectrum, labels=np.floor(r).astype(int),
                                index=np.arange(0, min(center)))

        features = [
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.max(magnitude_spectrum),
            entropy(magnitude_spectrum.flatten()),
            skew(radial_profile[~np.isnan(radial_profile)]),
            kurtosis(radial_profile[~np.isnan(radial_profile)])
        ]

        return np.array(features)

    def _extract_wavelet_features(self, image: np.ndarray) -> np.ndarray:
        """Wavelet transform features (simplified)"""
        # Simple wavelet-like decomposition using Gaussian pyramid
        features = []
        current = image.copy()

        for level in range(3):
            # Low-pass filter
            low_pass = gaussian(current, sigma=1.0)
            # High-pass (detail)
            high_pass = current - low_pass

            # Statistics of detail coefficients
            features.extend([
                np.mean(high_pass), np.std(high_pass), np.var(high_pass),
                np.max(high_pass), np.min(high_pass), entropy(np.abs(high_pass).flatten())
            ])

            # Downsample for next level
            current = low_pass[::2, ::2]
            if current.size < 4:  # Minimum size
                break

        return np.array(features)

    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Deep feature extraction using pre-trained models"""
        if not self.use_gpu:
            return np.array([])

        try:
            # Use multiple pre-trained models for diverse features
            models_to_use = {
                'resnet18': models.resnet18(pretrained=True),
                'efficientnet_b0': models.efficientnet_b0(pretrained=True)
            }

            features = []
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            for model_name, model in models_to_use.items():
                # Remove classification layer
                feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
                feature_extractor.eval()

                if self.use_gpu:
                    feature_extractor = feature_extractor.cuda()

                # Prepare image
                img_tensor = transform(image).unsqueeze(0)
                if self.use_gpu:
                    img_tensor = img_tensor.cuda()

                with torch.no_grad():
                    model_features = feature_extractor(img_tensor)
                    features.extend(model_features.cpu().numpy().flatten())

            return np.array(features)

        except Exception as e:
            print(f"Deep feature extraction failed: {e}")
            return np.array([])

    def _extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """Comprehensive statistical features"""
        flattened = image.flatten()

        features = [
            # Central tendencies
            np.mean(image), np.median(image),
            # Dispersion
            np.std(image), np.var(image), np.ptp(image),  # Peak-to-peak (range)
            # Percentiles
            np.percentile(image, 10), np.percentile(image, 25),
            np.percentile(image, 75), np.percentile(image, 90),
            # Shape
            skew(flattened), kurtosis(flattened),
            # Robust statistics
            measure.shannon_entropy(image),
            # Higher moments
            np.mean(np.abs(image - np.mean(image)) ** 3) ** (1/3),  # Third moment
            np.mean(np.abs(image - np.mean(image)) ** 4) ** (1/4)   # Fourth moment
        ]

        return np.array(features)

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Edge and corner detection features"""
        features = []

        # Canny edge detection at multiple thresholds
        for threshold in [0.1, 0.3, 0.5]:
            edges = feature.canny(image, sigma=1.0, low_threshold=threshold,
                                high_threshold=threshold * 2)
            features.extend([
                np.sum(edges),  # Total edge pixels
                np.mean(edges), # Edge density
                measure.shannon_entropy(edges)
            ])

        # Harris corner detection
        from skimage.feature import corner_harris, corner_peaks
        corner_response = corner_harris(image)
        corners = corner_peaks(corner_response, min_distance=5, threshold_rel=0.02)
        features.extend([
            len(corners),  # Number of corners
            np.mean(corner_response) if len(corners) > 0 else 0,
            np.std(corner_response) if len(corners) > 0 else 0
        ])

        return np.array(features)

    def fuse_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Intelligently fuse features from all heads"""
        all_features = []
        feature_metadata = []

        for head_name, features in feature_dict.items():
            if len(features) > 0:
                all_features.extend(features)
                feature_metadata.extend([f"{head_name}_{i}" for i in range(len(features))])

        print(f"🎯 Feature fusion: {len(all_features)} total features from {len(feature_dict)} heads")
        return np.array(all_features), feature_metadata

class IndustrialDefectFeaturePipeline:
    """Complete pipeline for industrial defect feature extraction and selection"""

    def __init__(self):
        self.extractor = MultiHeadFeatureExtractor()
        self.feature_names = []

    def process_images(self, images: List[np.ndarray], labels: List,
                      output_csv: str = "industrial_defect_features.csv") -> str:
        """Complete pipeline: extract features → fuse → save for DBNN"""
        print("🚀 Starting Industrial Defect Feature Extraction Pipeline")
        print("=" * 60)

        all_features = []
        all_metadata = []

        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 10 == 0:
                print(f"📊 Processing image {i+1}/{len(images)}...")

            # Extract features from all heads in parallel
            feature_dict = self.extractor.extract_features_parallel(image)

            # Fuse features
            fused_features, feature_metadata = self.extractor.fuse_features(feature_dict)

            all_features.append(fused_features)
            if i == 0:  # Store metadata only once
                self.feature_names = feature_metadata

        # Create DataFrame
        df = pd.DataFrame(all_features, columns=self.feature_names)
        df['defect_label'] = labels

        # Save for DBNN processing
        df.to_csv(output_csv, index=False)

        print(f"✅ Feature extraction complete: {len(all_features)} samples, {len(self.feature_names)} features")
        print(f"💾 Saved to: {output_csv}")

        return output_csv

def load_images_from_folder(folder_path: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load images from folder structure where subfolders represent classes

    Args:
        folder_path: Path to the root folder containing class subfolders

    Returns:
        Tuple of (images, labels, filenames)
    """
    images = []
    labels = []
    filenames = []

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    # Get all subdirectories (classes)
    class_folders = [f for f in folder_path.iterdir() if f.is_dir()]

    if not class_folders:
        raise ValueError(f"No class subfolders found in {folder_path}")

    print(f"📁 Found {len(class_folders)} classes:")

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"   📂 {class_name}")

        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # Get all image files in the class folder
        image_files = [f for f in class_folder.iterdir()
                      if f.suffix.lower() in valid_extensions and f.is_file()]

        print(f"      📷 Found {len(image_files)} images")

        for image_file in image_files:
            try:
                # Read image as grayscale
                image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"      ⚠️  Warning: Could not read {image_file}")
                    continue

                # Normalize image to [0, 1] range
                image = image.astype(np.float32) / 255.0

                images.append(image)
                labels.append(class_name)
                filenames.append(image_file.name)

            except Exception as e:
                print(f"      ❌ Error loading {image_file}: {e}")

    print(f"\n📊 Total: {len(images)} images loaded from {len(class_folders)} classes")
    return images, labels, filenames

def display_help_summary():
    """Display comprehensive help and usage information"""
    print("""
🎯 INDUSTRIAL DEFECT FEATURE EXTRACTOR
==================================================

📖 DESCRIPTION:
   Multi-Head Ensemble Feature Extractor for industrial defect detection.
   Extracts comprehensive features from images using parallel processing
   and saves them for DBNN (Deep Belief Neural Network) optimization.

🚀 USAGE:
   python industrial_feature_extractor.py --images <folder_path> [--output <csv_file>]

🔧 ARGUMENTS:
   --images FOLDER_PATH   Path to folder containing class subfolders with images
   --output CSV_FILE      Output CSV filename (optional, default: <folder_name>_features.csv)

📁 FOLDER STRUCTURE:
   Your image folder should be organized as:

   dataset_name/
   ├── normal/           # Class 1: Normal samples
   │   ├── img1.jpg
   │   ├── img2.png
   │   └── ...
   ├── defect_type1/     # Class 2: Defect type 1
   │   ├── defect1.jpg
   │   └── ...
   └── defect_type2/     # Class 3: Defect type 2
       └── ...

✨ FEATURE HEADS:
   The extractor uses 8 specialized feature extraction heads:

   1. 🔍 GABOR FILTERS     - Multi-scale, multi-orientation texture analysis
   2. 🔄 MULTI-RADIUS LBP  - Local Binary Patterns with various radii
   3. 🎯 GLCM FEATURES     - Gray-Level Co-occurrence Matrix textures
   4. 📊 MULTI-SCALE HOG   - Histogram of Oriented Gradients at different scales
   5. 🔷 MORPHOLOGICAL     - Shape and structural features
   6. 📡 FOURIER DOMAIN    - Frequency domain analysis
   7. 🌊 WAVELET FEATURES  - Multi-resolution analysis
   8. 📈 STATISTICAL       - Comprehensive statistical measures
   9. 🔗 DEEP FEATURES     - Pre-trained neural network features (GPU only)
   10. 📐 EDGE FEATURES    - Edge and corner detection

💡 EXPECTED OUTPUT:
   - CSV file with extracted features (1000+ features per image)
   - 'defect_label' column with class names from subfolder names
   - 'filename' column with original image filenames
   - Ready for DBNN feature selection and training

⏰ PERFORMANCE:
   - Parallel processing across all feature heads
   - 30-second timeout per feature head
   - Progress reporting every 10 images
   - Detailed feature breakdown and statistics

🔍 SUPPORTED FORMATS:
   JPEG, PNG, BMP, TIFF, and other common image formats

🎯 EXAMPLE COMMANDS:
   python industrial_feature_extractor.py --images ./industrial_dataset
   python industrial_feature_extractor.py --images ./defect_images --output my_features.csv
   python industrial_feature_extractor.py --images /path/to/steel_defects

📊 OUTPUT STATISTICS:
   - Class distribution analysis
   - Feature count by type
   - Processing time estimates
   - Quality control metrics

⚠️  REQUIREMENTS:
   - OpenCV, scikit-image, PyTorch, pandas, numpy
   - CUDA GPU recommended for deep features (optional)
   - Sufficient RAM for large datasets
""")

def main():
    """Main function with comprehensive help system"""
    parser = argparse.ArgumentParser(
        description='Industrial Defect Feature Extractor - Multi-Head Ensemble Feature Extraction',
        add_help=False
    )
    parser.add_argument('--images', type=str, required=False,
                       help='Path to folder containing class subfolders with images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename (default: <folder_name>_features.csv)')
    parser.add_argument('--help', '-h', action='store_true',
                       help='Show this comprehensive help message and exit')

    args = parser.parse_args()

    # Show help if no arguments or help flag
    if args.help or not args.images:
        display_help_summary()

        if not args.images:
            print("\n❌ ERROR: Please provide the --images argument")
            print("💡 Usage: python industrial_feature_extractor.py --images <folder_path>")
            print("🔍 For help: python industrial_feature_extractor.py --help")
        return

    print("🧪 Industrial Defect Feature Pipeline")
    print("=" * 50)

    # Load images from folder structure
    try:
        images, labels, filenames = load_images_from_folder(args.images)
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("   - Check if the folder path is correct")
        print("   - Ensure the folder contains subfolders for each class")
        print("   - Verify images are in supported formats (jpg, png, bmp, tiff)")
        print("   - Check file permissions")
        return

    if len(images) == 0:
        print("❌ No images found to process")
        print("\n💡 POSSIBLE SOLUTIONS:")
        print("   - Check if images are in supported formats")
        print("   - Verify subfolder structure exists")
        print("   - Ensure images are not corrupted")
        return

    # Set output filename
    if args.output is None:
        folder_name = Path(args.images).name
        output_csv = f"{folder_name}_features.csv"
    else:
        output_csv = args.output

    # Initialize pipeline
    pipeline = IndustrialDefectFeaturePipeline()

    # Process images and generate features
    print(f"\n🚀 Starting feature extraction for {len(images)} images...")
    start_time = time.time()

    feature_csv = pipeline.process_images(images, labels, output_csv)

    # Add filename information to the CSV
    df = pd.read_csv(feature_csv)
    df['filename'] = filenames
    df.to_csv(feature_csv, index=False)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n🎯 Feature extraction complete!")
    print(f"💾 Saved to: {feature_csv}")

    # Show comprehensive statistics
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   📈 Total images processed: {len(images)}")
    print(f"   🔧 Total features extracted: {len(pipeline.feature_names)}")
    print(f"   ⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"   🚀 Average time per image: {processing_time/len(images):.2f} seconds")

    label_counts = pd.Series(labels).value_counts()
    print(f"   📊 Class distribution:")
    for class_name, count in label_counts.items():
        percentage = (count / len(images)) * 100
        print(f"      📂 {class_name}: {count} images ({percentage:.1f}%)")

    # Show feature breakdown
    print(f"\n🔧 FEATURE BREAKDOWN BY TYPE:")
    feature_types = {}
    for name in pipeline.feature_names:
        head = name.split('_')[0]
        feature_types[head] = feature_types.get(head, 0) + 1

    total_features = sum(feature_types.values())
    for head, count in sorted(feature_types.items()):
        percentage = (count / total_features) * 100
        print(f"   {head:15}: {count:4} features ({percentage:.1f}%)")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Run DBNN feature selection:")
    print(f"      python dbnn_feature_selector.py -i {feature_csv} -t defect_label -fc 150")
    print(f"   2. Train your defect classification model")
    print(f"   3. Evaluate feature importance and optimize")

if __name__ == "__main__":
    main()
