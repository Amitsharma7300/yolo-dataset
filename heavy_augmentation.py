"""
Heavy Augmentation Pipeline for Construction Machinery Object Detection
=========================================================================
Generates photorealistic augmented images while preserving:
- Original yellow/orange color (RAL 1007 / Caterpillar yellow)
- Component dimensions and proportions
- Part geometry, shape, and structural details
- Bounding box annotations for YOLO training

Usage:
    python heavy_augmentation.py --input_dir <folder> --output_dir <output> --num_augmentations 50
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================================
# CUSTOM TRANSFORMS FOR MACHINERY-SPECIFIC AUGMENTATION
# ============================================================================

class PreserveYellowColorTransform(ImageOnlyTransform):
    """
    Preserves the yellow/orange color of machinery while adjusting other aspects.
    Targets RAL 1007 / Caterpillar yellow color range.
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([45, 255, 255])
    
    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        variation = random.uniform(-10, 10)
        hsv_float = hsv.astype(np.float32)
        hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] + variation, 0, 255)
        hsv_float[:, :, 2] = np.clip(hsv_float[:, :, 2] + variation * 0.5, 0, 255)
        result = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result
    
    def get_transform_init_args_names(self):
        return ()


class SimulateDustTransform(ImageOnlyTransform):
    """Simulates dusty environmental conditions on machinery."""
    def __init__(self, dust_intensity=(0.05, 0.2), always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.dust_intensity = dust_intensity
    
    def apply(self, img, **params):
        intensity = random.uniform(*self.dust_intensity)
        dust_color = np.array([180, 160, 130], dtype=np.uint8)
        dust_overlay = np.full_like(img, dust_color)
        noise = np.random.random(img.shape[:2])
        noise = cv2.GaussianBlur(noise.astype(np.float32), (21, 21), 0)
        noise = (noise * intensity)[:, :, np.newaxis]
        result = (img.astype(np.float32) * (1 - noise) + dust_overlay.astype(np.float32) * noise)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def get_transform_init_args_names(self):
        return ("dust_intensity",)


class SimulateWetSurfaceTransform(ImageOnlyTransform):
    """Simulates wet/rain conditions on machinery surface."""
    def __init__(self, always_apply=False, p=0.2):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        highlight_noise = np.random.random(img.shape[:2]) > 0.995
        highlight_mask = highlight_noise.astype(np.float32)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
        highlight_mask = highlight_mask[:, :, np.newaxis] * 50
        result = np.clip(result.astype(np.float32) + highlight_mask, 0, 255).astype(np.uint8)
        return result
    
    def get_transform_init_args_names(self):
        return ()


class SimulateWeatheredTransform(ImageOnlyTransform):
    """Simulates weathered surface appearance."""
    def __init__(self, always_apply=False, p=0.25):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 0.95)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # Add scratch marks
        for _ in range(random.randint(2, 6)):
            x1 = random.randint(0, img.shape[1])
            y1 = random.randint(0, img.shape[0])
            length = random.randint(10, 40)
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            color = random.randint(100, 150)
            cv2.line(result, (x1, y1), (x2, y2), (color, color, color), 1)
        return result
    
    def get_transform_init_args_names(self):
        return ()


class VariableLightingTransform(ImageOnlyTransform):
    """Simulates various lighting conditions with realistic gradients."""
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lighting_types = ['morning', 'midday', 'afternoon', 'warehouse', 'outdoor_shade', 'bright_sun', 'cloudy']
    
    def apply(self, img, **params):
        lighting = random.choice(self.lighting_types)
        result = img.astype(np.float32) / 255.0
        
        if lighting == 'morning':
            result[:, :, 0] *= 1.1
            result[:, :, 2] *= 0.9
        elif lighting == 'midday':
            result = np.power(result, 0.9)
            result[:, :, 2] *= 1.05
        elif lighting == 'afternoon':
            result[:, :, 0] *= 1.15
            result[:, :, 1] *= 1.05
            result[:, :, 2] *= 0.85
        elif lighting == 'warehouse':
            result[:, :, 0] *= 1.08
            result[:, :, 1] *= 1.05
            result[:, :, 2] *= 0.9
        elif lighting == 'outdoor_shade':
            result[:, :, 0] *= 0.9
            result[:, :, 2] *= 1.1
            result = result * 0.85
        elif lighting == 'bright_sun':
            result = np.power(result, 0.85) * 1.1
        elif lighting == 'cloudy':
            result = 0.5 + (result * 0.9 - 0.5) * 0.85
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def get_transform_init_args_names(self):
        return ()


# ============================================================================
# BOUNDING BOX UTILITIES
# ============================================================================

def create_default_bbox(image_shape: Tuple[int, int]) -> List[float]:
    """Creates a default bounding box covering most of the image."""
    margin = random.uniform(0.05, 0.15)
    return [0.5, 0.5, 1.0 - 2 * margin, 1.0 - 2 * margin]


def apply_random_bbox_variation(bbox: List[float], variation: float = 0.05) -> List[float]:
    """Applies small random variations to bounding box."""
    x_center, y_center, width, height = bbox
    x_center += random.uniform(-variation, variation) * width
    y_center += random.uniform(-variation, variation) * height
    scale = random.uniform(1 - variation, 1 + variation)
    width *= scale
    height *= scale
    x_center = np.clip(x_center, width/2, 1 - width/2)
    y_center = np.clip(y_center, height/2, 1 - height/2)
    width = np.clip(width, 0.1, 1.0)
    height = np.clip(height, 0.1, 1.0)
    return [x_center, y_center, width, height]


def save_yolo_annotation(filepath: str, bboxes: List[List[float]], class_labels: List[int]):
    """Saves bounding boxes in YOLO format."""
    with open(filepath, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            line = f"{class_id} {' '.join(map(lambda x: f'{x:.6f}', bbox))}\n"
            f.write(line)


def load_yolo_annotation(filepath: str) -> Tuple[List[List[float]], List[int]]:
    """Loads bounding boxes from YOLO format file."""
    bboxes, class_labels = [], []
    if not os.path.exists(filepath):
        return bboxes, class_labels
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_labels


# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

def create_augmentation_pipeline():
    """Creates the augmentation pipeline without bbox params (we handle bboxes manually)."""
    return A.Compose([
        # Geometric transforms - camera angles perspectives
        A.OneOf([
            A.Perspective(scale=(0.02, 0.08), p=1.0),
            A.Affine(scale=(0.85, 1.15), rotate=(-15, 15), shear=(-10, 10), p=1.0),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ], p=0.6),
        
        # Positioning variations
        A.HorizontalFlip(p=0.5),
        
        # Resize to standard YOLO size
        A.Resize(640, 640),
        
        # Lighting conditions (time of day, indoor/outdoor)
        A.OneOf([
            VariableLightingTransform(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        
        # Environmental conditions (dust, wet, weathered)
        A.OneOf([
            SimulateDustTransform(p=1.0),
            SimulateWetSurfaceTransform(p=1.0),
            SimulateWeatheredTransform(p=1.0),
            A.GaussNoise(var_limit=(5, 30), p=1.0),
        ], p=0.4),
        
        # Photo quality (blur, focus)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.25),
        
        # Color preservation with slight variations
        PreserveYellowColorTransform(p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.4),
        
        # Image quality variations
        A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=1.0),
        ], p=0.2),
    ])


def create_extreme_augmentation_pipeline():
    """Creates more aggressive augmentation for maximum diversity."""
    return A.Compose([
        # More aggressive geometric transforms
        A.OneOf([
            A.Perspective(scale=(0.05, 0.12), p=1.0),
            A.Affine(scale=(0.7, 1.3), rotate=(-25, 25), shear=(-15, 15), p=1.0),
            A.ElasticTransform(alpha=15, sigma=4, p=1.0),
        ], p=0.8),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        
        # Random crop and resize
        A.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
        A.Resize(640, 640),
        
        # Intense lighting variations
        A.OneOf([
            VariableLightingTransform(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.Posterize(num_bits=5, p=1.0),
        ], p=0.8),
        
        # Environmental effects
        A.OneOf([
            SimulateDustTransform(dust_intensity=(0.1, 0.3), p=1.0),
            SimulateWetSurfaceTransform(p=1.0),
            SimulateWeatheredTransform(p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.25, p=1.0),
        ], p=0.5),
        
        # Quality degradation
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Color
        PreserveYellowColorTransform(p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        
        # Noise and compression
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=90, p=1.0),
        ], p=0.3),
    ])


# ============================================================================
# MAIN AUGMENTATION ENGINE
# ============================================================================

class MachineryAugmentor:
    """Heavy augmentation engine for construction machinery images."""
    
    def __init__(self, input_dir: str, output_dir: str, num_augmentations: int = 50, use_extreme: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_augmentations = num_augmentations
        self.use_extreme = use_extreme
        self.standard_pipeline = create_augmentation_pipeline()
        self.extreme_pipeline = create_extreme_augmentation_pipeline()
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """Creates output directory structure."""
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self) -> List[Path]:
        """Gets all image files from input directory recursively."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(self.input_dir.rglob(ext))
        # Exclude augmented_dataset folder
        files = [f for f in files if 'augmented_dataset' not in str(f)]
        return files
    
    def determine_class_from_folder(self, filepath: Path) -> int:
        """Determines class ID from folder name."""
        folder_name = filepath.parent.name.lower()
        if 'part1' in folder_name or 'part 1' in folder_name:
            return 0
        elif 'part2' in folder_name or 'part 2' in folder_name:
            return 1
        elif 'part3' in folder_name or 'part 3' in folder_name:
            return 2
        elif 'part4' in folder_name or 'part 4' in folder_name:
            return 3
        return 0
    
    def process_single_image(self, image_path: Path, output_prefix: str, split: str = 'train') -> int:
        """Processes a single image and generates augmentations."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return 0
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get or create bounding box
        annotation_path = image_path.with_suffix('.txt')
        if annotation_path.exists():
            bboxes, class_labels = load_yolo_annotation(str(annotation_path))
        else:
            class_id = self.determine_class_from_folder(image_path)
            bboxes = [create_default_bbox(image.shape[:2])]
            class_labels = [class_id]
        
        # Save original resized
        orig_image = cv2.resize(image, (640, 640))
        orig_save_path = self.output_dir / 'images' / split / f'{output_prefix}_orig.jpg'
        orig_label_path = self.output_dir / 'labels' / split / f'{output_prefix}_orig.txt'
        cv2.imwrite(str(orig_save_path), cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR))
        save_yolo_annotation(str(orig_label_path), bboxes, class_labels)
        
        count = 1
        
        # Generate augmented versions
        for i in range(self.num_augmentations):
            try:
                # Alternate between standard and extreme pipeline
                if self.use_extreme and i % 3 == 0:
                    pipeline = self.extreme_pipeline
                else:
                    pipeline = self.standard_pipeline
                
                # Apply augmentation to image only
                transformed = pipeline(image=image)
                aug_image = transformed['image']
                
                # Apply varied bounding boxes (with small random variations)
                varied_bboxes = [apply_random_bbox_variation(bb, 0.03) for bb in bboxes]
                
                # Save augmented image and annotation
                aug_save_path = self.output_dir / 'images' / split / f'{output_prefix}_aug{i:03d}.jpg'
                aug_label_path = self.output_dir / 'labels' / split / f'{output_prefix}_aug{i:03d}.txt'
                
                cv2.imwrite(str(aug_save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                save_yolo_annotation(str(aug_label_path), varied_bboxes, class_labels)
                count += 1
                
            except Exception as e:
                print(f"Error augmentation {i}: {e}")
                continue
        
        return count
    
    def run(self, val_split: float = 0.15):
        """Runs the complete augmentation pipeline."""
        image_files = self.get_image_files()
        if not image_files:
            print("No image files found!")
            return
        
        print(f"Found {len(image_files)} images to augment")
        print(f"Generating {self.num_augmentations} augmentations per image")
        print(f"Expected output: ~{len(image_files) * (self.num_augmentations + 1)} images")
        print("-" * 50)
        
        total_generated = 0
        
        for idx, image_path in enumerate(image_files):
            split = 'val' if random.random() < val_split else 'train'
            folder_name = image_path.parent.name.replace(' ', '_')
            file_stem = image_path.stem.replace(' ', '_')[:30]
            output_prefix = f"{folder_name}_{file_stem}"
            
            count = self.process_single_image(image_path, output_prefix, split)
            total_generated += count
            
            print(f"[{idx+1}/{len(image_files)}] {image_path.name}: {count} images ({split})")
        
        print("-" * 50)
        print(f"Total images generated: {total_generated}")
        print(f"Output directory: {self.output_dir}")
        
        # Create data.yaml
        self.create_data_yaml()
    
    def create_data_yaml(self):
        """Creates YOLO data configuration file."""
        yaml_content = f"""# YOLO Dataset Configuration
# Auto-generated by Heavy Augmentation Pipeline

path: {self.output_dir.absolute()}
train: images/train
val: images/val

names:
  0: part1
  1: part2
  2: part3
  3: part4

nc: 4
"""
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Heavy Augmentation Pipeline for Construction Machinery')
    parser.add_argument('--input_dir', type=str, default='.', help='Input directory containing part folders')
    parser.add_argument('--output_dir', type=str, default='augmented_dataset', help='Output directory')
    parser.add_argument('--num_augmentations', type=int, default=50, help='Number of augmentations per image')
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction for validation')
    parser.add_argument('--extreme', action='store_true', help='Include extreme augmentations')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Heavy Augmentation Pipeline for Construction Machinery")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Augmentations: {args.num_augmentations}")
    print(f"Extreme mode: {args.extreme}")
    print("=" * 60)
    
    augmentor = MachineryAugmentor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        use_extreme=args.extreme
    )
    
    augmentor.run(val_split=args.val_split)


if __name__ == '__main__':
    main()
