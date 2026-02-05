"""
Visualize Bounding Boxes on All Images
=======================================
Draws bounding boxes on all augmented images for verification.
"""

import cv2
import os
from pathlib import Path
import argparse


# Class colors (BGR format for OpenCV)
COLORS = {
    0: (0, 255, 0),    # Green - Part 1
    1: (255, 0, 0),    # Blue - Part 2
    2: (0, 0, 255),    # Red - Part 3
    3: (0, 255, 255),  # Yellow - Part 4
}

CLASS_NAMES = ['part1', 'part2', 'part3', 'part4']


def draw_bboxes_on_image(image_path: str, label_path: str, output_path: str):
    """Draw bounding boxes on image and save."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return False
    
    h, w = image.shape[:2]
    
    # Load annotations
    if not os.path.exists(label_path):
        print(f"No label file: {label_path}")
        # Still save image without boxes
        cv2.imwrite(output_path, image)
        return True
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))  # Handle both int and float format
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_w = float(parts[3])
                bbox_h = float(parts[4])
                
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - bbox_w / 2) * w)
                y1 = int((y_center - bbox_h / 2) * h)
                x2 = int((x_center + bbox_w / 2) * w)
                y2 = int((y_center + bbox_h / 2) * h)
                
                # Get color for class
                color = COLORS.get(class_id, (255, 255, 255))
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Background for label
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 5, y1), color, -1)
                cv2.putText(image, label, (x1 + 2, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save image with bboxes
    cv2.imwrite(output_path, image)
    return True


def process_dataset(dataset_dir: str, output_dir: str = None):
    """Process all images in the dataset."""
    dataset_path = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = dataset_path / 'images_with_bboxes'
    else:
        output_dir = Path(output_dir)
    
    # Process train and val splits
    for split in ['train', 'val']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        output_split_dir = output_dir / split
        
        if not images_dir.exists():
            print(f"Skip {split}: {images_dir} not found")
            continue
        
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png'))
        
        print(f"\nProcessing {split}: {len(image_files)} images")
        
        for idx, img_path in enumerate(image_files):
            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            output_path = output_split_dir / img_path.name
            
            draw_bboxes_on_image(str(img_path), str(label_path), str(output_path))
            
            if (idx + 1) % 50 == 0 or idx == len(image_files) - 1:
                print(f"  Processed {idx + 1}/{len(image_files)}")
    
    print(f"\nDone! Images with bounding boxes saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on images')
    parser.add_argument('--dataset', type=str, default='augmented_dataset', 
                       help='Path to augmented dataset')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output directory (default: dataset/images_with_bboxes)')
    
    args = parser.parse_args()
    process_dataset(args.dataset, args.output)


if __name__ == '__main__':
    main()
