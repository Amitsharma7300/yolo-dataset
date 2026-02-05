"""
Bounding Box Annotation Tool for Machinery Parts
=================================================
Simple GUI tool to annotate machinery components with bounding boxes
before running augmentation.

Usage:
    python annotate_bboxes.py --input_dir <folder>

Controls:
    - Left click and drag: Draw bounding box
    - 'n' or Enter: Next image
    - 'p': Previous image
    - 'r': Reset current image annotations
    - 's': Save and continue
    - 'q': Quit (saves automatically)
    - '0-9': Set class ID for next box
"""

import cv2
import os
import sys
from pathlib import Path
import argparse
from typing import List, Tuple


class BoundingBoxAnnotator:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.image_files = self.get_image_files()
        self.current_idx = 0
        self.current_class = 0
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_box = None
        
        # Annotations for current image
        self.boxes = []  # [(x1, y1, x2, y2, class_id), ...]
        
        # Colors for different classes
        self.colors = [
            (0, 255, 0),    # Green - Part 1
            (255, 0, 0),    # Blue - Part 2
            (0, 0, 255),    # Red - Part 3
            (255, 255, 0),  # Cyan - Part 4
        ]
        
        self.class_names = ['part1', 'part2', 'part3', 'part4']
        
        if not self.image_files:
            print("No image files found!")
            sys.exit(1)
        
        print(f"Found {len(self.image_files)} images to annotate")
        self.print_instructions()
    
    def print_instructions(self):
        print("\n" + "=" * 50)
        print("ANNOTATION CONTROLS")
        print("=" * 50)
        print("Mouse: Click and drag to draw bounding box")
        print("'n' or Enter: Next image")
        print("'p': Previous image")
        print("'r': Reset current annotations")
        print("'s': Save current annotations")
        print("'q': Quit (auto-saves)")
        print("'0-3': Set class (0=part1, 1=part2, 2=part3, 3=part4)")
        print("=" * 50 + "\n")
    
    def get_image_files(self) -> List[Path]:
        """Gets all image files from input directory."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(self.input_dir.rglob(ext))
        return sorted(files)
    
    def load_annotations(self, image_path: Path) -> List[Tuple]:
        """Load existing annotations for an image."""
        annotation_path = image_path.with_suffix('.txt')
        boxes = []
        
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            boxes.append((x_center, y_center, width, height, class_id))
            except Exception as e:
                print(f"Error loading annotations: {e}")
        
        return boxes
    
    def save_annotations(self, image_path: Path, boxes: List[Tuple], img_shape: Tuple):
        """Save annotations in YOLO format."""
        annotation_path = image_path.with_suffix('.txt')
        h, w = img_shape[:2]
        
        with open(annotation_path, 'w') as f:
            for box in boxes:
                if len(box) == 5:
                    # Already in YOLO format (x_center, y_center, w, h, class)
                    x_center, y_center, width, height, class_id = box
                else:
                    # Pixel coordinates (x1, y1, x2, y2, class)
                    x1, y1, x2, y2, class_id = box
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = abs(x2 - x1) / w
                    height = abs(y2 - y1) / h
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Saved: {annotation_path}")
    
    def yolo_to_pixel(self, yolo_box: Tuple, img_shape: Tuple) -> Tuple:
        """Convert YOLO format to pixel coordinates."""
        x_center, y_center, width, height, class_id = yolo_box
        h, w = img_shape[:2]
        
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        return (x1, y1, x2, y2, class_id)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Normalize coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if box has some size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.boxes.append((x1, y1, x2, y2, self.current_class))
                
                self.current_box = None
    
    def draw_boxes(self, image):
        """Draw all bounding boxes on the image."""
        img = image.copy()
        
        # Draw saved boxes
        for box in self.boxes:
            if len(box) == 5 and box[0] <= 1:
                # YOLO format - convert
                box = self.yolo_to_pixel(box, img.shape)
            
            x1, y1, x2, y2, class_id = box
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{self.class_names[class_id]}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            color = self.colors[self.current_class % len(self.colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # Draw current class indicator
        cv2.putText(img, f"Class: {self.current_class} ({self.class_names[self.current_class]})", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Image: {self.current_idx + 1}/{len(self.image_files)}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Boxes: {len(self.boxes)}", 
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def run(self):
        """Main annotation loop."""
        cv2.namedWindow('Annotate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotate', 1000, 800)
        cv2.setMouseCallback('Annotate', self.mouse_callback)
        
        while True:
            image_path = self.image_files[self.current_idx]
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Failed to load: {image_path}")
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                continue
            
            # Load existing annotations
            existing = self.load_annotations(image_path)
            if existing and not self.boxes:
                self.boxes = list(existing)
                # Determine class from folder
                folder = image_path.parent.name.lower()
                if 'part1' in folder or 'part 1' in folder:
                    self.current_class = 0
                elif 'part2' in folder or 'part 2' in folder:
                    self.current_class = 1
                elif 'part3' in folder or 'part 3' in folder:
                    self.current_class = 2
                elif 'part4' in folder or 'part 4' in folder:
                    self.current_class = 3
            
            # Auto-set class based on folder if no boxes yet
            if not self.boxes:
                folder = image_path.parent.name.lower()
                if 'part1' in folder or 'part 1' in folder:
                    self.current_class = 0
                elif 'part2' in folder or 'part 2' in folder:
                    self.current_class = 1
                elif 'part3' in folder or 'part 3' in folder:
                    self.current_class = 2
                elif 'part4' in folder or 'part 4' in folder:
                    self.current_class = 3
            
            print(f"\nCurrent: {image_path.name} (Class: {self.current_class})")
            
            while True:
                display = self.draw_boxes(image)
                cv2.imshow('Annotate', display)
                
                key = cv2.waitKey(50) & 0xFF
                
                if key == ord('q'):
                    # Save and quit
                    if self.boxes:
                        self.save_annotations(image_path, self.boxes, image.shape)
                    cv2.destroyAllWindows()
                    print("\nAnnotation complete!")
                    return
                
                elif key == ord('n') or key == 13:  # 'n' or Enter
                    # Save and next
                    if self.boxes:
                        self.save_annotations(image_path, self.boxes, image.shape)
                    self.boxes = []
                    self.current_idx = (self.current_idx + 1) % len(self.image_files)
                    break
                
                elif key == ord('p'):
                    # Save and previous
                    if self.boxes:
                        self.save_annotations(image_path, self.boxes, image.shape)
                    self.boxes = []
                    self.current_idx = (self.current_idx - 1) % len(self.image_files)
                    break
                
                elif key == ord('r'):
                    # Reset
                    self.boxes = []
                    print("Annotations reset")
                
                elif key == ord('s'):
                    # Save
                    if self.boxes:
                        self.save_annotations(image_path, self.boxes, image.shape)
                
                elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                    self.current_class = int(chr(key))
                    print(f"Class set to: {self.current_class} ({self.class_names[self.current_class]})")
                
                elif key == ord('u') and self.boxes:
                    # Undo last box
                    self.boxes.pop()
                    print("Undid last box")


def main():
    parser = argparse.ArgumentParser(description='Bounding Box Annotation Tool')
    parser.add_argument('--input_dir', type=str, default='.', help='Directory containing images')
    args = parser.parse_args()
    
    annotator = BoundingBoxAnnotator(args.input_dir)
    annotator.run()


if __name__ == '__main__':
    main()
