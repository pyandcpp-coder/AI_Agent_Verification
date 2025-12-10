import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import requests
import numpy as np
from io import BytesIO


class DetectionVisualizer:
    def __init__(self, model_path="models/best4.pt"):
        """Initialize the detection visualizer with a YOLO model"""
        print(f"Loading Model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = 0.05
        self.retry_threshold = 0.3  # If best detection is below this, try zooming
        self.zoom_levels = [1.5, 2.0]  # Zoom factors to try
        self.tile_overlap = 0.2  # 20% overlap between tiles
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            'aadhar_front': (0, 255, 0),      # Green
            'aadhar_back': (255, 0, 0),       # Blue
            'aadhar_long_front': (0, 255, 255),  # Yellow
            'aadhar_long_back': (255, 0, 255),   # Magenta
            'default': (0, 165, 255)          # Orange
        }
    
    def load_image_from_url(self, url):
        """Download and load an image from a URL"""
        try:
            print(f"Downloading image from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Error: Could not decode image from URL")
                return None
            
            print(f"Successfully loaded image from URL")
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def create_tiles(self, img, zoom_factor=1.5):
        """
        Create overlapping tiles from an image for better detection
        
        Args:
            img: Input image
            zoom_factor: How much to zoom in (1.5 = 150% zoom)
        
        Returns:
            List of (tile_image, x_offset, y_offset, scale_factor)
        """
        h, w = img.shape[:2]
        tiles = []
        
        # Calculate tile size (zoomed region)
        tile_h = int(h / zoom_factor)
        tile_w = int(w / zoom_factor)
        
        # Calculate step size with overlap
        step_h = int(tile_h * (1 - self.tile_overlap))
        step_w = int(tile_w * (1 - self.tile_overlap))
        
        print(f"\n[Tiling] Creating tiles with zoom={zoom_factor}x, tile_size={tile_w}x{tile_h}")
        
        # Generate tiles
        for y in range(0, h - tile_h + 1, step_h):
            for x in range(0, w - tile_w + 1, step_w):
                # Extract tile
                tile = img[y:y+tile_h, x:x+tile_w]
                
                # Resize to original resolution (zoom effect)
                zoomed_tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_LINEAR)
                
                tiles.append((zoomed_tile, x, y, zoom_factor))
        
        print(f"[Tiling] Created {len(tiles)} tiles")
        return tiles
    
    def run_detection_with_retry(self, img):
        """
        Run detection with automatic retries using zooming if initial detection is poor
        
        Returns:
            List of detections with format: [(label, conf, [x1, y1, x2, y2], is_from_tile, tile_info), ...]
        """
        all_detections = []
        
        # First attempt: Full image detection
        print("\n[Detection] Running on full image...")
        results = self.model(img, verbose=False)
        
        max_conf = 0.0
        for box in results[0].boxes:
            conf = float(box.conf)
            if conf >= self.conf_threshold:
                cls_id = int(box.cls)
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                all_detections.append((label, conf, [x1, y1, x2, y2], False, None))
                max_conf = max(max_conf, conf)
        
        print(f"[Detection] Full image: {len(all_detections)} detections, max_conf={max_conf:.4f}")
        
        # If no detection or low confidence, try zooming
        if len(all_detections) == 0 or max_conf < self.retry_threshold:
            print(f"\n[Retry] Low confidence ({max_conf:.4f} < {self.retry_threshold}), trying zoom levels...")
            
            for zoom_level in self.zoom_levels:
                tiles = self.create_tiles(img, zoom_factor=zoom_level)
                
                for tile_idx, (tile_img, x_offset, y_offset, scale) in enumerate(tiles):
                    results = self.model(tile_img, verbose=False)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        if conf >= self.conf_threshold:
                            cls_id = int(box.cls)
                            label = self.model.names[cls_id]
                            
                            # Get coordinates from zoomed tile
                            tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Map back to original image coordinates
                            h, w = img.shape[:2]
                            tile_h = int(h / scale)
                            tile_w = int(w / scale)
                            
                            # Scale down from full resolution to tile size
                            x1 = int(tx1 / scale) + x_offset
                            y1 = int(ty1 / scale) + y_offset
                            x2 = int(tx2 / scale) + x_offset
                            y2 = int(ty2 / scale) + y_offset
                            
                            # Clamp to image bounds
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))
                            
                            tile_info = f"zoom_{scale}x_tile_{tile_idx}"
                            all_detections.append((label, conf, [x1, y1, x2, y2], True, tile_info))
                            
                            if conf > max_conf:
                                max_conf = conf
                                print(f"[Retry] Found better detection in {tile_info}: {label} @ {conf:.4f}")
        
        return all_detections

    def detect_and_visualize(self, image_path, save_path=None, show=True):
        """
        Run detection on an image and visualize results
        
        Args:
            image_path: Path to input image or URL
            save_path: Path to save annotated image (optional)
            show: Whether to display the image in a window
        """
        # Read image - check if it's a URL or local path
        if image_path.startswith('http://') or image_path.startswith('https://'):
            img = self.load_image_from_url(image_path)
            image_name = image_path.split('/')[-1] or "url_image"
        else:
            img = cv2.imread(image_path)
            image_name = Path(image_path).name
        
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        print(f"\n{'='*70}")
        print(f"Processing: {image_name}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        print('='*70)
        
        # Run detection with retry mechanism
        all_detections = self.run_detection_with_retry(img)
        
        # Create a copy for annotation
        annotated_img = img.copy()
        
        # Remove duplicates (same detection from different tiles)
        unique_detections = []
        seen_boxes = set()
        
        for label, conf, box, is_from_tile, tile_info in all_detections:
            # Create a key for deduplication (rounded coordinates)
            box_key = (label, round(box[0]/10), round(box[1]/10), round(box[2]/10), round(box[3]/10))
            
            if box_key not in seen_boxes:
                seen_boxes.add(box_key)
                unique_detections.append((label, conf, box, is_from_tile, tile_info))
        
        # Annotate image with all unique detections
        final_detections = []
        for label, conf, box, is_from_tile, tile_info in unique_detections:
            x1, y1, x2, y2 = box
            
            # Get color for this label
            color = self.colors.get(label, self.colors['default'])
            
            # Draw bounding box (thicker if from tile)
            thickness = 3 if is_from_tile else 2
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            source_tag = " [ZOOM]" if is_from_tile else ""
            label_text = f"{label}: {conf:.2f}{source_tag}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background for text
            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_img,
                label_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            final_detections.append({
                'label': label,
                'confidence': conf,
                'box': box,
                'from_zoom': is_from_tile,
                'tile_info': tile_info
            })
            
            source_str = f" (from {tile_info})" if is_from_tile else " (full image)"
            print(f"✓ Detected: {label} | Confidence: {conf:.4f} | Box: {box}{source_str}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: {len(final_detections)} unique detection(s)")
        zoom_count = sum(1 for d in final_detections if d['from_zoom'])
        if zoom_count > 0:
            print(f"  - {zoom_count} detection(s) found using zoom enhancement")
        print('='*70)
        
        # Save annotated image if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"Saved annotated image to: {save_path}")
        
        # Display image if requested
        if show:
            # Resize if image is too large
            height, width = annotated_img.shape[:2]
            max_dimension = 1200
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                annotated_img = cv2.resize(annotated_img, (new_width, new_height))
            
            cv2.imshow(f"Detections - {image_name}", annotated_img)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return final_detections, annotated_img

    def detect_batch(self, image_folder, output_folder=None, show=False):
        """
        Process multiple images from a folder
        
        Args:
            image_folder: Path to folder containing images
            output_folder: Path to save annotated images (optional)
            show: Whether to display each image
        """
        image_folder = Path(image_folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))
            image_files.extend(image_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
        # Create output folder if specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_path in image_files:
            print(f"\n{'='*60}")
            print(f"Processing: {img_path.name}")
            print('='*60)
            
            save_path = None
            if output_folder:
                save_path = output_folder / f"annotated_{img_path.name}"
            
            self.detect_and_visualize(str(img_path), save_path, show)


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO detection results')
    parser.add_argument('--model', type=str, default='models/best4.pt',
                        help='Path to YOLO model')
    parser.add_argument('--image', type=str, 
                        help='Path or URL to single image to process')
    parser.add_argument('--url', type=str,
                        help='URL to image (alternative to --image)')
    parser.add_argument('--folder', type=str,
                        help='Path to folder containing images')
    parser.add_argument('--output', type=str,
                        help='Path to save annotated images')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display images in window')
    parser.add_argument('--conf', type=float, default=0.40,
                        help='Confidence threshold (default: 0.40)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DetectionVisualizer(model_path=args.model)
    visualizer.conf_threshold = args.conf
    
    # Process single image or batch
    if args.image or args.url:
        image_source = args.image or args.url
        visualizer.detect_and_visualize(
            image_source,
            save_path=args.output,
            show=not args.no_show
        )
    elif args.folder:
        visualizer.detect_batch(
            args.folder,
            output_folder=args.output,
            show=not args.no_show
        )
    else:
        print("Error: Please provide either --image, --url, or --folder argument")
        parser.print_help()


if __name__ == "__main__":
    main()
