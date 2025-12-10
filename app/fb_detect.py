import cv2
from ultralytics import YOLO
import numpy as np

class DocAgent:
    def __init__(self, model_path="models/best4.pt"):
        # Load model once when server starts
        print(f"Loading Document Model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = 0.15
        self.retry_threshold = 0.30  # If best detection is below this, try zooming
        self.zoom_levels = [1.5, 2.0]  # Zoom factors to try
        self.tile_overlap = 0.2  # 20% overlap between tiles
    
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
        
        print(f"  [Tiling] Creating tiles with zoom={zoom_factor}x, tile_size={tile_w}x{tile_h}")
        
        # Generate tiles
        for y in range(0, h - tile_h + 1, step_h):
            for x in range(0, w - tile_w + 1, step_w):
                # Extract tile
                tile = img[y:y+tile_h, x:x+tile_w]
                
                # Resize to original resolution (zoom effect)
                zoomed_tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_LINEAR)
                
                tiles.append((zoomed_tile, x, y, zoom_factor))
        
        print(f"  [Tiling] Created {len(tiles)} tiles")
        return tiles

    def _detect(self, img, prefer_front=False):
        """Helper to run detection on a single loaded image with retry mechanism
        
        Args:
            img: Input image
            prefer_front: If True, prioritize aadhar_front over aadhar_back when both are detected
        """
        # First attempt: Full image detection
        print("  [Detection] Running on full image...")
        results = self.model(img, verbose=False)
        
        best_label = None
        max_conf = 0.0
        best_box = None
        all_detections = []

        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = self.model.names[cls_id]

            # Collect all detections above threshold
            if conf >= self.conf_threshold:
                box_coords = box.xyxy[0].cpu().numpy().astype(int)
                all_detections.append((label, conf, box_coords))
                
                # Track highest confidence
                if conf > max_conf:
                    max_conf = conf
                    best_label = label
                    best_box = box_coords

        print(f"  [Detection] Full image: found {len(all_detections)} detection(s), max_conf={max_conf:.4f}")
        
        # If prefer_front is enabled and we have multiple detections, prioritize front
        if prefer_front and len(all_detections) > 1:
            front_detections = [d for d in all_detections if 'front' in d[0].lower()]
            if front_detections:
                # Sort by confidence and take the best front detection
                front_detections.sort(key=lambda x: x[1], reverse=True)
                best_label, max_conf, best_box = front_detections[0]
                print(f"  [Priority] Prioritizing front detection: {best_label} @ {max_conf:.4f}")
        
        # If no detection or low confidence, try zooming
        if (best_label is None or max_conf < self.retry_threshold):
            print(f"  [Retry] Low confidence ({max_conf:.4f} < {self.retry_threshold}), trying zoom levels...")
            
            tile_detections = []
            
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
                            
                            mapped_box = np.array([x1, y1, x2, y2], dtype=int)
                            tile_detections.append((label, conf, mapped_box))
                            
                            if conf > max_conf:
                                max_conf = conf
                                best_label = label
                                best_box = mapped_box
                                print(f"  [Retry] Found better detection in zoom_{scale}x_tile_{tile_idx}: {label} @ {conf:.4f}")
            
            # If prefer_front and we found multiple detections in tiles, prioritize front
            if prefer_front and len(tile_detections) > 1:
                front_tile_detections = [d for d in tile_detections if 'front' in d[0].lower()]
                if front_tile_detections:
                    front_tile_detections.sort(key=lambda x: x[1], reverse=True)
                    best_label, max_conf, best_box = front_tile_detections[0]
                    print(f"  [Priority] Prioritizing front detection from tiles: {best_label} @ {max_conf:.4f}")

        return best_label, max_conf, best_box

    def verify_documents(self, front_path, back_path):
        """
        1. Reads images from local temp folder.
        2. Checks if Front and Back are present.
        3. Returns coordinates of the Front card for the next step.
        """
        # 1. Read Images
        img_f = cv2.imread(front_path)
        img_b = cv2.imread(back_path)

        if img_f is None:
            return {"success": False, "message": f"Could not read file: {front_path}"}
        if img_b is None:
            return {"success": False, "message": f"Could not read file: {back_path}"}

        # 2. Run Detection
        # For front image, prefer aadhar_front if both front and back are detected
        label_f, conf_f, box_f = self._detect(img_f, prefer_front=True)
        label_b, conf_b, box_b = self._detect(img_b, prefer_front=False)

        # Log detections
        print(f"\n=== Document Detection Results ===")
        print(f"Front Image: {front_path}")
        print(f"  - Detected: {label_f if label_f else 'None'}")
        print(f"  - Confidence: {conf_f:.4f}")
        if box_f is not None:
            print(f"  - Bounding Box: {box_f.tolist()}")
        
        print(f"\nBack Image: {back_path}")
        print(f"  - Detected: {label_b if label_b else 'None'}")
        print(f"  - Confidence: {conf_b:.4f}")
        if box_b is not None:
            print(f"  - Bounding Box: {box_b.tolist()}")
        print("=" * 35 + "\n")

        # 3. Validate Presence (Fail Fast)
        # Check for all possible front/back label variations
        front_labels = ["aadhar_front", "aadhar_long_front"]
        back_labels = ["aadhar_back", "aadhar_long_back"]
        
        front_detected = label_f and any(fl in label_f.lower() for fl in front_labels)
        back_detected = label_b and any(bl in label_b.lower() for bl in back_labels)

        if not front_detected:
            return {
                "success": False, 
                "message": f"Front Aadhaar not detected. Found: {label_f} ({conf_f:.2f})"
            }
        
        if not back_detected:
            return {
                "success": False, 
                "message": f"Back Aadhaar not detected. Found: {label_b} ({conf_b:.2f})"
            }

        # 4. Success - Return the coordinates!
        # We return box_f so the Entity Agent can crop perfectly.
        return {
            "success": True,
            "message": "Both documents verified",
            "front_coords": box_f.tolist()  # Convert numpy array to list [x1, y1, x2, y2]
        }