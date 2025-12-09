import cv2
from ultralytics import YOLO

class DocAgent:
    def __init__(self, model_path="models/best4.pt"):
        # Load model once when server starts
        print(f"Loading Document Model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = 0.40

    def _detect(self, img):
        """Helper to run detection on a single loaded image"""
        results = self.model(img, verbose=False)
        
        best_label = None
        max_conf = 0.0
        best_box = None

        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = self.model.names[cls_id]

            # We want the highest confidence detection that passes threshold
            if conf >= self.conf_threshold and conf > max_conf:
                max_conf = conf
                best_label = label
                # Box format: [x1, y1, x2, y2] (Standard for cropping)
                best_box = box.xyxy[0].cpu().numpy().astype(int)

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
        label_f, conf_f, box_f = self._detect(img_f)
        label_b, conf_b, box_b = self._detect(img_b)

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