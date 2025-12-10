import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from ultralytics import YOLO

class DocAgent:
    def __init__(self, model_path="models/best4.pt"):
        # Load model once when server starts
        print(f"Loading Document Model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = 0.40

    def _download_image(self, url):
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None

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

    def _draw_label(self, img, label, conf, box):
        """Draw bounding box and label on image"""
        img_copy = img.copy()
        
        if box is not None:
            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Prepare label text
            text = f"{label}: {conf:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_copy, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Put text
            cv2.putText(img_copy, text, (x1, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return img_copy

    def test_single_image(self, image_url):
        """Test a single image URL and display result"""
        print(f"\nTesting image: {image_url}")
        
        # Download image
        img = self._download_image(image_url)
        if img is None:
            print("Failed to download image")
            return
        
        # Detect
        label, conf, box = self._detect(img)
        
        # Print results
        if label:
            print(f"✓ Detected: {label} (confidence: {conf:.2f})")
            print(f"  Box coordinates: {box}")
        else:
            print(f"✗ No detection above threshold {self.conf_threshold}")
        
        # Draw label on image
        img_labeled = self._draw_label(img, label if label else "No Detection", 
                                       conf if label else 0.0, box)
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        img_rgb = cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"{label if label else 'No Detection'} - Confidence: {conf:.2f}" if label else "No Detection")
        plt.tight_layout()
        plt.show()

    def test_multiple_images(self, image_urls):
        """Test multiple image URLs and display results"""
        num_images = len(image_urls)
        fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 8))
        
        if num_images == 1:
            axes = [axes]
        
        for idx, url in enumerate(image_urls):
            print(f"\nTesting image {idx+1}: {url}")
            
            # Download image
            img = self._download_image(url)
            if img is None:
                print(f"Failed to download image {idx+1}")
                axes[idx].text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                axes[idx].axis('off')
                continue
            
            # Detect
            label, conf, box = self._detect(img)
            
            # Print results
            if label:
                print(f"✓ Detected: {label} (confidence: {conf:.2f})")
            else:
                print(f"✗ No detection above threshold {self.conf_threshold}")
            
            # Draw label on image
            img_labeled = self._draw_label(img, label if label else "No Detection", 
                                           conf if label else 0.0, box)
            
            # Display
            img_rgb = cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f"{label if label else 'No Detection'}\nConf: {conf:.2f}" if label else "No Detection")
        
        plt.tight_layout()
        plt.show()


# Test script
if __name__ == "__main__":
    # Initialize the agent
    agent = DocAgent(model_path="./models/best4.pt")
    
    # Example usage - replace with your image URLs
    print("\n" + "="*60)
    print("Document Detection Test")
    print("="*60)
    
    # Single image test
    image_url = "https://cdn.qoneqt.com/uploads/48732/aadhar_front_3Lcf3ZhxBB.jpeg"
    agent.test_single_image(image_url)
    
    # Multiple images test
    # image_urls = [
    #     "YOUR_FRONT_IMAGE_URL",
    #     "YOUR_BACK_IMAGE_URL"
    # ]
    agent.test_multiple_images(image_url)
    
    print("\n✓ Ready! Uncomment the test lines above and add your image URLs to test.")