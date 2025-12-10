import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import pytesseract
from PIL import Image
from datetime import datetime
import requests
from io import BytesIO
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderPipeline:
    """
    Complete Gender Detection Pipeline:
    1. Detect & crop Aadhar front using best.pt (DocAgent)
    2. Detect & crop gender field using gender_detection.pt
    3. Extract gender via OCR
    """
    
    def __init__(self, 
                 doc_model_path: str = "models/best4.pt",
                 gender_model_path: str = "models/gender_detection.pt"):
        """
        Initialize the Gender Detection Pipeline with both models.
        
        Args:
            doc_model_path: Path to best.pt for aadhar_front detection
            gender_model_path: Path to gender_detection.pt for gender field detection
        """
        logger.info("=" * 70)
        logger.info("Initializing Gender Detection Pipeline")
        logger.info("=" * 70)
        
        # Load Document Detection Model (for aadhar_front detection)
        doc_model_path = self._resolve_model_path(doc_model_path)
        logger.info(f"Loading Document Detection Model: {doc_model_path}")
        self.doc_model = YOLO(str(doc_model_path))
        self.doc_conf_threshold = 0.15
        self.doc_retry_threshold = 0.30
        logger.info(f"✓ Document Model loaded. Classes: {self.doc_model.names}")
        
        # Load Gender Detection Model
        gender_model_path = self._resolve_model_path(gender_model_path)
        logger.info(f"Loading Gender Detection Model: {gender_model_path}")
        self.gender_model = YOLO(str(gender_model_path))
        self.gender_conf_threshold = 0.25
        self.gender_retry_threshold = 0.30
        self.zoom_levels = [1.5, 2.0]
        self.tile_overlap = 0.2
        logger.info(f"✓ Gender Detection Model loaded. Classes: {self.gender_model.names}")
        
        # Setup OCR temp folder
        self.ocr_temp_folder = Path(__file__).parent.parent / "ocr_temp"
        self.ocr_temp_folder.mkdir(exist_ok=True)
        logger.info(f"✓ OCR temp folder: {self.ocr_temp_folder}")
        
        # Check Tesseract
        self._check_tesseract()
        
        logger.info("=" * 70)
        logger.info("Pipeline Initialization Complete")
        logger.info("=" * 70)
    
    def _resolve_model_path(self, model_path: str) -> Path:
        """
        Resolve model path - handle both relative and absolute paths.
        """
        model_path = Path(model_path)
        
        if not model_path.is_absolute() and not model_path.exists():
            parent_model_path = Path(__file__).parent.parent / model_path
            if parent_model_path.exists():
                model_path = parent_model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        return model_path
    
    def _check_tesseract(self):
        """Check if Tesseract is installed"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("✓ Tesseract OCR is available")
        except pytesseract.TesseractNotFoundError:
            logger.warning("⚠ Tesseract executable not found. OCR will not work.")
        except Exception as e:
            logger.warning(f"⚠ Error checking Tesseract: {e}")
    
    def _load_image(self, image_source: str) -> Tuple[Optional[np.ndarray], str, Optional[str]]:
        """
        Load image from file path or URL.
        
        Args:
            image_source: File path or URL
        
        Returns:
            tuple: (numpy array, source_type, error_msg)
        """
        try:
            # Check if it's a URL
            if image_source.startswith(('http://', 'https://')):
                logger.info(f"Loading image from URL: {image_source}")
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                img_data = Image.open(BytesIO(response.content))
                img = cv2.cvtColor(np.array(img_data), cv2.COLOR_RGB2BGR)
                logger.info(f"✓ Loaded image from URL: {img.shape}")
                return img, 'url', None
            
            # Otherwise treat as file path
            if isinstance(image_source, str) or isinstance(image_source, Path):
                img = cv2.imread(str(image_source))
                if img is None:
                    return None, 'file', f"Could not load image from {image_source}"
                logger.info(f"✓ Loaded image from file: {img.shape}")
                return img, 'file', None
        except requests.exceptions.RequestException as e:
            return None, 'url', f"Failed to download image from URL: {str(e)}"
        except Exception as e:
            return None, 'file', str(e)
    
    # ============================================================================
    # STAGE 1: Document Detection & Cropping (using best.pt)
    # ============================================================================
    
    def _detect_aadhar_front_with_retry(self, img: np.ndarray) -> Tuple[Optional[str], float, Optional[np.ndarray], str]:
        """
        Detect aadhar_front in the image with retry mechanism using zoom.
        
        Args:
            img: Input image
        
        Returns:
            tuple: (label, confidence, bounding_box, method_used)
        """
        logger.info("[Stage 1] Detecting Aadhar Front Card...")
        
        # First attempt: Full image detection
        logger.info("  [Attempt 1] Running detection on full image...")
        results = self.doc_model(img, verbose=False)
        
        all_detections = []  # Store all detections for visualization
        best_label = None
        max_conf = 0.0
        best_box = None
        detection_method = 'full_image'
        
        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = self.doc_model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            all_detections.append((label, conf, bbox))
            logger.info(f"    Found: {label} (conf: {conf:.4f})")
            
            if conf >= self.doc_conf_threshold:
                # Check if it's aadhar_front
                if 'front' in label.lower():
                    if conf > max_conf:
                        max_conf = conf
                        best_label = label
                        best_box = bbox
        
        # Visualize all detections from full image
        if all_detections:
            img_with_detections = self._draw_detections(
                img, 
                [d[2] for d in all_detections],
                [d[0] for d in all_detections],
                [d[1] for d in all_detections],
                "Stage 1: Full Image Detection"
            )
            self._save_detection_image(img_with_detections, "stage1_fullimage")
        
        logger.info(f"  [Attempt 1] Max confidence: {max_conf:.4f}, Total detections: {len(all_detections)}")
        
        # If good detection, return it
        if best_label and max_conf >= self.doc_conf_threshold:
            logger.info(f"  ✓ Found {best_label} with confidence {max_conf:.4f}")
            return best_label, max_conf, best_box, detection_method
        
        # Retry with zoom if no good detection
        if max_conf < self.doc_retry_threshold:
            logger.info(f"  [Attempt 2] Low confidence ({max_conf:.4f}), trying zoom levels...")
            
            for zoom_level in self.zoom_levels:
                tiles = self._create_tiles(img, zoom_factor=zoom_level)
                logger.info(f"  [Zoom {zoom_level}x] Created {len(tiles)} tiles")
                
                for tile_idx, (tile_img, x_offset, y_offset, scale) in enumerate(tiles):
                    results = self.doc_model(tile_img, verbose=False)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        if conf >= self.doc_conf_threshold:
                            cls_id = int(box.cls)
                            label = self.doc_model.names[cls_id]
                            
                            if 'front' in label.lower():
                                # Map coordinates back to original image
                                tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy().astype(int)
                                h, w = img.shape[:2]
                                
                                x1 = int(tx1 / scale) + x_offset
                                y1 = int(ty1 / scale) + y_offset
                                x2 = int(tx2 / scale) + x_offset
                                y2 = int(ty2 / scale) + y_offset
                                
                                # Clamp to image bounds
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                mapped_box = np.array([x1, y1, x2, y2], dtype=int)
                                
                                if conf > max_conf:
                                    max_conf = conf
                                    best_label = label
                                    best_box = mapped_box
                                    detection_method = f'zoom_{zoom_level}x'
                                    logger.info(f"    ✓ Better detection in zoom {zoom_level}x: {label} @ {conf:.4f}")
        
        if best_label:
            logger.info(f"  ✓ Final: {best_label} ({max_conf:.4f}) using {detection_method}")
            # Visualize final detection
            img_final = self._draw_detections(img, [best_box], [best_label], [max_conf], "Stage 1: Final Detection")
            self._save_detection_image(img_final, "stage1_final")
            return best_label, max_conf, best_box, detection_method
        else:
            logger.warning("  ✗ No aadhar_front detected")
            return None, 0.0, None, 'none'
    
    def _create_tiles(self, img: np.ndarray, zoom_factor: float = 1.5) -> List[Tuple]:
        """
        Create overlapping tiles from an image for zoom detection.
        
        Returns:
            List of (tile_image, x_offset, y_offset, scale_factor)
        """
        h, w = img.shape[:2]
        tiles = []
        
        tile_h = int(h / zoom_factor)
        tile_w = int(w / zoom_factor)
        
        step_h = int(tile_h * (1 - self.tile_overlap))
        step_w = int(tile_w * (1 - self.tile_overlap))
        
        for y in range(0, h - tile_h + 1, step_h):
            for x in range(0, w - tile_w + 1, step_w):
                tile = img[y:y+tile_h, x:x+tile_w]
                zoomed_tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_LINEAR)
                tiles.append((zoomed_tile, x, y, zoom_factor))
        
        return tiles
    
    def _crop_aadhar_front(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop the aadhar front from the image using bounding box.
        
        Args:
            img: Full image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Cropped image or None
        """
        try:
            x1, y1, x2, y2 = bbox
            cropped = img[y1:y2, x1:x2]
            
            if cropped is None or cropped.size == 0:
                logger.error("  ✗ Failed to crop aadhar front")
                return None
            
            logger.info(f"  ✓ Cropped aadhar front: {cropped.shape}")
            return cropped
        except Exception as e:
            logger.error(f"  ✗ Error cropping aadhar front: {e}")
            return None
    
    # ============================================================================
    # STAGE 2: Gender Field Detection & Cropping (using gender_detection.pt)
    # ============================================================================
    
    def _detect_gender_field_with_retry(self, img: np.ndarray) -> Tuple[Optional[str], float, Optional[np.ndarray], str]:
        """
        Detect gender field in the cropped aadhar image with retry mechanism.
        
        Args:
            img: Cropped aadhar front image
        
        Returns:
            tuple: (label, confidence, bounding_box, method_used)
        """
        logger.info("[Stage 2] Detecting Gender Field...")
        
        # First attempt: Full image detection
        logger.info("  [Attempt 1] Running detection on full image...")
        results = self.gender_model(img, verbose=False)
        
        all_detections = []  # Store all detections for visualization
        best_label = None
        max_conf = 0.0
        best_box = None
        detection_method = 'full_image'
        
        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = self.gender_model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            all_detections.append((label, conf, bbox))
            logger.info(f"    Found: {label} (conf: {conf:.4f})")
            
            if conf >= self.gender_conf_threshold:
                if conf > max_conf:
                    max_conf = conf
                    best_label = label
                    best_box = bbox
        
        # Visualize all detections from full image
        if all_detections:
            img_with_detections = self._draw_detections(
                img,
                [d[2] for d in all_detections],
                [d[0] for d in all_detections],
                [d[1] for d in all_detections],
                "Stage 2: Full Image Detection"
            )
            self._save_detection_image(img_with_detections, "stage2_fullimage")
        
        logger.info(f"  [Attempt 1] Max confidence: {max_conf:.4f}, Total detections: {len(all_detections)}")
        
        # If good detection, return it
        if best_label and max_conf >= self.gender_conf_threshold:
            logger.info(f"  ✓ Found gender field with confidence {max_conf:.4f}")
            return best_label, max_conf, best_box, detection_method
        
        # Retry with zoom if no good detection
        if max_conf < self.gender_retry_threshold:
            logger.info(f"  [Attempt 2] Low confidence ({max_conf:.4f}), trying zoom levels...")
            
            for zoom_level in self.zoom_levels:
                tiles = self._create_tiles(img, zoom_factor=zoom_level)
                logger.info(f"  [Zoom {zoom_level}x] Created {len(tiles)} tiles")
                
                for tile_idx, (tile_img, x_offset, y_offset, scale) in enumerate(tiles):
                    results = self.gender_model(tile_img, verbose=False)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        if conf >= self.gender_conf_threshold:
                            cls_id = int(box.cls)
                            label = self.gender_model.names[cls_id]
                            
                            # Map coordinates back to original image
                            tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy().astype(int)
                            h, w = img.shape[:2]
                            
                            x1 = int(tx1 / scale) + x_offset
                            y1 = int(ty1 / scale) + y_offset
                            x2 = int(tx2 / scale) + x_offset
                            y2 = int(ty2 / scale) + y_offset
                            
                            # Clamp to image bounds
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            mapped_box = np.array([x1, y1, x2, y2], dtype=int)
                            
                            if conf > max_conf:
                                max_conf = conf
                                best_label = label
                                best_box = mapped_box
                                detection_method = f'zoom_{zoom_level}x'
                                logger.info(f"    ✓ Better detection in zoom {zoom_level}x @ {conf:.4f}")
        
        if best_label:
            logger.info(f"  ✓ Final: Gender field ({max_conf:.4f}) using {detection_method}")
            # Visualize final detection
            img_final = self._draw_detections(img, [best_box], [best_label], [max_conf], "Stage 2: Final Detection")
            self._save_detection_image(img_final, "stage2_final")
            return best_label, max_conf, best_box, detection_method
        else:
            logger.warning("  ✗ Gender field not detected")
            return None, 0.0, None, 'none'
    
    def _crop_gender_field(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop the gender field from the image using bounding box.
        
        Args:
            img: Full image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Cropped image or None
        """
        try:
            x1, y1, x2, y2 = bbox
            cropped = img[y1:y2, x1:x2]
            
            if cropped is None or cropped.size == 0:
                logger.error("  ✗ Failed to crop gender field")
                return None
            
            logger.info(f"  ✓ Cropped gender field: {cropped.shape}")
            return cropped
        except Exception as e:
            logger.error(f"  ✗ Error cropping gender field: {e}")
            return None
    
    # ============================================================================
    # VISUALIZATION & DEBUG
    # ============================================================================
    
    def _draw_detections(self, img: np.ndarray, boxes: List, labels: List, confidences: List, title: str = "") -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            img: Input image
            boxes: List of bounding boxes
            labels: List of labels
            confidences: List of confidences
            title: Title for the image
        
        Returns:
            Image with drawn detections
        """
        img_copy = img.copy()
        
        for box, label, conf in zip(boxes, labels, confidences):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            text = f"{label} {conf:.3f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Get text size for background
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background for text
            cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0] + 5, y1), (0, 255, 0), -1)
            
            # Put text
            cv2.putText(img_copy, text, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        if title:
            cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img_copy
    
    def _save_detection_image(self, img: np.ndarray, stage: str, filename_suffix: str = "") -> str:
        """Save detection image for visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"detection_{stage}_{timestamp}{filename_suffix}.jpg"
        filepath = self.ocr_temp_folder / filename
        cv2.imwrite(str(filepath), img)
        logger.info(f"  📸 Saved visualization: {filepath}")
        return str(filepath)
    
    def _enhance_contrast(self, gray_img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray_img)
        except Exception:
            return gray_img
    
    def _apply_morphological_ops(self, gray_img: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance text."""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        except Exception:
            return gray_img
    
    def _normalize_gender(self, text: str) -> str:
        """Normalize gender text to 'Male', 'Female', or 'Other'."""
        if not text:
            return 'Other'
        
        text = text.strip().lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        
        if 'male' in text and 'female' not in text:
            return 'Male'
        elif 'female' in text or 'femal' in text:
            return 'Female'
        elif text in ['m', 'man', 'boy']:
            return 'Male'
        elif text in ['f', 'woman', 'girl', 'femlae', 'femaie']:
            return 'Female'
        else:
            return 'Other'
    
    def _extract_gender_via_ocr(self, cropped_gender_img: np.ndarray) -> Dict:
        """
        Extract gender text from cropped gender field image.
        
        Args:
            cropped_gender_img: Cropped gender field image
        
        Returns:
            dict: {
                'extracted_text': str,
                'normalized_gender': str,
                'confidence': float,
                'method': str,
                'image_path': str
            }
        """
        logger.info("[Stage 3] Extracting Gender via OCR...")
        
        try:
            # Save cropped gender image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            gender_filename = f"gender_cropped_{timestamp}.jpg"
            gender_image_path = self.ocr_temp_folder / gender_filename
            cv2.imwrite(str(gender_image_path), cropped_gender_img)
            logger.info(f"  ✓ Saved cropped gender image: {gender_image_path}")
            
            # Convert to grayscale
            if len(cropped_gender_img.shape) == 3:
                gray = cv2.cvtColor(cropped_gender_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cropped_gender_img
            
            # Try multiple preprocessing methods and PSM modes
            best_text = ''
            best_confidence = 0.0
            best_method = 'none'
            
            preprocessing_methods = {
                'original': lambda img: img,
                'upscale_2x': lambda img: cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                'upscale_3x': lambda img: cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                'contrast_enhance': lambda img: self._enhance_contrast(img),
                'denoise': lambda img: cv2.fastNlMeansDenoising(img, h=10),
                'upscale_2x + contrast': lambda img: self._enhance_contrast(
                    cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ),
                'morphology': lambda img: self._apply_morphological_ops(img),
            }
            
            psm_modes = [6, 7, 8, 11, 13]
            
            logger.info(f"  Testing {len(preprocessing_methods)} preprocessing methods × {len(psm_modes)} PSM modes...")
            
            for prep_name, prep_func in preprocessing_methods.items():
                try:
                    preprocessed = prep_func(gray)
                    
                    for psm in psm_modes:
                        custom_config = f'--psm {psm} -c tessedit_char_whitelist=MaleFfemale'
                        result = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DICT)
                        
                        # Get text and confidence
                        text = ' '.join([word for word in result['text'] if word.strip()])
                        
                        # Calculate average confidence (excluding 0 confidence)
                        confidences = [int(conf) for conf in result['confidence'] if int(conf) > 0]
                        confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_text = text
                            best_method = f'{prep_name} (PSM {psm})'
                            
                except Exception as e:
                    logger.debug(f"  Error with {prep_name}, PSM {psm}: {e}")
                    continue
            
            # Normalize the result
            normalized_gender = self._normalize_gender(best_text) if best_text else 'Other'
            
            logger.info(f"  ✓ Best OCR result: '{best_text}' → '{normalized_gender}'")
            logger.info(f"  ✓ Method: {best_method}, Confidence: {best_confidence:.2f}")
            
            return {
                'extracted_text': best_text,
                'normalized_gender': normalized_gender,
                'confidence': best_confidence,
                'method': best_method,
                'image_path': str(gender_image_path)
            }
        
        except Exception as e:
            logger.error(f"  ✗ OCR extraction failed: {e}")
            return {
                'extracted_text': '',
                'normalized_gender': 'Other',
                'confidence': 0.0,
                'method': 'failed',
                'image_path': '',
                'error': str(e)
            }
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def process(self, image_path: str) -> Dict:
        """
        Complete pipeline: best.pt → crop aadhar_front → gender_detection.pt → crop gender → OCR
        
        Args:
            image_path: Path to aadhar image
        
        Returns:
            dict: {
                'success': bool,
                'stage_1': {...},  # Document detection results
                'stage_2': {...},  # Gender field detection results
                'stage_3': {...},  # OCR extraction results
                'final_gender': str,
                'error': str (if failed)
            }
        """
        logger.info("=" * 70)
        logger.info("Starting Gender Pipeline")
        logger.info(f"Image: {image_path}")
        logger.info("=" * 70)
        
        try:
            # Load image
            img, source_type, error_msg = self._load_image(image_path)
            if img is None:
                logger.error(f"✗ Failed to load image: {error_msg}")
                return {
                    'success': False,
                    'stage_1': None,
                    'stage_2': None,
                    'stage_3': None,
                    'final_gender': 'Other',
                    'error': error_msg
                }
            
            logger.info(f"✓ Loaded image: {img.shape}")
            
            # ===== STAGE 1: Detect & Crop Aadhar Front =====
            label_1, conf_1, box_1, method_1 = self._detect_aadhar_front_with_retry(img)
            
            if not label_1 or box_1 is None:
                logger.error("✗ Stage 1 Failed: Could not detect aadhar_front")
                return {
                    'success': False,
                    'stage_1': {'success': False, 'reason': 'aadhar_front not detected'},
                    'stage_2': None,
                    'stage_3': None,
                    'final_gender': 'Other',
                    'error': 'Document detection failed'
                }
            
            # Crop aadhar front
            aadhar_cropped = self._crop_aadhar_front(img, box_1)
            if aadhar_cropped is None:
                logger.error("✗ Stage 1 Failed: Could not crop aadhar_front")
                return {
                    'success': False,
                    'stage_1': {'success': False, 'reason': 'Failed to crop aadhar_front'},
                    'stage_2': None,
                    'stage_3': None,
                    'final_gender': 'Other',
                    'error': 'Cropping failed'
                }
            
            stage_1_result = {
                'success': True,
                'label': label_1,
                'confidence': float(conf_1),
                'bounding_box': box_1.tolist(),
                'method': method_1,
                'cropped_shape': list(aadhar_cropped.shape)
            }
            logger.info(f"✓ Stage 1 Complete: {stage_1_result}")
            
            # ===== STAGE 2: Detect & Crop Gender Field =====
            label_2, conf_2, box_2, method_2 = self._detect_gender_field_with_retry(aadhar_cropped)
            
            if not label_2 or box_2 is None:
                logger.warning("⚠ Stage 2 Warning: Gender field not detected, setting to 'Other'")
                stage_2_result = {
                    'success': False,
                    'reason': 'Gender field not detected'
                }
                return {
                    'success': False,
                    'stage_1': stage_1_result,
                    'stage_2': stage_2_result,
                    'stage_3': None,
                    'final_gender': 'Other',
                    'error': 'Gender field detection failed'
                }
            
            # Crop gender field
            gender_cropped = self._crop_gender_field(aadhar_cropped, box_2)
            if gender_cropped is None:
                logger.warning("⚠ Stage 2 Warning: Could not crop gender field")
                stage_2_result = {
                    'success': False,
                    'reason': 'Failed to crop gender field'
                }
                return {
                    'success': False,
                    'stage_1': stage_1_result,
                    'stage_2': stage_2_result,
                    'stage_3': None,
                    'final_gender': 'Other',
                    'error': 'Gender field cropping failed'
                }
            
            stage_2_result = {
                'success': True,
                'label': label_2,
                'confidence': float(conf_2),
                'bounding_box': box_2.tolist(),
                'method': method_2,
                'cropped_shape': list(gender_cropped.shape)
            }
            logger.info(f"✓ Stage 2 Complete: {stage_2_result}")
            
            # ===== STAGE 3: Extract Gender via OCR =====
            stage_3_result = self._extract_gender_via_ocr(gender_cropped)
            logger.info(f"✓ Stage 3 Complete: {stage_3_result}")
            
            # Final result
            final_gender = stage_3_result['normalized_gender']
            
            return {
                'success': True,
                'stage_1': stage_1_result,
                'stage_2': stage_2_result,
                'stage_3': stage_3_result,
                'final_gender': final_gender
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline Error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'stage_1': None,
                'stage_2': None,
                'stage_3': None,
                'final_gender': 'Other',
                'error': str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    import sys
    import json
    
    logger.info("\n" + "=" * 70)
    logger.info("Gender Detection Pipeline Test")
    logger.info("=" * 70)
    logger.info("\nUsage:")
    logger.info("  python gender_pipeline.py <image_path_or_url>")
    logger.info("\nExamples:")
    logger.info("  File:  python gender_pipeline.py images/aadhar_front.jpg")
    logger.info("  URL:   python gender_pipeline.py https://example.com/aadhar.jpg")
    logger.info("=" * 70 + "\n")
    
    if len(sys.argv) > 1:
        image_source = sys.argv[1]
        
        try:
            # Initialize pipeline
            logger.info(f"Initializing pipeline...")
            pipeline = GenderPipeline()
            
            # Process image
            logger.info(f"Processing image: {image_source}")
            result = pipeline.process(image_source)
            
            # Print results
            logger.info("\n" + "=" * 70)
            logger.info("FINAL RESULTS")
            logger.info("=" * 70)
            logger.info(f"Success: {result['success']}")
            logger.info(f"Final Gender: {result['final_gender']}")
            
            if result['stage_1']:
                logger.info(f"\n{'='*70}")
                logger.info(f"Stage 1 (Document Detection):")
                logger.info(f"{'='*70}")
                logger.info(f"  Success: {result['stage_1'].get('success', False)}")
                if result['stage_1'].get('success'):
                    logger.info(f"  Detected: {result['stage_1']['label']}")
                    logger.info(f"  Confidence: {result['stage_1']['confidence']:.4f}")
                    logger.info(f"  Method: {result['stage_1']['method']}")
                    logger.info(f"  Bounding Box: {result['stage_1']['bounding_box']}")
                    logger.info(f"  Cropped Shape: {result['stage_1']['cropped_shape']}")
                else:
                    logger.info(f"  Reason: {result['stage_1'].get('reason', 'Unknown')}")
            
            if result['stage_2']:
                logger.info(f"\n{'='*70}")
                logger.info(f"Stage 2 (Gender Field Detection):")
                logger.info(f"{'='*70}")
                logger.info(f"  Success: {result['stage_2'].get('success', False)}")
                if result['stage_2'].get('success'):
                    logger.info(f"  Label: {result['stage_2']['label']}")
                    logger.info(f"  Confidence: {result['stage_2']['confidence']:.4f}")
                    logger.info(f"  Method: {result['stage_2']['method']}")
                    logger.info(f"  Bounding Box: {result['stage_2']['bounding_box']}")
                    logger.info(f"  Cropped Shape: {result['stage_2']['cropped_shape']}")
                else:
                    logger.info(f"  Reason: {result['stage_2'].get('reason', 'Unknown')}")
            
            if result['stage_3']:
                logger.info(f"\n{'='*70}")
                logger.info(f"Stage 3 (OCR Extraction):")
                logger.info(f"{'='*70}")
                logger.info(f"  Extracted Text: '{result['stage_3']['extracted_text']}'")
                logger.info(f"  Normalized Gender: '{result['stage_3']['normalized_gender']}'")
                logger.info(f"  Confidence: {result['stage_3']['confidence']:.2f}")
                logger.info(f"  Method: {result['stage_3']['method']}")
                logger.info(f"  Saved Image: {result['stage_3']['image_path']}")
            
            if not result['success']:
                logger.info(f"\nError: {result.get('error', 'Unknown error')}")
            
            logger.info("=" * 70)
            logger.info("\n📸 All detection visualizations saved to: ocr_temp/")
            logger.info("=" * 70 + "\n")
            
            # Print JSON summary
            print("\n" + "=" * 70)
            print("JSON OUTPUT:")
            print("=" * 70)
            print(json.dumps(result, indent=2))
            print("=" * 70)
        
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    else:
        logger.info("No image path or URL provided")
