import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import logging
import pytesseract
from PIL import Image
from typing import Optional, Any
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderDetector:
    """
    Simple gender detection class that uses a YOLO model to detect gender field on Aadhar cards.
    Supports retry with image zooming if initial detection fails.
    """
    
    def __init__(self, model_path="models/gender_detection.pt"):
        """
        Initialize the Gender Detector with the specified model.
        
        Args:
            model_path: Path to the gender_detection.pt YOLO model
        """
        print(f"Loading Gender Detection Model from {model_path}...")
        
        # Resolve path - handle both relative and absolute paths
        model_path = Path(model_path)
        
        # If relative path and doesn't exist from current dir, try parent directory
        if not model_path.is_absolute() and not model_path.exists():
            # Try from parent directory (in case we're in app/ subdirectory)
            parent_model_path = Path(__file__).parent.parent / model_path
            if parent_model_path.exists():
                model_path = parent_model_path
        
        # Check if model exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load YOLO model
        self.model = YOLO(str(model_path))
        self.conf_threshold = 0.25  # Confidence threshold for detection
        self.retry_threshold = 0.30  # If below this, try scanning/zooming
        self.zoom_levels = [1.5, 2.0]  # Zoom factors to try
        self.tile_overlap = 0.2  # 20% overlap between tiles
        self.scan_step_ratio = 0.15  # Scan window moves by 15% of image
        
        print("Gender Detection Model loaded successfully.")
        print(f"Model classes: {self.model.names}")
        
        # Create OCR temp folder
        self.ocr_temp_folder = Path(__file__).parent.parent / "ocr_temp"
        self.ocr_temp_folder.mkdir(exist_ok=True)
        logger.info(f"OCR temp folder: {self.ocr_temp_folder}")
        
        # Check Tesseract for OCR
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is installed"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("✓ Tesseract OCR is available")
        except pytesseract.TesseractNotFoundError:
            logger.warning("⚠ Tesseract executable not found. Please install Tesseract OCR for gender text extraction.")
        except Exception as e:
            logger.warning(f"⚠ Error checking Tesseract: {e}")
    
    def _load_image(self, image_source):
        """
        Load image from file path or URL.
        
        Args:
            image_source: File path (str/Path) or URL (str)
        
        Returns:
            tuple: (numpy array, source_type, error_msg)
        """
        try:
            # Check if it's a URL
            if isinstance(image_source, str) and (image_source.startswith('http://') or image_source.startswith('https://')):
                # Download from URL
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    return None, 'url', f"Could not decode image from URL: {image_source}"
                return img, 'url', None
            else:
                # Load from file path
                img = cv2.imread(str(image_source))
                if img is None:
                    return None, 'file', f"Could not read image: {image_source}"
                return img, 'file', None
        
        except requests.RequestException as e:
            return None, 'url', f"Error downloading image: {str(e)}"
        except Exception as e:
            return None, None, f"Error loading image: {str(e)}"
    
    def create_tiles(self, img, zoom_factor=1.5):
        """
        Create overlapping tiles from an image for better detection at zoomed resolution.
        
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
        
        print(f"    [Tiling] Creating tiles with zoom={zoom_factor}x, tile_size={tile_w}x{tile_h}")
        
        # Generate tiles
        for y in range(0, h - tile_h + 1, step_h):
            for x in range(0, w - tile_w + 1, step_w):
                # Extract tile
                tile = img[y:y+tile_h, x:x+tile_w]
                
                # Resize to original resolution (zoom effect)
                zoomed_tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_LINEAR)
                
                tiles.append((zoomed_tile, x, y, zoom_factor))
        
        print(f"    [Tiling] Created {len(tiles)} tiles")
        return tiles
    
    def scan_image(self, img, window_size_ratio=0.5, step_ratio=0.15):
        """
        Scan image with a sliding window to find gender field.
        
        Args:
            img: Input image
            window_size_ratio: Size of scan window as ratio of image (0.5 = 50%)
            step_ratio: Step size as ratio of image (0.15 = 15%)
        
        Returns:
            List of (window_image, x_offset, y_offset, description)
        """
        h, w = img.shape[:2]
        
        # Calculate window size
        window_h = int(h * window_size_ratio)
        window_w = int(w * window_size_ratio)
        
        # Calculate step size
        step_h = int(h * step_ratio)
        step_w = int(w * step_ratio)
        
        print(f"    [Scan] Window size: {window_w}x{window_h}, Step: {step_w}x{step_h}")
        
        windows = []
        scan_num = 0
        
        # Scan horizontally and vertically
        for y in range(0, h - window_h + 1, step_h):
            for x in range(0, w - window_w + 1, step_w):
                window = img[y:y+window_h, x:x+window_w]
                # Resize to original resolution for consistent detection
                resized_window = cv2.resize(window, (w, h), interpolation=cv2.INTER_LINEAR)
                
                scan_num += 1
                description = f"scan_{scan_num}(pos:{x},{y})"
                windows.append((resized_window, x, y, description))
        
        print(f"    [Scan] Created {len(windows)} scan windows")
        return windows
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, osd_confidence_threshold: float = 0.5) -> Optional[Image.Image]:
        """
        Enhanced preprocessing and orientation correction (from EntityAgent).
        Analyzes letter shapes and text lines to determine correct orientation.
        """
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Image data for {entity_key} is empty")
                return None
            
            h, w = img.shape[:2]
            if h < 100:
                scale_factor = 100 / h
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_for_analysis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_for_analysis = img

            # Try letter-based orientation detection
            best_rotation = self._detect_orientation_by_letters(img_for_analysis, entity_key)
            
            if best_rotation is None:
                try:
                    osd = pytesseract.image_to_osd(img_for_analysis, output_type=pytesseract.Output.DICT)
                    if osd['orientation_conf'] > osd_confidence_threshold:
                        best_rotation = osd['rotate']
                        logger.info(f"  Using Tesseract OSD for {entity_key}: {best_rotation}° (conf: {osd['orientation_conf']:.2f})")
                    else:
                        best_rotation = 0
                except pytesseract.TesseractError as e:
                    logger.warning(f"  OSD failed for {entity_key}. Assuming 0° rotation.")
                    best_rotation = 0
            
            corrected_img = img
            if best_rotation != 0:
                logger.info(f"  Correcting {entity_key} orientation by {best_rotation}°")
                if best_rotation == 90: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif best_rotation == 180: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif best_rotation == 270: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr:
                logger.info(f"  Rotating vertical entity {entity_key} to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            # Convert to grayscale PIL image
            gray_cv = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
            return Image.fromarray(gray_cv)
            
        except Exception as e:
            logger.error(f"  Error during orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
        """
        Detect the correct orientation by analyzing letter shapes and OCR confidence.
        """
        try:
            rotations = [0, 90, 180, 270]
            rotation_scores = {}
            
            for rotation in rotations:
                if rotation == 0:
                    rotated_img = img
                elif rotation == 90:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                score = self._calculate_orientation_score(rotated_img, rotation)
                rotation_scores[rotation] = score
                logger.debug(f"    Rotation {rotation}°: score = {score:.3f}")
            
            best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
            best_score = rotation_scores[best_rotation]
            
            if best_score > 0.1:
                logger.info(f"  Letter-based orientation for {entity_key}: {best_rotation}° (score: {best_score:.3f})")
                return best_rotation
            else:
                return None
                
        except Exception as e:
            logger.warning(f"  Error in letter-based orientation detection: {e}")
            return None

    def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
        """
        Calculate a comprehensive score for how likely this orientation is correct.
        Combines OCR confidence, letter shapes, and text line analysis.
        """
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            ocr_score = self._get_ocr_confidence_score(gray)
            shape_score = self._analyze_letter_shapes(gray)
            line_score = self._analyze_text_lines(gray)
            
            total_score = (ocr_score * 0.5 + shape_score * 0.3 + line_score * 0.2)
            return total_score
            
        except Exception as e:
            logger.debug(f"    Error calculating orientation score: {e}")
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
        """
        Get OCR confidence and text quality score by trying multiple PSM modes.
        Uses PSM modes: 6, 7, 8, 13 (same as EntityAgent).
        """
        try:
            psm_modes = [6, 7, 8, 13]
            best_confidence = 0.0
            best_text_length = 0
            
            for psm in psm_modes:
                try:
                    data = pytesseract.image_to_data(gray_img, config=f'--psm {psm}', output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        text_length = sum(len(text.strip()) for text in data['text'] if text.strip())
                        
                        if avg_confidence > best_confidence or (avg_confidence == best_confidence and text_length > best_text_length):
                            best_confidence = avg_confidence
                            best_text_length = text_length
                            
                except pytesseract.TesseractError:
                    continue
            
            confidence_factor = best_confidence / 100.0
            length_factor = min(best_text_length / 10.0, 1.0)
            return confidence_factor * 0.7 + length_factor * 0.3
            
        except Exception:
            return 0.0

    def _analyze_letter_shapes(self, gray_img: np.ndarray) -> float:
        """
        Analyze the shapes of detected contours to determine if they look like upright letters.
        Same logic as EntityAgent.
        """
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            upright_score = 0.0
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8:
                    continue
                
                aspect_ratio = h / w
                if 0.3 <= aspect_ratio <= 4.0:
                    valid_contours += 1
                    if 1.0 <= aspect_ratio <= 2.5:
                        upright_score += 1.0
                    elif 0.5 <= aspect_ratio <= 3.5:
                        upright_score += 0.7
                    else:
                        upright_score += 0.3
            
            if valid_contours == 0:
                return 0.0
            return min(upright_score / valid_contours, 1.0)
            
        except Exception:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        """
        Analyze text line orientation using morphological operations.
        Horizontal text should have more horizontal lines than vertical lines.
        Same logic as EntityAgent.
        """
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            total_pixels = horizontal_pixels + vertical_pixels
            if total_pixels == 0:
                return 0.5
            
            horizontal_ratio = horizontal_pixels / total_pixels
            return horizontal_ratio
            
        except Exception:
            return 0.5
    
    def detect_gender(self, image_source):
        """
        Detect gender from an Aadhar card image with retry mechanism using scan + zoom.
        
        Strategy:
        1. Try full image detection
        2. If fails, scan image with sliding window
        3. If still fails, try zooming with tile detection
        4. Return best result found
        
        Args:
            image_source: Path to the Aadhar image file or URL (string or Path object)
        
        Returns:
            dict: {
                'success': bool,
                'gender': str or None,
                'confidence': float or None,
                'bbox': list or None  # [x1, y1, x2, y2]
                'method': str  # 'full_image', 'scan_X', 'zoom_1.5x', 'zoom_2.0x', or 'none'
            }
        """
        try:
            # Load image from path or URL
            img, source_type, error_msg = self._load_image(image_source)
            if img is None:
                return {
                    'success': False,
                    'gender': None,
                    'confidence': None,
                    'bbox': None,
                    'method': 'none',
                    'error': error_msg
                }
            
            # First attempt: Full image detection
            print("  [Detection] Running on full image...")
            results = self.model(img, verbose=False)
            
            best_gender = None
            max_conf = 0.0
            best_box = None
            detection_method = 'full_image'
            
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= self.conf_threshold and conf > max_conf:
                    cls_id = int(box.cls)
                    label = self.model.names[cls_id]
                    max_conf = conf
                    best_gender = label
                    best_box = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            print(f"  [Detection] Full image: max_conf={max_conf:.4f}")
            
            # If good detection on full image, return it
            if best_gender and max_conf >= self.conf_threshold:
                return {
                    'success': True,
                    'gender': best_gender,
                    'confidence': max_conf,
                    'bbox': best_box,
                    'method': detection_method
                }
            
            # If no detection or low confidence, try scanning
            if (best_gender is None or max_conf < self.retry_threshold):
                print(f"  [Retry] Low/no confidence ({max_conf:.4f}), trying image scanning...")
                
                scan_windows = self.scan_image(img, window_size_ratio=0.5, step_ratio=0.15)
                
                for scan_idx, (window_img, x_offset, y_offset, description) in enumerate(scan_windows):
                    results = self.model(window_img, verbose=False)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        if conf >= self.conf_threshold:
                            cls_id = int(box.cls)
                            label = self.model.names[cls_id]
                            
                            # Get coordinates from scanned window
                            tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Map back to original image coordinates
                            h, w = img.shape[:2]
                            window_h = int(h * 0.5)
                            window_w = int(w * 0.5)
                            
                            # Scale from full resolution back to original region
                            x1 = int((tx1 / w) * window_w) + x_offset
                            y1 = int((ty1 / h) * window_h) + y_offset
                            x2 = int((tx2 / w) * window_w) + x_offset
                            y2 = int((ty2 / h) * window_h) + y_offset
                            
                            # Clamp to image bounds
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))
                            
                            mapped_box = [x1, y1, x2, y2]
                            
                            if conf > max_conf:
                                max_conf = conf
                                best_gender = label
                                best_box = mapped_box
                                detection_method = description
                                print(f"  [Scan] Found at {description}: {label} @ {conf:.4f}")
                
                # If we found detection via scanning, return it
                if best_gender and max_conf >= self.conf_threshold:
                    return {
                        'success': True,
                        'gender': best_gender,
                        'confidence': max_conf,
                        'bbox': best_box,
                        'method': detection_method
                    }
            
            # If still no good detection, try zooming
            if (best_gender is None or max_conf < self.retry_threshold):
                print(f"  [Retry] Still no detection, trying zoom levels...")
                
                for zoom_level in self.zoom_levels:
                    print(f"  [Zoom] Trying {zoom_level}x zoom...")
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
                                
                                mapped_box = [x1, y1, x2, y2]
                                
                                if conf > max_conf:
                                    max_conf = conf
                                    best_gender = label
                                    best_box = mapped_box
                                    detection_method = f'zoom_{zoom_level}x'
                    
                    # If we found a good detection, stop trying
                    if best_gender and max_conf >= self.conf_threshold:
                        print(f"  [Zoom] Found detection at {zoom_level}x: {best_gender} @ {max_conf:.4f}")
                        break
            
            if best_gender:
                return {
                    'success': True,
                    'gender': best_gender,
                    'confidence': max_conf,
                    'bbox': best_box,
                    'method': detection_method
                }
            else:
                return {
                    'success': False,
                    'gender': None,
                    'confidence': None,
                    'bbox': None,
                    'method': 'none',
                    'error': f"No gender detected (tried: full image + scan + zoom levels {self.zoom_levels})"
                }
        
        except Exception as e:
            return {
                'success': False,
                'gender': None,
                'confidence': None,
                'bbox': None,
                'method': 'none',
                'error': str(e)
            }
    
    def extract_gender_text_from_bbox(self, img, bbox):
        """
        Extract and perform OCR on the detected gender region using EntityAgent settings.
        
        Steps:
        1. Crop the gender region from image
        2. Save cropped image to ocr_temp folder
        3. Apply multiple image enhancements
        4. Try multiple OCR configurations (PSM modes, preprocessing)
        5. Return best extracted text with confidence
        
        Args:
            img: Full image (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            dict: {
                'extracted_text': str,
                'raw_text': str,
                'ocr_confidence': float,
                'cropped_image_path': str,
                'psm_used': int,
                'preprocessing_method': str
            }
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Step 1: Crop the gender region
            cropped_image = img[y1:y2, x1:x2]
            if cropped_image is None or cropped_image.size == 0:
                logger.warning(f"  Could not crop gender region from bbox {bbox}")
                return {'extracted_text': '', 'ocr_confidence': 0.0, 'cropped_image_path': '', 'psm_used': 0}
            
            logger.info(f"  [OCR] Cropped gender region size: {cropped_image.shape}")
            
            # Step 2: Save original cropped image to ocr_temp folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cropped_image_filename = f"gender_cropped_{timestamp}.jpg"
            cropped_image_path = self.ocr_temp_folder / cropped_image_filename
            
            cv2.imwrite(str(cropped_image_path), cropped_image)
            logger.info(f"  [Save] Saved cropped gender image to: {cropped_image_path}")
            
            # Step 3: Apply orientation correction
            corrected_img = self._correct_entity_orientation_and_preprocess(
                cropped_image, 
                entity_key="gender_field"
            )
            
            if corrected_img is None:
                logger.warning(f"  OCR orientation correction failed")
                return {
                    'extracted_text': '', 
                    'ocr_confidence': 0.0, 
                    'cropped_image_path': str(cropped_image_path),
                    'raw_text': '',
                    'psm_used': 0,
                    'preprocessing_method': 'none'
                }
            
            # Convert PIL to cv2 for preprocessing
            corrected_cv2 = np.array(corrected_img)
            
            # Step 4: Try multiple preprocessing and PSM combinations
            best_text = ''
            best_confidence = 0.0
            best_psm = 6
            best_preprocessing = 'none'
            
            # Multiple preprocessing techniques
            preprocessing_methods = {
                'original': lambda img: img,
                'upscale_2x': lambda img: cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                'upscale_3x': lambda img: cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                'contrast_enhance': lambda img: self._enhance_contrast(img),
                'denoise': lambda img: cv2.fastNlMeansDenoising(img, h=10),
                'upscale_2x + contrast': lambda img: self._enhance_contrast(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
                'upscale_3x + contrast': lambda img: self._enhance_contrast(cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
                'denoise + contrast': lambda img: self._enhance_contrast(cv2.fastNlMeansDenoising(img, h=10)),
                'morphology': lambda img: self._apply_morphological_ops(img),
            }
            
            # PSM modes to try
            psm_modes = [6, 7, 8, 11, 13]  # Single block, single text, word, sparse text, raw
            
            logger.info(f"  [OCR] Testing {len(preprocessing_methods)} preprocessing methods × {len(psm_modes)} PSM modes...")
            
            for prep_name, prep_func in preprocessing_methods.items():
                try:
                    # Apply preprocessing
                    preprocessed = prep_func(corrected_cv2)
                    pil_img = Image.fromarray(preprocessed)
                    
                    for psm in psm_modes:
                        try:
                            # Extract text with this PSM
                            text = pytesseract.image_to_string(pil_img, lang='eng', config=f'--psm {psm}')
                            extracted_text = ' '.join(text.split()).strip().lower()
                            
                            # Get confidence
                            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                            ocr_confidence = (sum(confidences) / len(confidences)) / 100.0 if confidences else 0.0
                            
                            # Normalize gender
                            normalized = self._normalize_gender(extracted_text)
                            
                            # Check if this is a valid gender result
                            is_valid_gender = normalized in ['Male', 'Female']
                            
                            logger.debug(f"    {prep_name:20} + PSM {psm}: '{extracted_text}' → '{normalized}' (conf: {ocr_confidence:.2f})")
                            
                            # Prefer valid gender results, then by confidence
                            if is_valid_gender:
                                if ocr_confidence > best_confidence:
                                    best_text = normalized
                                    best_confidence = ocr_confidence
                                    best_psm = psm
                                    best_preprocessing = prep_name
                                    logger.info(f"    ✓ New best: {normalized} (PSM {psm}, {prep_name}, conf: {ocr_confidence:.2f})")
                            else:
                                # Still track non-gender results in case nothing valid is found
                                if best_text == '' and ocr_confidence > best_confidence:
                                    best_text = normalized
                                    best_confidence = ocr_confidence
                                    best_psm = psm
                                    best_preprocessing = prep_name
                        
                        except pytesseract.TesseractError as e:
                            logger.debug(f"    PSM {psm} failed: {str(e)[:50]}")
                            continue
                
                except Exception as e:
                    logger.debug(f"    Preprocessing {prep_name} failed: {str(e)[:50]}")
                    continue
            
            # Normalize final result
            normalized_gender = self._normalize_gender(best_text) if best_text else 'Other'
            
            logger.info(f"  [OCR] Best result: '{best_text}' → '{normalized_gender}'")
            logger.info(f"  [OCR] Configuration: PSM {best_psm}, {best_preprocessing}, Confidence: {best_confidence:.2f}")
            logger.info(f"  [OCR] Using cropped image: {cropped_image_path}")
            
            return {
                'extracted_text': normalized_gender,
                'raw_text': best_text,
                'ocr_confidence': best_confidence,
                'cropped_image_path': str(cropped_image_path),
                'psm_used': best_psm,
                'preprocessing_method': best_preprocessing
            }
        
        except Exception as e:
            logger.error(f"  Error extracting gender text from bbox: {e}")
            return {'extracted_text': '', 'ocr_confidence': 0.0, 'cropped_image_path': '', 'raw_text': '', 'psm_used': 0}
    
    def _enhance_contrast(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)
            return enhanced
        except Exception:
            return gray_img
    
    def _apply_morphological_ops(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to enhance text.
        """
        try:
            # Threshold
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return cleaned
        except Exception:
            return gray_img
    
    def _normalize_gender(self, text: str) -> str:
        """
        Normalize gender text to 'Male', 'Female', or 'Other'.
        """
        if not text:
            return ''
        
        text = text.strip().lower()
        # Remove special characters
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        
        # Check for male variations
        if 'male' in text and 'female' not in text:
            return 'Male'
        # Check for female variations
        elif 'female' in text or 'femal' in text:
            return 'Female'
        # Check for single letter variations
        elif text in ['m', 'man', 'boy']:
            return 'Male'
        elif text in ['f', 'woman', 'girl', 'femlae', 'femaie']:
            return 'Female'
        else:
            return text if text else 'Other'
        """
        Detect gender from a numpy array image.
        
        Args:
            img_array: Numpy array of the image (BGR format)
        
        Returns:
            dict: Same format as detect_gender()
        """
        try:
            if img_array is None or not isinstance(img_array, np.ndarray):
                return {
                    'success': False,
                    'gender': None,
                    'confidence': None,
                    'bbox': None,
                    'error': "Invalid image array"
                }
            
            # Run detection
            results = self.model(img_array, verbose=False)
            
            # Find best gender detection
            best_gender = None
            max_conf = 0.0
            best_box = None
            
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= self.conf_threshold and conf > max_conf:
                    cls_id = int(box.cls)
                    label = self.model.names[cls_id]
                    max_conf = conf
                    best_gender = label
                    best_box = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            if best_gender:
                return {
                    'success': True,
                    'gender': best_gender,
                    'confidence': max_conf,
                    'bbox': best_box
                }
            else:
                return {
                    'success': False,
                    'gender': None,
                    'confidence': None,
                    'bbox': None,
                    'error': f"No gender detected above threshold {self.conf_threshold}"
                }
        
        except Exception as e:
            return {
                'success': False,
                'gender': None,
                'confidence': None,
                'bbox': None,
                'error': str(e)
            }
    
    def visualize_detection(self, image_source, output_path=None, show=True):
        """
        Detect gender, extract OCR text, and visualize the result with bounding box.
        
        Args:
            image_source: Path to the Aadhar image or URL
            output_path: Optional path to save the annotated image
            show: Whether to display the image (default: True)
        
        Returns:
            dict: Detection and OCR results
        """
        # Load image from path or URL
        img, source_type, error_msg = self._load_image(image_source)
        if img is None:
            print(f"Error: {error_msg}")
            return None
        
        # Run detection with retry
        result = self.detect_gender(image_source)
        
        # Draw results
        if result['success']:
            gender = result['gender']
            conf = result['confidence']
            bbox = result['bbox']
            method = result['method']
            
            # Extract OCR text from the detected region
            ocr_result = self.extract_gender_text_from_bbox(img, bbox)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with both detection and OCR info
            ocr_text = ocr_result.get('extracted_text', 'N/A')
            ocr_conf = ocr_result.get('ocr_confidence', 0.0)
            label = f"{ocr_text} | Det:{conf:.2f} OCR:{ocr_conf:.2f} ({method})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            print(f"✓ Detected: {gender} (confidence: {conf:.2f}) via {method}")
            print(f"✓ OCR Extracted: '{ocr_result.get('raw_text', '')}' → '{ocr_text}' (OCR confidence: {ocr_conf:.2f})")
            print(f"✓ OCR Configuration: PSM {ocr_result.get('psm_used', '?')}, {ocr_result.get('preprocessing_method', 'unknown')}")
            print(f"✓ Cropped image saved to: {ocr_result.get('cropped_image_path', 'N/A')}")
            
            result['ocr'] = ocr_result
        else:
            print(f"✗ {result.get('error', 'No detection')}")
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), img)
            print(f"Saved annotated image to: {output_path}")
        
        # Display if requested
        if show:
            self._display_image(img, result)
        
        return result
    
    def _display_image(self, img, result):
        """
        Display image with detection and OCR results using matplotlib.
        
        Args:
            img: Image array (BGR format)
            result: Detection result dict
        """
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with title
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Add result text
        if result['success']:
            method = result.get('method', 'unknown')
            ocr_data = result.get('ocr', {})
            ocr_text = ocr_data.get('extracted_text', 'N/A')
            ocr_conf = ocr_data.get('ocr_confidence', 0.0)
            cropped_path = ocr_data.get('cropped_image_path', '')
            psm = ocr_data.get('psm_used', '?')
            preprocessing = ocr_data.get('preprocessing_method', 'unknown')
            
            title = f"Detection: {result['gender']} ({result['confidence']:.2%}) | OCR: '{ocr_text}' ({ocr_conf:.2%})\nMethod: {method} | PSM: {psm} | Preprocessing: {preprocessing}\nCropped: {Path(cropped_path).name if cropped_path else 'N/A'}"
        else:
            title = f"No Detection - {result.get('error', 'Unknown error')}"
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize detector
    detector = GenderDetector(model_path="models/gender_detection.pt")
    
    # Test with an image or URL
    if len(sys.argv) > 1:
        image_source = sys.argv[1]
        print(f"\nTesting gender detection on: {image_source}")
        
        # Simple detection (with auto retry via zoom)
        result = detector.detect_gender(image_source)
        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Gender: {result['gender']}")
        print(f"  Confidence: {result['confidence']:.4f}" if result['confidence'] else "  Confidence: None")
        print(f"  Detection Method: {result['method']}")
        print(f"  Error: {result.get('error', 'None')}")
        
        # Visualize with display
        if result['success']:
            print("\nDisplaying image with detection...")
            detector.visualize_detection(image_source, show=True)
    else:
        print("\n" + "="*70)
        print("Gender Detection from Aadhar Images (with Auto-Zoom Retry)")
        print("="*70)
        print("\nUsage:")
        print("  python gender_detection.py <image_path_or_url>")
        print("\nFeatures:")
        print("  • Detects gender field on full image")
        print("  • If detection fails, scans image with sliding window (left→right, top→bottom)")
        print("  • If still fails, tries 1.5x and 2.0x zoom levels")
        print("  • Displays result with bounding box and detection method")
        print("\nExamples:")
        print("  # From file:")
        print("  python gender_detection.py images/aadhar_front.jpg")
        print("\n  # From URL:")
        print("  python gender_detection.py https://example.com/aadhar.jpg")
        print("\n" + "="*70)
