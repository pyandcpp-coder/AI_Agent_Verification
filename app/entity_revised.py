import logging
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityAgent:
    def __init__(self, model1_path="models/best4.pt", model2_path="models/best.pt", other_lang_code='hin+tel+ben'):
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")

        self.model1_path = model1_path
        self.model2_path = model2_path
        
        logger.info("Checking for YOLO models on filesystem...")
        if not Path(self.model1_path).exists():
             if Path(f"../{self.model1_path}").exists():
                 self.model1_path = f"../{self.model1_path}"
             else:
                 logger.warning(f"Model1 not found at {self.model1_path}. Assuming it will be available at runtime.")

        if not Path(self.model2_path).exists():
             if Path(f"../{self.model2_path}").exists():
                 self.model2_path = f"../{self.model2_path}"
             else:
                 logger.warning(f"Model2 not found at {self.model2_path}. Assuming it will be available at runtime.")

        logger.info("Loading models directly from filesystem...")
        try:
            self.model1 = YOLO(self.model1_path)
            self.model2 = YOLO(self.model2_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
            # Do not raise here to allow main.py to handle initialization errors if needed
            
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("YOLOv8 models loaded successfully.")
        
        # Safely get names if models loaded
        if hasattr(self, 'model1'):
            self.card_classes = {i: name for i, name in self.model1.names.items()}
        if hasattr(self, 'model2'):
            self.entity_classes = {
                0: 'aadharnumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
                4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
                8: 'name', 9: 'name_otherlang', 10: 'pincode'
            }
        
        logger.info(f"Pipeline initialized to use '{self.other_lang_code}' for other language fields.")

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    def detect_and_crop_cards(self, image_paths: List[str], confidence_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Step 1: Detect Aadhaar front/back cards and pass cropped image data (np.array) forward."""
        logger.info(f"\nStep 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}
        for image_path in image_paths:
            logger.info(f"  Processing: {Path(image_path).name}")
            results = self.model1(str(image_path), device=self.device, verbose=False)
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem
            detected = False
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")
                if class_name == 'print_aadhar':
                    raise ValueError("print_aadhar_detected")
                if class_name not in ['aadhar_front', 'aadhar_back']: continue
                detected = True
                crop = img[y1:y2, x1:x2]
                card_type = class_name.replace('aadhar_', '')
                card_data = {
                    "image": crop,
                    "name": f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}",
                    "type": card_type
                }
                cropped_cards[card_type].append(card_data)
                logger.info(f" Detected {class_name}")
            if not detected:
                logger.warning(f" No Aadhaar card detected in {Path(image_path).name}.")
        return cropped_cards

    def detect_entities_in_cards(self, cropped_cards: Dict[str, List[Dict[str, Any]]], confidence_threshold: float):
        """Step 2: Detect entities using in-memory card image data."""
        all_card_data = cropped_cards.get('front', []) + cropped_cards.get('back', [])
        logger.info(f"\nStep 2: Detecting entities in {len(all_card_data)} cards (Threshold: {confidence_threshold})")
        all_detections = {}

        for card_info in all_card_data:
            card_name = card_info['name']
            img = card_info['image']
            card_type = card_info['type']
            logger.info(f"  Processing: {card_name}")
            
            results = self.model2(img, device=self.device, verbose=False)
            card_detections = []
            
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                card_detections.append({'bbox': (x1, y1, x2, y2), 'class_name': class_name, 'confidence': float(box.conf[0])})
                
            logger.info(f"  Detected {len(card_detections)} entities in {card_name}")
            all_detections[card_name] = {
                "card_image": img,
                "card_type": card_type,
                "detections": card_detections
            }
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 3: Crop individual entities and add their image data to the detection dictionary."""
        logger.info(f"\nStep 3: Cropping individual entities")
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                
                # Bounds checking
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                entity_key = f"{card_name}_{detection['class_name']}_{i}"
                detection['entity_key'] = entity_key
                logger.info(f"   Cropped entity: {entity_key}")
        return all_detections
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, osd_confidence_threshold: float = 0.5) -> Optional[Image.Image]:
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Entity image data for {entity_key} is empty, skipping.")
                return None
            
            h, w = img.shape[:2]
            if h < 100:
                scale_factor = 100 / h
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_for_analysis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_for_analysis = img

            best_rotation = self._detect_orientation_by_letters(img_for_analysis, entity_key)
            
            if best_rotation is None:
                try:
                    osd = pytesseract.image_to_osd(img_for_analysis, output_type=pytesseract.Output.DICT)
                    if osd['orientation_conf'] > osd_confidence_threshold:
                        best_rotation = osd['rotate']
                        logger.info(f" Using Tesseract OSD for {entity_key}: {best_rotation}° (conf: {osd['orientation_conf']:.2f})")
                    else:
                        best_rotation = 0
                except pytesseract.TesseractError as e:
                    logger.warning(f" Both letter-based and OSD failed for {entity_key}. Assuming 0° rotation. Details: {e}")
                    best_rotation = 0
            
            corrected_img = img
            if best_rotation != 0:
                logger.info(f"   Correcting entity {entity_key} orientation by {best_rotation}°")
                if best_rotation == 90: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif best_rotation == 180: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif best_rotation == 270: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in entity_key:
                logger.info(f"   Rotating vertical entity {entity_key} to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            pil_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            return pil_img
            
        except Exception as e:
            logger.error(f"   Unhandled error during entity orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
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
                logger.debug(f"      Rotation {rotation}°: score = {score:.3f}")
            
            best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
            best_score = rotation_scores[best_rotation]
            
            if best_score > 0.1:
                logger.info(f" Letter-based analysis for {entity_key}: {best_rotation}° (score: {best_score:.3f})")
                return best_rotation
            else:
                logger.warning(f"   Letter-based analysis inconclusive for {entity_key} (best score: {best_score:.3f})")
                return None
                
        except Exception as e:
            logger.warning(f"  Error in letter-based orientation detection for {entity_key}: {e}")
            return None

    def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
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
            logger.debug(f"      Error calculating orientation score: {e}")
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
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
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.0
            
            upright_score = 0.0
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20: continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8: continue
                
                aspect_ratio = h / w
                if 0.3 <= aspect_ratio <= 4.0:
                    valid_contours += 1
                    if 1.0 <= aspect_ratio <= 2.5: upright_score += 1.0
                    elif 0.5 <= aspect_ratio <= 3.5: upright_score += 0.7
                    else: upright_score += 0.3
            
            if valid_contours == 0: return 0.0
            return min(upright_score / valid_contours, 1.0)
            
        except Exception:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            total_pixels = horizontal_pixels + vertical_pixels
            if total_pixels == 0: return 0.5
            
            horizontal_ratio = horizontal_pixels / total_pixels
            return horizontal_ratio
            
        except Exception:
            return 0.5

    def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
        logger.info(f"\nStep 4: Correcting Entity Orientation & Performing Multi-Language OCR")
        ocr_results = {}
        for card_name, card_data in all_detections.items():
            for detection in card_data['detections']:
                cropped_image = detection.get('cropped_image')
                entity_key = detection.get('entity_key')
                class_name = detection.get('class_name')

                if cropped_image is None or entity_key is None: continue

                logger.info(f"  Processing entity: {entity_key} (Class: {class_name})")
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_key] = extracted_text
                    except Exception as e:
                        logger.error(f" OCR failed for {entity_key}: {e}")
                        ocr_results[entity_key] = None
        return ocr_results

    def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
        logger.info("\nStep 5: Organizing final results")
        organized_results = {
            'front': {}, 'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold
            }
        }
        for card_name, card_data in all_detections.items():
            card_type = card_data['card_type']
            
            if card_name not in organized_results[card_type]:
                organized_results[card_type][card_name] = {'entities': {}}
            
            for detection in card_data['detections']:
                 entity_name = detection['class_name']
                 entity_key = detection.get('entity_key')
                 extracted_text = ocr_results.get(entity_key)

                 if entity_name not in organized_results[card_type][card_name]['entities']:
                      organized_results[card_type][card_name]['entities'][entity_name] = []
                 
                 organized_results[card_type][card_name]['entities'][entity_name].append({
                     'confidence': detection['confidence'], 
                     'bbox': detection['bbox'],
                     'extracted_text': extracted_text
                 })
        return organized_results
            
    def process_images(self, image_paths, user_id: str, task_id: str, confidence_threshold: float, verbose=True):
        """Main pipeline function to process images in-memory without saving any intermediate or final files."""
        try:
            if verbose:
                logger.info(f"[user_id={user_id}] Starting In-Memory Pipeline for task {task_id}")

            # Step 1: Detect and Crop Cards
            cropped_cards = self.detect_and_crop_cards(image_paths, confidence_threshold)
            if not cropped_cards.get('front', []) and not cropped_cards.get('back', []):
                logger.warning(f"[user_id={user_id}] No Aadhaar card detected in any provided images.")
                return {'error': 'no_aadhar_detected', 'success': False}

            # Step 2: Detect Entities
            all_detections = self.detect_entities_in_cards(cropped_cards, confidence_threshold)

            # Step 3: Crop Entities
            self.crop_entities(all_detections)

            # Step 4: OCR
            ocr_results = self.perform_multi_language_ocr(all_detections)

            # Step 5: Organize
            organized_results = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)

            # Step 6: Extract Main Fields
            final_data = self.extract_main_fields(organized_results)

            if verbose:
                logger.info(f"[user_id={user_id}] Pipeline processing completed successfully in memory.")
            
            return {'organized_results': organized_results, 'data': final_data, 'success': True}

        except ValueError as ve:
            logger.error(f"[user_id={user_id}] SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection', 'success': False}
        except Exception as e:
            logger.error(f"[user_id={user_id}] Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown', 'success': False}

    def extract_main_fields(self, organized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts only Aadhaar Number, DOB, and Gender."""
        fields = ['aadharnumber', 'dob', 'gender']
        data = {key: "" for key in fields}
        
        for side in ['front', 'back']:
            for card in organized_results.get(side, {}).values():
                for field in fields:
                    if field in card['entities'] and card['entities'][field]:
                        all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                        first_valid_text = next((text for text in all_texts if text), '')
                        if first_valid_text:
                            data[field] = first_valid_text
                            
        if data.get('aadharnumber'):
            aad = data['aadharnumber']
            # detect masked aadhaar
            if re.search(r'X{4}', aad, re.IGNORECASE):
                data['aadhar_status'] = "masked_aadhar" # Avoid returning simple string, update dict
            
            # clean spaces for real aadhaar
            data['aadharnumber'] = aad.replace(" ", "")
        
        if data.get('dob'):
            digit_groups = re.findall(r'\d+', data['dob'])
            digits_only = ''.join(digit_groups)
            if len(digits_only) == 8:
                try:
                    parsed_date = datetime.strptime(digits_only, '%d%m%Y')
                    data['dob'] = parsed_date.strftime('%d-%m-%Y')
                except ValueError:
                    data['dob'] = 'Invalid Format'
            else:
                year = next((g for g in digit_groups if len(g) == 4), None)
                if year:
                    data['dob'] = year
                else:
                    data['dob'] = 'Invalid Format'
        
        # Gender normalization
        if data['gender']:
            gender = data['gender'].strip().lower()
            if gender == 'male':
                data['gender'] = 'Male'
            elif gender == 'female':
                data['gender'] = 'Female'
            else:
                data['gender'] = 'Other'
            
        return data

if __name__ == "__main__":
    try:
        agent = EntityAgent(model1_path="models/best4.pt", model2_path="models/best.pt")
        print("✅ Entity Agent initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")