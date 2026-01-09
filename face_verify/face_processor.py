# Enhanced face_processor.py with multiple face detection
import logging
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace

import config

logger = logging.getLogger(__name__)

# Custom Exception for Multiple Faces
class MultipleFacesError(ValueError):
    """Custom exception raised when multiple prominent faces are detected."""
    def __init__(self, message, face_count, faces_info):
        super().__init__(message)
        self.face_count = face_count
        self.faces_info = faces_info

# Custom Exception for Blurry Images
class BlurryImageError(ValueError):
    """Custom exception raised when an image's blurriness is below the threshold."""
    def __init__(self, message, score):
        super().__init__(message)
        self.score = score

def calculate_face_area(facial_area):
    """Calculate the area of a face bounding box."""
    return facial_area['w'] * facial_area['h']

def calculate_image_area(img_array):
    """Calculate the total image area."""
    return img_array.shape[0] * img_array.shape[1]

def is_face_prominent(face, img_array, min_size_ratio=0.02, min_confidence=0.7):
    """
    Determine if a face is prominent enough to be considered a main subject.
    
    Args:
        face: Face object from DeepFace
        img_array: Original image array
        min_size_ratio: Minimum ratio of face area to image area (default 2%)
        min_confidence: Minimum confidence score for prominence
    
    Returns:
        bool: True if face is prominent
    """
    facial_area = face.get('facial_area', {})
    confidence = face.get('confidence', 0.0)
    
    # Calculate face area ratio
    face_area = calculate_face_area(facial_area)
    image_area = calculate_image_area(img_array)
    area_ratio = face_area / image_area
    
    # For dlib (no confidence), use only size
    if confidence == 0.0:  # dlib case
        return area_ratio >= min_size_ratio
    
    # For other detectors, use both confidence and size
    is_confident = confidence >= min_confidence
    is_large_enough = area_ratio >= min_size_ratio
    
    logger.debug(f"Face analysis - Area ratio: {area_ratio:.4f}, Confidence: {confidence:.4f}, "
                f"Is confident: {is_confident}, Is large enough: {is_large_enough}")
    
    return is_confident and is_large_enough

def analyze_faces_in_image(img_array, detection_backends=None):
    """
    Analyze all faces in an image and categorize them as prominent or background.
    
    Args:
        img_array: Image array
        detection_backends: List of detection backends to try
    
    Returns:
        tuple: (all_faces, prominent_faces, background_faces)
    """
    if detection_backends is None:
        detection_backends = [config.PRIMARY_DETECTOR, config.SECONDARY_DETECTOR, config.TERTIARY_DETECTOR]
    
    all_faces = []
    
    for backend in detection_backends:
        try:
            logger.info(f"Analyzing faces with '{backend}'...")
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend=backend,
                enforce_detection=False  # Don't enforce to get all faces
            )
            
            if faces:
                # Add backend info to each face
                for face in faces:
                    face['detection_backend'] = backend
                all_faces = faces
                break
                
        except Exception as e:
            logger.warning(f"Face analysis failed with '{backend}': {e}")
            continue
    
    if not all_faces:
        return [], [], []
    
    # Categorize faces
    prominent_faces = []
    background_faces = []
    
    for face in all_faces:
        if is_face_prominent(face, img_array):
            prominent_faces.append(face)
        else:
            background_faces.append(face)
    
    logger.info(f"Face analysis complete - Total: {len(all_faces)}, "
               f"Prominent: {len(prominent_faces)}, Background: {len(background_faces)}")
    
    return all_faces, prominent_faces, background_faces

def is_image_blurry(image_array: np.ndarray):
    """
    Detects if an image is too blurry by calculating the variance of the Laplacian.
    This should be run on a CROPPED FACE, not the whole image.
    """
    # Convert to grayscale for blur detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.info(f"Image blur score (on face crop): {laplacian_var:.2f} (Threshold: {config.BLUR_THRESHOLD})")
    
    if laplacian_var < config.BLUR_THRESHOLD:
        return True, laplacian_var
    return False, laplacian_var

def resize_image(image: Image.Image, max_size=1024) -> Image.Image:
    """Resizes an image to a maximum dimension while preserving aspect ratio."""
    if image.width > max_size or image.height > max_size:
        logger.info(f"Image is large ({image.width}x{image.height}), resizing to max {max_size}px.")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def get_embedding(image_bytes: bytes):
    """
    Generates a facial embedding with enhanced multiple face detection:
    1. Resize image.
    2. Analyze all faces in the image.
    3. Check for multiple prominent faces.
    4. If single prominent face, proceed with processing.
    5. Run blur detection on the selected face.
    6. Generate embedding.
    """
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError("Invalid or corrupted image file.")

    # Step 1: Resize image for normalization
    pil_image = resize_image(pil_image)
    img_array = np.array(pil_image)
    
    # Step 2: Analyze all faces in the image
    detection_backends = [config.PRIMARY_DETECTOR, config.SECONDARY_DETECTOR, config.TERTIARY_DETECTOR]
    all_faces, prominent_faces, background_faces = analyze_faces_in_image(img_array, detection_backends)
    
    if not all_faces:
        raise ValueError("No face detected with sufficient confidence across all detectors.")
    
    # Step 3: Check for multiple prominent faces
    if len(prominent_faces) > 1:
        faces_info = []
        for i, face in enumerate(prominent_faces):
            facial_area = face.get('facial_area', {})
            confidence = face.get('confidence', 0.0)
            area = calculate_face_area(facial_area)
            faces_info.append({
                'face_index': i + 1,
                'confidence': round(confidence, 4),
                'area': area,
                'position': {
                    'x': facial_area.get('x', 0),
                    'y': facial_area.get('y', 0),
                    'width': facial_area.get('w', 0),
                    'height': facial_area.get('h', 0)
                }
            })
        
        message = f"Multiple prominent faces detected ({len(prominent_faces)} faces). Please ensure only one person is clearly visible in the selfie."
        raise MultipleFacesError(
            message=message,
            face_count=len(prominent_faces),
            faces_info=faces_info
        )
    
    if len(prominent_faces) == 0:
        raise ValueError("No prominent face detected. Please ensure your face is clearly visible and takes up a reasonable portion of the image.")
    
    # Step 4: Use the single prominent face
    selected_face = prominent_faces[0]
    facial_area = selected_face['facial_area']
    
    # Step 5: Blur detection on the cropped face
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    face_crop = img_array[y:y+h, x:x+w]
    
    is_blurry, score = is_image_blurry(face_crop)
    if is_blurry:
        message = f"Image is too blurry to process (score: {score:.2f}, threshold: {config.BLUR_THRESHOLD})."
        raise BlurryImageError(message=message, score=score)

    # Step 6: Generate the embedding
    try:
        embedding_obj = DeepFace.represent(
            img_path=img_array,
            model_name=config.MODEL_NAME,
            detector_backend=selected_face['detection_backend'],
            enforce_detection=False
        )
        
        logger.info(f"Successfully processed image with {len(all_faces)} total faces "
                   f"({len(prominent_faces)} prominent, {len(background_faces)} background)")
        
        return embedding_obj[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding generation failed after successful detection: {e}")
        raise ValueError("Face was detected, but embedding could not be generated.")