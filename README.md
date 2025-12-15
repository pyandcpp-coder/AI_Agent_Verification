# AI Agent Verification System

An automated identity verification system that validates using computer vision, OCR, and face recognition technology. The system performs multi-stage verification including document detection, entity extraction, face similarity matching, and compliance scoring.

## Overview

This system automates the Aadhaar verification process by analyzing front and back Aadhaar card images along with a selfie photo. It validates the authenticity of documents, extracts key information, and provides an approval decision based on a comprehensive scoring mechanism.

## Features

- **Document Verification**: Detects and validates front and back Aadhaar cards using YOLOv8
- **Face Similarity Matching**: Compares selfie with Aadhaar photo using InsightFace (97%+ accuracy)
- **Entity Extraction**: OCR-based extraction of Aadhaar number, DOB, and gender
- **Multi-Language Support**: Supports Hindi, Telugu, and Bengali for regional text
- **Age Verification**: Automatic age calculation and 18+ validation
- **Gender Detection**: Dual-method gender verification (OCR + specialized ML pipeline)
- **Smart Scoring System**: Weighted scoring with approval thresholds
- **Cloudflare Bypass**: Downloads images from protected URLs using cloudscraper

## Architecture

### Verification Pipeline

```
Input (URLs/Paths) → Document Detection → Entity Extraction → Face Matching → Scoring → Decision
                           ↓                    ↓                  ↓           ↓
                    Front/Back Check      Aadhaar/DOB/Gender   Similarity    APPROVED
                                                                Score         REVIEW
                                                                              REJECTED
```

### Scoring System

| Component | Weight | Description |
|-----------|--------|-------------|
| Face Similarity | 40 points | Selfie vs Aadhaar photo match (16-100% range) |
| Aadhaar Validity | 20 points | Valid 12-digit number & proper masking |
| DOB/Age Check | 20 points | Valid age (18+) & DOB match |
| Gender Match | 20 points | Gender consistency across inputs |
| **Total** | **100 points** | |

### Decision Thresholds

- **APPROVED** (Score ≥ 65): Verification successful
- **REVIEW** (40 ≤ Score < 65): Manual review required
- **REJECTED** (Score < 40): Verification failed

**Critical Failures** (Automatic Rejection):
- DOB mismatch between input and Aadhaar
- Gender mismatch between input and detected gender
- Age under 18
- Document detection failure
- Gender verification failure

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Tesseract OCR installed on system

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-ben
sudo apt-get install libgl1-mesa-glx
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang
```

**Windows:**
- Download and install Tesseract from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
- Add Tesseract to PATH

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Model Setup

Place the following pre-trained models in the `models/` directory:

- `best.pt` - Entity detection model (YOLOv8)
- `best4.pt` - Document detection model (YOLOv8)
- InsightFace models (downloaded automatically on first run)

## Usage

### Starting the Server

```bash
python main.py
```

Server runs on `http://localhost:8000`

### API Endpoints

#### 1. Full Verification (Development/Testing)

**Endpoint:** `POST /verification/verify`

**Request Body:**
```json
{
  "user_id": "12345",
  "dob": "15-08-1995",
  "passport_first": "https://example.com/aadhaar_front.jpg",
  "passport_old": "https://example.com/aadhaar_back.jpg",
  "selfie_photo": "https://example.com/selfie.jpg",
  "gender": "Male"
}
```

**Response:**
```json
{
  "user_id": "12345",
  "final_decision": "APPROVED",
  "status_code": 2,
  "score": 78.5,
  "breakdown": {
    "face_score": 35.2,
    "aadhar_score": 20,
    "dob_score": 20,
    "gender_score": 20
  },
  "extracted_data": {
    "aadhaar": "123456789012",
    "dob": "1995",
    "gender": "Male"
  },
  "input_data": {
    "dob": "15-08-1995",
    "gender": "Male"
  },
  "rejection_reasons": []
}
```

#### 2. Production Verification (Streamlined)

**Endpoint:** `POST /verification/verify/agent/`

**Request Body:** Same as above

**Response (Minimal):**
```json
{
  "user_id": "12345",
  "final_decision": "APPROVED",
  "status_code": 2,
  "score": 78.5,
  "breakdown": {
    "face_score": 35.2,
    "aadhar_score": 20,
    "dob_score": 20,
    "gender_score": 20
  },
  "extracted_data": {
    "aadhaar": "123456789012",
    "dob": "1995",
    "gender": "Male"
  },
  "rejection_reasons": []
}
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | int/string | Yes | Unique identifier for the user |
| `dob` | string | No | Date of birth (dd-mm-yyyy format) |
| `passport_first` | string | Yes | Aadhaar front image (URL or local path) |
| `passport_old` | string | Yes | Aadhaar back image (URL or local path) |
| `selfie_photo` | string | Yes | User selfie (URL or local path) |
| `gender` | string | No | Expected gender (Male/Female/Other) |

### Status Codes

- `2` - APPROVED
- `1` - REJECTED
- `0` - REVIEW

## Example Usage

### Python Client

```python
import requests

url = "http://localhost:8000/verification/verify"
payload = {
    "user_id": "USR001",
    "dob": "15-08-1995",
    "passport_first": "https://example.com/front.jpg",
    "passport_old": "https://example.com/back.jpg",
    "selfie_photo": "https://example.com/selfie.jpg",
    "gender": "Male"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Decision: {result['final_decision']}")
print(f"Score: {result['score']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/verification/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR001",
    "dob": "15-08-1995",
    "passport_first": "https://example.com/front.jpg",
    "passport_old": "https://example.com/back.jpg",
    "selfie_photo": "https://example.com/selfie.jpg",
    "gender": "Male"
  }'
```

## Verification Logic

### 1. Document Detection Stage
- Validates presence of front and back Aadhaar cards
- Uses YOLO-based detection with retry mechanism
- Implements zoom-and-tile strategy for low-confidence detections
- **Fail Fast**: Rejects immediately if documents aren't detected

### 2. Entity Extraction Stage
- Detects and crops Aadhaar number, DOB, and gender fields
- Applies orientation correction for rotated text
- Performs multi-language OCR (English + Hindi/Telugu/Bengali)
- Cross-validates Aadhaar number from both front and back

### 3. Face Similarity Stage
- Extracts face embeddings using InsightFace (buffalo_l model)
- Computes cosine similarity between selfie and Aadhaar photo
- Scales score from 16-100% to 0-40 points (below 16% = 0 points)

### 4. Compliance Checks
- **Aadhaar Validation**: 12-digit number, no masking
- **Age Verification**: Extracts birth year, validates 18+
- **DOB Match**: If provided, must match extracted DOB
- **Gender Match**: If provided, must match detected gender

### 5. Scoring & Decision
- Aggregates all component scores
- Applies critical failure overrides
- Returns final decision with detailed breakdown

## Project Structure

```
project/
├── main.py                 # FastAPI server & orchestration
├── scoring.py              # Scoring logic & decision rules
├── requirements.txt        # Python dependencies
├── models/
│   ├── best.pt            # Entity detection model
│   └── best4.pt           # Document detection model
├── app/
│   ├── face_sim.py        # Face similarity (InsightFace)
│   ├── fb_detect.py       # Document detection (YOLO)
│   ├── entity.py          # Entity extraction (YOLO + OCR)
│   └── gender_pipeline.py # Specialized gender detection
└── temp/                  # Temporary file storage (auto-cleanup)
```

## Key Components

### FaceAgent (`face_sim.py`)
- InsightFace-based face comparison
- 640×640 detection size for optimal accuracy
- Returns similarity score (0-100%)

### DocAgent (`fb_detect.py`)
- YOLOv8-based Aadhaar card detection
- Multi-scale detection with zoom levels
- Overlapping tile strategy for partial cards

### EntityAgent (`entity.py`)
- YOLOv8 entity localization
- Tesseract OCR with orientation correction
- Multi-language support (English + Indian languages)
- Smart Aadhaar number reconciliation from front/back

### GenderPipeline (`gender_pipeline.py`)
- Fallback gender detection when OCR fails
- Specialized ML model for gender classification
- Integrates with face detection pipeline

### VerificationScorer (`scoring.py`)
- Weighted scoring system
- Critical failure handling
- Threshold-based decision logic

## Performance Optimization

- **Async Processing**: Parallel execution of face matching and document processing
- **Model Caching**: All models loaded once at startup
- **Temporary Files**: Automatic cleanup after processing
- **GPU Acceleration**: CUDA support for faster inference
- **Cloudflare Bypass**: Handles protected image URLs

## Error Handling

The system handles various failure scenarios:

- **File Retrieval Failure**: Invalid URLs or paths
- **Document Not Detected**: Missing front/back cards
- **No Face Detected**: Selfie or Aadhaar photo issues
- **OCR Failure**: Unreadable text or poor image quality
- **Validation Errors**: Invalid Aadhaar format, age issues, mismatches

## Limitations

- Requires clear, well-lit images
- Masked Aadhaar cards may be flagged
- OCR accuracy depends on image quality
- Face similarity threshold may need tuning
- GPU recommended for production workloads

## Security Considerations

- Temporary files are automatically deleted after processing
- No data persistence (stateless API)
- Supports both URLs and local paths for flexibility
- Validates file types and image formats

## Troubleshooting

### Common Issues

**Tesseract not found:**
```bash
# Check installation
tesseract --version

# Verify language packs
tesseract --list-langs
```

**CUDA errors:**
- Verify CUDA toolkit installation
- Check PyTorch CUDA compatibility
- System falls back to CPU automatically

**Model loading failures:**
- Ensure model files exist in `models/` directory
- Check file permissions
- Verify model file integrity

**Low accuracy:**
- Improve image quality
- Ensure proper lighting
- Use higher resolution images
- Check for document orientation issues

## Contributing

Contributions are welcome! Areas for improvement:

- Additional document types support
- Enhanced OCR preprocessing
- Improved gender detection accuracy
- Multi-card verification
- Liveness detection integration

## License

[Specify your license here]

## Contact

[Add contact information or support details]

---

**Note**: This system is designed for educational and development purposes. Ensure compliance with local data protection and privacy regulations when handling identity documents.
=======
# Automated Identity Verification System (KYC)

This project provides a comprehensive, automated **Know Your Customer (KYC)** verification API. It utilizes advanced Deep Learning models to verify user identity by analyzing Aadhaar cards and matching them against live selfies.

The system is built with **FastAPI** and integrates **YOLOv8** for object detection, **InsightFace** for facial recognition, and **Tesseract OCR** for text extraction.

---

## 🚀 Key Features

*   **Document Validation**: Automatically detects and validates Aadhaar Front and Back sides.
*   **Face Verification**: Compares the user's selfie with the photo extracted from the ID card using ArcFace.
*   **Data Extraction**: Extracts Aadhaar Number, Date of Birth (DOB), and Gender.
*   **Automated Scoring**: Assigns a trust score (0-100) based on weighted verification metrics.
*   **Asynchronous Processing**: Handles file downloads and heavy inference tasks efficiently.

---

## ⚙️ How It Works

The system follows a strict pipeline to validate a user:

1.  **Document Detection**: Checks if `passport_first` (Front) and `passport_old` (Back) are valid Aadhaar cards. **If this fails, the request is immediately rejected.**
2.  **Parallel Processing**:
    *   **Face Matching**: Calculates similarity between the Selfie and the ID Photo.
    *   **Entity Extraction**: OCRs the card to extract Name, DOB, Gender, and Aadhaar Number.
3.  **Cross-Verification**: Compares extracted data against user input (DOB, Gender).
4.  **Final Scoring**: Calculates a weighted score to determine the final status.

---

## 📊 Scoring & Decision Logic

The system calculates a **Total Score** based on four weighted criteria. The final decision is determined by strict thresholds.

### Decision Thresholds

| Total Score | Status | Status Code | Description |
| :--- | :--- | :--- | :--- |
| **> 65** | **APPROVED** | `2` | High confidence. Face matches and data is valid. |
| **40 - 65** | **IN_REVIEW** | `0` | Medium confidence. Manual verification recommended. |
| **< 40** | **REJECTED** | `1` | Low confidence, data mismatch, or potential fraud. |

### Scoring Breakdown (Total: 100)

| Criterion | Weight | Notes |
| :--- | :--- | :--- |
| **Face Similarity** | **40** | < 15% similarity triggers automatic rejection. |
| **Aadhaar Validity** | **20** | Checks for 12 digits and ensures number is not masked. |
| **DOB Check** | **20** | User must be 18+. Extracted DOB must match Input DOB. |
| **Gender Match** | **20** | Extracted Gender must match Input Gender. |

---

## 🔌 API Documentation

### Endpoint: `/verification/verify`
**Method:** `POST`


The API accepts **URLs** or **Local File Paths** for images.

```json
{
  "user_id": "1001",                 // Unique User Identifier
  "dob": "15-08-1998",               // User provided DOB (DD-MM-YYYY)
  "gender": "Male",                  // User provided Gender
  "passport_first": "path/front.jpg",// Aadhaar Card FRONT Image
  "passport_old": "path/back.jpg",   // Aadhaar Card BACK Image
  "selfie_photo": "path/selfie.jpg"  // User Selfie
}
```

* passport_first: Corresponds to Aadhaar Front.
* passport_old: Corresponds to Aadhaar Back.
* dob: Expected format DD-MM-YYYY (Optional, used for cross-verification).
* gender: Male, Female, or Other.

### Success Response (Approved)

>>>>>>> Stashed changes
>>>>>>> f4ec8c4 (WIP: describe changes briefly)
