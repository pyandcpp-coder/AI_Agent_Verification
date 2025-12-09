# Aadhaar Entity Extraction API

A FastAPI-based service that extracts three key entities directly from Aadhaar images:
- **Aadhaar Number**
- **Date of Birth (DOB)**
- **Gender**

**Note:** This API assumes the input image is already a valid Aadhaar card. No card validation is performed.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/yrevash/work_qoneqt/verfication_agent
source venv/bin/activate
pip install -r requirements.txt
pip install uvicorn  # If not already installed
```

### 2. Verify Tesseract Installation

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

### 3. Run the Server

```bash
# Method 1: Direct execution
python app/entity.py

# Method 2: Using uvicorn
python -m uvicorn app.entity:app --host 0.0.0.0 --port 8111 --reload
```

The API will be available at: **http://localhost:8111**

## 📚 API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8111/docs
- **ReDoc**: http://localhost:8111/redoc

## 🔌 API Endpoints

### 1. Extract Entities (POST)
**Endpoint**: `/extract_entities`

**Request Body**:
```json
{
  "image_url": "https://your-cdn.com/aadhaar_front.jpg",
  "confidence_threshold": 0.45
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "entities_extracted",
  "data": {
    "aadharnumber": "123456789012",
    "dob": "01-01-1990",
    "gender": "Male"
  }
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "no_aadhar_front_detected",
  "data": null
}
```

### 2. Health Check (GET)
**Endpoint**: `/health`

**Response**:
```json
{
  "success": true,
  "data": {
    "pipeline_status": "initialized",
    "inference_device": "cuda",
    "torch_version": "2.0.0",
    "cuda_available": true
  },
  "message": "service_healthy"
}
```

## 🧪 Testing

### Using the Test Script

```bash
python test_entity_api.py
```

### Using cURL

```bash
# Health check
curl http://localhost:8111/health

# Extract entities
curl -X POST http://localhost:8111/extract_entities \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-cdn.com/aadhaar_front.jpg",
    "confidence_threshold": 0.45
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8111/extract_entities"
payload = {
    "image_url": "https://your-cdn.com/aadhaar_front.jpg",
    "confidence_threshold": 0.45
}

response = requests.post(url, json=payload)
result = response.json()

if result["success"]:
    print(f"Aadhaar: {result['data']['aadharnumber']}")
    print(f"DOB: {result['data']['dob']}")
    print(f"Gender: {result['data']['gender']}")
```

## 📁 Project Structure

```
verfication_agent/
├── app/
│   └── entity.py          # Main FastAPI application
├── models/
│   └── best.pt           # YOLO model for entity detection (only model used)
├── downloads/            # Temporary image downloads (auto-created)
├── .env                  # Environment configuration
├── requirements.txt      # Python dependencies
├── test_entity_api.py    # Test script
└── README.md            # This file
```

## ⚙️ Configuration

Edit `.env` file to configure:

```properties
MODEL2_PATH=models/best.pt              # Entity detection model
DOWNLOAD_DIR=downloads                   # Temporary downloads folder
DEFAULT_CONFIDENCE_THRESHOLD=0.45        # Default detection threshold
```

**Note:** `MODEL1_PATH` is no longer used since card detection is skipped.

## 📊 Response Messages

| Message | Description |
|---------|-------------|
| `entities_extracted` | Successfully extracted all entities |
| `no_entities_detected` | No entities found in image |
| `failed_to_download_image` | Could not download image from URL |
| `processing_error` | Internal processing error |
| `service_unavailable` | Pipeline not initialized |

## 🔧 Troubleshooting

### Models Not Found
```bash
# Ensure models exist in the models/ directory
ls -la models/
```

### Tesseract Not Found
```bash
# Install Tesseract
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

### Port Already in Use
```bash
# Change port in entity.py (last line) or use a different port
uvicorn app.entity:app --port 8112
```

## 🎯 Features

- ✅ Simple single image endpoint
- ✅ Only extracts 3 key entities (Aadhaar, DOB, Gender)
- ✅ No card validation - assumes valid Aadhaar input
- ✅ No database storage
- ✅ Automatic image cleanup
- ✅ GPU support (CUDA if available)
- ✅ Multi-language OCR support
- ✅ Orientation correction
- ✅ Direct entity detection (skips card detection)

## 📝 Notes

- The API assumes input is a **valid Aadhaar image** (no validation performed)
- Images are downloaded temporarily and deleted after processing
- No data is stored (no CSV, JSON, or PKL files)
- Supports CDN URLs for image input
- Confidence threshold can be adjusted per request (default: 0.15)
- Only uses the entity detection model (MODEL2_PATH) - card detection model is not needed

## 🐛 Known Issues

- Masked Aadhaar detection may return "masked_aadhar" string
- Invalid date formats may return "Invalid Format"
- Gender values normalized to: Male, Female, or Other
