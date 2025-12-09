"""
Test script for the Entity Extraction API
"""
import requests
import json

# API endpoint
API_URL = "http://localhost:8111/extract_entities"
HEALTH_URL = "http://localhost:8111/health"

def test_health():
    """Test if the API is running"""
    print("Testing Health Endpoint...")
    response = requests.get(HEALTH_URL)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_entity_extraction(image_url):
    """Test entity extraction from an Aadhaar image"""
    print("Testing Entity Extraction...")
    
    payload = {
        "image_url": image_url,
        "confidence_threshold": 0.45  # Optional, defaults to 0.15
    }
    
    response = requests.post(API_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    
    return response.json()

if __name__ == "__main__":
    # Test health endpoint
    test_health()
    
    # Replace with your actual image URL (CDN link)
    # Example: "https://example.com/aadhaar_front.jpg"
    IMAGE_URL = "YOUR_IMAGE_CDN_URL_HERE"
    
    if IMAGE_URL != "YOUR_IMAGE_CDN_URL_HERE":
        result = test_entity_extraction(IMAGE_URL)
        
        if result.get("success"):
            print("✅ Extraction Successful!")
            print(f"Aadhaar Number: {result['data']['aadharnumber']}")
            print(f"DOB: {result['data']['dob']}")
            print(f"Gender: {result['data']['gender']}")
        else:
            print("❌ Extraction Failed!")
            print(f"Message: {result.get('message')}")
    else:
        print("⚠️  Please set IMAGE_URL to test entity extraction")
