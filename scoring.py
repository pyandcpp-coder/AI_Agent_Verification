# scoring.py
import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100)
        self.weights = {
            "face": 40,      # High weightage for Face
            "aadhar": 20,    # Valid Aadhaar Number
            "dob": 20,       # Valid Age/DOB
            "gender": 20     # Gender Match
        }

    def calculate_score(self, face_data, entity_data, expected_gender):
        score = 0
        breakdown = {}
        rejection_reasons = []

        # --- 1. Face Similarity (0 - 40 points) ---
        # Assumption: Face sim returns a score 0-100.
        # We map 0-100 similarity to 0-40 points.
        face_sim = face_data.get("similarity", 0)
        face_points = 0
        
        if face_sim >= 40: # Threshold for "Same Person"
            # Linear scaling: 40 sim -> 20 pts, 80+ sim -> 40 pts
            face_points = min(40, (face_sim / 100) * 40 * 1.5) 
            if face_sim > 70: face_points = 40 # Max points for very high match
        else:
            rejection_reasons.append(f"Face mismatch (Similarity: {face_sim:.2f}%)")
        
        score += face_points
        breakdown["face_score"] = round(face_points, 2)

        # --- 2. Aadhaar Validation (0 - 20 points) ---
        aadhaar_num = entity_data.get("aadharnumber", "").replace(" ", "")
        if len(aadhaar_num) == 12 and aadhaar_num.isdigit():
            # (Optional) Add Verhoeff algorithm here for extra strictness
            score += self.weights["aadhar"]
            breakdown["aadhar_score"] = self.weights["aadhar"]
        else:
            breakdown["aadhar_score"] = 0
            rejection_reasons.append("Invalid Aadhaar Number format")

        # --- 3. DOB / Age Validation (0 - 20 points) ---
        # entity_data should have 'age_status': 'age_approved'
        if entity_data.get("age_status") == "age_approved":
            score += self.weights["dob"]
            breakdown["dob_score"] = self.weights["dob"]
        else:
            breakdown["dob_score"] = 0
            if entity_data.get("age") and entity_data.get("age") < 18:
                rejection_reasons.append(f"User is under 18 ({entity_data.get('age')})")
            else:
                rejection_reasons.append("DOB invalid or unreadable")

        # --- 4. Gender Check (0 - 20 points) ---
        extracted_gender = entity_data.get("gender", "").lower()
        expected_gender = expected_gender.lower() if expected_gender else ""
        
        # Normalize gender strings (e.g., 'm' -> 'male')
        if "male" in extracted_gender: extracted_gender = "male"
        if "female" in extracted_gender: extracted_gender = "female"
        
        if expected_gender and extracted_gender:
            if expected_gender == extracted_gender:
                score += self.weights["gender"]
                breakdown["gender_score"] = self.weights["gender"]
            else:
                breakdown["gender_score"] = 0
                rejection_reasons.append(f"Gender mismatch (Expected: {expected_gender}, Found: {extracted_gender})")
        elif not expected_gender:
             # If no expected gender provided, give benefit of doubt or partial points
             score += 10 
             breakdown["gender_score"] = 10
        else:
             breakdown["gender_score"] = 0

        # Final Verdict
        status = "APPROVED" if score >= 70 else "REJECTED"
        if rejection_reasons and score >= 70:
            status = "MANUAL_REVIEW" # High score but some issues

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }