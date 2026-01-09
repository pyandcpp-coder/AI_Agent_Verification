#main.py
import logging
import uuid
import json
import time
import base64
import asyncio
import requests
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, HttpUrl
import uvicorn

import config
import embedding_manager as db
import face_processor
from face_processor import BlurryImageError, MultipleFacesError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - API - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Modular Face Recognition API",
    description="Redis-Queued GPU Processing",
    version="3.5.0"
)

# --- Pydantic Models ---
class ImageURLRequest(BaseModel):
    user_id: str
    name: str
    email_id: str
    mob_no: str
    image_url: HttpUrl

# --- Helper to download image ---
def download_image(url: HttpUrl) -> bytes:
    try:
        response = requests.get(str(url), timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException:
        logger.error(f"Failed to download image from URL: {url}")
        raise HTTPException(status_code=400, detail="failed_to_download_image")

# --- Face Processing Logic ---
def process_face_logic(payload):
    try:
        user_id = payload.get("user_id")
        name = payload.get("name")
        email_id = payload.get("email_id")
        mob_no = payload.get("mob_no")
        image_b64 = payload.get("image_b64")
        logger.info(f"Processing image for user_id: {user_id}, name: {name}, email_id: {email_id}, mob_no: {mob_no}")
        image_bytes = base64.b64decode(image_b64)

        embedding = face_processor.get_embedding(image_bytes)

        # 1. Block employee faces
        employee_match = db.check_is_employee(embedding)
        if employee_match:
            return JSONResponse(
                status_code=403,
                content={
                    "success": True,
                    "data": {
                        "matched_id": employee_match.get("employee_id", ""),
                        "name": employee_match.get("employee_name", "")
                    },
                    "message": "employee_face_detected"
                }
            )

        # 2. Find duplicate users
        duplicate = db.find_duplicate_user(embedding)
        if duplicate:
            user_df = db.load_user_embeddings()
            duplicate_details = {}
            if not user_df.empty:
                row = user_df[user_df["user_id"] == duplicate.get("matched_user_id", "")]
                if not row.empty:
                    duplicate_details = row.iloc[0].to_dict()
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "matched_id": duplicate.get("matched_user_id", ""),
                        "name": duplicate_details.get("name", ""),
                        "email_id": duplicate_details.get("email_id", ""),
                        "mob_no": duplicate_details.get("mob_no", ""),
                        "similarity_score": round(duplicate.get("score", 0), 3)
                    },
                    "message": "duplicate_face_found"
                }
            )

        # 3. Register new user
        user_df = db.load_user_embeddings()
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "name": name,
            "email_id": email_id,
            "mob_no": mob_no,
            "embedding": embedding
        }])
        user_df = pd.concat([user_df, new_row], ignore_index=True)
        db.save_user_embeddings(user_df)
        logger.info("new_face_registered")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "name": name,
                    "email_id": email_id,
                    "mob_no": mob_no,
                    "threshold": getattr(config, "FACE_CONFIDENCE_THRESHOLD", 0.4)
                },
                "message": "new_face_registered"
            }
        )

    except MultipleFacesError as e:
        logger.info("multiple_faces_detected")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {"face_count": e.face_count},
                "message": "multiple_faces_detected"
            }
        )
    except BlurryImageError as e:
        logger.info("blur_image_detected")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {"blur_score": round(e.score, 2)},
                "message": "blur_image_detected"
            }
        )
    except ValueError as e:
        error_str = str(e).lower()
        # Match more variants for no face detected
        no_face_keywords = [
            "no face detected", "no faces detected", "no prominent face detected", "no face found", "no valid face", "not appear to be a valid", "no face", "no faces"
        ]
        if any(x in error_str for x in no_face_keywords):
            logger.info("no_face_detected")
            return JSONResponse(status_code=400, content={"success": False, "data": {}, "message": "no_face_detected"})
        elif "invalid" in error_str:
            logger.info("invalid_image_file")
            return JSONResponse(status_code=400, content={"success": False, "data": {}, "message": "invalid_image_file"})
        else:
            logger.info("image_processing_error")
            logger.error(f"Processing Error: {error_str}")
            return JSONResponse(status_code=400, content={"success": False, "data": {}, "message": "image_processing_error"})
    except Exception as e:
        logger.error(f"Worker Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "data": {}, "message": "internal_server_error"})

# --- ENDPOINTS ---

@app.post("/check-face-file")
async def check_face_file(
    user_id: str = Form(...), name: str = Form(...), 
    email_id: str = Form(...), mob_no: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "user_id": user_id,
        "name": name,
        "email_id": email_id,
        "mob_no": mob_no,
        "image_b64": image_b64
    }
    return process_face_logic(payload)

@app.post("/check-face-url")
async def check_face_url(request: ImageURLRequest):
    try:
        image_bytes = await run_in_threadpool(download_image, request.image_url)
    except HTTPException as e:
        return JSONResponse(status_code=400, content={"detail": "failed_to_download_image"})
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "user_id": request.user_id,
        "name": request.name,
        "email_id": request.email_id,
        "mob_no": request.mob_no,
        "image_b64": image_b64
    }
    return process_face_logic(payload)

@app.get("/employees")
async def list_employees():
    logger.info("Admin requested Employee List")
    df = await run_in_threadpool(db.load_employee_embeddings)
    data = df[["employee_id", "employee_name"]].to_dict(orient="records") if not df.empty else []
    return {"success": True, "data": {"employees": data}, "message": "retrieved"}

@app.get("/users")
async def list_users():
    logger.info("Admin requested User List")
    df = await run_in_threadpool(db.load_user_embeddings)
    data = df[["user_id", "name", "email_id", "mob_no"]].to_dict(orient="records") if not df.empty else []
    return {"success": True, "data": {"users": data}, "message": "retrieved"}

@app.delete("/user/{user_id}")
async def remove_user(user_id: str):
    logger.info(f" Request to delete User: {user_id}")
    def _delete_sync(uid):
        df = db.load_user_embeddings()
        if uid not in df["user_id"].values: return None
        row = df[df["user_id"] == uid].iloc[0].to_dict()
        db.save_user_embeddings(df[df["user_id"] != uid])
        return row

    deleted_data = await run_in_threadpool(_delete_sync, user_id)
    if not deleted_data:
        logger.warning(f"Delete failed. User {user_id} not found.")
        return JSONResponse(status_code=404, content={"success": False, "message": "user_not_found"})
    logger.info(f"User {user_id} deleted successfully.")
    return {
        "success": True,
        "data": {k: deleted_data.get(k, "") for k in ["user_id", "name", "email_id"]},
        "message": "user_removed"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8108, reload=True)