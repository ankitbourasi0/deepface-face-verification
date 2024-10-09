from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uvicorn
from pydantic import BaseModel
import os
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
import shutil
app = FastAPI(title="DeepFace Image Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
class VerificationResult(BaseModel):
    verified: bool
    distance: float = None
    threshold: float = None
    model: str = None
    detector_backend: str = None

class ImageAnalysisResult(BaseModel):
    image1_has_face: bool
    image2_has_face: bool
    verification_result: VerificationResult = None
    message: str

# Update this path to the location of your comparison image
COMPARISON_IMAGE_PATH = "./image2.jpg"



# def is_jpg(file: UploadFile) -> bool:
#     return file.content_type == "image/jpeg"

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        return tmp_path
    finally:
        upload_file.file.close()

def detect_face(image_path: str) -> bool:
    try:
        DeepFace.detect_face(image_path)
        return True
    except ValueError:
        return False

def is_valid_jpg(file_content):
    try:
        img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
        return img is not None
    except:
        return False

def detect_face(img):
    try:
        DeepFace.detectFace(img)
        return True
    except:
        return False

@app.post("/verify-faces/")
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    temp_file_path1 = None
    temp_file_path2 = None

    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await image1.read())
            temp_file_path1 = temp_file.name
            
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await image2.read())
            temp_file_path2 = temp_file.name

        # Detect faces in both images
        detected_img1 = detect_face(temp_file_path1)
        detected_img2 = detect_face(temp_file_path2)

        result = ImageAnalysisResult(
            image1_has_face=detected_img1,
            image2_has_face=detected_img2,
            message=""
        )

        # Proceed with verification only if both images have faces
        if detected_img1 and detected_img2:
            verification = DeepFace.verify(temp_file_path1, temp_file_path2)
            result.verification_result = VerificationResult(**verification)
            result.message = "Face verification completed successfully."
        else:
            if not detected_img1:
                result.message = "No face detected in the uploaded image."
            elif not detected_img2:
                result.message = "No face detected in the comparison image."
            else:
                result.message = "No faces detected in either image."

        return result

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path1 and os.path.exists(temp_file_path1):
            os.unlink(temp_file_path1)
            
        if temp_file_path2 and os.path.exists(temp_file_path2):
            os.unlink(temp_file_path2)

@app.post("/analyze_single", response_model=ImageAnalysisResult)
async def analyze_single_image(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Detect faces in both images
        uploaded_image_has_face = detect_face(temp_file_path)
        comparison_image_has_face = detect_face(COMPARISON_IMAGE_PATH)

        result = ImageAnalysisResult(
            image1_has_face=uploaded_image_has_face,
            image2_has_face=comparison_image_has_face,
            message=""
        )

        # Proceed with verification only if both images have faces
        if uploaded_image_has_face and comparison_image_has_face:
            verification = DeepFace.verify(temp_file_path, COMPARISON_IMAGE_PATH)
            result.verification_result = VerificationResult(**verification)
            result.message = "Face verification completed successfully."
        else:
            if not uploaded_image_has_face:
                result.message = "No face detected in the uploaded image."
            elif not comparison_image_has_face:
                result.message = "No face detected in the comparison image."
            else:
                result.message = "No faces detected in either image."

        return result

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the DeepFace Image Verification API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")