from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uvicorn
from pydantic import BaseModel
import os
from tempfile import NamedTemporaryFile
from PIL import Image
import io
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

def detect_face(image_path):
    try:
        DeepFace.extract_faces(img_path=image_path, enforce_detection=True)
        return True
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return False
        raise

def is_jpg(file: UploadFile) -> bool:
    return file.content_type == "image/jpeg"

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        return tmp_path
    finally:
        upload_file.file.close()



@app.post("/compare_faces")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # Validate image format
    if not (is_jpg(image1) and is_jpg(image2)):
        raise HTTPException(status_code=400, detail="Both images must be in JPG format")

    try:
        # Save uploaded files temporarily
        path1 = save_upload_file_tmp(image1)
        path2 = save_upload_file_tmp(image2)

        # Check for faces in both images
        if not (detect_face(path1) and detect_face(path2)):
            raise HTTPException(status_code=400, detail="Both images must contain a human face")

        # Verify faces and get similarity
        result = DeepFace.verify(path1, path2, enforce_detection=False,)
        
        # Clean up temporary files
        os.remove(path1)
        os.remove(path2)

        return JSONResponse(content={"similarity": result["distance"], "same_person": result["verified"]})

    except Exception as e:
        # Clean up temporary files in case of an error
        if 'path1' in locals(): os.remove(path1)
        if 'path2' in locals(): os.remove(path2)
        raise HTTPException(status_code=500, detail=str(e))


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
    uvicorn.run(app, host="0.0.0.0", port=8000)