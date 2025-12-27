"""
Face Recognition Endpoints
Sử dụng face_recognition và OpenCV cho nhận diện khuôn mặt
"""
import os
import uuid
import json
import base64
import numpy as np
from typing import List, Optional
from io import BytesIO
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user, create_access_token, create_refresh_token
from app.schemas.auth import Token, UserInfo
from app.core.config import get_face_images_dir
from app.core.timezone import now_vn

# Thư mục lưu ảnh khuôn mặt - Lấy từ config
FACE_IMAGES_DIR = str(get_face_images_dir())

# Import face_recognition (cần cài đặt: pip install face_recognition dlib opencv-python)
try:
    import face_recognition
    import cv2
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not installed. Face recognition features will be disabled.")

router = APIRouter()


# Schemas
class FaceImageBase64(BaseModel):
    image: str  # Base64 encoded image


class FaceRegisterRequest(BaseModel):
    images: List[str]  # List of base64 encoded images


class FaceLoginRequest(BaseModel):
    image: str  # Base64 encoded image
    username: Optional[str] = None  # Optional username to verify against


class FaceDetectionResponse(BaseModel):
    success: bool
    face_detected: bool
    face_count: int
    face_locations: Optional[List[dict]] = None
    message: str


class FaceRegisterResponse(BaseModel):
    success: bool
    message: str
    images_saved: int
    face_registered: bool


# Helper functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (BGR format for OpenCV)"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    return img_array


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def detect_and_crop_face(image: np.ndarray, padding: float = 0.3) -> tuple:
    """
    Detect face in image and return cropped face with padding
    
    Returns:
        tuple: (success, face_locations, cropped_faces)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return False, [], []
    
    # Detect faces
    face_locations = face_recognition.face_locations(image, model="hog")  # hoặc "cnn" nếu có GPU
    
    if not face_locations:
        return False, [], []
    
    cropped_faces = []
    height, width = image.shape[:2]
    
    for (top, right, bottom, left) in face_locations:
        # Add padding
        face_height = bottom - top
        face_width = right - left
        
        pad_h = int(face_height * padding)
        pad_w = int(face_width * padding)
        
        # Ensure within bounds
        top = max(0, top - pad_h)
        bottom = min(height, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(width, right + pad_w)
        
        cropped_face = image[top:bottom, left:right]
        cropped_faces.append(cropped_face)
    
    return True, face_locations, cropped_faces


def get_face_encoding(image: np.ndarray) -> Optional[np.ndarray]:
    """Get face encoding from image"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    
    # Detect face locations
    face_locations = face_recognition.face_locations(image, model="hog")
    
    if not face_locations:
        return None
    
    # Get encoding for the first face
    encodings = face_recognition.face_encodings(image, face_locations)
    
    if not encodings:
        return None
    
    return encodings[0]


def compare_faces(known_encoding: np.ndarray, unknown_encoding: np.ndarray, tolerance: float = 0.5) -> tuple:
    """
    Compare two face encodings
    
    Returns:
        tuple: (match, distance)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return False, 1.0
    
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    match = distance <= tolerance
    
    return match, float(distance)


# API Endpoints
@router.get("/status")
async def face_recognition_status():
    """Check if face recognition is available"""
    return {
        "available": FACE_RECOGNITION_AVAILABLE,
        "message": "Face recognition is available" if FACE_RECOGNITION_AVAILABLE else "Face recognition libraries not installed"
    }


@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_face(request: FaceImageBase64):
    """
    Detect faces in an image
    Returns face locations and count
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition is not available"
        )
    
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        # Detect faces
        success, face_locations, _ = detect_and_crop_face(image)
        
        # Convert face_locations to list of dicts
        locations_list = []
        for (top, right, bottom, left) in face_locations:
            locations_list.append({
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            })
        
        return FaceDetectionResponse(
            success=True,
            face_detected=success,
            face_count=len(face_locations),
            face_locations=locations_list,
            message=f"Detected {len(face_locations)} face(s)" if success else "No face detected"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error detecting face: {str(e)}"
        )


@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    request: FaceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Register face images for current user
    Expects multiple images for better recognition accuracy
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition is not available"
        )
    
    if len(request.images) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 image is required for face registration"
        )
    
    try:
        # Create user's face images folder
        user_face_folder = os.path.join(FACE_IMAGES_DIR, str(current_user.id))
        os.makedirs(user_face_folder, exist_ok=True)
        
        saved_count = 0
        all_encodings = []
        
        for i, base64_image in enumerate(request.images):
            # Decode image
            image = decode_base64_image(base64_image)
            
            # Detect and validate face
            success, face_locations, cropped_faces = detect_and_crop_face(image)
            
            if not success:
                continue
            
            if len(face_locations) > 1:
                # Skip images with multiple faces
                continue
            
            # Get face encoding
            encoding = get_face_encoding(image)
            
            if encoding is None:
                continue
            
            all_encodings.append(encoding.tolist())
            
            # Save cropped face image
            face_image = cropped_faces[0]
            face_path = os.path.join(user_face_folder, f"face_{i+1}.jpg")
            
            # Convert RGB to BGR for OpenCV
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_path, face_bgr)
            
            saved_count += 1
        
        if saved_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid face images found. Please ensure each image contains exactly one clearly visible face."
            )
        
        # Calculate average encoding for better matching
        avg_encoding = np.mean(all_encodings, axis=0).tolist()
        
        # Update user record
        current_user.face_images_folder = user_face_folder
        current_user.face_encoding = json.dumps({
            "average": avg_encoding,
            "all": all_encodings
        })
        current_user.face_registered = True
        current_user.face_registered_at = now_vn()
        
        db.commit()
        
        return FaceRegisterResponse(
            success=True,
            message=f"Successfully registered {saved_count} face image(s)",
            images_saved=saved_count,
            face_registered=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering face: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def face_login(
    request: FaceLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Login using face recognition
    Can optionally provide username to verify against specific user
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition is not available"
        )
    
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        # Get face encoding from input image
        input_encoding = get_face_encoding(image)
        
        if input_encoding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        # If username provided, verify against that user only
        if request.username:
            user = db.query(User).filter(User.username == request.username).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            if not user.face_registered or not user.face_encoding:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Face not registered for this user"
                )
            
            users_to_check = [user]
        else:
            # Search all users with registered faces
            users_to_check = db.query(User).filter(
                User.face_registered == True,
                User.face_encoding.isnot(None),
                User.is_active == True
            ).all()
        
        if not users_to_check:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No users with registered faces found"
            )
        
        # Find best match
        best_match_user = None
        best_match_distance = float('inf')
        TOLERANCE = 0.5  # Lower is stricter
        
        for user in users_to_check:
            try:
                encoding_data = json.loads(user.face_encoding)
                stored_encoding = np.array(encoding_data.get("average", encoding_data.get("all", [[]])[0]))
                
                match, distance = compare_faces(stored_encoding, input_encoding, TOLERANCE)
                
                if match and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_user = user
            except:
                continue
        
        if best_match_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Face not recognized. Please try again or use password login."
            )
        
        if not best_match_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        # Generate tokens
        access_token = create_access_token(best_match_user.id, best_match_user.role)
        refresh_token = create_refresh_token(best_match_user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=UserInfo(
                id=best_match_user.id,
                username=best_match_user.username,
                email=best_match_user.email,
                role=best_match_user.role,
                full_name=best_match_user.full_name,
                avatar=best_match_user.avatar,
                phone=best_match_user.phone,
                is_active=best_match_user.is_active,
                created_at=best_match_user.created_at
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during face login: {str(e)}"
        )


@router.post("/register-during-signup")
async def register_face_during_signup(
    images: str = Form(...),  # JSON string of base64 images
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Register face during user signup
    Called after user is created but before completing registration
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition is not available"
        )
    
    try:
        # Parse images
        image_list = json.loads(images)
        
        if not image_list or len(image_list) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 1 face image is required"
            )
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create user's face images folder
        user_face_folder = os.path.join(FACE_IMAGES_DIR, str(user.id))
        os.makedirs(user_face_folder, exist_ok=True)
        
        saved_count = 0
        all_encodings = []
        
        for i, base64_image in enumerate(image_list):
            # Decode image
            image = decode_base64_image(base64_image)
            
            # Detect and validate face
            success, face_locations, cropped_faces = detect_and_crop_face(image)
            
            if not success or len(face_locations) != 1:
                continue
            
            # Get face encoding
            encoding = get_face_encoding(image)
            
            if encoding is None:
                continue
            
            all_encodings.append(encoding.tolist())
            
            # Save cropped face image
            face_image = cropped_faces[0]
            face_path = os.path.join(user_face_folder, f"face_{i+1}.jpg")
            
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_path, face_bgr)
            
            saved_count += 1
        
        if saved_count == 0:
            # Don't fail registration, just don't enable face login
            return {
                "success": False,
                "message": "No valid face images found. Face login will not be enabled.",
                "images_saved": 0,
                "face_registered": False
            }
        
        # Calculate average encoding
        avg_encoding = np.mean(all_encodings, axis=0).tolist()
        
        # Update user record
        user.face_images_folder = user_face_folder
        user.face_encoding = json.dumps({
            "average": avg_encoding,
            "all": all_encodings
        })
        user.face_registered = True
        user.face_registered_at = now_vn()
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Successfully registered {saved_count} face image(s)",
            "images_saved": saved_count,
            "face_registered": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering face: {str(e)}"
        )


@router.delete("/unregister")
async def unregister_face(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove face registration for current user"""
    try:
        # Remove face images folder
        if current_user.face_images_folder and os.path.exists(current_user.face_images_folder):
            import shutil
            shutil.rmtree(current_user.face_images_folder)
        
        # Clear face data
        current_user.face_images_folder = None
        current_user.face_encoding = None
        current_user.face_registered = False
        current_user.face_registered_at = None
        
        db.commit()
        
        return {"success": True, "message": "Face registration removed"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing face registration: {str(e)}"
        )


@router.get("/status/{user_id}")
async def get_user_face_status(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get face registration status for a user"""
    # Only allow users to check their own status or admins
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this information"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "user_id": str(user.id),
        "face_registered": user.face_registered,
        "registered_at": user.face_registered_at.isoformat() if user.face_registered_at else None
    }


class FaceVerifyLoginRequest(BaseModel):
    face_image: str  # Base64 encoded image
    username: Optional[str] = None  # Optional: verify against specific user


@router.post("/verify-login")
async def verify_face_login(
    request: FaceVerifyLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Verify face and login
    Returns success status, user info and tokens
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition is not available"
        )
    
    try:
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Get face encoding from input image
        input_encoding = get_face_encoding(image)
        
        if input_encoding is None:
            return {
                "success": False,
                "message": "Không phát hiện khuôn mặt trong ảnh. Vui lòng thử lại.",
                "face_detected": False
            }
        
        # If username provided, verify against that user only
        if request.username:
            user = db.query(User).filter(User.username == request.username).first()
            
            if not user:
                return {
                    "success": False,
                    "message": "Không tìm thấy người dùng",
                    "face_detected": True
                }
            
            if not user.face_registered or not user.face_encoding:
                return {
                    "success": False,
                    "message": "Người dùng chưa đăng ký khuôn mặt",
                    "face_detected": True
                }
            
            users_to_check = [user]
        else:
            # Search all users with registered faces
            users_to_check = db.query(User).filter(
                User.face_registered == True,
                User.face_encoding.isnot(None),
                User.is_active == True
            ).all()
        
        if not users_to_check:
            return {
                "success": False,
                "message": "Không tìm thấy người dùng nào đã đăng ký khuôn mặt",
                "face_detected": True
            }
        
        # Find best match
        best_match_user = None
        best_match_distance = float('inf')
        TOLERANCE = 0.5  # Lower is stricter
        
        for user in users_to_check:
            try:
                encoding_data = json.loads(user.face_encoding)
                stored_encoding = np.array(encoding_data.get("average", encoding_data.get("all", [[]])[0]))
                
                match, distance = compare_faces(stored_encoding, input_encoding, TOLERANCE)
                
                if match and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_user = user
            except:
                continue
        
        if best_match_user is None:
            return {
                "success": False,
                "message": "Không nhận dạng được khuôn mặt. Vui lòng thử lại hoặc đăng nhập bằng mật khẩu.",
                "face_detected": True
            }
        
        if not best_match_user.is_active:
            return {
                "success": False,
                "message": "Tài khoản đã bị vô hiệu hóa",
                "face_detected": True
            }
        
        # Generate tokens
        access_token = create_access_token(best_match_user.id, best_match_user.role)
        refresh_token = create_refresh_token(best_match_user.id)
        
        return {
            "success": True,
            "message": f"Xác thực thành công! Xin chào {best_match_user.full_name or best_match_user.username}",
            "face_detected": True,
            "confidence": round((1 - best_match_distance) * 100, 2),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(best_match_user.id),
                "username": best_match_user.username,
                "email": best_match_user.email,
                "role": best_match_user.role,
                "full_name": best_match_user.full_name,
                "avatar": best_match_user.avatar,
                "phone": best_match_user.phone,
                "is_active": best_match_user.is_active,
                "created_at": best_match_user.created_at.isoformat() if best_match_user.created_at else None,
                "face_registered": best_match_user.face_registered
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi xác thực: {str(e)}",
            "face_detected": False
        }
