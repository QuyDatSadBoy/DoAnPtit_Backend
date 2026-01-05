"""
Face Recognition Endpoints - DeepFace with GPU Support
Logic: Detect Face → Crop Face → Get Embedding từ Face (không phải cả ảnh)
"""
import os
import shutil
import json
import base64
import numpy as np
from typing import List, Optional
from io import BytesIO
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user, create_access_token, create_refresh_token
from app.schemas.auth import Token, UserInfo
from app.core.config import get_face_images_dir
from app.core.timezone import now_vn

# =====================================================
# CONFIGURATION
# =====================================================
FACE_IMAGES_DIR = str(get_face_images_dir())
MAX_IMAGES = 3

# DeepFace config
DEEPFACE_MODEL = "ArcFace"  # Best: ArcFace, Facenet512
DEEPFACE_DETECTOR = "yunet"  # Fast: yunet, opencv | Accurate: retinaface, mtcnn
DEEPFACE_METRIC = "cosine"
MIN_CONFIDENCE = 0.9  # Minimum face detection confidence
ANTI_SPOOFING = True  # Chống giả mạo ảnh/video

# Ngưỡng verify - Càng thấp càng nghiêm ngặt (khó match hơn)
# Mặc định DeepFace: 0.68 (ArcFace + cosine)
# Khuyến nghị: 0.4-0.5 để an toàn hơn
FACE_VERIFY_THRESHOLD = 0.36

# =====================================================
# DEEPFACE INITIALIZATION WITH GPU
# =====================================================
FACE_RECOGNITION_AVAILABLE = False
DeepFace = None

try:
    # Import và config TensorFlow cho GPU
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    
    # Import DeepFace
    from deepface import DeepFace as DF
    DeepFace = DF
    
    # Pre-load models để cache (sẽ tự động dùng GPU nếu có)
    print("🔄 Loading DeepFace models...")
    DeepFace.build_model(DEEPFACE_MODEL)
    print(f"✅ DeepFace ready - Model: {DEEPFACE_MODEL}, Detector: {DEEPFACE_DETECTOR}")
    
    FACE_RECOGNITION_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ DeepFace not installed: {e}")
except Exception as e:
    print(f"❌ DeepFace init error: {e}")

router = APIRouter()


# =====================================================
# SCHEMAS
# =====================================================
class FaceImageBase64(BaseModel):
    image: str

class FaceRegisterRequest(BaseModel):
    images: List[str]

class FaceLoginRequest(BaseModel):
    image: str
    username: Optional[str] = None

class FaceVerifyLoginRequest(BaseModel):
    face_image: str
    username: Optional[str] = None

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


# =====================================================
# CORE FUNCTIONS - Dùng 100% DeepFace built-in
# =====================================================

def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """Decode base64 thành numpy array (RGB) - DeepFace hỗ trợ trực tiếp numpy"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)


def save_numpy_to_file(img_array: np.ndarray, path: str):
    """Save numpy array to file"""
    Image.fromarray(img_array.astype(np.uint8)).save(path, 'JPEG', quality=95)


def extract_face_and_embedding(img_input) -> dict:
    """
    LOGIC CHUẨN:
    1. Detect face trong ảnh
    2. Crop face ra 
    3. Lấy embedding từ FACE (không phải cả ảnh)
    
    Args:
        img_input: numpy array hoặc file path
        
    Returns:
        {
            "success": bool,
            "face_detected": bool,
            "face_count": int,
            "face_img": numpy array of cropped face (nếu có),
            "embedding": list of float (nếu có),
            "confidence": float,
            "facial_area": dict
        }
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "face_detected": False, "error": "DeepFace not available"}
    
    try:
        # Step 1: Detect và extract faces (với anti-spoofing)
        face_objs = DeepFace.extract_faces(
            img_path=img_input,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,  # Không raise error nếu không tìm thấy
            align=True,  # Align face cho accuracy cao hơn
            expand_percentage=0,  # Không expand, chỉ lấy face
            anti_spoofing=ANTI_SPOOFING  # Chống giả mạo
        )
        
        # Filter faces với confidence cao và không phải spoofing
        valid_faces = []
        for f in face_objs:
            if f.get('confidence', 0) >= MIN_CONFIDENCE:
                # Nếu bật anti-spoofing, kiểm tra is_real
                if ANTI_SPOOFING and not f.get('is_real', True):
                    print(f"[ANTI-SPOOF] Detected fake face! Antispoof score: {f.get('antispoof_score', 0)}")
                    continue
                valid_faces.append(f)
        
        if len(valid_faces) == 0:
            return {
                "success": True,
                "face_detected": False,
                "face_count": 0,
                "error": "No face detected with sufficient confidence"
            }
        
        if len(valid_faces) > 1:
            return {
                "success": True,
                "face_detected": True,
                "face_count": len(valid_faces),
                "error": "Multiple faces detected"
            }
        
        # Lấy face đầu tiên (và duy nhất)
        face_obj = valid_faces[0]
        face_img = face_obj.get('face')  # Numpy array của cropped face (đã normalized 0-1)
        confidence = face_obj.get('confidence', 0)
        facial_area = face_obj.get('facial_area', {})
        
        # Convert face từ 0-1 về 0-255 để lưu và hiển thị
        if face_img is not None:
            face_img_uint8 = (face_img * 255).astype(np.uint8)
        else:
            face_img_uint8 = None
        
        # Step 2: Lấy embedding từ FACE IMAGE (không phải original image)
        # DeepFace.represent() sẽ tự detect face trong ảnh input
        # Vì face_img đã là face rồi, ta dùng detector='skip' để không detect lại
        embedding_objs = DeepFace.represent(
            img_path=face_img,  # Input là cropped face
            model_name=DEEPFACE_MODEL,
            detector_backend="skip",  # Skip detection vì đã là face rồi
            enforce_detection=False,
            align=False  # Đã align ở bước extract
        )
        
        if not embedding_objs or len(embedding_objs) == 0:
            return {
                "success": True,
                "face_detected": True,
                "face_count": 1,
                "face_img": face_img_uint8,
                "embedding": None,
                "confidence": confidence,
                "facial_area": facial_area,
                "error": "Failed to extract embedding"
            }
        
        embedding = embedding_objs[0].get('embedding', [])
        
        return {
            "success": True,
            "face_detected": True,
            "face_count": 1,
            "face_img": face_img_uint8,
            "embedding": embedding,
            "confidence": confidence,
            "facial_area": facial_area
        }
        
    except Exception as e:
        print(f"[FACE ERROR] extract_face_and_embedding: {e}")
        return {"success": False, "face_detected": False, "error": str(e)}


def verify_face_with_stored(input_face_img, stored_face_path: str) -> dict:
    """
    Verify 2 cropped faces sử dụng DeepFace.verify()
    Cả 2 đều là cropped face nên dùng detector='skip'
    
    Sử dụng custom threshold (FACE_VERIFY_THRESHOLD) thay vì mặc định của DeepFace
    
    Returns:
        {"verified": bool, "distance": float, "threshold": float}
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {"verified": False, "distance": 1.0, "threshold": FACE_VERIFY_THRESHOLD}
    
    try:
        result = DeepFace.verify(
            img1_path=input_face_img,  # Cropped face từ input
            img2_path=stored_face_path,  # Cropped face đã lưu
            model_name=DEEPFACE_MODEL,
            detector_backend="skip",  # Skip vì cả 2 đều là cropped face
            distance_metric=DEEPFACE_METRIC,
            enforce_detection=False,
            align=False  # Đã align khi extract
        )
        
        distance = result.get("distance", 1.0)
        # Sử dụng custom threshold thay vì mặc định của DeepFace
        verified = distance <= FACE_VERIFY_THRESHOLD
        
        return {
            "verified": verified,
            "distance": distance,
            "threshold": FACE_VERIFY_THRESHOLD
        }
    except Exception as e:
        print(f"[FACE ERROR] verify: {e}")
        return {"verified": False, "distance": 1.0, "threshold": FACE_VERIFY_THRESHOLD}


# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_face(
    request: FaceImageBase64,
    current_user: User = Depends(get_current_user)
):
    """Detect faces trong ảnh"""
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Face recognition not available")
    
    try:
        img_array = decode_base64_to_numpy(request.image)
        result = extract_face_and_embedding(img_array)
        
        face_locations = []
        if result.get("facial_area"):
            area = result["facial_area"]
            face_locations.append({
                "x": area.get("x", 0),
                "y": area.get("y", 0),
                "w": area.get("w", 0),
                "h": area.get("h", 0),
                "confidence": result.get("confidence", 0)
            })
        
        return FaceDetectionResponse(
            success=True,
            face_detected=result.get("face_detected", False),
            face_count=result.get("face_count", 0),
            face_locations=face_locations if face_locations else None,
            message=f"Phát hiện {result.get('face_count', 0)} khuôn mặt" 
                    if result.get("face_detected") else result.get("error", "Không phát hiện khuôn mặt")
        )
    except Exception as e:
        return FaceDetectionResponse(
            success=False, face_detected=False, face_count=0,
            face_locations=None, message=f"Lỗi: {str(e)}"
        )


@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    request: FaceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Đăng ký khuôn mặt - Logic:
    1. Decode ảnh từ base64
    2. Detect face → Crop face
    3. Lấy embedding từ FACE
    4. Lưu FACE image (không phải original) và embedding
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Face recognition not available")
    
    user_folder = os.path.join(FACE_IMAGES_DIR, str(current_user.id))
    os.makedirs(user_folder, exist_ok=True)
    
    # Check existing
    existing = [f for f in os.listdir(user_folder) if f.startswith("face_") and f.endswith(".jpg")]
    if len(existing) >= MAX_IMAGES:
        return FaceRegisterResponse(
            success=False,
            message=f"Đã đạt giới hạn {MAX_IMAGES} ảnh. Xóa ảnh cũ trước.",
            images_saved=0,
            face_registered=current_user.face_registered
        )
    
    # Load existing embeddings
    existing_embeddings = []
    if current_user.face_encoding:
        try:
            data = json.loads(current_user.face_encoding)
            existing_embeddings = data.get("embeddings", [])
        except:
            pass
    
    # Get next index
    indices = []
    for f in existing:
        try:
            idx = int(f.replace("face_", "").replace(".jpg", ""))
            indices.append(idx)
        except:
            pass
    next_idx = max(indices) + 1 if indices else 1
    
    saved = 0
    new_embeddings = []
    slots = MAX_IMAGES - len(existing)
    
    for i, b64_img in enumerate(request.images[:slots]):
        try:
            # Decode
            img_array = decode_base64_to_numpy(b64_img)
            
            # Extract face và embedding
            result = extract_face_and_embedding(img_array)
            
            if not result.get("face_detected"):
                print(f"[REGISTER] Image {i+1}: {result.get('error', 'No face')}")
                continue
            
            if result.get("face_count", 0) > 1:
                print(f"[REGISTER] Image {i+1}: Multiple faces")
                continue
            
            face_img = result.get("face_img")
            embedding = result.get("embedding")
            
            if face_img is None or embedding is None:
                print(f"[REGISTER] Image {i+1}: No face_img or embedding")
                continue
            
            # Lưu FACE IMAGE (đã crop) - không phải original
            face_path = os.path.join(user_folder, f"face_{next_idx + saved}.jpg")
            save_numpy_to_file(face_img, face_path)
            
            new_embeddings.append(embedding)
            saved += 1
            
            print(f"[REGISTER] Image {i+1}: ✅ Saved face_{next_idx + saved - 1}.jpg, confidence={result.get('confidence', 0):.2f}")
            
        except Exception as e:
            print(f"[REGISTER] Image {i+1} error: {e}")
            continue
    
    if saved == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh khuôn mặt hợp lệ")
    
    # Update DB
    all_embeddings = existing_embeddings + new_embeddings
    current_user.face_images_folder = user_folder
    current_user.face_encoding = json.dumps({
        "model": DEEPFACE_MODEL,
        "detector": DEEPFACE_DETECTOR,
        "embeddings": all_embeddings
    })
    current_user.face_registered = True
    if not current_user.face_registered_at:
        current_user.face_registered_at = now_vn()
    
    db.commit()
    
    total = len(existing) + saved
    return FaceRegisterResponse(
        success=True,
        message=f"Đã thêm {saved} ảnh. Tổng: {total}/{MAX_IMAGES}",
        images_saved=saved,
        face_registered=True
    )


@router.post("/verify-login")
async def verify_face_login(
    request: FaceVerifyLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Login bằng khuôn mặt - Dùng DeepFace.verify()
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Face recognition not available")
    
    try:
        # Decode input
        input_img = decode_base64_to_numpy(request.face_image)
        
        # Check có face không và lấy cropped face
        result = extract_face_and_embedding(input_img)
        if not result.get("face_detected"):
            return {
                "success": False,
                "message": result.get("error", "Không phát hiện khuôn mặt"),
                "face_detected": False
            }
        
        # Lấy cropped face để verify (quan trọng!)
        input_face = result.get("face_img")
        if input_face is None:
            return {
                "success": False,
                "message": "Không thể extract khuôn mặt",
                "face_detected": False
            }
        
        # Get users to check
        if request.username:
            user = db.query(User).filter(User.username == request.username).first()
            if not user:
                return {"success": False, "message": "Không tìm thấy người dùng", "face_detected": True}
            if not user.face_registered:
                return {"success": False, "message": "Người dùng chưa đăng ký khuôn mặt", "face_detected": True}
            users = [user]
        else:
            users = db.query(User).filter(
                User.face_registered == True,
                User.is_active == True
            ).all()
        
        if not users:
            return {"success": False, "message": "Không có người dùng nào đăng ký khuôn mặt", "face_detected": True}
        
        # Find best match
        best_user = None
        best_distance = float('inf')
        best_threshold = 0.4
        
        for user in users:
            if not user.face_images_folder or not os.path.exists(user.face_images_folder):
                continue
            
            face_files = [f for f in os.listdir(user.face_images_folder) if f.startswith("face_") and f.endswith(".jpg")]
            
            for face_file in face_files:
                face_path = os.path.join(user.face_images_folder, face_file)
                
                # Verify cropped faces với nhau
                verify_result = verify_face_with_stored(input_face, face_path)
                
                print(f"[LOGIN] {user.username}/{face_file}: verified={verify_result['verified']}, dist={verify_result['distance']:.4f}")
                
                if verify_result.get("verified"):
                    dist = verify_result.get("distance", 1.0)
                    if dist < best_distance:
                        best_distance = dist
                        best_threshold = verify_result.get("threshold", 0.4)
                        best_user = user
        
        if best_user is None:
            return {
                "success": False,
                "message": "Không nhận dạng được. Thử lại hoặc đăng nhập bằng mật khẩu.",
                "face_detected": True
            }
        
        if not best_user.is_active:
            return {"success": False, "message": "Tài khoản đã bị vô hiệu hóa", "face_detected": True}
        
        # Generate tokens
        access_token = create_access_token(best_user.id, best_user.role)
        refresh_token = create_refresh_token(best_user.id)
        
        confidence = max(0, (1 - best_distance / best_threshold) * 100) if best_threshold > 0 else 90
        
        return {
            "success": True,
            "message": f"Xin chào {best_user.full_name or best_user.username}!",
            "face_detected": True,
            "confidence": round(confidence, 2),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(best_user.id),
                "username": best_user.username,
                "email": best_user.email,
                "role": best_user.role,
                "full_name": best_user.full_name,
                "avatar": best_user.avatar,
                "phone": best_user.phone,
                "is_active": best_user.is_active,
                "created_at": best_user.created_at.isoformat() if best_user.created_at else None,
                "face_registered": best_user.face_registered
            }
        }
        
    except Exception as e:
        print(f"[LOGIN ERROR] {e}")
        return {"success": False, "message": f"Lỗi: {str(e)}", "face_detected": False}


@router.post("/login", response_model=Token)
async def login_with_face(
    request: FaceLoginRequest,
    db: Session = Depends(get_db)
):
    """Login endpoint cũ - wrap verify-login"""
    result = await verify_face_login(
        FaceVerifyLoginRequest(face_image=request.image, username=request.username),
        db
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("message", "Face not recognized"))
    
    return Token(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type="bearer",
        user=UserInfo(
            id=result["user"]["id"],
            username=result["user"]["username"],
            email=result["user"]["email"],
            role=result["user"]["role"],
            full_name=result["user"]["full_name"],
            avatar=result["user"]["avatar"],
            phone=result["user"]["phone"],
            is_active=result["user"]["is_active"],
            created_at=result["user"]["created_at"],
            face_registered=result["user"]["face_registered"]
        )
    )


@router.get("/my-images")
async def get_my_face_images(current_user: User = Depends(get_current_user)):
    """Get danh sách ảnh khuôn mặt của user"""
    folder = os.path.join(FACE_IMAGES_DIR, str(current_user.id))
    
    if not os.path.exists(folder):
        return {"success": True, "images": [], "count": 0, "max_images": MAX_IMAGES, "can_add_more": True}
    
    images = []
    for f in sorted(os.listdir(folder)):
        if f.startswith("face_") and f.endswith(".jpg"):
            # Trả về URL string trực tiếp để FE dùng với config.getFileUrl()
            images.append(f"face_images/{current_user.id}/{f}")
    
    return {
        "success": True,
        "images": images,
        "count": len(images),
        "max_images": MAX_IMAGES,
        "can_add_more": len(images) < MAX_IMAGES
    }


@router.delete("/images/{filename}")
async def delete_face_image(
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Xóa ảnh khuôn mặt"""
    folder = os.path.join(FACE_IMAGES_DIR, str(current_user.id))
    file_path = os.path.join(folder, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Ảnh không tồn tại")
    
    os.remove(file_path)
    
    # Recalculate embeddings
    remaining = [f for f in os.listdir(folder) if f.startswith("face_") and f.endswith(".jpg")]
    
    if not remaining:
        current_user.face_registered = False
        current_user.face_encoding = None
    else:
        new_embeddings = []
        for f in remaining:
            path = os.path.join(folder, f)
            # Read saved face image và get embedding
            img = np.array(Image.open(path))
            # Face đã được crop rồi, dùng skip detector
            try:
                emb_objs = DeepFace.represent(
                    img_path=img,
                    model_name=DEEPFACE_MODEL,
                    detector_backend="skip",
                    enforce_detection=False
                )
                if emb_objs:
                    new_embeddings.append(emb_objs[0].get("embedding", []))
            except:
                pass
        
        if new_embeddings:
            current_user.face_encoding = json.dumps({
                "model": DEEPFACE_MODEL,
                "detector": DEEPFACE_DETECTOR,
                "embeddings": new_embeddings
            })
        else:
            current_user.face_registered = False
            current_user.face_encoding = None
    
    db.commit()
    
    return {
        "success": True,
        "message": "Đã xóa ảnh",
        "remaining_count": len(remaining),
        "max_images": MAX_IMAGES,
        "can_add_more": len(remaining) < MAX_IMAGES
    }


@router.get("/status")
async def get_face_status(current_user: User = Depends(get_current_user)):
    """Get trạng thái đăng ký khuôn mặt"""
    return {
        "user_id": str(current_user.id),
        "face_registered": current_user.face_registered,
        "registered_at": current_user.face_registered_at.isoformat() if current_user.face_registered_at else None,
        "model": DEEPFACE_MODEL,
        "detector": DEEPFACE_DETECTOR,
        "gpu_available": FACE_RECOGNITION_AVAILABLE
    }


@router.post("/register-during-signup")
async def register_face_during_signup(
    user_id: str = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Đăng ký khuôn mặt trong quá trình signup"""
    if not FACE_RECOGNITION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Face recognition not available")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    folder = os.path.join(FACE_IMAGES_DIR, str(user.id))
    os.makedirs(folder, exist_ok=True)
    
    saved = 0
    embeddings = []
    
    for i, img_file in enumerate(images[:MAX_IMAGES]):
        try:
            content = await img_file.read()
            img = Image.open(BytesIO(content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            result = extract_face_and_embedding(img_array)
            
            if not result.get("face_detected") or result.get("face_count", 0) != 1:
                continue
            
            face_img = result.get("face_img")
            embedding = result.get("embedding")
            
            if face_img is None or embedding is None:
                continue
            
            # Save cropped face
            save_numpy_to_file(face_img, os.path.join(folder, f"face_{saved + 1}.jpg"))
            embeddings.append(embedding)
            saved += 1
            
        except Exception as e:
            print(f"[SIGNUP] Image {i+1} error: {e}")
    
    if saved == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh khuôn mặt hợp lệ")
    
    user.face_images_folder = folder
    user.face_encoding = json.dumps({
        "model": DEEPFACE_MODEL,
        "detector": DEEPFACE_DETECTOR,
        "embeddings": embeddings
    })
    user.face_registered = True
    user.face_registered_at = now_vn()
    
    db.commit()
    
    return {"success": True, "message": f"Đã đăng ký {saved} ảnh khuôn mặt", "images_saved": saved}
