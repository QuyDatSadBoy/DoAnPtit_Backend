"""
Authentication Endpoints
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import Token, LoginRequest, RefreshTokenRequest, ChangePasswordRequest, UserInfo
from app.schemas.user import UserResponse
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        access_token = create_access_token(user.id, user.role)
        refresh_token = create_refresh_token(user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                full_name=user.full_name,
                avatar=user.avatar,
                phone=user.phone,
                is_active=user.is_active,
                created_at=user.created_at,
                face_registered=user.face_registered
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi đăng nhập: {str(e)}")


@router.post("/login/json", response_model=Token)
async def login_json(
    request: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with JSON body"""
    try:
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        access_token = create_access_token(user.id, user.role)
        refresh_token = create_refresh_token(user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                full_name=user.full_name,
                avatar=user.avatar,
                phone=user.phone,
                is_active=user.is_active,
                created_at=user.created_at,
                face_registered=user.face_registered
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during JSON login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi đăng nhập: {str(e)}")


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    try:
        payload = decode_token(request.refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        access_token = create_access_token(user.id, user.role)
        new_refresh_token = create_refresh_token(user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi refresh token: {str(e)}")


@router.post("/register")
async def register(
    user_data: dict,
    db: Session = Depends(get_db)
):
    """Register new user account"""
    import uuid
    import json
    import os
    import httpx
    from app.models.doctor import Doctor
    from app.core.config import get_face_images_dir
    
    try:
        # Check if username exists
        existing_user = db.query(User).filter(User.username == user_data.get("username")).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email exists
        existing_email = db.query(User).filter(User.email == user_data.get("email")).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Create user
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            username=user_data.get("username"),
            email=user_data.get("email"),
            password_hash=get_password_hash(user_data.get("password")),
            role=user_data.get("role", "doctor"),
            full_name=user_data.get("full_name"),
            is_active=True
        )
        db.add(user)
        db.flush()
        
        # If doctor role, create doctor profile
        # Note: Use user.role instead of user_data.get("role") since default might be applied
        if user.role == "doctor":
            doctor_info = user_data.get("doctor_info", {})
            doctor = Doctor(
                id=uuid.uuid4(),
                user_id=user_id,
                specialty=doctor_info.get("specialty"),
                phone=doctor_info.get("phone"),
                hospital=doctor_info.get("hospital")
            )
            db.add(doctor)
        
        db.commit()
        
        # Handle face registration if face_images provided
        face_images = user_data.get("face_images")
        face_registered = False
        
        if face_images and len(face_images) >= 3:
            try:
                # Call face registration endpoint internally
                from app.api.endpoints.face_recognition import (
                    decode_base64_image, 
                    detect_and_crop_face, 
                    get_face_encoding,
                    FACE_RECOGNITION_AVAILABLE
                )
                import cv2
                import numpy as np
                from app.core.timezone import now_vn
                
                if FACE_RECOGNITION_AVAILABLE:
                    FACE_IMAGES_DIR = str(get_face_images_dir())
                    user_face_folder = os.path.join(FACE_IMAGES_DIR, str(user_id))
                    os.makedirs(user_face_folder, exist_ok=True)
                    
                    saved_count = 0
                    all_encodings = []
                    
                    for i, base64_image in enumerate(face_images):
                        try:
                            image = decode_base64_image(base64_image)
                            success, face_locations, cropped_faces = detect_and_crop_face(image)
                            
                            if not success or len(face_locations) != 1:
                                continue
                            
                            encoding = get_face_encoding(image)
                            if encoding is None:
                                continue
                            
                            all_encodings.append(encoding.tolist())
                            
                            face_image = cropped_faces[0]
                            face_path = os.path.join(user_face_folder, f"face_{i+1}.jpg")
                            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(face_path, face_bgr)
                            
                            saved_count += 1
                        except Exception as e:
                            logger.error(f"Error processing face image {i}: {e}")
                            continue
                    
                    if saved_count > 0 and all_encodings:
                        avg_encoding = np.mean(all_encodings, axis=0).tolist()
                        
                        # Update user with face data
                        user.face_images_folder = user_face_folder
                        user.face_encoding = json.dumps({
                            "average": avg_encoding,
                            "all": all_encodings
                        })
                        user.face_registered = True
                        user.face_registered_at = now_vn()
                        db.commit()
                        face_registered = True
            except Exception as e:
                logger.error(f"Error registering face during signup: {e}")
                # Don't fail the registration, just skip face registration
        
        return {
            "message": "Registration successful", 
            "user_id": str(user_id),
            "face_registered": face_registered
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi đăng ký: {str(e)}")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        if not verify_password(request.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect current password"
            )
        
        current_user.password_hash = get_password_hash(request.new_password)
        db.commit()
        
        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi đổi mật khẩu: {str(e)}")
