"""
Authentication Endpoints
"""
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

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
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
            full_name=user.full_name
        )
    )


@router.post("/login/json", response_model=Token)
async def login_json(
    request: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with JSON body"""
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
            full_name=user.full_name
        )
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
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


@router.post("/register")
async def register(
    user_data: dict,
    db: Session = Depends(get_db)
):
    """Register new user account"""
    import uuid
    from app.models.doctor import Doctor
    
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
    if user_data.get("role") == "doctor":
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
    
    return {"message": "Registration successful", "user_id": str(user_id)}


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
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    current_user.password_hash = get_password_hash(request.new_password)
    db.commit()
    
    return {"message": "Password changed successfully"}
