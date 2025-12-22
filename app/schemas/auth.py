"""
Authentication Schemas
"""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class UserInfo(BaseModel):
    id: UUID
    username: str
    email: str
    role: str
    full_name: Optional[str] = None


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Optional[UserInfo] = None


class TokenPayload(BaseModel):
    sub: str
    role: str
    type: str
    exp: int


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
