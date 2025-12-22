# Medical Imaging System - Hướng dẫn cài đặt và chạy

## Tổng quan hệ thống

Hệ thống tái tạo ảnh CT 3D từ X-ray bao gồm:
- **Backend**: FastAPI + SQLAlchemy + Celery + Socket.IO
- **Frontend**: React + MUI + Socket.IO Client
- **Database**: PostgreSQL
- **Message Queue**: RabbitMQ + Redis

## Yêu cầu hệ thống

- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis (Docker hoặc native)
- RabbitMQ (Docker hoặc native)
- CUDA 12.x (cho inference)
- GPU NVIDIA với >= 12GB VRAM

## 1. Cài đặt Database

### PostgreSQL
```bash
# Tạo database
sudo -u postgres psql
CREATE DATABASE medical_imaging;
CREATE USER medical_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE medical_imaging TO medical_user;
\q
```

### Redis (Docker)
```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

### RabbitMQ (Docker)
```bash
docker run -d --name rabbitmq \
  -p 5670:5672 \
  -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=user \
  -e RABBITMQ_DEFAULT_PASS=adminace123 \
  rabbitmq:3-management
```

## 2. Cài đặt Backend

```bash
cd DoAnPtit_Backend

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Tạo file .env
cat > .env << EOF
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medical_imaging
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5670
RABBITMQ_USER=user
RABBITMQ_PASSWORD=adminace123
SECRET_KEY=your-super-secret-key-change-in-production
EOF
```

## 3. Chạy Backend

### Terminal 1 - FastAPI Server
```bash
cd DoAnPtit_Backend
source venv/bin/activate
uvicorn app.main:application --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Celery Worker (cho inference)
```bash
cd DoAnPtit_Backend
source venv/bin/activate

# Kích hoạt conda environment cho PyTorch
conda activate Xray2CT

# Chạy Celery worker
celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

## 4. Cài đặt Frontend

```bash
cd DoAnPtit_FrontEnd

# Cài đặt dependencies
npm install

# Tạo file .env
cat > .env << EOF
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_SOCKET_URL=http://localhost:8000
EOF

# Chạy development server
npm start
```

## 5. Truy cập hệ thống

- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **RabbitMQ Management**: http://localhost:15672 (user/adminace123)

## 6. Tạo tài khoản admin đầu tiên

```python
# Chạy script tạo admin
python -c "
from app.db.session import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash
import uuid

db = SessionLocal()
admin = User(
    id=uuid.uuid4(),
    username='admin',
    email='admin@medical.local',
    password_hash=get_password_hash('admin123'),
    role='admin',
    full_name='System Admin',
    is_active=True
)
db.add(admin)
db.commit()
print('Admin created: admin / admin123')
"
```

## Cấu trúc thư mục

```
DoAnPtit_Backend/
├── app/
│   ├── api/           # API routes
│   │   └── endpoints/
│   ├── core/          # Config, security
│   ├── db/            # Database session
│   ├── models/        # SQLAlchemy models
│   ├── schemas/       # Pydantic schemas
│   ├── socket/        # Socket.IO manager
│   ├── worker/        # Celery tasks
│   └── main.py
├── patient_files/     # Uploaded files
├── uploads/           # Temporary uploads
└── requirements.txt

DoAnPtit_FrontEnd/
├── src/
│   ├── components/    # Reusable components
│   ├── contexts/      # React contexts
│   ├── layouts/       # Page layouts
│   ├── pages/         # Page components
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── patients/
│   │   └── medical-records/
│   └── services/      # API & Socket services
└── package.json
```

## Quy trình sử dụng

1. **Đăng nhập** với tài khoản admin hoặc doctor
2. **Thêm bệnh nhân** mới
3. **Tạo hồ sơ bệnh án** cho bệnh nhân
4. **Tải ảnh X-ray** lên hồ sơ
5. **Bấm "Bắt đầu tái tạo CT"** - hệ thống sẽ:
   - Đẩy task vào RabbitMQ
   - Celery worker nhận task và chạy inference
   - Gửi thông báo realtime qua Socket.IO khi hoàn thành
6. **Xem kết quả CT 3D** trong CT Viewer

## Troubleshooting

### Lỗi kết nối database
```bash
# Kiểm tra PostgreSQL
sudo systemctl status postgresql
```

### Lỗi Celery không nhận task
```bash
# Kiểm tra RabbitMQ
docker logs rabbitmq
# Kiểm tra Redis
docker logs redis
```

### Lỗi CUDA out of memory
- Giảm batch size trong inference
- Sử dụng FP16 inference

## API Endpoints chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | /api/v1/auth/login | Đăng nhập |
| POST | /api/v1/auth/register | Đăng ký |
| GET | /api/v1/patients | Danh sách bệnh nhân |
| POST | /api/v1/patients | Thêm bệnh nhân |
| GET | /api/v1/medical-records/{id} | Chi tiết hồ sơ |
| POST | /api/v1/inference/upload | Upload X-ray & bắt đầu inference |
| GET | /api/v1/inference/status/{id} | Trạng thái inference |
