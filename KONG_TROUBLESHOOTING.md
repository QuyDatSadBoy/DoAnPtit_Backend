# Kong Gateway - Troubleshooting Guide

## Mục lục
- [Vấn đề gặp phải](#vấn-đề-gặp-phải)
- [Phân tích nguyên nhân](#phân-tích-nguyên-nhân)
- [Giải pháp khắc phục](#giải-pháp-khắc-phục)
- [Cấu hình Kong cho Medical Imaging API](#cấu-hình-kong-cho-medical-imaging-api)

---

## Vấn đề gặp phải

### Triệu chứng
Khi gọi API thông qua Kong Gateway (port 8000), nhận được lỗi:
```json
{
  "message": "The upstream server is timing out",
  "request_id": "..."
}
```

Kong logs hiển thị:
```
upstream timed out (110: Connection timed out) while connecting to upstream
```

### Môi trường
- **Kong Gateway**: Chạy trong Docker container (port 8000/8001)
- **Backend API**: FastAPI chạy trên host machine (port 8999)
- **Docker network của Kong**: `172.19.0.0/16`
- **Docker bridge network (docker0)**: `172.17.0.0/16`

---

## Phân tích nguyên nhân

### 1. Docker Network Isolation

Khi Kong chạy trong Docker container, nó tạo ra một network riêng biệt:

```
┌─────────────────────────────────────────────────────────────┐
│                     Host Machine                            │
│  ┌─────────────────┐                                        │
│  │ Backend API     │◄── Firewall chặn traffic từ 172.19.x.x │
│  │ (localhost:8999)│                                        │
│  └─────────────────┘                                        │
│          ▲                                                  │
│          │ Connection Timeout!                              │
│          │                                                  │
│  ┌───────┴───────────────────────────────────────────────┐  │
│  │ Docker Network: 172.19.0.0/16                         │  │
│  │  ┌──────────────────┐                                 │  │
│  │  │ Kong Container   │                                 │  │
│  │  │ IP: 172.19.0.x   │──────► Gọi 172.17.0.1:8999      │  │
│  │  └──────────────────┘        hoặc host.docker.internal│  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2. Firewall Rules

Hệ thống (iptables) chỉ cho phép traffic từ một số network cụ thể:

```bash
# Kiểm tra firewall rules
sudo iptables -L INPUT -n --line-numbers
```

**Vấn đề**: Firewall cho phép `172.17.0.0/16` (docker0 bridge) nhưng KHÔNG cho phép `172.19.0.0/16` (Kong network).

### 3. Xác định Docker Network của Kong

```bash
# Tìm container Kong
docker ps | grep kong

# Kiểm tra IP của Kong container
docker inspect <kong_container_id> | grep -A 20 "Networks"

# Hoặc kiểm tra tất cả Docker networks
docker network ls
docker network inspect <network_name>
```

---

## Giải pháp khắc phục

### Bước 1: Mở Firewall cho Kong Network

```bash
# Thêm rule cho phép traffic từ Kong network
sudo iptables -I INPUT 1 -s 172.19.0.0/16 -j ACCEPT

# Verify rule đã được thêm
sudo iptables -L INPUT -n --line-numbers | head -10
```

### Bước 2: Lưu Firewall Rules (Persistent)

**Option A: Sử dụng iptables-save**
```bash
sudo iptables-save | sudo tee /etc/iptables.rules
```

**Option B: Sử dụng UFW (Ubuntu)**
```bash
sudo ufw allow from 172.19.0.0/16
```

**Option C: Thêm vào rc.local**
```bash
echo "iptables -I INPUT 1 -s 172.19.0.0/16 -j ACCEPT" | sudo tee -a /etc/rc.local
```

### Bước 3: Test kết nối

```bash
# Test từ host
curl http://localhost:8999/api/v1/medical/health

# Test qua Kong
curl http://localhost:8000/api/v1/medical/health
```

---

## Cấu hình Kong cho Medical Imaging API

### Tạo Service

```bash
curl -i -X POST http://localhost:8001/services/ \
  --data name=api-medical \
  --data protocol=http \
  --data host=192.168.1.175 \
  --data port=8999
```

> **Lưu ý**: Sử dụng IP thực của host machine (`192.168.1.175`), KHÔNG dùng `localhost` hoặc `host.docker.internal` nếu gặp vấn đề.

### Tạo Route

```bash
curl -i -X POST http://localhost:8001/services/api-medical/routes \
  --data name=api-medical-route \
  --data paths[]=/api/v1/medical \
  --data strip_path=false
```

> **Quan trọng**: `strip_path=false` để giữ nguyên path `/api/v1/medical` khi forward đến backend.

### Kiểm tra cấu hình

```bash
# List services
curl -s http://localhost:8001/services | jq

# List routes
curl -s http://localhost:8001/routes | jq

# Kiểm tra chi tiết route
curl -s http://localhost:8001/routes/api-medical-route | jq '{paths, strip_path}'
```

### Cập nhật Route (nếu cần)

```bash
# Cập nhật strip_path
curl -X PATCH http://localhost:8001/routes/api-medical-route \
  --data strip_path=false

# Cập nhật paths
curl -X PATCH http://localhost:8001/routes/api-medical-route \
  --data paths[]=/api/v1/medical
```

### Xóa và tạo lại (nếu cần)

```bash
# Xóa route
curl -X DELETE http://localhost:8001/routes/api-medical-route

# Xóa service
curl -X DELETE http://localhost:8001/services/api-medical
```

---

## Lưu ý quan trọng

### 1. FastAPI root_path vs prefix

**KHÔNG sử dụng `root_path`** trong FastAPI khi có StaticFiles:

```python
# ❌ SAI - root_path gây lỗi 404 cho static files
app = FastAPI(root_path="/api/v1/medical")

# ✅ ĐÚNG - Sử dụng prefix cho router
app = FastAPI()
app.include_router(api_router, prefix="/api/v1/medical")
```

### 2. Kong strip_path

| strip_path | Request đến Kong | Forward đến Backend |
|------------|------------------|---------------------|
| true       | /api/v1/medical/health | /health |
| false      | /api/v1/medical/health | /api/v1/medical/health |

Với cấu hình hiện tại, cần `strip_path=false`.

### 3. Kiểm tra Kong logs

```bash
# Xem logs real-time
docker logs -f <kong_container_id>

# Hoặc nếu dùng docker-compose
docker-compose logs -f kong
```

---

## Quick Reference Commands

```bash
# === FIREWALL ===
# Mở firewall cho Kong network
sudo iptables -I INPUT 1 -s 172.19.0.0/16 -j ACCEPT

# === KONG CONFIG ===
# Tạo service
curl -X POST http://localhost:8001/services/ \
  -d name=api-medical \
  -d host=192.168.1.175 \
  -d port=8999

# Tạo route
curl -X POST http://localhost:8001/services/api-medical/routes \
  -d name=api-medical-route \
  -d 'paths[]=/api/v1/medical' \
  -d strip_path=false

# === TEST ===
# Health check qua Kong
curl http://localhost:8000/api/v1/medical/health
```

---

## Troubleshooting Checklist

- [ ] Backend đang chạy và accessible tại `localhost:8999`
- [ ] Firewall cho phép traffic từ Docker network của Kong
- [ ] Kong service trỏ đúng host IP và port
- [ ] Kong route có `strip_path=false` 
- [ ] Backend sử dụng `prefix` thay vì `root_path`
- [ ] Test health endpoint qua Kong thành công

---

*Cập nhật lần cuối: 24/12/2024*
