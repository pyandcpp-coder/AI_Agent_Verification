# Production Deployment Guide - AI Agent Verification System

## 📋 Prerequisites

1. **Docker & Docker Compose** installed
2. **NVIDIA Docker Runtime** (for GPU support)
3. **Model Files** (best4.pt, best.pt) placed in `models/` directory
4. **Minimum 8GB RAM**, 20GB disk space
5. **GPU recommended** (NVIDIA with CUDA support)

## 🚀 Quick Start

### 1. Prepare Environment

```bash
# Clone repository
cd /path/to/AI_Agent_Verification

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Ensure models are in place
ls -lh models/
# Should see: best4.pt, best.pt
```

### 2. Start Services

```bash
# Build and start all services
./docker-start.sh

# Or manually:
docker-compose up -d --build
```

### 3. Verify Deployment

```bash
# Check health
curl http://localhost:8101/health

# Expected response:
# {"status":"healthy","gpu_enabled":true,"components":{...}}

# View logs
docker-compose logs -f app
```

## 📊 Service Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Application   │────▶│     Redis       │
│   (Port 8101)   │     │   (Port 6379)   │
└─────────────────┘     └─────────────────┘
         │
         │ (optional debug)
         ▼
┌─────────────────┐
│ Redis Commander │
│   (Port 8081)   │
└─────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | 0.0.0.0 | Server bind address |
| `APP_PORT` | 8101 | Application port |
| `WORKERS` | 1 | Gunicorn workers (keep at 1 for GPU) |
| `USE_GPU` | true | Enable GPU acceleration |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `REDIS_HOST` | redis | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `REDIS_TTL_HOURS` | 24 | Cache TTL in hours |
| `ENABLE_QWEN_FALLBACK` | true | Enable Qwen VL fallback |
| `ENABLE_DEBUG_IMAGES` | false | Save debug images |

### Resource Limits

Edit `docker-compose.yml` to adjust:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

## 📝 Logging

Logs are stored in `./logs/` directory:

- `application.log` - All application logs (50MB max, 10 backups)
- `errors.log` - Error logs only (20MB max, 5 backups)
- `verifications.log` - All verification requests (100MB max, 20 backups)
- `access.log` - HTTP access logs
- `error.log` - HTTP error logs

View live logs:
```bash
# Application logs
docker-compose logs -f app

# All logs
tail -f logs/application.log

# Errors only
tail -f logs/errors.log

# Verification tracking
tail -f logs/verifications.log
```

## 🔍 Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8101/health
```

Response:
```json
{
  "status": "healthy",
  "gpu_enabled": true,
  "components": {
    "face_agent": true,
    "entity_agent": true,
    "gender_pipeline": true,
    "scorer": true,
    "redis": true
  }
}
```

### Redis Monitoring (Debug Mode)

Start with debug profile to access Redis Commander:

```bash
docker-compose --profile debug up -d
```

Access Redis Commander: http://localhost:8081

## 🔐 Security Considerations

1. **Non-root User**: Application runs as `appuser` (UID 1000)
2. **Read-only Models**: Models mounted as read-only
3. **Network Isolation**: Services in isolated Docker network
4. **Resource Limits**: CPU and memory limits enforced
5. **Temporary File Cleanup**: Automatic cleanup of temp files

## 🛠 Maintenance

### View Service Status

```bash
docker-compose ps
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart app only
docker-compose restart app
```

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up -d --build
```

### Clear Cache

```bash
# Access Redis CLI
docker-compose exec redis redis-cli

# In Redis CLI:
FLUSHDB  # Clear current database
FLUSHALL # Clear all databases
```

### Backup Data

```bash
# Backup Aadhaar records
docker cp kyc_verification_app:/app/data/aadhaar_records.pkl ./backup/

# Backup Redis data
docker-compose exec redis redis-cli SAVE
docker cp kyc_redis:/data/dump.rdb ./backup/
```

## 📈 Performance Tuning

### For High Load

1. **Increase Workers** (CPU-only mode):
   ```yaml
   environment:
     - WORKERS=4
   ```

2. **Increase Redis Memory**:
   ```yaml
   command: redis-server --maxmemory 1024mb
   ```

3. **Adjust Cleanup Intervals**:
   ```yaml
   environment:
     - CLEANUP_INTERVAL=60  # More frequent
     - TEMP_FILE_TTL=180     # Shorter TTL
   ```

### For GPU Optimization

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specific GPU
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU ID
          capabilities: [gpu]
```

## 🐛 Troubleshooting

### Application Won't Start

```bash
# Check logs
docker-compose logs app

# Common issues:
# 1. Model files missing
ls -lh models/

# 2. Port already in use
sudo lsof -i :8101

# 3. GPU not available
nvidia-smi
```

### Redis Connection Failed

```bash
# Check Redis status
docker-compose ps redis

# Test connection
docker-compose exec app nc -zv redis 6379

# Check Redis logs
docker-compose logs redis
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
# Or reduce model batch sizes
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Ensure nvidia-docker2 installed
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and Deploy
        run: |
          docker-compose build
          docker-compose up -d
      
      - name: Health Check
        run: |
          sleep 30
          curl -f http://localhost:8101/health || exit 1
```

## 📊 Scaling

### Horizontal Scaling (Load Balancer)

```nginx
# nginx.conf
upstream kyc_backend {
    least_conn;
    server kyc_app_1:8101;
    server kyc_app_2:8101;
    server kyc_app_3:8101;
}

server {
    listen 80;
    location / {
        proxy_pass http://kyc_backend;
    }
}
```

### Redis Cluster (High Availability)

Use Redis Sentinel or Redis Cluster for production HA.

## 📞 Support

- Check logs: `./logs/`
- Health endpoint: `/health`
- Debug images: `./debug_output/` (if enabled)

## 🔄 Shutdown

```bash
# Stop services
./docker-stop.sh

# Or manually
docker-compose down

# Remove volumes (CAUTION: Deletes data)
docker-compose down -v
```
