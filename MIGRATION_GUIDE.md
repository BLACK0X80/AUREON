# AUREON Migration Guide

## Overview

This guide helps you migrate from the basic AUREON setup to the production-grade version with all advanced features.

## Migration Steps

### 1. Database Migration (SQLite → PostgreSQL)

#### Backup Current Data

```bash
# Backup SQLite database
cp aureon.db aureon_backup.db
```

#### Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql postgresql-server postgresql-contrib

# macOS
brew install postgresql
```

#### Create Database

```sql
-- Connect to PostgreSQL
sudo -u postgres psql

-- Create database and user
CREATE DATABASE aureon;
CREATE USER aureon WITH PASSWORD 'aureon_password';
GRANT ALL PRIVILEGES ON DATABASE aureon TO aureon;
\q
```

#### Run Migrations

```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

### 2. Environment Configuration

#### Update Environment Variables

```bash
# Copy example environment file
cp env.example .env

# Edit configuration
nano .env
```

#### Key Changes

```bash
# Database URL
AUREON_DATABASE_URL=postgresql://aureon:aureon_password@localhost:5432/aureon

# Redis URL
AUREON_REDIS_URL=redis://localhost:6379/0

# Monitoring
AUREON_PROMETHEUS_ENDPOINT=http://localhost:9090
AUREON_GRAFANA_ENDPOINT=http://localhost:3000
```

### 3. Service Dependencies

#### Install Redis

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# CentOS/RHEL
sudo yum install redis

# macOS
brew install redis

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis
```

#### Install Monitoring Stack

```bash
# Start monitoring services
docker-compose up -d prometheus grafana
```

### 4. Application Updates

#### Install New Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Update Application Code

The new version includes:

- Async database operations
- Redis caching
- Celery background tasks
- WebSocket support
- Advanced ML features

### 5. Configuration Migration

#### Update config.yaml

```yaml
# Add new configuration sections
monitoring:
  prometheus_endpoint: "http://localhost:9090"
  grafana_endpoint: "http://localhost:3000"

ml:
  model_cache_size: 100
  prediction_cache_ttl: 3600

automl:
  max_trials: 100
  timeout_minutes: 60
```

### 6. Data Migration

#### Migrate Existing Data

```python
# Migration script
import sqlite3
import psycopg2
import pandas as pd

# Connect to SQLite
sqlite_conn = sqlite3.connect('aureon_backup.db')

# Connect to PostgreSQL
pg_conn = psycopg2.connect(
    host="localhost",
    database="aureon",
    user="aureon",
    password="aureon_password"
)

# Migrate experiments
experiments_df = pd.read_sql_query("SELECT * FROM experiments", sqlite_conn)
experiments_df.to_sql('experiments', pg_conn, if_exists='append', index=False)

# Migrate models
models_df = pd.read_sql_query("SELECT * FROM models", sqlite_conn)
models_df.to_sql('models', pg_conn, if_exists='append', index=False)

# Close connections
sqlite_conn.close()
pg_conn.close()
```

### 7. Testing Migration

#### Run Tests

```bash
# Test database connection
python -c "from aureon.services.database import DatabaseService; print('Database OK')"

# Test Redis connection
python -c "from aureon.services.redis_service import RedisService; print('Redis OK')"

# Run full test suite
pytest tests/
```

#### Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Prometheus metrics
curl http://localhost:9090/metrics

# Check Grafana
curl http://localhost:3000/api/health
```

### 8. Deployment Migration

#### Docker Deployment

```bash
# Build new image
docker build -t aureon:latest .

# Stop old containers
docker-compose down

# Start new services
docker-compose up -d
```

#### Kubernetes Deployment

```bash
# Apply new manifests
kubectl apply -f k8s/

# Or use Helm
helm upgrade aureon ./helm/aureon
```

### 9. Feature Enablement

#### Enable Advanced Features

```python
# AutoML
from aureon.services.automl_service import AutoMLService
automl_service = AutoMLService()

# Feature Engineering
from aureon.services.feature_engineering_service import FeatureEngineeringService
fe_service = FeatureEngineeringService()

# Model Compression
from aureon.services.model_compression_service import ModelCompressionService
compression_service = ModelCompressionService()

# Federated Learning
from aureon.services.federated_learning_service import FederatedLearningService
fl_service = FederatedLearningService()
```

### 10. Monitoring Setup

#### Configure Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Import dashboards from `monitoring/grafana/dashboards/`
3. Configure data sources

#### Set Up Alerts

1. Configure AlertManager
2. Set up notification channels
3. Test alert rules

### 11. Performance Optimization

#### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_models_created_at ON models(created_at);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
```

#### Redis Optimization

```bash
# Configure Redis
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### 12. Security Hardening

#### Update Secrets

```bash
# Generate new secrets
openssl rand -hex 32  # For AUREON_SECRET_KEY
openssl rand -hex 32  # For AUREON_JWT_SECRET
```

#### Configure SSL

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### 13. Backup Strategy

#### Database Backup

```bash
# Create backup script
#!/bin/bash
pg_dump -h localhost -U aureon -d aureon > backup_$(date +%Y%m%d_%H%M%S).sql
```

#### Model Backup

```bash
# Backup model files
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz models/
```

### 14. Rollback Plan

#### If Migration Fails

```bash
# Stop new services
docker-compose down

# Restore SQLite
cp aureon_backup.db aureon.db

# Start old version
python -m aureon.api.main
```

### 15. Post-Migration Verification

#### Check All Services

```bash
# API
curl http://localhost:8000/health

# Database
psql -h localhost -U aureon -d aureon -c "SELECT COUNT(*) FROM experiments;"

# Redis
redis-cli ping

# Prometheus
curl http://localhost:9090/api/v1/targets

# Grafana
curl http://localhost:3000/api/health
```

#### Performance Testing

```bash
# Run benchmarks
python scripts/benchmark_mlflow.py

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U aureon -d aureon
```

#### Redis Connection Issues

```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping
```

#### Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :5432
netstat -tulpn | grep :6379
```

### Performance Issues

#### High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head
```

#### Slow Database Queries

```sql
-- Check slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## Support

If you encounter issues during migration:

1. Check the logs: `docker-compose logs aureon`
2. Review the troubleshooting section
3. Open an issue on GitHub
4. Contact support: support@aureon.com

## Migration Checklist

- [ ] Backup existing data
- [ ] Install PostgreSQL
- [ ] Install Redis
- [ ] Update environment variables
- [ ] Run database migrations
- [ ] Install new dependencies
- [ ] Update configuration files
- [ ] Test all services
- [ ] Configure monitoring
- [ ] Set up alerts
- [ ] Test advanced features
- [ ] Performance testing
- [ ] Security hardening
- [ ] Backup strategy
- [ ] Documentation update
