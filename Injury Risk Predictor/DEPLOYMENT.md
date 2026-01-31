# Deployment Guide

This guide covers deploying the Football Injury Risk Predictor to various cloud platforms.

---

## ⚠️ Pre-Deployment Checklist

Before deploying, ensure:

- [ ] All models are trained and saved in `models/` directory
- [ ] Required data files are available or will be uploaded
- [ ] `requirements.txt` is up to date
- [ ] Environment variables are configured
- [ ] Disclaimers are prominent in the UI
- [ ] You have read and understood the limitations (see README.md)

---

## Deployment Options

### 1. Streamlit Cloud (Recommended for Quick Deployment)

Streamlit Cloud offers free hosting for Streamlit apps with GitHub integration.

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))
- App code in a GitHub repository

#### Steps

1. **Prepare your repository**:
   ```bash
   # Ensure these files are in your repo root:
   # - app.py (main Streamlit application)
   # - requirements.txt
   # - .streamlit/config.toml (optional customization)
   ```

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Choose branch (usually `main` or `master`)
   - Set main file path: `app.py`
   - Click "Deploy"

4. **Configure secrets** (if needed):
   - In Streamlit Cloud dashboard, go to app settings
   - Click "Secrets"
   - Add any API keys or sensitive data in TOML format:
   ```toml
   # Example secrets
   [database]
   host = "your-db-host"
   password = "your-password"
   ```

#### Environment Variables for Streamlit Cloud

Create `.streamlit/secrets.toml` locally (DO NOT commit to git):
```toml
# API Keys (if using external data sources)
api_key = "your-api-key"

# Database credentials (if applicable)
[database]
host = "db-host"
port = 5432
database = "injury_db"
username = "user"
password = "password"

# Model paths (if using cloud storage)
model_path = "s3://bucket/models/"
```

Add `.streamlit/secrets.toml` to `.gitignore`:
```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

#### Resource Limits
- Free tier: 1 GB RAM, 1 CPU core
- For larger models, consider upgrading or using model compression

---

### 2. Render

Render offers free hosting for web services with automatic deploys from GitHub.

#### Steps

1. **Create `render.yaml`** in project root:
   ```yaml
   services:
     - type: web
       name: injury-risk-predictor
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
       envVars:
         - key: PYTHON_VERSION
           value: 3.9.16
   ```

2. **Push to GitHub**:
   ```bash
   git add render.yaml
   git commit -m "Add Render configuration"
   git push origin main
   ```

3. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Click "Create Web Service"

4. **Configure environment variables**:
   - In Render dashboard, go to "Environment"
   - Add any required variables

#### Environment Variables for Render

Set in Render dashboard:
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=10000
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

### 3. Railway

Railway offers simple deployment with GitHub integration and generous free tier.

#### Steps

1. **Install Railway CLI** (optional):
   ```bash
   npm install -g railway
   ```

2. **Deploy via Railway Dashboard**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway auto-detects Python and installs dependencies

3. **Configure start command**:
   - In Railway dashboard, go to "Settings"
   - Set start command:
   ```bash
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

4. **Add environment variables**:
   - Go to "Variables" tab
   - Add required environment variables

#### Environment Variables for Railway

```
PORT=8080
PYTHON_VERSION=3.9
STREAMLIT_SERVER_HEADLESS=true
```

---

### 4. Fly.io

Fly.io offers global deployment with Docker containers.

#### Prerequisites
- Fly.io account
- Flyctl CLI installed

#### Steps

1. **Install Flyctl**:
   ```bash
   # macOS
   brew install flyctl

   # Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login to Fly.io**:
   ```bash
   flyctl auth login
   ```

3. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application
   COPY . .

   # Expose port
   EXPOSE 8080

   # Run Streamlit
   CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
   ```

4. **Create `fly.toml`**:
   ```toml
   app = "injury-risk-predictor"

   [build]
     dockerfile = "Dockerfile"

   [[services]]
     internal_port = 8080
     protocol = "tcp"

     [[services.ports]]
       handlers = ["http"]
       port = 80

     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443

   [env]
     STREAMLIT_SERVER_HEADLESS = "true"
   ```

5. **Deploy**:
   ```bash
   flyctl launch
   flyctl deploy
   ```

6. **Set secrets**:
   ```bash
   flyctl secrets set API_KEY=your-api-key
   ```

---

## Optimizing for Deployment

### Model Size Reduction

Large models can cause deployment issues. Consider:

1. **Model compression**:
   ```python
   import joblib
   from sklearn.tree import DecisionTreeClassifier

   # Save with compression
   joblib.dump(model, 'model.pkl', compress=3)
   ```

2. **Use smaller models**:
   - Reduce `n_estimators` for tree-based models
   - Prune less important features
   - Use model distillation

3. **Lazy loading**:
   ```python
   import streamlit as st

   @st.cache_resource
   def load_models():
       import joblib
       cat_model = joblib.load('models/catboost_classifier.pkl')
       lgb_model = joblib.load('models/lightgbm_classifier.pkl')
       xgb_model = joblib.load('models/xgboost_classifier.pkl')
       return cat_model, lgb_model, xgb_model
   ```

### Data Handling

For large datasets:

1. **Use cloud storage**:
   ```python
   import boto3
   import pandas as pd

   def load_data_from_s3(bucket, key):
       s3 = boto3.client('s3')
       obj = s3.get_object(Bucket=bucket, Key=key)
       return pd.read_csv(obj['Body'])
   ```

2. **Use parquet format**:
   ```python
   # Convert CSV to Parquet (smaller, faster)
   df.to_parquet('data.parquet', compression='gzip')
   df = pd.read_parquet('data.parquet')
   ```

3. **Sample data for demos**:
   ```python
   # Use subset for faster loading
   df = pd.read_csv('large_data.csv', nrows=10000)
   ```

### Performance Optimization

1. **Use caching**:
   ```python
   import streamlit as st

   @st.cache_data
   def expensive_computation(data):
       # This will only run once per unique input
       return data.groupby('player').mean()
   ```

2. **Async data loading**:
   ```python
   import streamlit as st

   with st.spinner('Loading models...'):
       models = load_models()
   ```

3. **Optimize imports**:
   ```python
   # Import only what you need
   from sklearn.metrics import roc_auc_score  # Not: import sklearn
   ```

---

## Monitoring and Maintenance

### Health Checks

Add a health check endpoint:
```python
# Add to app.py
import streamlit as st

def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

### Logging

Use Streamlit's built-in logging:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")
logger.error("Prediction failed", exc_info=True)
```

### Error Handling

Graceful error handling:
```python
import streamlit as st

try:
    prediction = model.predict(features)
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    logger.error(f"Prediction failed: {e}", exc_info=True)
```

---

## Security Best Practices

1. **Never commit secrets**:
   - Use `.gitignore` for sensitive files
   - Use environment variables for secrets
   - Use platform-specific secret management

2. **Validate inputs**:
   ```python
   def validate_player_input(player_data):
       required_fields = ['name', 'age', 'position']
       for field in required_fields:
           if field not in player_data:
               raise ValueError(f"Missing required field: {field}")
   ```

3. **Add rate limiting** (for production):
   ```python
   from functools import wraps
   import time

   def rate_limit(max_calls, period):
       calls = []
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               now = time.time()
               calls[:] = [c for c in calls if c > now - period]
               if len(calls) >= max_calls:
                   raise Exception("Rate limit exceeded")
               calls.append(now)
               return func(*args, **kwargs)
           return wrapper
       return decorator
   ```

---

## Troubleshooting

### Common Issues

1. **Memory errors**:
   - Reduce model size
   - Use model compression
   - Upgrade to paid tier with more RAM

2. **Slow loading**:
   - Use `@st.cache_resource` for models
   - Use `@st.cache_data` for data
   - Lazy load heavy dependencies

3. **Port issues**:
   - Ensure app uses `$PORT` environment variable
   - Default to 8080 if not set:
   ```python
   import os
   port = int(os.environ.get('PORT', 8080))
   ```

4. **Module import errors**:
   - Verify all dependencies in `requirements.txt`
   - Pin versions to avoid conflicts
   - Use virtual environment locally to test

5. **Model file not found**:
   - Ensure models are committed (if small)
   - Use Git LFS for large files
   - Or use cloud storage (S3, GCS)

---

## Quick Deployment Checklist

Before deploying:

- [ ] Test app locally: `streamlit run app.py`
- [ ] Check `requirements.txt` is complete
- [ ] Add `.streamlit/config.toml` for customization
- [ ] Set up `.gitignore` for secrets
- [ ] Add prominent disclaimers in UI
- [ ] Test with sample data
- [ ] Configure environment variables
- [ ] Set up error logging
- [ ] Add health check (optional)
- [ ] Document any manual setup steps
- [ ] Test mobile responsiveness

---

## Platform Comparison

| Platform | Free Tier | Ease of Setup | Best For |
|----------|-----------|---------------|----------|
| **Streamlit Cloud** | 1GB RAM | ⭐⭐⭐⭐⭐ | Quick demos, prototypes |
| **Render** | 512MB RAM | ⭐⭐⭐⭐ | Production apps, APIs |
| **Railway** | $5 credit/month | ⭐⭐⭐⭐ | Full-stack apps |
| **Fly.io** | 3 VMs free | ⭐⭐⭐ | Global apps, Docker |

---

## Next Steps

After deployment:
1. Test all features in production
2. Monitor for errors and performance
3. Gather user feedback
4. Iterate and improve
5. Consider adding:
   - User authentication
   - Database integration
   - API endpoints
   - Advanced analytics
   - A/B testing

---

**Need help?** Check platform-specific documentation or open an issue on GitHub.
