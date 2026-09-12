# PULMO·AI — Free-Tier Deployment Guide

Comprehensive guide to deploying PULMO·AI on free cloud platforms with zero infrastructure costs.

---

## Quick Start: Recommended Path for Free Deployment

**Fastest option (2-5 min setup):**
1. Push repo to GitHub
2. Deploy Streamlit UI to Streamlit Community Cloud (free tier)
3. Done — live at `https://yourname-pulmo-ai.streamlit.app`

**Why this works:**
- Streamlit Community Cloud: Free tier for public repos, unlimited deployments
- TensorFlow Lite models: Lightweight (~120MB total), cached on first load
- Minimal design: Low memory footprint, fast startup
- No servers to manage, no database, no API keys needed

---

## Option 1: Streamlit Community Cloud (RECOMMENDED ✓)

### Setup (5 minutes)

1. **Push code to GitHub (public repo)**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to [Streamlit Cloud Dashboard](https://share.streamlit.io/)**
   - Click "New app"
   - Select your repo, branch, and main file: `app.py`
   - Click "Deploy"

3. **Wait 2-3 minutes for first build**
   - Streamlit downloads dependencies from `requirements.txt`
   - First load: 60-90 seconds (models cached after)
   - Subsequent loads: <5 seconds

### Features

| Feature | Included |
| :--- | :---: |
| **Hosting** | ✓ Free |
| **Bandwidth** | ✓ Unlimited public traffic |
| **Compute** | ✓ 1 vCPU, 1 GB RAM |
| **Storage** | ✓ 1 GB temporary (models cached in session) |
| **Concurrency** | Up to 3 simultaneous users on free tier |
| **Custom Domain** | ✗ Not on free tier |
| **SSL/HTTPS** | ✓ Automatic |
| **Uptime SLA** | ~99% community standard |

### Limits & Workarounds

| Limit | Value | Workaround |
| :--- | :--- | :--- |
| **Memory** | 1 GB | TFLite models (28-92MB) fit comfortably; Streamlit caches well |
| **Session Duration** | ~48 hours | Automatic restart; no user data persisted |
| **Concurrent Users** | 3 on free | Sufficient for portfolio/demo; upgrade to Pro for higher |
| **Cold Start** | 60-90s first load | Models cached after; subsequent <5s |

### Optimization Tips for Streamlit Cloud

1. **Use `@st.cache_resource` for model loading** (already implemented)
   ```python
   @st.cache_resource
   def load_model():
       detector = TFLiteDetector(...)
       return detector
   ```

2. **Lazy load models on first use, not page load**
   - Streamlit Cloud startup slower if models load immediately
   - Current implementation: loads only when "Analyze" button clicked ✓

3. **Compress images before processing**
   - Already done in `streamlit_minimal.py` ✓

4. **Set secrets (if API keys needed) via Streamlit Cloud UI**
   - Settings → Secrets → paste `.env` contents
   - Not needed for current app (no external APIs)

### Deployment Steps

```bash
# 1. Add Streamlit config for cloud optimization
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'EOF'
[client]
showErrorDetails = false
maxMessageSize = 200

[logger]
level = "error"

[server]
maxUploadSize = 500
headless = true
runOnSave = true
EOF

# 2. Ensure requirements.txt is updated
pip freeze > requirements.txt

# 3. Commit and push
git add requirements.txt
git commit -m "Update requirements for deployment"
git push origin main

# 4. Deploy via Streamlit Cloud UI (no CLI needed)
# Go to https://share.streamlit.io/ → New App
```

### Live Deployment URL Format

```
https://{github-username}-pulmo-ai-{repo-slug}.streamlit.app
# Example: https://anurag-amrev-7557-pneumonia-detection-dl-7a9k2x.streamlit.app
```

---

## Option 2: HuggingFace Spaces (FREE Alternative)

### Setup (8 minutes)

1. **Create HuggingFace account** (free, no credit card)

2. **Create new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `pulmo-ai-detector`
   - Space type: **Streamlit**
   - Visibility: Public
   - Click "Create space"

3. **Upload files**
   ```bash
   git clone https://huggingface.co/spaces/yourusername/pulmo-ai-detector
   cd pulmo-ai-detector
   
   # Copy all project files here
   cp -r /path/to/pneumonia-detection/* .
   
   git add .
   git commit -m "Initial commit"
   git push
   ```

4. **Wait for build** (3-5 minutes)
   - HF Spaces builds automatically from `app.py` + `requirements.txt`
   - Live at: `https://huggingface.co/spaces/yourusername/pulmo-ai-detector`

### Features

| Feature | HF Spaces | Streamlit Cloud | Winner |
| :--- | :---: | :---: | :---: |
| **Free Tier** | ✓ Yes | ✓ Yes | Tie |
| **No Credit Card** | ✓ Yes | ✓ Yes | Tie |
| **Startup Speed** | ~60s | ~60s | Tie |
| **Memory** | 16 GB (generous!) | 1 GB (tight) | **HF** |
| **Ease of Use** | Medium | Very Easy | **Streamlit** |
| **Community** | Large | Larger | **Streamlit** |
| **Custom Domain** | Paid only | Paid (Pro+) | Tie |

### When to Use HF Spaces

- You want more memory (16GB vs 1GB)
- You prefer Git-based deployment workflow
- You want integrated model/dataset versioning
- Models take >5 minutes to load

### Known Issues & Fixes

**Issue**: HF Spaces requires `requirements.txt` with all dependencies
```bash
# Generate clean requirements
pip freeze > requirements.txt

# Or manually specify key deps (cleaner)
cat > requirements.txt << 'EOF'
streamlit>=1.28.0
tensorflow>=2.13.0
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
plotly>=5.17.0
pandas>=2.0.0
EOF
```

**Issue**: First load slow due to model download
- Same as Streamlit Cloud; models cached after first load

---

## Option 3: Replicate.com (Great for Heavy Computation)

**Best for**: Running inference with GPU acceleration

### Setup (10 minutes)

1. **Create Replicate account** (free tier includes)
   - 50 free predictions/month
   - GPU access (A100, A40)

2. **Create `cog.yaml` configuration**
   ```yaml
   build:
     cuda: "11.8"
     python_version: "3.11"
     python_packages:
       - tensorflow==2.13.1
       - opencv-python>=4.8.0
       - numpy>=1.24.0

   predict: "predict.py:predict"
   ```

3. **Create `predict.py` API wrapper**
   ```python
   import cog
   from pathlib import Path
   from src.models.tflite_inference import TFLiteDetector

   class Predictor(cog.Predictor):
       def setup(self):
           self.detector = TFLiteDetector(
               model_path=Path("models/current/best_model.tflite"),
               secondary_model_path=Path("models/current/densenet121_best.tflite")
           )

       @cog.input("image", type=cog.Path, description="Input chest X-ray")
       def predict(self, image):
           result = self.detector.predict(str(image))
           return {
               "diagnosis": result["label"],
               "confidence": result["confidence"],
               "processing_time_ms": result["processing_time"]
           }
   ```

4. **Push to GitHub with Cog config**
5. **Deploy via Replicate**
   - Go to [replicate.com](https://replicate.com)
   - Click "Push a model"
   - Select repo with `cog.yaml`
   - Automatic deployment

### Pros & Cons

| Aspect | Rating | Notes |
| :--- | :---: | :--- |
| **Ease** | ⭐⭐ | Need to learn Cog framework |
| **Cost** | ⭐⭐⭐⭐⭐ | 50 free predictions/month |
| **Speed** | ⭐⭐⭐⭐⭐ | GPU inference much faster |
| **UI** | ⭐⭐ | No web UI; API only |
| **Best For** | High-accuracy demands | Need fast inference |

### When to Use

- You want GPU acceleration (10-50x faster inference)
- You're building an API-first service
- You don't need a web UI
- You want serverless scaling per request

---

## Option 4: Railway.app (Server Deployment)

**Best for**: Running FastAPI server with more control

### Setup (12 minutes)

1. **Create Railway account** (free tier: $5/month credit)

2. **Create `railway.json`**
   ```json
   {
     "build": {
       "builder": "nixpacks"
     },
     "deploy": {
       "startCommand": "python run_app.py"
     }
   }
   ```

3. **Set environment variables in Railway UI**
   - `PORT=8000`
   - `PYTHONUNBUFFERED=1`

4. **Connect GitHub repo → Railway**
   - New Project → GitHub Repo → Deploy
   - Auto-deploys on every push to main

5. **Get live URL**
   ```
   https://pulmo-ai-production.up.railway.app
   ```

### Comparison: Free-Tier Hosting Options

| Platform | Type | Ease | Free Memory | Best For | Setup Time |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Streamlit Cloud** | Web UI | ★★★★★ | 1 GB | Quick demos | 5 min |
| **HF Spaces** | Web UI | ★★★★ | 16 GB | Heavy models | 8 min |
| **Replicate** | API | ★★★ | 50 calls/mo | GPU inference | 10 min |
| **Railway** | Server | ★★★★ | $5 credit | Full control | 12 min |
| **Fly.io** | Server | ★★★★ | 3 shared-cpu | Production | 12 min |

---

## RECOMMENDED DEPLOYMENT PATH

### Tier 1: Quick Portfolio Demo (Today)
```
GitHub → Streamlit Community Cloud
- URL: yourusername-pulmo-ai.streamlit.app
- Time: 5 minutes
- Perfect for: Portfolios, resumes, interviews
- Cost: $0/month
```

### Tier 2: More Reliable (Next Week)
```
GitHub → HuggingFace Spaces (Backup)
- URL: huggingface.co/spaces/yourusername/pulmo-ai
- Time: 8 minutes
- Perfect for: Backup, higher memory needs
- Cost: $0/month
```

### Tier 3: Production API (Optional)
```
GitHub → Railway.app (FastAPI)
- URL: pulmo-ai-production.up.railway.app
- Time: 12 minutes
- Perfect for: Hospital/clinic integration
- Cost: $0-5/month
```

---

## Step-by-Step: Deploy to Streamlit Cloud NOW

### Prerequisites
- GitHub account (free)
- Repo pushed to GitHub (public)
- No model files in git (too large) — they're downloaded at runtime

### Steps

1. **Verify GitHub repo is public**
   ```bash
   cd /path/to/pneumonia-detection
   git remote -v
   # Should show: origin https://github.com/yourusername/pneumonia-detection-dl.git
   ```

2. **Ensure `.gitignore` excludes large files**
   ```bash
   cat .gitignore | grep -E "(models|\.h5|\.tflite|data)"
   # Should see: models/ *.h5 *.tflite data/
   ```

3. **Clean up uncommitted changes**
   ```bash
   git status
   # If any large files: git rm --cached <file>
   git add .
   git commit -m "Final deployment commit"
   git push origin main
   ```

4. **Go to [share.streamlit.io](https://share.streamlit.io/)**
   - Sign in with GitHub
   - Click "New app"
   - Repo: `yourusername/pneumonia-detection-dl`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

5. **Monitor deployment logs**
   - Takes 2-3 minutes on first deploy
   - Watch for model download progress
   - If errors, check "Logs" tab at top right

6. **Share your URL**
   - Copy URL from app header
   - Share in portfolio, resume, interviews
   - Live at: `https://yourusername-pneumonia-detection-dl-xxxxx.streamlit.app`

---

## Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'src'"

**Cause**: `sys.path` not configured correctly

**Fix** (already in `app.py`):
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
```

### Issue: "Memory limit exceeded"

**Cause**: Loading both Keras H5 models (too large for 1GB RAM)

**Fix** (already implemented): Use TFLite models only in Streamlit
- TFLite ResNet: 92MB vs Keras: 211MB
- TFLite DenseNet: 28MB vs Keras: 36MB
- Total: 120MB (vs 247MB)

### Issue: "First prediction takes 90 seconds"

**Expected behavior**: Model caching on first load
- Streamlit caches with `@st.cache_resource`
- Subsequent predictions: <1 second
- Browser refresh clears session cache (cold start again)

**Optimization**: Nothing needed; this is normal

### Issue: "Connection timeout downloading models"

**Cause**: Hugging Face Hub temporarily unavailable

**Fix**: Manually download models to `models/current/` before deployment
```bash
python -c "from src.utils.model_loader import ensure_models_downloaded; ensure_models_downloaded()"
```

### Issue: "App takes 5+ minutes to start"

**Likely causes**:
1. Building dependencies from scratch (happens once)
2. Streamlit Cloud rebuilding image (happens on deploy)
3. Model download failing silently

**Solution**:
- Check Logs tab in Streamlit Cloud dashboard
- Re-deploy: top-right menu → "Rerun app"
- Nuclear option: Delete app, redeploy

---

## Cost Breakdown: Annual Estimate

| Service | Free Tier | Typical Cost (Annual) |
| :--- | :--- | :--- |
| **GitHub** | ✓ Unlimited public repos | $0 |
| **Streamlit Cloud** | ✓ 1 concurrent user, unlimited traffic | $0 (upgrade: $20/mo) |
| **HF Spaces** | ✓ Compute + storage | $0 (upgrade: $20/mo) |
| **Railway** | $5/month credit | $0 (then $10-50/mo) |
| **Domain** (optional) | Included (`.streamlit.app`) | $0 (custom: $12/yr) |
| **Total (free tier)** | | **$0/year** ✓ |

---

## Next Steps After Deployment

1. **Test live app**
   - Upload sample X-ray
   - Verify Grad-CAM visualization
   - Download JSON report

2. **Share in portfolio**
   - LinkedIn: "Just deployed PULMO·AI clinical ML project"
   - Portfolio website: Link to live demo
   - Resume: Add project with live URL

3. **Monitor performance**
   - Streamlit Cloud provides usage analytics
   - Check logs weekly for errors
   - Update code → auto-redeploy

4. **Upgrade when needed**
   - Streamlit Pro: $20/mo (custom domain, more resources)
   - HF Spaces Upgrade: $20/mo (more compute)
   - Railway Pro: Add budget as you scale

---

## Summary: The 5-Minute Deployment

```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy to Streamlit"
git push origin main

# 2. Go to https://share.streamlit.io/
# 3. Click "New App"
# 4. Select repo, app.py
# 5. Click "Deploy"
# 6. Wait 2-3 min

# DONE! Live at: https://yourusername-pneumonia-detection-dl-xxxxx.streamlit.app
```

**Cost**: $0  
**Time**: 5 minutes  
**Result**: Live clinical AI app with zero infrastructure overhead

---

**Questions?** Check the repo [Issues](https://github.com/Anurag-amrev-7557/pneumonia-detection-dl/issues) or see [main README](README.md).
