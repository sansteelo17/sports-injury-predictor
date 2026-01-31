# Deployment Checklist

Use this checklist to ensure you're ready to deploy the Football Injury Risk Predictor.

---

## Pre-Deployment Checklist

### Essential Files

- [ ] `app.py` - Main Streamlit application exists and is functional
- [ ] `requirements.txt` - All dependencies listed with versions
- [ ] `.streamlit/config.toml` - Streamlit configuration present
- [ ] `.gitignore` - Sensitive files and large datasets excluded
- [ ] `README.md` - Comprehensive documentation complete
- [ ] `DEPLOYMENT.md` - Deployment instructions ready

### Code Quality

- [ ] Application runs locally without errors
- [ ] All imports resolve correctly
- [ ] No hardcoded paths or credentials in code
- [ ] Error handling implemented for critical functions
- [ ] Loading states and spinners added for long operations
- [ ] Input validation in place

### Models & Data

- [ ] Models trained and saved (or using demo mode)
- [ ] Model files compressed if large (>100MB)
- [ ] Data files prepared or using sample data
- [ ] Data loading tested and works
- [ ] Model prediction pipeline tested

### Security

- [ ] No API keys or secrets committed to git
- [ ] `.env.example` created with template variables
- [ ] `.streamlit/secrets.toml` in `.gitignore`
- [ ] Sensitive data handling reviewed
- [ ] User input validation implemented

### Documentation

- [ ] README.md includes prominent disclaimers
- [ ] Installation instructions are clear
- [ ] Usage examples provided
- [ ] Limitations clearly stated
- [ ] Use cases and non-use cases documented

### User Experience

- [ ] Disclaimers prominent in UI (NOT medical advice)
- [ ] Input forms are intuitive
- [ ] Results are clearly displayed
- [ ] Help sections available
- [ ] Mobile-responsive design tested

---

## Platform-Specific Checklists

### Streamlit Cloud

- [ ] GitHub repository created and pushed
- [ ] Repository is public or you have Streamlit Cloud Teams
- [ ] `app.py` in repository root
- [ ] `requirements.txt` complete and tested
- [ ] `.streamlit/config.toml` configured
- [ ] Secrets configured in Streamlit Cloud dashboard (if needed)
- [ ] Resource limits checked (1GB RAM on free tier)

### Render

- [ ] `render.yaml` created (optional but recommended)
- [ ] Start command configured: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- [ ] Environment variables set in dashboard
- [ ] Health check endpoint configured (optional)
- [ ] Resource plan selected (free or paid)

### Railway

- [ ] GitHub repository connected
- [ ] Start command set: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
- [ ] Environment variables configured
- [ ] `PORT` variable set to 8080 or Railway default
- [ ] Build and deploy tested

### Fly.io

- [ ] `Dockerfile` created and tested locally
- [ ] `fly.toml` configured
- [ ] Flyctl CLI installed and authenticated
- [ ] App launched with `flyctl launch`
- [ ] Secrets set with `flyctl secrets set`
- [ ] Health check configured
- [ ] Regions selected for deployment

### Docker (Self-Hosted)

- [ ] `Dockerfile` created and tested
- [ ] `docker-compose.yml` configured (optional)
- [ ] Image builds successfully
- [ ] Container runs and app accessible
- [ ] Volumes configured for persistence (if needed)
- [ ] Environment variables passed correctly
- [ ] Health check implemented

---

## Testing Checklist

### Local Testing

- [ ] App starts without errors: `streamlit run app.py`
- [ ] All pages/tabs load correctly
- [ ] Forms accept input and validate correctly
- [ ] Predictions generate successfully
- [ ] Results display properly
- [ ] No console errors or warnings
- [ ] Memory usage is reasonable
- [ ] Load times are acceptable

### Production Testing (After Deploy)

- [ ] App accessible at deployment URL
- [ ] All features work in production
- [ ] Environment variables loaded correctly
- [ ] Secrets/credentials working (if applicable)
- [ ] Error handling works as expected
- [ ] Logging captures errors
- [ ] Performance is acceptable
- [ ] Mobile view works correctly

---

## Performance Optimization

### Before Deploying

- [ ] Large models compressed or stored externally
- [ ] Data files optimized (parquet instead of CSV)
- [ ] Caching implemented with `@st.cache_data` and `@st.cache_resource`
- [ ] Lazy loading for heavy dependencies
- [ ] Unnecessary imports removed
- [ ] Images optimized for web

### Monitoring Setup

- [ ] Error tracking configured (optional)
- [ ] Usage analytics set up (optional, privacy-respecting)
- [ ] Performance monitoring enabled (optional)
- [ ] Health check endpoint working
- [ ] Logs accessible and reviewed

---

## Security Review

### Code Security

- [ ] No sensitive data in code
- [ ] Input validation on all user inputs
- [ ] SQL injection prevention (if using database)
- [ ] XSS prevention (Streamlit handles this)
- [ ] HTTPS enabled in production

### Data Security

- [ ] No real patient/player data exposed
- [ ] Data privacy compliance (GDPR if applicable)
- [ ] Secure data storage (if applicable)
- [ ] Secure API communication (if applicable)

---

## Legal & Compliance

### Disclaimers

- [ ] "NOT medical advice" prominently displayed
- [ ] Educational use only clearly stated
- [ ] Limitations explicitly documented
- [ ] Appropriate use cases listed
- [ ] Inappropriate use cases listed

### Licensing

- [ ] License file added (if needed)
- [ ] Third-party licenses reviewed
- [ ] Data usage rights confirmed
- [ ] Model deployment rights confirmed

---

## Post-Deployment

### Immediate (First 24 Hours)

- [ ] Test all features in production
- [ ] Monitor for errors/crashes
- [ ] Check performance metrics
- [ ] Verify analytics (if enabled)
- [ ] Test from different devices/browsers

### Short-term (First Week)

- [ ] Gather user feedback
- [ ] Monitor resource usage
- [ ] Check for edge cases
- [ ] Review logs for issues
- [ ] Update documentation if needed

### Long-term (Ongoing)

- [ ] Regular security updates
- [ ] Dependency updates
- [ ] Feature improvements
- [ ] Performance optimization
- [ ] User feedback incorporation

---

## Rollback Plan

### If Deployment Fails

- [ ] Know how to revert to previous version
- [ ] Have backup of working code
- [ ] Document rollback procedure
- [ ] Test rollback process before deploying

### Emergency Contacts

- [ ] Platform support documentation bookmarked
- [ ] Community forums identified
- [ ] Backup deployment method planned

---

## Quick Deployment Commands

### Streamlit Cloud
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
# Then deploy via Streamlit Cloud dashboard
```

### Render
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
# Render auto-deploys on push
```

### Railway
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
# Railway auto-deploys on push
```

### Fly.io
```bash
flyctl launch  # First time
flyctl deploy  # Subsequent deployments
```

### Docker
```bash
docker build -t injury-risk-predictor .
docker run -p 8501:8501 injury-risk-predictor
```

---

## Verification Commands

Run these to verify readiness:

```bash
# Check Python version
python --version  # Should be 3.9+

# Test imports
python -c "import pandas, numpy, sklearn, lightgbm, xgboost, catboost, shap, streamlit"

# Test app locally
streamlit run app.py

# Check file sizes
du -sh models/*  # Should be reasonable for platform
du -sh data/*    # Should be reasonable for platform

# Run deployment check (if using Makefile)
make deploy-check
```

---

## Common Issues & Solutions

### Out of Memory
- [ ] Reduce model size
- [ ] Use model compression
- [ ] Implement lazy loading
- [ ] Upgrade to paid tier

### Slow Loading
- [ ] Add caching
- [ ] Optimize data loading
- [ ] Use parquet instead of CSV
- [ ] Reduce model complexity

### Build Failures
- [ ] Check requirements.txt for conflicts
- [ ] Pin all dependency versions
- [ ] Test build locally
- [ ] Check platform build logs

### Runtime Errors
- [ ] Check environment variables
- [ ] Verify file paths
- [ ] Review error logs
- [ ] Test locally with production config

---

## Sign-Off

Before deploying to production:

- [ ] All items in this checklist completed
- [ ] Local testing passed
- [ ] Documentation reviewed
- [ ] Team approved (if applicable)
- [ ] Backup plan in place

**Deployed by**: _______________
**Date**: _______________
**Platform**: _______________
**URL**: _______________

---

**Remember**: This is an educational project. Ensure all disclaimers are prominent and users understand the limitations!
