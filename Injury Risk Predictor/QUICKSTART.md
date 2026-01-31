# Injury Risk Predictor - Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 min)
```bash
pip install -r requirements.txt
```

### Step 2: Launch Dashboard (30 seconds)
```bash
./run_dashboard.sh
```
Or:
```bash
streamlit run app.py
```

### Step 3: Open Browser (automatic)
The dashboard opens at: `http://localhost:8501`

---

## 📝 Your First Prediction (1 minute)

1. **Fill in the sidebar form:**
   - Player Name: `Kevin De Bruyne`
   - Age: `33` (use slider)
   - Position: `Attacking Midfielder` (dropdown)
   - Team: `Man City` (dropdown)
   - Recent Matches: `10` (number input)
   - FIFA Rating: `91` (slider)

2. **Click "Predict Injury Risk"**

3. **Review the results:**
   - ✅ Risk Level & Probability
   - ✅ Projected Severity (days)
   - ✅ Player Archetype
   - ✅ Top Risk Factors
   - ✅ Recommendations

---

## 📊 Understanding Your Results

### Risk Levels
| Level | Probability | Action Required |
|-------|-------------|----------------|
| 🔴 **High** | 60%+ | Immediate load reduction |
| 🟡 **Moderate** | 40-60% | Monitor closely |
| 🟢 **Low** | <40% | Normal training |

### Severity Categories
- **Minor**: <7 days
- **Moderate**: 7-14 days
- **Major**: 14-30 days
- **Catastrophic**: 30+ days

### Confidence Levels
- **Very High**: Trust the prediction
- **High**: Follow recommendations
- **Medium**: Monitor and adjust
- **Low**: Informational only

---

## 🎯 Common Use Cases

### Fantasy Football
- Check players before transfers
- Avoid high-risk players before busy periods
- Safe captain selection (low-risk players)

### Amateur Coaching
- Rotate high-risk players
- Manage training loads
- Prevent injury before it happens

### Learning & Research
- Explore injury risk factors
- Understand ML in sports
- Experiment with different profiles

---

## ⚠️ Important Reminders

1. **NOT medical advice** - Educational/entertainment only
2. **No liability** - Statistical predictions, not certainties
3. **Demo mode** - Uses mock predictions (models not trained yet)
4. **Privacy** - All data stays local, nothing sent to servers

---

## 🎮 Try These Examples

### High-Risk Winger
```
Name: Marcus Rashford
Age: 27
Position: Left Winger
Team: Man United
Recent Matches: 13
→ Expected: High risk (65-75%)
```

### Low-Risk Goalkeeper
```
Name: Ederson
Age: 31
Position: Goalkeeper
Team: Man City
Recent Matches: 8
→ Expected: Low risk (20-30%)
```

### Aging Striker
```
Name: Jamie Vardy
Age: 37
Position: Center Forward
Team: Leicester
Recent Matches: 10
→ Expected: Moderate-High risk (50-65%)
```

---

## 🛠️ Troubleshooting

### Dashboard won't start
```bash
# Check Streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit
```

### Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Module errors
```bash
# Ensure you're in project root
cd "/path/to/Injury Risk Predictor"
streamlit run app.py
```

---

## 📚 Next Steps

1. **Explore the dashboard**
   - Try different player profiles
   - Read the "How It Works" section
   - Check out "Limitations & Caveats"

2. **Read the documentation**
   - `DASHBOARD_README.md` - Full technical docs
   - `USAGE_EXAMPLES.md` - Detailed examples
   - `README.md` - Project overview

3. **Train real models** (optional)
   - Follow notebooks in `/notebooks`
   - Save models to `/models` directory
   - Update `load_models()` in `app.py`

4. **Customize the dashboard**
   - Edit CSS styles in `app.py`
   - Adjust risk thresholds
   - Add new features

---

## 💡 Pro Tips

### 1. Realistic Inputs
- Use actual recent match counts for better predictions
- Set FIFA rating based on player quality (70-75 average, 80+ top)

### 2. Batch Analysis
- Take notes on multiple players
- Compare risk levels for squad decisions
- Track predictions over time

### 3. Interpretation
- Focus on risk level + severity together
- Consider archetype recommendations
- Look at top risk factors for insights

### 4. Fantasy Football Strategy
**Before busy period:**
1. Run predictions for all squad members
2. Transfer out High-risk players (60%+)
3. Captain Low-risk players (<40%)
4. Bench Moderate-risk players (40-60%)

---

## 🎓 Learning Resources

### In the Dashboard
- **How It Works** - ML methodology
- **Data Sources** - Training data info
- **Limitations** - What it can/can't do
- **Use Cases** - Recommended applications

### External Reading
- [ACWR explained](https://en.wikipedia.org/wiki/Acute:chronic_workload_ratio)
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- [SHAP values](https://github.com/slundberg/shap)

---

## 📞 Support

### Issues
1. Check `DASHBOARD_README.md` troubleshooting
2. Review error messages carefully
3. Ensure all dependencies installed
4. Verify you're in correct directory

### Questions
- Review in-app help sections
- Check `USAGE_EXAMPLES.md`
- Read code comments in `app.py`

---

## ✅ Quick Checklist

Before reporting issues, verify:
- [ ] Python 3.9+ installed
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] Running from project root directory
- [ ] No other app using port 8501
- [ ] Browser allows localhost connections

---

**That's it! You're ready to explore injury risk prediction. Have fun! ⚽**

---

*Remember: This is a proof-of-concept for educational purposes. Always consult qualified professionals for actual health decisions.*
