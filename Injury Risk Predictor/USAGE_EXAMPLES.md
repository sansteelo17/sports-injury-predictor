# Dashboard Usage Examples

This guide shows common usage scenarios for the Injury Risk Predictor dashboard.

## Example 1: High-Risk Player Assessment

### Input
```
Player Name: Marcus Rashford
Age: 27
Position: Left Winger
Team: Man United
Recent Matches: 12
FIFA Rating: 86
```

### Expected Output
- **Risk Level**: High (65-75% probability)
- **Severity**: 25-35 days (Major)
- **Archetype**: High-Risk Frequent or Moderate-Load High-Variance
- **Top Risk Factors**:
  - Recent match load (acute workload) - HIGH impact
  - Position injury rate (winger) - MEDIUM impact
  - Workload variability - MEDIUM impact

### Recommendations
- ⚠️ Reduce training load by 20-30%
- ⏱️ Limit match minutes to 60-70 mins
- 🔄 Add extra recovery day between sessions
- 🏆 Fantasy football: High transfer risk

---

## Example 2: Low-Risk Goalkeeper

### Input
```
Player Name: Alisson Becker
Age: 31
Position: Goalkeeper
Team: Liverpool
Recent Matches: 6
FIFA Rating: 90
```

### Expected Output
- **Risk Level**: Low (20-30% probability)
- **Severity**: 5-10 days (Minor)
- **Archetype**: Low-Severity Stable
- **Top Risk Factors**:
  - Age-related risk - LOW impact
  - Position injury rate (GK) - LOW impact
  - Training load monotony - MINIMAL impact

### Recommendations
- ✅ Continue current training structure
- ⏱️ Full availability under normal rotation
- 📊 Track daily wellness markers
- 🏆 Fantasy football: Safe pick

---

## Example 3: Aging Striker

### Input
```
Player Name: Jamie Vardy
Age: 37
Position: Center Forward
Team: Leicester
Recent Matches: 10
FIFA Rating: 79
```

### Expected Output
- **Risk Level**: Moderate to High (50-65% probability)
- **Severity**: 15-25 days (Moderate to Major)
- **Archetype**: Moderate-Risk Recurrent or Catastrophic + Re-aggravation
- **Top Risk Factors**:
  - Age-related risk factor - HIGH impact
  - Recent match load - MEDIUM impact
  - Position injury rate (striker) - MEDIUM impact

### Recommendations
- ⚡ Monitor workload carefully
- ⏱️ Manage to 60-70 mins per match
- 🔄 Avoid consecutive full matches
- 🎯 Reinforcement work + continuity blocks
- 🏆 Fantasy football: Rotation risk - monitor closely

---

## Example 4: Young Prospect

### Input
```
Player Name: Jude Bellingham
Age: 21
Position: Central Midfielder
Team: Other (Real Madrid)
Recent Matches: 8
FIFA Rating: 90
```

### Expected Output
- **Risk Level**: Low to Moderate (35-45% probability)
- **Severity**: 8-15 days (Moderate)
- **Archetype**: Low-Severity Stable or Moderate-Load High-Variance
- **Top Risk Factors**:
  - Recent match load - MEDIUM impact
  - Age-related risk - LOW impact (young)
  - Sprint intensity exposure - LOW to MEDIUM impact

### Recommendations
- ✅ Good tissue adaptation
- ⚡ Avoid sudden increases in intensity
- ⏱️ Can handle 70-80 mins per match
- 🏆 Fantasy football: Good long-term pick

---

## Example 5: Heavy Rotation Player

### Input
```
Player Name: Phil Foden
Age: 24
Position: Attacking Midfielder
Team: Man City
Recent Matches: 15
FIFA Rating: 85
```

### Expected Output
- **Risk Level**: High (60-70% probability)
- **Severity**: 20-30 days (Major)
- **Archetype**: High-Risk Frequent
- **Top Risk Factors**:
  - Recent match load (VERY HIGH) - HIGH impact
  - Workload variability - HIGH impact
  - Position injury rate - MEDIUM impact

### Recommendations
- ⚠️ Immediate reduction in high-intensity exposure
- ⏱️ Maximum 60 mins per match
- 🔄 Rotate where possible
- 🎯 Reduce high-intensity exposure; recovery emphasis
- 🏆 Fantasy football: Bench or transfer - very high injury risk

---

## Use Case Scenarios

### Fantasy Football Manager
**Goal**: Minimize injury risk in squad selection

**Workflow**:
1. Run predictions for all players in transfer considerations
2. Filter out High-risk players before busy fixture periods
3. Prioritize Low-risk players for captain selection
4. Keep Moderate-risk players on bench during tough fixtures

**Example Decision**:
- **Player A**: 70% risk, 30 days severity → Transfer out before DGW
- **Player B**: 25% risk, 8 days severity → Safe captain choice
- **Player C**: 45% risk, 15 days severity → Keep but bench for tough fixtures

---

### Amateur Team Coach
**Goal**: Manage squad rotation and prevent injuries

**Workflow**:
1. Input all squad players at start of season
2. Update recent matches weekly
3. Identify high-risk players for load management
4. Rotate high-risk players proactively

**Example Plan**:
- **Winger (65% risk)**: Reduce from 90 mins to 60 mins per match
- **Striker (35% risk)**: Full availability, monitor weekly
- **Defender (50% risk)**: Rotate every other match during busy period

---

### Sports Science Student
**Goal**: Learn about injury risk factors and ML in sports

**Workflow**:
1. Test different player profiles to see how factors change predictions
2. Compare positions to understand position-specific risks
3. Experiment with age ranges to see age-related patterns
4. Analyze which features have highest impact

**Learning Insights**:
- Wingers/Forwards have higher base risk than goalkeepers
- Age 30+ shows increased age-related risk factors
- Recent match load (acute workload) is often top risk factor
- Risk and severity are correlated but distinct predictions

---

## Tips for Best Results

### 1. Accurate Input Data
- Use actual recent match counts (last 4 weeks)
- Be precise with age and position
- Use correct team names for better reference

### 2. Interpreting Confidence
- **Very High**: Strong model agreement, trust the prediction
- **High**: Good agreement, recommendation should be followed
- **Medium**: Moderate confidence, monitor and adjust
- **Low**: Treat as informational, not directive

### 3. Understanding Risk Levels
- **High (60%+)**: Significant action needed
- **Moderate (40-60%)**: Monitor closely, adjust load
- **Low (<40%)**: Normal training, low concern

### 4. Severity Context
- **Minor (<7 days)**: Short-term concern
- **Moderate (7-14 days)**: 1-2 weeks impact
- **Major (14-30 days)**: 2-4 weeks out
- **Catastrophic (30+ days)**: Month+ recovery

### 5. Archetype Awareness
Each archetype has specific training interventions:
- **High-Risk Frequent**: Focus on recovery
- **Moderate-Load High-Variance**: Stabilize workload
- **Moderate-Risk Recurrent**: Continuity blocks
- **Low-Severity Stable**: Maintain structure
- **Catastrophic + Re-aggravation**: High control

---

## Common Questions

### Q: Why do similar players get different predictions?
**A**: Small differences in age, recent matches, or position can shift risk probability. The model is sensitive to workload and age factors.

### Q: Is a 45% risk probability High or Moderate?
**A**: 45% is classified as Moderate (40-60% range). The exact threshold is configurable.

### Q: Can I trust predictions for non-Premier League players?
**A**: The model is trained on Premier League data. Predictions for other leagues should be treated as estimates with potentially lower accuracy.

### Q: What if I don't know the FIFA rating?
**A**: FIFA rating is optional. Use the slider to set a reasonable estimate (70-75 for average players, 80+ for top players).

### Q: How often should I update predictions?
**A**: Re-run predictions weekly or after significant changes in match load (e.g., busy fixture periods, return from injury).

---

## Integration Examples

### Python API (Future Feature)
```python
from injury_predictor import predict_risk

player = {
    'name': 'Bruno Fernandes',
    'age': 29,
    'position': 'Central Midfielder',
    'team': 'Man United',
    'recent_matches': 11,
    'fifa_rating': 87
}

prediction = predict_risk(player)
print(f"Risk: {prediction['risk_level']} ({prediction['risk_probability']:.1%})")
print(f"Severity: {prediction['severity_days']} days")
print(f"Archetype: {prediction['archetype']}")
```

### Batch Processing (Future Feature)
```python
import pandas as pd

# Upload squad CSV
squad_df = pd.read_csv('my_squad.csv')
results = predict_batch(squad_df)

# Filter high-risk players
high_risk = results[results['risk_probability'] > 0.6]
print(f"High-risk players: {len(high_risk)}")
```

---

**For more information, see:**
- `DASHBOARD_README.md` - Full technical documentation
- `README.md` - Project overview
- In-app help sections - Interactive guidance
