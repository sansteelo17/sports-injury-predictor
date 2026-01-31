# Contributing to Football Injury Risk Predictor

Thank you for your interest in contributing to this educational project! This guide will help you get started.

---

## Code of Conduct

This is an educational project focused on learning and sharing knowledge. Please be respectful, constructive, and helpful in all interactions.

---

## How Can I Contribute?

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### 2. Suggesting Enhancements

We welcome suggestions for:
- New features
- Performance improvements
- Documentation improvements
- Better visualizations
- Additional model types

Please create an issue describing:
- What you want to add/change
- Why it would be useful
- How it maintains the educational focus

### 3. Contributing Code

#### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/injury-risk-predictor.git
   cd injury-risk-predictor
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Workflow

1. **Make your changes**
   - Write clean, well-documented code
   - Follow existing code style
   - Add docstrings to functions
   - Keep functions focused and modular

2. **Test your changes**
   ```bash
   # Run your code to verify it works
   python -m src.your_module

   # Test the Streamlit app (when available)
   streamlit run app.py
   ```

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template (if available)
   - Describe your changes clearly

---

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where appropriate

Example:
```python
def calculate_injury_risk(
    player_data: pd.DataFrame,
    model: Any
) -> float:
    """
    Calculate injury risk probability for a player.

    Args:
        player_data: DataFrame with player features
        model: Trained ML model

    Returns:
        Injury risk probability (0-1)
    """
    # Your code here
    pass
```

### Documentation

- Add docstrings to all functions
- Use clear, concise comments
- Update README.md if adding new features
- Include examples where helpful

### File Organization

```
src/
├── data_loaders/       # Data loading utilities
├── preprocessing/      # Data cleaning
├── feature_engineering/  # Feature creation
├── models/             # Model training
├── inference/          # Prediction pipeline
└── dashboard/          # UI components
```

---

## What We're Looking For

### High Priority
- Bug fixes
- Documentation improvements
- Performance optimizations
- Better error handling
- Unit tests (future)
- Streamlit dashboard features

### Medium Priority
- New feature engineering techniques
- Additional model types
- Visualization improvements
- Data validation utilities

### Low Priority (But Still Welcome!)
- Code refactoring
- Type hints
- Logging improvements
- Configuration management

---

## Important Reminders

### Educational Focus
This is an educational project. Contributions should:
- Be well-documented and easy to understand
- Serve as learning examples
- Not sacrifice clarity for performance (unless necessary)

### Disclaimers
Any new features must maintain the project's disclaimers:
- NOT medical advice
- Educational use only
- Not for professional medical decisions

### Data Privacy
- Never commit real patient data
- Use synthetic or publicly available data only
- Respect data licensing

---

## Testing Guidelines

Currently, this project uses manual testing. When adding code:

1. **Test locally**:
   ```bash
   # Test your specific module
   python -c "from src.your_module import your_function; your_function(test_data)"
   ```

2. **Integration testing**:
   - Run the full pipeline end-to-end
   - Verify outputs make sense
   - Check for errors/warnings

3. **Future**: We plan to add pytest-based unit tests

---

## Documentation Standards

### README Updates

If adding a new feature, update:
- Installation instructions (if dependencies change)
- Architecture overview (if structure changes)
- Quick start guide (if workflow changes)

### Code Comments

```python
# Good comment
# Calculate ACWR using 7-day acute and 28-day chronic load
acwr = acute_load / chronic_load

# Bad comment
# Divide acute by chronic
acwr = acute_load / chronic_load
```

### Docstrings

Use Google-style docstrings:

```python
def train_injury_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "catboost"
) -> Any:
    """
    Train injury risk classification model.

    This function trains a gradient boosting model to predict injury risk
    based on player features including workload, injury history, and match data.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (injury=1, no injury=0)
        model_type: Type of model to train ('catboost', 'lightgbm', 'xgboost')

    Returns:
        Trained model object ready for predictions

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> X_train, y_train = prepare_data(df)
        >>> model = train_injury_model(X_train, y_train, model_type='catboost')
        >>> predictions = model.predict(X_test)
    """
    # Implementation here
    pass
```

---

## Pull Request Process

1. **Before submitting**:
   - [ ] Code works locally
   - [ ] No new errors/warnings
   - [ ] Documentation updated
   - [ ] Follows code style
   - [ ] Maintains educational focus

2. **PR Description**:
   ```markdown
   ## Description
   Brief description of what this PR does

   ## Changes Made
   - Added feature X
   - Fixed bug Y
   - Updated documentation for Z

   ## Testing
   Describe how you tested this

   ## Screenshots (if applicable)
   Add screenshots for UI changes
   ```

3. **Review process**:
   - Maintainer will review within 1 week
   - Address feedback if requested
   - Once approved, changes will be merged

---

## Questions?

If you have questions:
- Check existing issues
- Review documentation
- Create a new issue with your question

---

## Recognition

All contributors will be recognized in the project README (if you'd like to be!).

---

Thank you for helping make this educational project better!
