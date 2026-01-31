# Logging Framework

This project uses Python's built-in `logging` module with a centralized configuration for consistent logging across all modules.

## Overview

The logging framework provides:
- **Dual output**: Console (INFO+) and file (DEBUG+) logging
- **Structured format**: Timestamps, module names, and log levels
- **Easy integration**: Simple `get_logger()` function
- **Production-ready**: Proper error tracking and debugging capabilities

## Usage

### Basic Usage

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error message")
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems (file only)
- **INFO**: General informational messages (console + file)
- **WARNING**: Something unexpected but non-critical (console + file)
- **ERROR**: Serious problems that need attention (console + file)

### Log Output

**Console Output** (INFO and above):
```
2026-01-23 22:54:39 - src.inference.inference_pipeline - INFO - Starting inference pipeline for 150 matches
```

**File Output** (DEBUG and above, saved to `logs/injury_predictor.log`):
```
2026-01-23 22:54:39 - src.inference.inference_pipeline - INFO - Starting inference pipeline for 150 matches
2026-01-23 22:54:39 - src.inference.inference_pipeline - DEBUG - Built inference features: (150, 45)
2026-01-23 22:54:40 - src.inference.inference_pipeline - INFO - Generated predictions for 150 records
```

## Integration Points

The logging framework is integrated in key modules:

### 1. Inference Pipeline (`src/inference/inference_pipeline.py`)
- Logs prediction pipeline start/end
- Logs number of matches processed
- Logs errors with full stack traces

### 2. Risk Card Generation (`src/inference/risk_card.py`)
- Logs risk card generation for each player
- Includes risk probability and severity in logs

### 3. Model Training (`src/models/classification.py`)
- Logs training start/end for each model
- Logs hyperparameter tuning progress
- Logs final model performance metrics

## Configuration

The logger is configured in `src/utils/logger.py`:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)  # Use module name
```

Configuration details:
- **Log directory**: `logs/` (auto-created)
- **Log file**: `logs/injury_predictor.log`
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Date format**: `%Y-%m-%d %H:%M:%S`

## Best Practices

1. **Use module name**: Always use `__name__` when getting a logger
   ```python
   logger = get_logger(__name__)
   ```

2. **Log at appropriate levels**:
   - Use `DEBUG` for detailed diagnostic info
   - Use `INFO` for general progress updates
   - Use `WARNING` for recoverable issues
   - Use `ERROR` for serious problems

3. **Include context**: Add relevant information to log messages
   ```python
   logger.info(f"Processing {len(data)} records")
   logger.error(f"Failed to load model: {str(e)}", exc_info=True)
   ```

4. **Use exc_info for errors**: Include stack traces for exceptions
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.error(f"Operation failed: {str(e)}", exc_info=True)
       raise
   ```

## Example Demo

Run the logging demonstration:

```bash
python examples/logging_demo.py
```

This will show logging in action and create entries in the log file.

## Log File Management

- Logs accumulate in `logs/injury_predictor.log`
- Consider implementing log rotation for production
- Current implementation: append-only (manual cleanup needed)

## Future Enhancements

Potential improvements:
- Rotating file handlers (size-based or time-based)
- Different log files for different modules
- JSON-formatted logs for machine parsing
- Log aggregation for distributed systems
