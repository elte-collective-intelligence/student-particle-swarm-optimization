# Test Suite

This directory contains unit tests and integration tests for the PSO environment implementation.

## Files

### `test_env.py`
**Environment functionality tests.** This file:
- Tests core environment operations (reset, step)
- Verifies TorchRL interface compliance
- Tests observation and action spaces
- Validates reward computation

**Test Cases:**

**`test_env_creation`**
- Purpose: Verify environment initializes without errors
- What it tests:
  - Environment creation succeeds
  - Parameters are set correctly
  - Observation and action specs are valid
- Why important: Catches initialization bugs early

**`test_env_reset`**
- Purpose: Verify reset returns valid observations
- What it tests:
  - Reset returns TensorDict with correct keys
  - Observations have correct shapes
  - All agents have valid initial states
- Expected: Properly formatted TensorDict

**`test_env_step`**
- Purpose: Verify basic step operation works
- What it tests:
  - Environment accepts valid actions
  - Step returns proper observation format
  - Rewards are computed correctly
  - Done flags work properly
- Why important: Ensures core simulation loop functions

**`test_observation_space`**
- Purpose: Verify observation structure
- What it tests:
  - Observation contains required keys (position, velocity, personal_best, global_best)
  - Shapes match expected dimensions
  - Values are within valid ranges
- Expected: Correctly formatted multi-agent observations

**`test_action_space`**
- Purpose: Verify action handling
- What it tests:
  - Actions have correct shape [n_particles, 4]
  - Action values are properly bounded
  - Invalid actions are handled gracefully
- Expected: Proper action processing

**`test_reward_computation`**
- Purpose: Verify reward calculation
- What it tests:
  - Rewards are computed based on fitness improvement
  - Better positions yield positive rewards
  - Rewards are properly normalized
- Expected: Correct incentive structure

**`test_vectorized_env`**
- Purpose: Test parallel environment execution
- What it tests:
  - Multiple environments run in parallel
  - Batch operations work correctly
  - Results are properly aggregated
- Why important: Training uses vectorized environments

**`test_objective_functions`**
- Purpose: Verify different optimization functions
- What it tests:
  - Sphere, Rastrigin, Rosenbrock, Ackley functions work
  - Functions return correct shapes
  - Known optima are handled correctly
- Expected: Valid fitness values for all functions

## Usage

### Running All Tests
```bash
# From project root
pytest test/ -v

# With coverage
pytest test/ --cov=src --cov-report=html
```

### Running Specific Test File
```bash
pytest test/test_env.py -v
```

### Running Specific Test
```bash
pytest test/test_env.py::test_env_reset -v
```

### Quick Smoke Test
```bash
pytest test/test_env.py -v -x --tb=short
```

## What to Check if Tests Fail

### Environment Creation Fails
- Check `src/envs/env.py` for initialization errors
- Verify Hydra config values are valid
- Check TorchRL version compatibility

### Step Fails
- Verify action shapes match expected [n_particles, 4]
- Check observation processing
- Verify reward calculation doesn't have NaN/Inf

### Reward Tests Fail
- Check fitness function implementation in `src/envs/dynamic_functions.py`
- Verify reward normalization logic
- Check for numerical stability issues

### Vectorized Tests Fail
- Verify batch dimension handling
- Check TensorDict structure
- Verify parallel reset/step implementations

## Test Configuration

Tests use fixtures defined in `conftest.py` (if present) or inline:
```python
@pytest.fixture
def env():
    """Create test environment with minimal config."""
    return make_env(n_particles=5, n_dims=2, max_steps=10)
```

## Coverage Goals

Target coverage: >80% for core modules
- `src/envs/env.py`: Environment implementation
- `src/utils.py`: Action extraction and transformation
- `src/visualization.py`: Visualization functions

## Adding New Tests

When adding new tests:
1. Follow naming convention: `test_<feature>_<scenario>`
2. Use descriptive docstrings
3. Test both success and failure cases
4. Keep tests fast (mock expensive operations)
5. Add to appropriate test file or create new one
