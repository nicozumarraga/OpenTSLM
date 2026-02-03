# Improved Existence & Localization Tasks: Distractor Insertion

## Problem Statement

The current implementation of EXISTENCE and LOCALIZATION tasks has a vulnerability: when the background signal is homogeneous (e.g., all "sleep"), the model can identify the needle simply by detecting a **change in variance**, rather than learning to recognize activity patterns. This allows the model to "cheat" on both positive and negative cases.

### Example of the Problem

```
Background: 100 seconds of "sleep" (very flat, low variance signal)
Inserted needle: 5 seconds of "walking" (rhythmic, higher variance)

Model can solve: "Is there walking?" → Just detect where variance increases
```

## Solution Overview

Insert **multiple needles from the SAME activity regime** (activities with similar signal properties). This forces the model to distinguish between similar-looking activities rather than just detecting variance changes.

### Activity Regimes (2 Groups)

We group WillettsSpecific2018 activities by signal characteristics:

| Regime | Activities | Signal Characteristics |
|--------|-----------|------------------------|
| **Sedentary** | sleep, sitting, vehicle, standing, household-chores | Low-to-moderate variance, minimal rhythmic patterns |
| **Active** | walking, mixed-activity, bicycling, manual-work, sports | Higher variance, rhythmic/dynamic patterns |

Using 2 broad regimes (instead of more granular groupings) ensures **5 activities per group**, enabling meaningful distractor sampling.

---

## Task 1: EXISTENCE - Updated Algorithm

### Pseudocode

```
Algorithm: Existence Task with Distractor Insertion

1. Sample background window
2. Decide positive (50%) or negative (50%) formulation
3. Randomly select a regime (sedentary or active)
4. Sample N needles from that regime (excluding activities already in background)
5. Insert all needles at non-overlapping positions
6. Positive case: ask about one of the inserted activities
7. Negative case: ask about an activity IN the same regime but NOT an inserted needle
```

### Example Scenarios

**Scenario A: Sedentary background with sedentary needles**
```
Background: "sitting" (sedentary regime)
Selected regime: sedentary
Inserted needles: "sleep", "vehicle" (2 needles from sedentary)
Positive question: "Is there sleep?" → Yes
Negative question: "Is there standing?" → No (sedentary, but not inserted)
```

**Scenario B: Active background with active needles**
```
Background: "walking" (active regime)
Selected regime: active
Inserted needles: "bicycling", "sports" (2 needles from active)
Positive question: "Is there bicycling?" → Yes
Negative question: "Is there manual-work?" → No (active, but not inserted)
```

**Scenario C: Cross-regime (sedentary background, active needles)**
```
Background: "sitting" (sedentary regime)
Selected regime: active (randomly chosen)
Inserted needles: "walking", "bicycling", "sports" (3 needles from active)
Positive question: "Is there walking?" → Yes
Negative question: "Is there mixed-activity?" → No (active, but not inserted)
```

### Key Design Decisions

1. **Regime selection is random** - Can be same or different from background regime
2. **Needles are always from ONE regime** - Forces within-regime discrimination
3. **Negative targets from SAME regime** - Cannot use regime-level signal differences
4. **Background activities are excluded** - Needles are always "foreign" to background

---

## Task 2: LOCALIZATION - Updated Algorithm

### Pseudocode

```
Algorithm: Localization Task with Distractor Insertion

1. Sample background window
2. Randomly select a regime (sedentary or active)
3. Sample N needles from that regime (excluding activities already in background)
4. Insert all needles at non-overlapping positions
5. Randomly select ONE inserted needle as the target
6. Ask: "When did the {target_activity} bout occur?"
7. Answer: timestamp range of the target needle
```

### Example Scenario

```
Background: "sitting" (sedentary regime)
Selected regime: sedentary
Inserted needles:
  - "sleep" at position 2000-2500 (5:30 AM - 5:35 AM)
  - "vehicle" at position 5000-5800 (6:00 AM - 6:08 AM)
  - "standing" at position 8000-8300 (6:30 AM - 6:33 AM)

Target: "vehicle" (randomly selected)
Question: "When did the vehicle bout occur?"
Answer: "The vehicle bout occurred from 6:00 AM to 6:08 AM."
```

### Key Design Decisions

1. **Always multiple needles** - Creates distractors with similar signal properties
2. **Single target** - Question asks about exactly one activity
3. **Model must identify correct activity** - Cannot rely on variance detection alone

---

## Implementation Plan

### Phase 1: Create Activity Regimes Module

**New file:** `src/opentslm/time_series_datasets/ts_haystack/core/activity_regimes.py`

```python
WILLETTS_ACTIVITY_REGIMES = {
    "sedentary": ["sleep", "sitting", "vehicle", "standing", "household-chores"],
    "active": ["walking", "mixed-activity", "bicycling", "manual-work", "sports"],
}

ACTIVITY_TO_REGIME = {activity: regime for regime, activities in ...}

# Utility functions:
# - get_regime(activity) -> str
# - get_regime_activities(regime) -> Set[str]
# - get_same_regime_activities(activity) -> Set[str]
# - get_distractor_candidates(target) -> Set[str]
```

**Update:** `src/opentslm/time_series_datasets/ts_haystack/core/__init__.py`
- Export new module functions

---

### Phase 2: Extend NeedleSampler

**Modify:** `src/opentslm/time_series_datasets/ts_haystack/core/needle_sampler.py`

Add new method:

```python
def sample_needles_for_regime(
    self,
    regime_activities: Set[str],
    n_needles: int,
    min_duration_ms: int = 0,
    exclude_pids: Optional[Set[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[NeedleSample]:
    """
    Sample multiple needles from a set of activities (typically same regime).

    Tries to get diversity by sampling one needle per activity first,
    then fills remaining slots from any activity in the set.
    """
```

---

### Phase 3: Rewrite Existence Task Generator

**Modify:** `src/opentslm/time_series_datasets/ts_haystack/tasks/task_existence.py`

Key changes to `generate_sample()`:

1. Add task-specific parameters: `min_distractors`, `max_distractors`, `min_gap_samples`
2. Randomly select regime (not based on background)
3. Compute `insertable_activities = regime_activities - background.activities_present`
4. For negative case, ensure at least 1 activity reserved for target
5. Sample N needles using `sample_needles_for_regime()`
6. Insert all needles at non-overlapping positions
7. Select target:
   - Positive: random from `inserted_activities`
   - Negative: random from `insertable_activities - inserted_activities`

**New difficulty_config fields:**
```python
{
    "target_activity": str,
    "is_positive": bool,
    "inserted_regime": str,  # "sedentary" or "active"
    "inserted_activities": List[str],
    "n_needles_inserted": int,
    "background_activities": List[str],
}
```

---

### Phase 4: Rewrite Localization Task Generator

**Modify:** `src/opentslm/time_series_datasets/ts_haystack/tasks/task_localization.py`

Key changes to `generate_sample()`:

1. Add task-specific parameters: `min_distractors`, `max_distractors`, `min_gap_samples`
2. Randomly select regime
3. Sample N needles from regime (excluding background activities)
4. Insert all needles at non-overlapping positions
5. Randomly select one needle as target
6. Generate Q/A with target needle's timestamps

**New difficulty_config fields:**
```python
{
    "target_activity": str,
    "target_needle_index": int,
    "inserted_regime": str,
    "inserted_activities": List[str],
    "n_needles_inserted": int,
    "background_activities": List[str],
}
```

---

### Phase 5: Update Configuration

**Modify:** `configs/default_generation_config.yaml`

```yaml
tasks:
  existence:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]
    background_purity: pure
    margin_samples: 100
    # Distractor insertion parameters
    min_distractors: 1
    max_distractors: 3
    min_gap_samples: 100

  localization:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]
    background_purity: pure
    margin_samples: 100
    # Distractor insertion parameters
    min_distractors: 2  # At least 2 to have distractors
    max_distractors: 4
    min_gap_samples: 100
```

---

### Phase 6: Update Tests

**Modify:** `test/tasks/test_task_existence.py`

Add new test class `TestExistenceDistractorInsertion`:

```python
def test_multiple_needles_inserted():
    """Verify multiple needles are inserted when configured."""

def test_needles_from_same_regime():
    """Verify all inserted needles are from the same activity regime."""

def test_negative_from_same_regime():
    """Verify negative targets are from same regime as inserted needles."""

def test_positive_negative_balance():
    """Verify ~50/50 balance of positive/negative samples."""
```

**Modify:** `test/tasks/test_task_localization.py`

Add new test class `TestLocalizationDistractorInsertion`:

```python
def test_multiple_needles_single_target():
    """Verify multiple needles inserted but answer refers to specific one."""

def test_needles_from_same_regime():
    """Verify all inserted needles are from the same activity regime."""

def test_target_timestamp_in_answer():
    """Verify answer contains correct target needle timestamps."""
```

---

### Phase 7: Update Documentation

**Modify:** `src/opentslm/time_series_datasets/ts_haystack/README.md`

Add section:

```markdown
## Distractor Insertion (Existence & Localization)

To prevent models from "cheating" by detecting variance changes...

### Activity Regimes
| Regime | Activities |
|--------|-----------|
| Sedentary | sleep, sitting, vehicle, standing, household-chores |
| Active | walking, mixed-activity, bicycling, manual-work, sports |

### Existence Task
- Inserts N needles from ONE regime
- Positive: asks about inserted activity
- Negative: asks about non-inserted activity IN SAME REGIME

### Localization Task
- Inserts N needles from ONE regime (distractors)
- Asks about specific inserted needle
- Model must identify correct activity among similar signals
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `core/activity_regimes.py` | **CREATE** | Activity regime definitions and utilities |
| `core/__init__.py` | **MODIFY** | Export activity regime utilities |
| `core/needle_sampler.py` | **MODIFY** | Add `sample_needles_for_regime()` method |
| `tasks/task_existence.py` | **REWRITE** | Implement distractor insertion logic |
| `tasks/task_localization.py` | **REWRITE** | Implement distractor insertion logic |
| `configs/default_generation_config.yaml` | **MODIFY** | Add distractor parameters |
| `test/tasks/test_task_existence.py` | **MODIFY** | Add distractor tests |
| `test/tasks/test_task_localization.py` | **MODIFY** | Add distractor tests |
| `README.md` | **MODIFY** | Document distractor mechanism |

---

## Validation Checklist

After implementation, verify:

- [ ] Activity regimes module loads correctly
- [ ] `sample_needles_for_regime()` returns needles from specified activities
- [ ] Existence: multiple needles inserted from same regime
- [ ] Existence: negative targets are from same regime as needles
- [ ] Existence: ~50/50 positive/negative balance maintained
- [ ] Localization: multiple needles inserted, single target in answer
- [ ] Localization: answer timestamps match target needle
- [ ] All existing tests still pass
- [ ] New distractor tests pass
- [ ] Visual inspection of generated samples shows expected behavior
