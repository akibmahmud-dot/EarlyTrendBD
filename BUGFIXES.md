# Bug Fixes Applied (every fix verified by actually running the code)

This branch (`capstone-fix`) is the original `develop` branch with 6 real
bugs found and fixed. Every claim below was verified by actually executing
the code, not just by reading it.

**Before any fix:** running `python main.py train` or hitting `/predict`
crashed immediately. **After all fixes:** `pytest tests/ -v` → **11/11
passed**, full training pipeline runs clean with no warnings, all 6 API
endpoints return correct status codes.

---

## Bug 1 — CRITICAL: merge key mismatch (`data/processor.py`)
**Symptom:** `KeyError: 'product_id'` — crashed on the very first run of
`data/processor.py`, `ml/train.py`, or `/api/v1/predict`.
**Cause:** `data/generators.py` creates each product with key `"id"`, but
`processor.py` merged `metrics_df` with `products_df` using `on='product_id'`
— a column that never existed on the products side.
**Fix:** `left_on='product_id', right_on='id'`.

## Bug 2 — CRITICAL: virality label formula produced only one class
**Symptom:** `ValueError: Invalid classes inferred from unique values of y.
Expected: [0], got [1]` — XGBoost refused to train at the project's own
default settings (150 products, 30 days).
**Cause:** `social_score = total_social_engagement / 1000` — but that raw
sum scales with `days`, so at 30 days it's already in the thousands. Every
single one of 150 generated products ended up with `virality_score = 1.0`
— zero variance in the label.
**Fix:** normalize each driver (social engagement, sales growth, sentiment)
to a 0–1 range *within the batch*, then label the top 30% by score as
viral via a percentile threshold — robust to any `num_products`/`days`
setting.
**Verified after fix:** XGBoost accuracy 90–96%, Ensemble accuracy 97–98%,
AUC 0.94–1.0 (logged in `logs/training.log`).

## Bug 3 — Ensemble silently dropped the LSTM's contribution
**Symptom:** `LSTM prediction failed: operands could not be broadcast
together with shapes (150,) (143,) (150,)` (a caught warning, not a
crash — but the LSTM's 40% weight was effectively discarded every time, so
the "Ensemble (LSTM+XGBoost)" was really just `0.6 × XGBoost` and predicted
scores never rose above ~0.6).
**Cause:** the LSTM needs `sequence_length` (7) rows of history before its
first prediction, so for a batch of N rows it returns only N-7 predictions
— too few to add to the XGBoost array.
**Fix:** pad the LSTM's output to match length (using its first available
prediction for the missing leading rows), and renormalize the ensemble
weights to whichever model(s) actually returned predictions instead of
always assuming both contributed.
**Verified after fix:** predicted virality scores now properly range up to
~0.87–0.90 for genuinely strong products instead of capping at ~0.6.

## Bug 4 — Data generator crashed on short histories
**Symptom:** `ValueError: empty range in randrange(5, 3)` — this is exactly
what `tests/test_data.py::test_data_generation` hit (it uses `days=7`).
**Cause:** `viral_day = random.randint(5, days - 5)` is only valid when
`days > 10`. Anyone training with `DAYS_HISTORY` under 11 (or running the
existing test suite!) crashed.
**Fix:** scale the margin down for short histories: `margin = min(5,
max(1, days // 3))`, and only assign a viral day when the resulting range
is valid.

## Bug 5 — `/api/v1/feature-importance` always returned 500
**Symptom:** `{"detail":"ModelTrainer.get_feature_importance() got an
unexpected keyword argument 'top_n'"}`
**Cause:** `api/routes.py` calls `trainer.get_feature_importance(top_n=top_n)`,
but `ml/train.py`'s `ModelTrainer.get_feature_importance()` was defined
with no parameters at all.
**Fix:** added `top_n: int = 15` parameter, passed through to the
underlying XGBoost model. Also fixed `evaluate_models()` so it evaluates
the already-trained model instead of silently retraining XGBoost from
scratch on every `/model-metrics` call.

## Bug 6 — A real 404 was being reported as a 500
**Symptom:** asking `/api/v1/analyze/{product_name}` for a product that
doesn't exist returned **HTTP 500** with body `{"detail":"404: Product not
found"}` instead of a proper **HTTP 404**.
**Cause:** the route raises `HTTPException(status_code=404, ...)` inside a
`try` block whose `except Exception` clause catches *everything*,
including the `HTTPException` it just raised, and re-wraps it as a 500.
**Fix:** added `except HTTPException: raise` before the generic
`except Exception` in every route handler, so intentional HTTP errors
pass through correctly.

---

## Also cleaned up (not bugs, but noisy / fragile)
- Replaced all deprecated `datetime.utcnow()` calls with
  `datetime.now(timezone.utc)` (Python 3.12+ prints a `DeprecationWarning`
  for every single one — over 4,000 of them during a `pytest` run).
- `requirements.txt` pinned `tensorflow==2.14.0` etc. with `==`, which only
  has wheels for Python 3.9–3.11. Changed to `>=` so `pip install` picks a
  compatible version automatically on Python 3.12/3.13 too. (Docker was
  never affected — `Dockerfile` already pins `python:3.9-slim`.)

## Known limitation left as-is (worth mentioning in your defense, not a crash)
- `api/routes.py`'s `load_models()` retrains everything from scratch on the
  first request after every server restart (fresh random synthetic data
  each time) rather than loading a previously saved model file, even
  though `save_model()`/`load_model()` exist on each model class. Wiring
  that up is a good "future work" item.
- `db/models.py` (Product, DailyMetrics, Prediction tables) and
  `db/migrations.py` are fully written and `python db/migrations.py` works,
  but nothing in `api/routes.py` actually writes predictions there yet.

## How this was verified
```
pytest tests/ -v          # 11 passed, 0 failed
python main.py train      # completes cleanly, no warnings in logs/training.log
python main.py api        # then tested all endpoints:
                           #   GET  /api/v1/health             -> 200
                           #   GET  /                           -> 200
                           #   POST /api/v1/predict              -> 200
                           #   GET  /api/v1/viral-products       -> 200
                           #   GET  /api/v1/feature-importance   -> 200
                           #   GET  /api/v1/model-metrics        -> 200
                           #   GET  /api/v1/analyze/{valid}      -> 200
                           #   GET  /api/v1/analyze/{invalid}    -> 404 (correct!)
```
