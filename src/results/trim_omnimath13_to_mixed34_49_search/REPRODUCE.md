# Reproduce: trim_omnimath13_to_mixed34_49_search

This archive captures the code and small router checkpoints needed to reproduce
the OmniMath 1-3 trained router evaluation on the mixed OmniMath 3-4/4-9 test
set.

## Included artifacts

- `src/results/trim_omnimath13_to_mixed34_49_search/`
  - final plots, selected points, summaries, dense BA80 sweep outputs, and mixed
    test manifest.
- `src/data/episodes/omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl`
  - the mixed 200-example test episode file used by the final eval.
- `src/data/episodes/omnimath_diff1_3_train_200_episodes.jsonl`
- `src/data/episodes/omnimath_diff3_4_test_200_episodes.jsonl`
- `src/data/episodes/omnimath_diff4_9_stratified_test_100_episodes.jsonl`
  - source episodes for rebuilding/training the OmniMath 1-3 to 3-4 router and
    the mixed 3-4/4-9 test set.
- `src/checkpoints/trim_{agg,rubric,rubric_v2b}_omnimath13_to34_point_search_*/`
  - router checkpoints used by the mixed eval script.
- `src/data/rubrics/omnimath13_to34/`
- `src/data/rubrics/omnimath13_to34_v2b_alpha01_corr005/`
  - rubric weights and router-feedback weights needed to reproduce rubric
    router training.

The large base SRM/LRM model weights are not included here. The offline final
evaluation uses stored episodes and router checkpoints, so it does not require
re-generating model rollouts.

## Quick final re-evaluation

From the repository root on a machine with the Python environment installed:

```bash
cd src
PYTHON_BIN="$(command -v python)" \
DEVICE="cuda:0" \
OUTPUT_DIR="results/trim_omnimath13_to_mixed34_49_search/repro_final" \
bash scripts/eval_omnimath_mixed34_49_search_final.sh
```

For CPU-only verification, use `DEVICE=cpu`. This is slower but should work for
the small router checkpoints.

## Rebuild the mixed episode file

```bash
cd src
python scripts/build_omnimath_mixed34_49_test.py \
  --omnimath34_episodes data/episodes/omnimath_diff3_4_test_200_episodes.jsonl \
  --omnimath49_episodes data/episodes/omnimath_diff4_9_stratified_test_100_episodes.jsonl \
  --output data/episodes/omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl \
  --manifest results/trim_omnimath13_to_mixed34_49_search/mixed_test_manifest.json \
  --seed 1 \
  --n34 100 \
  --n49 100
```

## End-to-end scripts

- `src/scripts/run_omnimath13_to34_search_pipeline.sh`
  - generation, search, and final evaluation for the original OmniMath 1-3 to
    3-4 experiment.
- `src/scripts/run_omnimath_mixed34_49_search_pipeline.sh`
  - waits for source episodes, builds the mixed 3-4/4-9 test file, then runs the
    final evaluation with the existing OmniMath 1-3 to 3-4 router checkpoints.

On a new server, override `PYTHON_BIN` if the default local path does not exist.
