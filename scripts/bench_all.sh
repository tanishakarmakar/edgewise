#!/bin/sh
# Benchmark the model catalog on this device, smallest model first, one at a time.
#
# For numbers worth publishing: plug in the charger and close other apps first.
# The 3B 8-bit variants are skipped by default: with normal apps open they do not fit
# comfortably in 8 GB. Include them with:  WITH_LARGE=1 scripts/bench_all.sh
#
# Usage:  scripts/bench_all.sh            standard decoding, every task
#         scripts/bench_all.sh accel      also the accelerated decodings on the 1.5B/3B models
set -eu
cd "$(dirname "$0")/.."

MODELS="
qwen2.5-0.5b-instruct@mlx-4bit
qwen2.5-0.5b-instruct@mlx-8bit
llama3.2-1b-instruct@mlx-4bit
llama3.2-1b-instruct@mlx-8bit
qwen2.5-1.5b-instruct@mlx-4bit
qwen2.5-1.5b-instruct@mlx-8bit
qwen2.5-3b-instruct@mlx-4bit
llama3.2-3b-instruct@mlx-4bit
"
if [ "${WITH_LARGE:-0}" = "1" ]; then
  MODELS="$MODELS
qwen2.5-3b-instruct@mlx-8bit
llama3.2-3b-instruct@mlx-8bit"
fi

if pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  echo "On battery power. Plug in the charger for publishable numbers (continuing in 5s)."
  sleep 5
fi

# caffeinate keeps the Mac awake for the duration of each run.
for model in $MODELS; do
  echo "=== $model"
  caffeinate -i uv run edgewise bench -m "$model" --all-tasks || echo "skipped: $model"
done

if [ "${1:-}" = "accel" ]; then
  DRAFT=qwen2.5-0.5b-instruct@mlx-4bit
  for model in qwen2.5-1.5b-instruct@mlx-4bit qwen2.5-3b-instruct@mlx-4bit; do
    echo "=== $model: prompt lookup"
    caffeinate -i uv run edgewise bench -m "$model" --all-tasks --decoding prompt_lookup \
      || echo "skipped"
    echo "=== $model: speculative (draft $DRAFT)"
    caffeinate -i uv run edgewise bench -m "$model" --all-tasks --decoding speculative \
      --draft "$DRAFT" || echo "skipped"
  done
fi

uv run edgewise report --local qwen2.5-1.5b-instruct@mlx-4bit --strong qwen2.5-3b-instruct@mlx-4bit
