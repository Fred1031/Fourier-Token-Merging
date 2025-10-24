#!/usr/bin/env bash
set -euo pipefail

RESULTS="results_adaptive.csv"
#echo "use_tomesd,tomesd_ratio,use_dft,trunc_ratio,time_s,fid" >> "$RESULTS"

dry_run=0
debug=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    dry_run=1
  fi
  if [[ "$arg" == "--debug" ]]; then
    debug=1
  fi
done

if [[ "$dry_run" == 1 ]]; then
  echo "[Dry Run] Will only print the commands without running them."
fi
if [[ "$debug" == 1 ]]; then
  echo "[Debug Mode] Verbose output will be printed."
fi

RESULTS="results_adaptive.csv"
debug=0
dry_run=0
use_dft=1
adaptive_trunc=1
trunc_ratio=1.0

tomesd_ratios=(0.1 0.4)
adaptive_bases=("layer" "timestep")
adaptive_directions=("decreasing" "increasing")
trunc_ratio_mins=(0.1 0.3)
trunc_ratio_maxs=(0.5 0.8)

RESULTS="results_adaptive.csv"

for basis in "${adaptive_bases[@]}"; do
  for direction in "${adaptive_directions[@]}"; do
    for ratio in "${tomesd_ratios[@]}"; do
      for trunc_ratio_min in "${trunc_ratio_mins[@]}"; do
        for trunc_ratio_max in "${trunc_ratio_maxs[@]}"; do

          awk_res=$(awk "BEGIN{print (${trunc_ratio_min} >= ${trunc_ratio_max})}")
          

          out_dir="EXP/adaptive/basis${basis}_dir${direction}_r${ratio}_min${trunc_ratio_min}_max${trunc_ratio_max}"
          echo "=== ratio=${ratio}, basis=${basis}, dir=${direction}, min=${trunc_ratio_min}, max=${trunc_ratio_max} → ${out_dir} ==="

          CMD="python generate_images.py \
            --use_tomesd 1 \
            --tomesd_ratio ${ratio} \
            --use_dft ${use_dft} \
            --trunc_ratio ${trunc_ratio} \
            --adaptive_trunc \
            --adaptive_basis ${basis} \
            --adaptive_direction ${direction} \
            --trunc_ratio_min ${trunc_ratio_min} \
            --trunc_ratio_max ${trunc_ratio_max} \
            --output_dir ${out_dir}"

          [[ "$debug" == 1 ]] && CMD="${CMD} --debug 1"

          

          OUTPUT=$(eval "$CMD" 2>&1)
          echo "$OUTPUT"

          TIME=$(echo "$OUTPUT" \
            | grep 'Latency' \
            | sed -E 's/.*Latency: *([0-9]+\.[0-9]+) 秒.*/\1/')
          echo "Extracted TIME: ${TIME}s"

          # calc FID
          FID=$(python calc_fid.py \
            --gen_dir  "${out_dir}" \
            --real_dir "data/imagenet_val_1000flat_resized")

          echo "${ratio},${use_dft},${adaptive_trunc},${basis},${direction},${trunc_ratio_min},${trunc_ratio_max},${trunc_ratio},${TIME},${FID}" >> "$RESULTS"
          echo

        done
      done
    done
  done
done

