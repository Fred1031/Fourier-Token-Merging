#!/usr/bin/env bash
set -euo pipefail

RESULTS="results.csv"
#echo "use_tomesd,tomesd_ratio,use_dft,trunc_ratio,time_s,fid" >> "$RESULTS"


use_tomesd_vals=(1)
tomesd_ratios=(0.1 0.2 0.3 0.4 0.5)
use_dft_vals=(1)
trunc_ratios=(0.6 0.7 0.8 0.9 1)
#trunc_ratios=(0.2 0.3 0.4 0.5 0.6)

for use_tomesd in "${use_tomesd_vals[@]}"; do
  for tomesd_ratio in "${tomesd_ratios[@]}"; do
    for use_dft in "${use_dft_vals[@]}"; do
      for trunc_ratio in "${trunc_ratios[@]}"; do
        if [ "$tomesd_ratio" = "0.7" ] && { [ "$trunc_ratio" = "0.8" ] || [ "$trunc_ratio" = "0.9" ]; }; then
          echo "skip use_tomesd=${use_tomesd}, tomesd_ratio=${tomesd_ratio}, use_dft=${use_dft}, trunc_ratio=${trunc_ratio}"
          continue
        fi
        out_dir="gen_ut${use_tomesd}_tr${tomesd_ratio}_dft${use_dft}_t${trunc_ratio}"
        echo "=== use_tomesd=${use_tomesd}, tomesd_ratio=${tomesd_ratio}, use_dft=${use_dft}, trunc_ratio=${trunc_ratio} → ${out_dir} ==="

        #OUTPUT=$(python generate_images.py \
        #  --use_tomesd   "${use_tomesd}" \
        #  --tomesd_ratio "${tomesd_ratio}" \
        #  --use_dft      "${use_dft}" \
        #  --trunc_ratio  "${trunc_ratio}" \
        #  --output_dir   "${out_dir}" 2>&1)
        #echo "$OUTPUT"

        python generate_images.py \
          --use_tomesd   "${use_tomesd}" \
          --tomesd_ratio "${tomesd_ratio}" \
          --use_dft      "${use_dft}" \
          --trunc_ratio  "${trunc_ratio}" \
          --output_dir   "${out_dir}"

        
        TIME=$(echo "$OUTPUT" \
          | grep 'Latency' \
          | sed -E 's/.*Latency: *([0-9]+\.[0-9]+) s.*/\1/')
        echo "Extracted TIME: ${TIME}s"



        FID=$(python calc_fid.py \
          --gen_dir  ${out_dir} \
          --real_dir data/imagenet_val_1000flat_resized)

        echo "${use_tomesd},${tomesd_ratio},${use_dft},${trunc_ratio},${TIME},${FID}" >> "$RESULTS"
        echo
      done
    done
  done
done
