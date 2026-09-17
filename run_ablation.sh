#!/usr/bin/env bash
# =============================================================================
#  Feature ablation runner — executes all eight feature configurations.
#
#  Each mode is passed on the command line, so no file editing is required.
#  Results from all modes and all three validation schemes append to a single
#  CSV, with deduplication on (Mode, Validation, Model).
#
#  Usage:
#     bash run_ablation.sh              # interactive file dialog per mode
#     bash run_ablation.sh --no-gui     # batch, uses the default data folder
#
#  Output:
#     results/ablation_results.csv      combined metrics, all modes
#     results/ablation_<MODE>/          figures per mode
#     ablation_<MODE>.log               console output per mode
# =============================================================================

SCRIPT="chatter_detection.py"
EXTRA="$@"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in the current directory."
  exit 1
fi

for M in A B C D E F G H; do
  echo ""
  echo "==============================================="
  echo "  ABLATION MODE $M"
  echo "==============================================="
  MPLBACKEND=Agg python3 "$SCRIPT" --mode "$M" $EXTRA 2>&1 | tee "ablation_$M.log"
done

echo ""
echo "==============================================="
echo "  ALL MODES COMPLETE"
echo "==============================================="
echo "  Combined results: results/ablation_results.csv"
