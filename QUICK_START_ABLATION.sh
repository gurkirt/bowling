#!/bin/bash
# QUICK START: Ablation Study

# 1. START ALL 4 EXPERIMENTS (2 per GPU)
# ./run_ablation_experiments.sh

# 2. MONITOR PROGRESS
# Option A: Real-time GPU monitoring
# nvidia-smi -i 0,1 -l 1

# Option B: Watch log files
tail -f ablation_logs/exp*.log

# Option C: Progress summary
python analyze_ablation_results.py progress

# 3. STOP ALL EXPERIMENTS (if needed)
pkill -f train_classifier.py

# 4. VIEW FINAL RESULTS
python analyze_ablation_results.py

# 5. COMPARE BEST MODELS
# Results are in trainings/exp{1,2,3,4}_*/ directories
# Each contains:
#   - best_model.pth (best checkpoint)
#   - training_history.png (loss/accuracy curves)
#   - confusion_matrix.png (performance matrix)
#   - metrics.json (numerical results)

echo "See ABLATION_STUDY.md for detailed documentation"
