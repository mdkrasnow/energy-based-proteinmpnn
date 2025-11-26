#!/bin/bash
set -euo pipefail

USER="mkrasnow"
HOST="login.rc.fas.harvard.edu"
REMOTE="${USER}@${HOST}"

# Control socket path for connection sharing
CTRL_PATH="$HOME/.ssh/ctl_%h_%p_%r"

echo "Opening master SSH connection to ${REMOTE}..."
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath="${CTRL_PATH}" \
  -o ControlPersist=600 \
  "${REMOTE}"

echo "Running SCP transfers using shared connection..."

# Copy main training and evaluation Python scripts
scp -o ControlPath="${CTRL_PATH}" hybrid/evaluation/run_comprehensive_evaluation.py "${REMOTE}:~/hybrid_evaluation/run_comprehensive_evaluation.py"
scp -o ControlPath="${CTRL_PATH}" hybrid/training/train_energy.py "${REMOTE}:~/hybrid_training/train_energy.py"

# Copy Slurm batch scripts
scp -o ControlPath="${CTRL_PATH}" train_hybrid_proteinmpnn.sh "${REMOTE}:~/train_hybrid_proteinmpnn.sh"
scp -o ControlPath="${CTRL_PATH}" eval_hybrid_proteinmpnn.sh "${REMOTE}:~/eval_hybrid_proteinmpnn.sh"

echo "Closing master SSH connection..."
ssh -O exit -o ControlPath="${CTRL_PATH}" "${REMOTE}"

echo "All files copied."