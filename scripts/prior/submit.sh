#!/bin/bash
#
# Submit one SLURM job per non-comment line of the `jobs` file beside this
# script. Upstream Quetzal utility, used for pretraining the prior -- the
# steering experiments are driven by scripts/01..08 instead.
#
# Usage:  bash scripts/prior/submit.sh
#         JOBS=path/to/other_jobs bash scripts/prior/submit.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
JOBS="${JOBS:-$HERE/jobs}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm

if [[ ! -f "$JOBS" ]]; then
    echo "[fatal] no job file at $JOBS" >&2
    exit 1
fi

# Read job commands line by line, skipping comments and empty lines.
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    echo "Submitting: $line"
    echo

    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --job-name=train
#SBATCH --output=slurm/%j.out

source ~/.bashrc
conda activate quetzal

wandb offline
$line
EOF

    sleep 2

done < "$JOBS"
