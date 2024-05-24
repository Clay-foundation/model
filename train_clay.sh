#!/bin/bash
#SBATCH --job-name=train_clay_v0.3.5
#SBATCH --partition=g4-queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0            # Set memory request (0 for all available)
#SBATCH --time=02:00:00    # Set maximum time request
#SBATCH --output=model_train_%j.log  # Standard output and error log

# Load any required modules (environments, libraries etc.)
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)"

# initialize conda
conda activate claymodel

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Run the model on multi node parallel cluster
srun python trainer.py fit --model ClayMAEModule --data ClayDataModule --config configs/config.yaml --data.data_dir /fsx
