#!/bin/bash
#SBATCH --account=m4287
#SBATCH --image=nersc/fair_universe:1298f0a8 
#SBATCH --qos=premium
#SBATCH -C gpu&hbm80g
#SBATCH -N 6
#SBATCH --gpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -J codabench
#SBATCH --output=codabench_%j.log   # Standard output and error log

now=$(date +"%Y%m%d_%H%M%S")

echo
echo "Start"
echo "Date and Time      : $now"
echo "SLURM_JOB_ID       : $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST : $SLURM_JOB_NODELIST"
echo "SLURM_NNODES       : $SLURM_NNODES"
echo

working_dir="/global/cfs/cdirs/m4287/hep"

if [ $# -eq 0 ]; then
  echo "No Model name provided"
  model_name="simple_one_syst_model"
else
  echo "Model name provided: $1"
  model_name=$1
fi

srun -n 2 shifter bash -c "
    List_systematics=(tes jes soft_met bkg_scale ttbar_scale diboson_scale)
    systematics=\${List_systematics[\$SLURM_PROCID]}
    export MODEL_SYST_NAME=\$systematics
    sh ${working_dir}/HEP-Challenge/test/run_test.sh \$systematics ${model_name}
    "
echo "End"