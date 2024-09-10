#!/bin/bash
#SBATCH --account=m4287
#SBATCH --image=nersc/fair_universe:1298f0a8 
#SBATCH --qos=debug
#SBATCH -C gpu&hbm80g
#SBATCH -N 6
#SBATCH --gpus-per-task=4
#SBATCH -t 5
#SBATCH -J codabench
#SBATCH --mail-user=ragansu@nersc.gov  # Where to send mail
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
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


srun -n 6 -c 64 shifter bash -c "
    List_systematics=(tes jes soft_met bkg_scale ttbar_scale diboson_scale)
    systematics=\${List_systematics[\$SLURM_PROCID]}
    export MODEL_SYST_NAME=\$systematics
    sh ${working_dir}/HEP-Challenge/test/run_test.sh \$systematics
    "
echo "End"