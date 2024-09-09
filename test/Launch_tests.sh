#!/bin/bash
#SBATCH --account=m4287
#SBATCH --image=nersc/fair_universe:1298f0a8 
#SBATCH --qos=regular
#SBATCH -C gpu&hbm80g
#SBATCH -N 6
#SBATCH --gpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -J codabench
#SBATCH --mail-user=ragansu@nersc.gov  # Where to send mail
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=codabench_%j.log   # Standard output and error log

now=$(date +"%Y%m%d_%H%M%S")

echo "Start"
echo "Date and Time : $now"

working_dir="/global/cfs/cdirs/m4287/hep"

input_dir="/global/cfs/projectdirs/m4287/data"
reference_dir="${working_dir}/public_data_neurips/reference_data"

export MAX_WORKERS=30



srun shifter python3 ${working_dir}/HEP-Challenge/ingestion_program/run_ingestion.py --use-random-mus --input $input_dir --num-pseudo-experiments 100 --num-of-sets 1 --parallel --submission ${working_dir}/HEP-Challenge/test_model

srun shifter python3 ${working_dir}/HEP-Challenge/scoring_program/run_scoring.py


srun -n 6 -c 64 shifter bach -c "
    List_systematics=[tes,jes,soft_met,bkg_scale,ttbar_scale,diboson_scale]
    systematics=\${List_systematics[\$SLURM_PROCID]}
    export MODEL_SYST_NAME=\$systematics
    systematics_flag=\$(echo \$systematics | tr '_' '-')
    use_systematics=--systematics-\${systematics_flag}
    mkdir -p ${working_dir}/HEP-Challenge/test/simple_one_syst_model_\${systematics}
    python3 ${working_dir}/HEP-Challenge/ingestion_program/run_ingestion.py \\
    --use-random-mus --input $input_dir --num-pseudo-experiments 100 --num-of-sets 1 --parallel \\
    --submission ${working_dir}/HEP-Challenge/simple_one_syst_model --systematics \$systematics \\
    --output ${working_dir}/HEP-Challenge/simple_one_syst_model_\${systematics}
    
    python3 ${working_dir}/HEP-Challenge/scoring_program/run_scoring.py --prediction ${working_dir}/HEP-Challenge/simple_one_syst_model_\${systematics} \\
"

echo "End"