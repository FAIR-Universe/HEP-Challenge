echo "Running $@"

working_dir="/global/cfs/cdirs/m4287/hep"

input_dir="/global/cfs/projectdirs/m4287/data"
reference_dir="${working_dir}/public_data_neurips/reference_data"
ingestion_program_dir=${working_dir}/HEP-Challenge/ingestion_program/run_ingestion.py 
scoring_program_dir=${working_dir}/HEP-Challenge/scoring_program/run_scoring.py
model_dir=${working_dir}/HEP-Challenge/simple_one_syst_model
export MAX_WORKERS=30

systematics=$1
export MODEL_SYST_NAME=\$systematics

systematics_flag=$(echo $systematics | tr '_' '-')
output_dir=${working_dir}/HEP-Challenge/test/temp/simple_one_syst_model_${systematics}

mkdir -p ${output_dir}
mkdir -p ${output_dir}/submission

cp -r ${model_dir}/* ${output_dir}/submission

python3 ${ingestion_program_dir} \
    --use-random-mus --input $input_dir \
    --num-pseudo-experiments 100 \
    --num-of-sets 1 \
    --parallel \
    --submission ${output_dir}/submission \
    --systematics-${systematics_flag} \
    --output ${output_dir} 2> ${output_dir}/ingestion.log

reco_rc=$?
if [ $reco_rc != 0 ]; then
    exit $reco_rc
fi

python3 ${scoring_program_dir} --prediction ${output_dir} 2> ${output_dir}/scoring.log

rm -rf ${output_dir}/submission