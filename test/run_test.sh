# Description: Run the test for a model with one systematic
systematics=$1
model_name=$2

echo Running test for model ${model_name} with systematic ${systematics}

working_dir="/global/cfs/cdirs/m4287/hep"
input_dir="/global/cfs/projectdirs/m4287/data"
reference_dir="${working_dir}/public_data_neurips/reference_data"
ingestion_program_dir=${working_dir}/HEP-Challenge/ingestion_program/run_ingestion.py 
scoring_program_dir=${working_dir}/HEP-Challenge/scoring_program/run_scoring.py
test_dir=${working_dir}/HEP-Challenge/test
export MAX_WORKERS=30


if [ "$model_name" = "simple_one_syst_model" ]; then
    export MODEL_SYST_NAME=$systematics
fi



model_dir=${working_dir}/HEP-Challenge/${model_name}

systematics_flag=$(echo $systematics | tr '_' '-')
output_dir=${working_dir}/HEP-Challenge/test/temp/${model_name}_${systematics}

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

python3 ${scoring_program_dir} --prediction ${output_dir} --output ${output_dir} 2> ${output_dir}/scoring.log

reco_rc=$?
if [ $reco_rc != 0 ]; then
    exit $reco_rc
fi

python3 ${test_dir}/run_performance_test.py  --result-file ${output_dir}/scores.json

rm -rf ${output_dir}/submission