#!/bin/bash

# Init logging dir and common file
train_dir="/scratch/project/open-26-5/msml/msml/experiments/pre_training/"
outdir="${train_dir}/submissions"
mkdir -p "${outdir}"
outfile="${outdir}/submissions.csv"

# Generate random key for a job
job_key=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10 ; echo '')

# Submit
job_id=$(qsub \
  -A "${PROJECT_ID_TOMAS_SOUCEK}" \
  -q qnvidia \
  -l walltime=24:00:00 \
  -o "${outdir}/${job_key}"_stdout.txt \
  -e "${outdir}/${job_key}"_errout.txt \
  -N "${job_key}" \
  -v job_key="${job_key}" \
  "${train_dir}/fine_tune_karolina.sh"
)

# Log
submission="$(date),${job_id},${job_key}"
echo "${submission}" >> "${outfile}"
echo "${submission}"
