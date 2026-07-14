"""sbatch script templates for the two-container `.mimosepkg` evaluation
pipeline (docker/preprocess/ -> docker/test/), run via SLURM's pyxis/enroot
`srun --container-image=...` integration (this cluster has no plain `docker`
on compute nodes). See docker/slurm/submit_pipeline.py for the orchestrator.
"""
from __future__ import annotations

PREPROCESS_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=mimose-preprocess-{run_id}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={preprocess_cpus}
#SBATCH --mem={preprocess_mem}
#SBATCH --time={preprocess_time}
#SBATCH -e {log_dir}/preprocess-{run_id}.err
#SBATCH -o {log_dir}/preprocess-{run_id}.out
#SBATCH --gres=gpu:0
#SBATCH --account={account}

srun --container-image={preprocess_image} \\
     --container-mounts={package_dir}:/submission:ro,{raw_data_dir}:/input:ro,{shared_dir}:/shared \\
     --no-container-mount-home \\
     python -m harness.preprocess_run \\
       --package /submission/{package_name} \\
       --raw-data-dir /input \\
       --output-dir /shared \\
       --num-workers {preprocess_cpus}
"""

TEST_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=mimose-test-{run_id}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={test_cpus}
#SBATCH --mem={test_mem}
#SBATCH --time={test_time}
#SBATCH -e {log_dir}/test-{run_id}.err
#SBATCH -o {log_dir}/test-{run_id}.out
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --account={account}
#SBATCH --dependency=afterok:{preprocess_job_id}

srun --container-image={test_image} \\
     --container-mounts={package_dir}:/submission:ro,{shared_dir}:/input:ro,{results_dir}:/output \\
     --no-container-mount-home \\
     python -m harness.run \\
       --package /submission/{package_name} \\
       --data-dir /input/preprocessed \\
       --output-dir /output \\
       --dataset-type {dataset_type} \\
       --split-file /input/split.json \\
       --num-workers {test_cpus}
"""
