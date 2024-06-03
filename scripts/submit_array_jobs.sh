#!/bin/bash
#SBATCH --partition=main          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nct_xr         # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=8         # Cores per task (>1 if multithread tasks)
#SBATCH --mem=4000                # Real memory (RAM) required (MB)
#SBATCH --time=12:00:00           # Total run time limit (D-HH:MM:SS)
#SBATCH --output=slurm_%A_%a.out  # STDOUT output file
#SBATCH --array=0-365
# SBATCH --array=366-727

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID:" ${SLURM_ARRAY_TASK_ID}

sleep $((SLURM_ARRAY_TASK_ID))

########################################################################################################################
# directories
projdir='/home/lp756/projects/f_ah1491_1/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
outdir=${projdir}'/results'
fmri_clusters_file='hcp_fmri_clusters_k-7.npy'
A_file='hcp_schaefer400-7_A-features_schaefer_streamcount_areanorm_log.npy'
file_prefix='hcp'

python ${scriptsdir}/compute_optimized_control_energy.py \
    --indir ${indir} --outdir ${outdir} \
    --fmri_clusters_file ${fmri_clusters_file} --A_file ${A_file} --subj_idx ${SLURM_ARRAY_TASK_ID}

########################################################################################################################
