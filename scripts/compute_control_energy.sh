########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
outdir=${projdir}'/results'
########################################################################################################################

########################################################################################################################
# main analysis: Average connectome; k=7 
A_file=${indir}'/hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy'
fmri_clusters_file=${outdir}'/hcp_fmri_clusters_k-7.npy'
file_prefix='hcp-Am'
reference_state='xf'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state} \
--run_rand_control_set 'True' --run_yeo_control_set 'True'
########################################################################################################################

########################################################################################################################
# replication analysis: Average connectome; k=2-19
A_file=${indir}'/hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy'
file_prefix='hcp-Am'
reference_state='xf'
outdir2=${projdir}'/results/replication_krange'

for n_clusters in $(seq 2 20); do
    fmri_clusters_file=${outdir}'/hcp_fmri_clusters_k-'${n_clusters}'.npy'
    echo ${fmri_clusters_file}

    python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir2} \
    --A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
    --file_prefix ${file_prefix} --reference_state ${reference_state} \
    --run_rand_control_set 'False' --run_yeo_control_set 'False'
done
########################################################################################################################