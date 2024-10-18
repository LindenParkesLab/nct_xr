########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
outdir=${projdir}'/results'
########################################################################################################################

########################################################################################################################
reference_state='xf'

# main analysis: Average connectome; k=7 
A_file=${indir}'/hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy'
fmri_clusters_file=${outdir}'/hcp_fmri_clusters_k-7.npy'
file_prefix='hcp-Am'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state} \
--run_rand_control_set 'True'
########################################################################################################################

########################################################################################################################
outdir_replication=${outdir}'/replication'

# replication analysis: Lausanne average connectome; k=7 
A_file=${indir}'/lausanne_connectome.npy'
fmri_clusters_file=${indir}'/lausanne_brain_states.npy'
file_prefix='laus-Am'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir_replication} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state}

# replication analysis: MICA-MICs average connectome; k=7 
A_file=${indir}'/mics_schaefer400-7_Am.npy'
fmri_clusters_file=${outdir_replication}'/mics_fmri_clusters_k-7.npy'
file_prefix='mics-Am'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir_replication} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state}
########################################################################################################################

########################################################################################################################
# replication analysis: Average connectome; k=2-19
A_file=${indir}'/hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy'
file_prefix='hcp-Am'

for n_clusters in $(seq 2 14); do
    fmri_clusters_file=${outdir_replication}'/hcp_fmri_clusters_k-'${n_clusters}'.npy'
    echo ${fmri_clusters_file}

    python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir_replication} \
    --A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
    --file_prefix ${file_prefix} --reference_state ${reference_state}
done
########################################################################################################################
