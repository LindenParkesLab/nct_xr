########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
reference_state='xf'
########################################################################################################################

########################################################################################################################
# main analysis: hcp average connectome; k=7 (RE-RUN DONE)
outdir=${projdir}'/results/HCPYA'
A_file=${indir}'/HCPYA_Schaefer4007_Am.npy'
fmri_clusters_file=${outdir}'/HCPYA_Schaefer4007_rsts_fmri_clusters_k-7.npy'
file_prefix='HCPYA-Schaefer4007-Am'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state}

########################################################################################################################
