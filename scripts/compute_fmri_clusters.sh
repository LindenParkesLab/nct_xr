########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
########################################################################################################################

########################################################################################################################
# main analysis: hcp data
outdir=${projdir}'/results/HCPYA'
for n_clusters in $(seq 2 20); do
    python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCPYA_Schaefer4007_rsts.npy' --file_prefix 'HCPYA_Schaefer4007_rsts_' --n_clusters ${n_clusters}
done
########################################################################################################################

########################################################################################################################
# main analysis: mica-mics data
outdir=${projdir}'/results/MICS'
for n_clusters in $(seq 2 20); do
    python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'MICS_Schaefer4007_rsts.npy' --file_prefix 'MICS_Schaefer4007_rsts_' --n_clusters ${n_clusters}
done
########################################################################################################################
