########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data'
outdir=${projdir}'/results/replication'
########################################################################################################################

########################################################################################################################
# main analysis: hcp data
for n_clusters in $(seq 2 20); do
    python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'hcp_schaefer400-7_rsts.npy' --file_prefix 'hcp_' --n_clusters ${n_clusters}
done
########################################################################################################################

########################################################################################################################
# main analysis: mica-mics data
for n_clusters in $(seq 2 20); do
    python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'mics_schaefer400-7_rsts.npy' --file_prefix 'mics_' --n_clusters ${n_clusters}
done
########################################################################################################################
