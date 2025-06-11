########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data/int_deflections'
########################################################################################################################

########################################################################################################################
# int deflections: hcp data, group concatenation
outdir=${projdir}'/results/int_deflections/HCP-YA'
n_clusters=7
python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} \
    --fmri_file 'HCP-YA_Schaefer4007_rsts.npy' --file_prefix 'HCP-YA_Schaefer4007_rsts_' \
    --n_clusters ${n_clusters}
########################################################################################################################

########################################################################################################################
# int deflections: hcp data, group concatenation
outdir=${projdir}'/results/int_deflections/HCP-YA/subjects/kmeans'
n_clusters=7
for i in {0..945}; do
    file_prefix='HCP-YA_Schaefer4007_rsts_subject-'${i}'_'
    echo ${file_prefix}
    python ${scriptsdir}/compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} \
        --fmri_file 'HCP-YA_Schaefer4007_rsts.npy' --file_prefix ${file_prefix} \
        --n_clusters ${n_clusters} --subject_idx ${i}
done
########################################################################################################################