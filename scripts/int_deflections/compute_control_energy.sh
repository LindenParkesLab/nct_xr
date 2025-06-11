########################################################################################################################
# directories
projdir='/home/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data/int_deflections'
reference_state='xf'
########################################################################################################################

########################################################################################################################
# int deflections
outdir=${projdir}'/results/int_deflections/HCP-YA'
n_parcels=400
A_file=${indir}'/HCP-YA_Schaefer'${n_parcels}'7_Am.npy'

# task contrasts
fmri_clusters_file=${outdir}'/HCP-YA_Schaefer'${n_parcels}'7_states.npy'
file_prefix='HCP-YA-Schaefer'${n_parcels}'7-Am'

python ${scriptsdir}/compute_optimized_control_energy.py --outdir ${outdir} \
--A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
--file_prefix ${file_prefix} --reference_state ${reference_state} \
--run_rand_control_set 'False'
########################################################################################################################
