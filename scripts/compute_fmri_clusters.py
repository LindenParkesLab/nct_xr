# import
import os, sys, warnings, argparse, time
sys.path.extend(['/home/lindenmp/research_projects/nct_xr'])
sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])

import numpy as np
import scipy as sp
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from snaplab_tools.plotting.plotting import surface_plot

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def run(config):
    start = time.time()
    indir = config['indir']
    outdir = config['outdir']
    fmri_file = config['fmri_file']
    file_prefix = config['file_prefix']
    n_clusters = config['n_clusters']

    def get_concat_timeseries(fmri_data, retain_subjects=None):
        [n_trs, n_nodes, n_scans, n_subs] = fmri_data.shape
        # if n_scans == 2:
        #     fmri_data = fmri_data[:, :, 0, :]  # get the LR phase encoding scan
        #     fmri_data = fmri_data[:, :, np.newaxis, :]
        # elif n_scans == 4:
        #     fmri_data = fmri_data[:, :, [0, 2], :]  # both scans of LR phase encoding
        fmri_data = fmri_data[:, :, 0, :]  # get the LR phase encoding scan
        fmri_data = fmri_data[:, :, np.newaxis, :]
        
        if retain_subjects is not None:
            fmri_data = fmri_data[:, :, :, :retain_subjects]

        [n_trs, n_nodes, n_scans, n_subs] = fmri_data.shape
        print('n_trs, {0}; n_nodes, {1}; n_scans {2}; n_subs, {3}'.format(n_trs, n_nodes, n_scans, n_subs))

        # reshape data to collapse scans
        fmri_concat = np.zeros((n_trs * n_subs * n_scans, n_nodes))
        fmri_concat[:] = np.nan
        print('Concatenating time series. Output shape: {0}'.format(fmri_concat.shape))
        fmri_concat_subjidx = np.zeros((n_trs * n_subs * n_scans, 1))
        fmri_concat_subjidx[:] = np.nan
        start_idx = 0
        for i in tqdm(np.arange(n_subs)):
            for j in np.arange(n_scans):
                if i == 0 and j == 0:
                    pass
                else:
                    start_idx += n_trs
                end_idx = start_idx + n_trs
                fmri_concat[start_idx:end_idx, :] = fmri_data[:, :, j, i]
                fmri_concat_subjidx[start_idx:end_idx, :] = i

        return fmri_concat, fmri_concat_subjidx

    ####################################################################################################################
    # load fMRI time series
    print('Loading fMRI data')
    if type(fmri_file) == list:
        print('\tlooping over list of fmri files')
        for i in np.arange(len(fmri_file)):
            print('\t\t...{0}'.format(fmri_file[i]))
            fmri_data = np.load(os.path.join(indir, fmri_file[i]))
            fmri_concat_scan, fmri_concat_subjidx_scan = get_concat_timeseries(fmri_data=fmri_data)
            if i == 0:
                fmri_concat = np.array([], dtype=np.float64).reshape(0, fmri_concat_scan.shape[1])
                fmri_concat_subjidx = np.array([], dtype=np.float64).reshape(0, 1)
            fmri_concat = np.vstack((fmri_concat, fmri_concat_scan))       
            fmri_concat_subjidx = np.vstack((fmri_concat_subjidx, fmri_concat_subjidx_scan))       
    else:
        print('\tfound one fmri file: {0}'.format(fmri_file))
        fmri_data = np.load(os.path.join(indir, fmri_file))
        fmri_concat, fmri_concat_subjidx = get_concat_timeseries(fmri_data=fmri_data)
        
    ####################################################################################################################

    ####################################################################################################################
    print('Computing k-means solution. Input shape: {0}, n_clusters = {1}'.format(fmri_concat.shape, n_clusters))
    
    # extract clusters of activity
    n_timepoints, n_nodes = fmri_concat.shape
    fmri_concat = sp.stats.zscore(fmri_concat, axis=0)
    nan_mask = np.sum(np.isnan(fmri_concat), axis=0) > 0
    # extract cluster centers. These represent dominant patterns of recurrent activity over time
    if np.any(nan_mask):
        print('WARNING: Found NaNs in centroids... masking out of k-means')
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fmri_concat[:, ~nan_mask])
        centroids = np.zeros((n_clusters, n_nodes))
        centroids[:] = np.nan
        centroids[:, ~nan_mask] = kmeans.cluster_centers_
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fmri_concat)
        centroids = kmeans.cluster_centers_

    labels = kmeans.labels_
    inertia = kmeans.inertia_
    print(centroids.shape, labels.shape)
    ####################################################################################################################

    ####################################################################################################################
    # print('Calculating silhouette score')
    # sil_score = silhouette_score(fmri_concat, labels)
    # print('\t...silhouette score = {:.2f}'.format(sil_score))
    ####################################################################################################################

    ####################################################################################################################
    print('Calculating subject fraction')
    # For each cluster, compute fraction of subjects that have frames representing that cluster
    centroid_subj_frac = []
    for i in np.arange(n_clusters):
        centroid_subj_frac.append(len(np.unique(fmri_concat_subjidx[labels == i])) / len(np.unique(fmri_concat_subjidx)))
    print('\t...fraction of subjects with frames in each cluster = {0}'.format(centroid_subj_frac))
    ####################################################################################################################

    ####################################################################################################################
    print('Calculating variance explained')
    n_labels = labels.shape[0]
    average_centroid = np.zeros(centroids.shape)
    for i in np.arange(n_clusters):
        average_centroid[i, :] = np.sum(labels == i) * centroids[i, :]
    average_centroid = np.sum(average_centroid, 0) / n_labels
    
    within_cluster_variance = np.zeros(n_clusters)
    between_cluster_variance = np.zeros(n_clusters)
    for i in np.arange(n_clusters):
        within_cluster_variance[i] = sp.spatial.distance.cdist(centroids[i, :][np.newaxis, :], fmri_concat[labels == i, :], metric='seuclidean').sum()
        between_cluster_variance[i] = sp.spatial.distance.cdist(centroids[i, :][np.newaxis, :], average_centroid[np.newaxis, :], metric='seuclidean')[0][0] * np.sum(labels == i)

    within_cluster_variance = np.sum(within_cluster_variance) / n_labels
    between_cluster_variance = np.sum(between_cluster_variance) / n_labels
    variance_explained = between_cluster_variance / (within_cluster_variance + between_cluster_variance)
    print('\t...variance explained = {:.2f}'.format(variance_explained * 100))
    ####################################################################################################################

    ####################################################################################################################
    print('Saving outputs')
    # save outpputs
    log_args = {
        'centroids': centroids,
        'labels': labels,
        'inertia': inertia,
        'centroid_subj_frac': centroid_subj_frac,
        'variance_explained': variance_explained,
    }
    file_str = '{0}fmri_clusters_k-{1}'.format(file_prefix, n_clusters)
    np.save(os.path.join(outdir, file_str), log_args)
    if n_clusters == 7:
        np.save(os.path.join(outdir, '{0}fmri_clusters_fmri_concat'.format(file_prefix)), fmri_concat)
        np.save(os.path.join(outdir, '{0}fmri_clusters_fmri_concat_subjidx'.format(file_prefix)), fmri_concat_subjidx)
    ####################################################################################################################

    end = time.time()
    print('Done in {:.2f} minutes.'.format((end - start) / 60))

# %%
def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='/home/lindenmp/research_projects/nct_xr/data')
    parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    parser.add_argument('--fmri_file', type=str, default='hcp_schaefer400-7_rsts.npy')
    # parser.add_argument('--fmri_file', type=str, default=['hcp_schaefer400-7_rsts.npy',
                                                        #   'hcp_schaefer400-7_taskts_wm.npy'])

    # parser.add_argument('--fmri_file', type=str, default=['hcp_schaefer400-7_rsts.npy',
    #                                                       'hcp_schaefer400-7_taskts_emotion.npy',
    #                                                       'hcp_schaefer400-7_taskts_gambling.npy',
    #                                                       'hcp_schaefer400-7_taskts_language.npy',
    #                                                       'hcp_schaefer400-7_taskts_motor.npy',
    #                                                       'hcp_schaefer400-7_taskts_relational.npy',
    #                                                       'hcp_schaefer400-7_taskts_social.npy',
    #                                                       'hcp_schaefer400-7_taskts_wm.npy'])
    parser.add_argument('--file_prefix', type=str, default='hcp_')

    # settings
    parser.add_argument('--n_clusters', type=int, default=7)

    args = parser.parse_args()
    args.indir = os.path.expanduser(args.indir)
    args.outdir = os.path.expanduser(args.outdir)

    return args

# %%
if __name__ == '__main__':
    args = get_args()

    config = {
        'indir': args.indir,
        'outdir': args.outdir,
        'fmri_file': args.fmri_file,
        'file_prefix': args.file_prefix,

        # settings
        'n_clusters': args.n_clusters,
    }

    run(config=config)
