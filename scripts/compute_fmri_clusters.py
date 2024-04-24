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
from nilearn import datasets
import matplotlib.pyplot as plt
from snaplab_tools.plotting.plotting import surface_plot

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def run(config):
    indir = config['indir']
    outdir = config['outdir']
    rsfmri_file = config['rsfmri_file']
    tfmri_file = config['tfmri_file']
    file_prefix = config['file_prefix']
    n_clusters = config['n_clusters']
    gen_figs = config['gen_figs']
    print('\nn_clusters, {0}'.format(n_clusters))

    ####################################################################################################################
    # load resting-state time series
    print('Loading resting-state fMRI data...')
    rsfmri = np.load(os.path.join(indir, rsfmri_file))
    [n_trs, n_nodes, n_scans, n_subs] = rsfmri.shape
    print('n_trs, {0}; n_nodes, {1}; n_scans {2}; n_subs, {3}'.format(n_trs, n_nodes, n_scans, n_subs))

    # take first scan
    print('Retaining first scan...')
    rsfmri = rsfmri[:, :, 0, :]
    rsfmri = rsfmri[:, :, np.newaxis, :]
    [n_trs, n_nodes, n_scans, n_subs] = rsfmri.shape
    print('n_trs, {0}; n_nodes, {1}; n_scans {2}; n_subs, {3}'.format(n_trs, n_nodes, n_scans, n_subs))

    # reshape data to collapse scans
    print('concatenating time series...')
    rsfmri_concat = np.zeros((n_trs * n_subs * n_scans, n_nodes))
    print(rsfmri_concat.shape)
    rsfmri_concat[:] = np.nan
    start_idx = 0
    for i in tqdm(np.arange(n_subs)):
        for j in np.arange(n_scans):
            if i == 0 and j == 0:
                pass
            else:
                start_idx += n_trs
            end_idx = start_idx + n_trs
            rsfmri_concat[start_idx:end_idx, :] = rsfmri[:, :, j, i]
    print(np.any(np.isnan(rsfmri_concat)))
    del rsfmri
    ####################################################################################################################

    ####################################################################################################################
    # load task time series
    if tfmri_file is not None:
        print('Loading task-based fMRI data...')
        tfmri = np.load(os.path.join(indir, tfmri_file))
        [n_trs, n_nodes, n_scans, n_subs] = tfmri.shape
        print('n_trs, {0}; n_nodes, {1}; n_scans {2}; n_subs, {3}'.format(n_trs, n_nodes, n_scans, n_subs))

        # take first scan
        print('Retaining first scan...')
        tfmri = tfmri[:, :, 0, :]
        tfmri = tfmri[:, :, np.newaxis, :]
        [n_trs, n_nodes, n_scans, n_subs] = tfmri.shape
        print('n_trs, {0}; n_nodes, {1}; n_scans {2}; n_subs, {3}'.format(n_trs, n_nodes, n_scans, n_subs))

        # reshape data to collapse scans
        print('concatenating time series...')
        tfmri_concat = np.zeros((n_trs * n_subs * n_scans, n_nodes))
        print(tfmri_concat.shape)
        tfmri_concat[:] = np.nan
        start_idx = 0
        for i in tqdm(np.arange(n_subs)):
            for j in np.arange(n_scans):
                if i == 0 and j == 0:
                    pass
                else:
                    start_idx += n_trs
                end_idx = start_idx + n_trs
                tfmri_concat[start_idx:end_idx, :] = tfmri[:, :, j, i]
        print(np.any(np.isnan(tfmri_concat)))
        del tfmri
    ####################################################################################################################

    ####################################################################################################################
    print('Computing k-means solution...')
    start = time.time()
    if tfmri_file is not None:
        ts_concat = np.concatenate((rsfmri_concat, tfmri_concat), axis=0)
    else:
        ts_concat = rsfmri_concat
    print(ts_concat.shape)

    # extract clusters of activity
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(rsfmri_concat)

    # extract cluster centers. These represent dominant patterns of recurrent activity over time
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(centroids.shape, labels.shape)

    # plot centroids on brain surface
    if gen_figs:
        print('Plotting k-means solution...')
        lh_annot_file = (
            "/home/lindenmp/research_projects/connectome_loader/data/schaefer_parc/"
            "fsaverage5/lh.Schaefer2018_400Parcels_7Networks_order.annot"
        )
        rh_annot_file = (
            "/home/lindenmp/research_projects/connectome_loader/data/schaefer_parc/"
            "fsaverage5/rh.Schaefer2018_400Parcels_7Networks_order.annot"
        )
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

        for cluster in np.arange(n_clusters):
            f = surface_plot(
                data=centroids[cluster, :],
                lh_annot_file=lh_annot_file,
                rh_annot_file=rh_annot_file,
                fsaverage=fsaverage,
                order="lr",
                cmap="coolwarm",
            )
            f.savefig(
                os.path.join(outdir, "{0}k-{1}_cluster_{2}.png".format(file_prefix, n_clusters, cluster)),
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
            )
    ####################################################################################################################

    # save outpputs
    log_args = {
        'centroids': centroids,
        'labels': labels,
    }
    file_str = '{0}fmri_clusters_k-{1}'.format(file_prefix, n_clusters)
    np.save(os.path.join(outdir, file_str), log_args)

    end = time.time()
    print('...done in {:.2f} seconds.'.format(end - start))

# %%
def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='/home/lindenmp/research_projects/nct_xr/data')
    parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    parser.add_argument('--rsfmri_file', type=str, default='hcp_schaefer400-7_rsts.npy')
    # parser.add_argument('--tfmri_file', type=str, default='hcp_schaefer400-7_taskts.npy')
    parser.add_argument('--tfmri_file', type=str, default=None)
    parser.add_argument('--file_prefix', type=str, default='hcp_')

    # settings
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--gen_figs', type=bool, default=True)

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
        'rsfmri_file': args.rsfmri_file,
        'tfmri_file': args.tfmri_file,
        'file_prefix': args.file_prefix,

        # settings
        'n_clusters': args.n_clusters,
        'gen_figs': args.gen_figs,
    }

    run(config=config)
