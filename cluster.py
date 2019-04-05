import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
import numpy as np
from heapq import nsmallest
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from itertools import groupby

"""
Uses the SpectralBiclustering method for biclustering the data.
If use_limits=False, then use the full range of data values for color mapping. Otherwise, use the minimum and maximum 
value of the 80% of data that is closest to the mean value.
"""
def spectral_bicluster(data, n_clusters=(2, 2), method='log', s_name=None, plot_results=True, plot_title=None,
                       n_components=6, n_best=3, svd_method="randomized", n_svd_vecs=None, init="k-means++", n_init=10,
                       use_limits=False):
    model = SpectralBiclustering(n_clusters=n_clusters, method=method, n_components=n_components, n_best=n_best,
                                 svd_method=svd_method, n_svd_vecs=n_svd_vecs, init=init, n_init=n_init, random_state=0)
    model.fit(data)

    closest_80p = None
    if use_limits:
        mean = data.mean()
        # Total number of values in the data
        n_values = len(data) * len(data[0])
        # Finds the 80% of values that are closest to the mean by absolute value
        closest_80p = sorted(nsmallest(round(n_values * 0.8), data.flatten(), key=lambda x: abs(x - mean)))
        print("80% of values closest to the mean in", (closest_80p[0], closest_80p[-1]))
        print("Full range:", (sorted(data.flatten())[0], sorted(data.flatten())[-1]))

    if plot_results:
        fig, axes = plt.subplots(3, 1, figsize=(150, 80))
        axes = axes.flatten()

        fig.subplots_adjust(wspace=0)
        img1 = axes[0].matshow(data, cmap=plt.cm.Blues)
        if s_name is not None:
            axes[0].set_title("Subject %s, original dataset" % s_name)
        else:
            axes[0].set_title("Original dataset")

        fit_data = data[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]

        img2 = axes[1].matshow(fit_data, cmap=plt.cm.Blues)
        if s_name is not None:
            axes[1].set_title("Subject %s, after biclustering; rearranged to show biclusters" % s_name)
        else:
            axes[1].set_title("After biclustering; rearranged to show biclusters")

        if use_limits:
            img1.set_clim(closest_80p[0], closest_80p[-1])
            img2.set_clim(closest_80p[0], closest_80p[-1])

        axes[2].matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1), cmap=plt.cm.Blues)
        if s_name is not None:
            axes[2].set_title("Subject %s, checkerboard structure of rearranged data" % s_name)
        else:
            axes[2].set_title("Checkerboard structure of rearranged data")

        if plot_title is not None:
            fig.savefig(plot_title + ".png", bbox_inches='tight')

    return model.row_labels_, model.column_labels_, model.rows_, model.columns_


"""
Uses the SpectralCoclustering method for biclustering the data.
"""
def bicocluster(data, n_clusters=5, svd_method="randomized", s_name=None, plot_results=True, plot_title=None,
                use_limits=False):
    model = SpectralCoclustering(n_clusters=n_clusters, svd_method=svd_method, random_state=0)
    model.fit(data)

    closest_80p = None
    if use_limits:
        mean = data.mean()
        # Total number of values in the data
        n_values = len(data) * len(data[0])
        # Finds the 80% of values that are closest to the mean by absolute value
        closest_80p = sorted(nsmallest(round(n_values * 0.8), data.flatten(), key=lambda x: abs(x - mean)))
        print("80% of values closest to the mean in", (closest_80p[0], closest_80p[-1]))
        print("Full range:", (sorted(data.flatten())[0], sorted(data.flatten())[-1]))

    if plot_results:
        fig, axes = plt.subplots(3, 1, figsize=(150, 80))
        axes = axes.flatten()

        fig.subplots_adjust(wspace=0)
        img1 = axes[0].matshow(data, cmap=plt.cm.Blues)
        if s_name is not None:
            axes[0].set_title("Subject %s, original dataset" % s_name)
        else:
            axes[0].set_title("Original dataset")

        fit_data = data[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]

        img2 = axes[1].matshow(fit_data, cmap=plt.cm.Blues)
        if s_name is not None:
            axes[1].set_title("Subject %s, after biclustering; rearranged to show biclusters" % s_name)
        else:
            axes[1].set_title("After biclustering; rearranged to show biclusters")

        if use_limits:
            img1.set_clim(closest_80p[0], closest_80p[-1])
            img2.set_clim(closest_80p[0], closest_80p[-1])

        axes[2].matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1), cmap=plt.cm.Blues)
        if s_name is not None:
            axes[2].set_title("Subject %s, checkerboard structure of rearranged data" % s_name)
        else:
            axes[2].set_title("Checkerboard structure of rearranged data")

        if plot_title is not None:
            fig.savefig(plot_title + ".png", bbox_inches='tight')

    return model.row_labels_, model.column_labels_, model.rows_, model.columns_


"""
Returns the biclusters in the data which correspond to the given row and column cluster labels.  
"""
def get_biclusters(data, row_labels, col_labels):
    # Sort the data to create biclusters
    fit_data = data[np.argsort(row_labels)]
    fit_data = fit_data[:, np.argsort(col_labels)]

    # Get the indices which sort the data into biclusters
    sorted_rlabels = np.sort(row_labels)
    sorted_clabels = np.sort(col_labels)

    clusters = []
    for r_label in set(sorted_rlabels):  # for row cluster
        r_label_idx = np.argwhere(sorted_rlabels == r_label).flatten()  # get the indices of the rows in cluster r_label
        for c_label in set(sorted_clabels):  # for col cluster
            c_label_idx = np.argwhere(
                sorted_clabels == c_label).flatten()  # get the indices of the columns in cluster c_label
            clusters.append(fit_data[r_label_idx][:, c_label_idx])

    return clusters


"""
Return the ASR value for the given bicluster.
"""
def ASR(data, row_labels, col_labels):
    clusters = get_biclusters(data, row_labels, col_labels)

    sum_asr = 0
    for cl in clusters:
        n_rows = len(cl)
        n_cols = len(cl[0])

        if n_rows == 1 or n_cols == 1:
            continue  # because the asr would be nan

        sum_corrs_r = 0
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                spr = spearmanr(cl[i], cl[j])[0]
                sum_corrs_r += (0 if np.isnan(spr) else spr)

        rho_rows = sum_corrs_r / ((n_rows * (n_rows - 1)) if n_rows > 1 else 1)

        sum_corrs_c = 0
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                sum_corrs_c += spearmanr(cl[:, i], cl[:, j])[0]

        rho_cols = sum_corrs_c / ((n_cols * (n_cols - 1)) if n_cols > 1 else 1)

        asr = max([rho_rows, rho_cols]) * 2
        sum_asr += asr

    return round(sum_asr / len(clusters), 3)


"""
Get the electrode time windows and trials which belong to the given bicluster (as defined by the given row and column
labels). Return list of electrodes, represented as strings in the form 'elec<electrode nr> w<time window nr>", 
and a list of trials, represented as strings in the form 'trial<trial nr>'.
"""
def get_bicluster_trials_elecs(tw_per_el, row_cl, col_cl, row_labels, col_labels, get_trials=True, get_elecs=True,
                               trials_as_labels=True):
    # row_cl -- a number 0..n_row_clusters - 1
    # col_cl -- a number 0..n_col_clusters - 1

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    sorted_elecs = None
    col_idx = None
    if get_elecs:
        n_elecs = n_cols // tw_per_el
        elecs = np.array(
            [[("elec%d w%d" % (i, j)) for j in range(1, tw_per_el + 1)] for i in range(1, n_elecs + 1)]).flatten()
        sorted_elecs = elecs[np.argsort(col_labels)]

        # First we need to find what is the label of the column cluster at index col_cl (it might not be col_cl + 1)
        colcl_idx_grouped = [list(j) for i, j in groupby(np.sort(col_labels))]
        col_label = colcl_idx_grouped[col_cl][0]
        # Now we can find the indices where the labels are
        col_idx = np.argwhere(np.sort(col_labels) == col_label).flatten()

        if not get_trials:
            return sorted_elecs[col_idx]

    sorted_trials = None
    row_idx = None
    if get_trials:
        if trials_as_labels:
            trials = np.array(["trial" + str(i) for i in range(1, n_rows + 1)])
        else:
            trials = np.array(range(1, n_rows + 1))
        sorted_trials = trials[np.argsort(row_labels)]

        # We need to find what is the label of the row cluster at index row_cl (it might not be row_cl + 1)
        rowcl_idx_grouped = [list(j) for i, j in groupby(np.sort(row_labels))]
        row_label = rowcl_idx_grouped[row_cl][0]
        # Now we can find the indices where the labels are
        row_idx = np.argwhere(np.sort(row_labels) == row_label).flatten()

        if not get_elecs:
            return sorted_trials[row_idx]

    return sorted_elecs[col_idx], sorted_trials[row_idx]


def show_tsne_plot(data, plot_trials, n=20):
    n_features = len(data[0])

    data_tsne = None
    if n_features >= 35:
        data_reduced = TruncatedSVD(n_components=30, random_state=0).fit_transform(data)
        data_tsne = TSNE(perplexity=30).fit_transform(data_reduced)
    else:
        data_tsne = TSNE(perplexity=30).fit_transform(data)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    if plot_trials and len(data) > 2 * n:
        # paint the first 20 trials green, the last 20 trials pink and all other trials blue
        colors = ["green" for _ in range(n)] + ["blue" for _ in range(len(data) - 2 * n)] + ["pink" for _ in range(n)]
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors)
    else:
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.show()


def compare_first_and_last_trials(row_labels, col_labels, n=20):
    n_rows = len(row_labels)
    assert n_rows > 2 * n, "Cannot compare the first %d and last %d trials, when data length is %d!" % (n, n, n_rows)

    n_row_clusters = len(set(row_labels))
    n_col_clusters = len(set(col_labels))

    # first_bicluster_counts[i][j] shows how many of the first n trials belong to bicluster ij
    first_bicluster_counts = np.zeros((n_row_clusters, n_col_clusters))
    # same for the last n trials
    last_bicluster_counts = np.zeros((n_row_clusters, n_col_clusters))
    for i in range(n_row_clusters):
        for j in range(n_col_clusters):
            sorted_trials = get_bicluster_trials_elecs(tw_per_el=None, row_cl=i, col_cl=j, row_labels=row_labels,
                                                       col_labels=col_labels, get_elecs=False, trials_as_labels=False)

            # Count the number of trials in the bicluster where number of trial is <= n
            first_n_count = len(list(filter(lambda t: t <= n, sorted_trials)))
            first_bicluster_counts[i][j] = first_n_count

            # Count the number of trials in the bicluster where number of trial is > n_trials - n
            last_n_count = len(list(filter(lambda t: t > (n_rows - n), sorted_trials)))
            last_bicluster_counts[i][j] = last_n_count

    return first_bicluster_counts, last_bicluster_counts

def pretty_print_arrays(arr1, arr2):
    assert len(arr1) == len(arr2), "The lengths don't match!"

    out = ""
    for i in range(len(arr1)):
        out += "[%s]    [%s]\n" % (" ".join('{:4}'.format(x) for x in arr1[i]),
                                   " ".join('{:4}'.format(x) for x in arr2[i]))

    print(out)

