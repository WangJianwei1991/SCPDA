import os
import sys
import random
import pandas as pd
import numpy as np
from collections import Counter
import math
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.1'
#import rpy2.robjects as robjects 
#from rpy2.robjects import pandas2ri


from matplotlib.legend_handler import HandlerPathCollection
from matplotlib import cm
import matplotlib.collections as mcol
import matplotlib.transforms as mtransforms

from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix

#from mpl_colors import cnames
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

Reduction = "pca"  # FindNeighbors和FindClusters的降维方法
PC1_2_Ratio = None  # 用于记录2个主成分的贡献率

# Draw confidence ellipse
def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
 
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
 
def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
 
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


def purity(labels_true, labels_pred):
    cm = contingency_matrix(labels_true, labels_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)


# Draw the Purity Score version of the clustering diagram from the csv table of clustering results - Part 4 of the article
# Distinguish Clusters
def Replot_Cluster_Result_From_File_Version2(filepath, 
                                    Groups = ['T', 'C'], 
                                    Batches = ['Batch1', 'Batch2', 'Batch3'],
                                    min_samples = 6,
                                    group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
                                                          [51/255, 157/255, 255/255, 1.0]],
                                    group_edge_color = [[139/255, 44/255, 60/255, 1.0],
                                                        [63/255, 109/255, 150/255, 1.0]]):

    
    # 2 groups' fill colors: red, blue
    #group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
    #                      [51/255, 157/255, 255/255, 1.0]]
    # 2 groups' edge colors: dark red, dark blue
    #group_edge_color = [[139/255, 44/255, 60/255, 1.0],
    #                    [63/255, 109/255, 150/255, 1.0]]
    # Batch Marker
    batch_marker = ['o', '^', 's', 'd', '*', 'P', 'v', '<', 'X', '>', 'h', 'H']


    df = pd.read_csv(filepath)
    df_sorted = df.sort_values(by='Cluster Label', ascending=True)
    cluster_list = df_sorted['Cluster Label'].unique().tolist()
    cluster_ellipses_color = []

    sample_index_of_each_group = {}
    Compared_groups_label = [] 

    for group in Groups:
        sample_index_of_each_group[group + '1'] = []
        sample_index_of_each_group[group + '2'] = []


    for cluster in cluster_list:
        df_cluster = df_sorted[df_sorted['Cluster Label'] == cluster]
        group_list = df_cluster['Group'].values.tolist()
        count_elements = Counter(group_list)
        result = max(count_elements.elements(), key=count_elements.get)  # The sample group that accounts for the majority of the cluster

        # The name of the sample that accounts for the majority of the cluster
        run_name_list = (df_cluster[df_cluster['Group'] == result])['Run Name'].values.tolist()
        # If the number of samples is >= 6
        if (len(run_name_list) >= min_samples):
            if (sample_index_of_each_group[result + '1'] == []):
                sample_index_of_each_group[result + '1'] = (df_cluster[df_cluster['Group'] == result]).index.tolist()

                if (sample_index_of_each_group[Groups[0] + '1'] != []) & (sample_index_of_each_group[Groups[1] + '1'] != []):
                    Compared_groups_label.append('{0}/{1}'.format(Groups[0]+'1', Groups[1]+'1'))

            elif (sample_index_of_each_group[result + '2'] == []):
                sample_index_of_each_group[result + '2'] = (df_cluster[df_cluster['Group'] == result]).index.tolist()
                Compared_groups_label.append('{0}/{1}'.format(result + '1', result + '2'))



        ellipses_color = group_edge_color[Groups.index(result)]
        # If the color of the ellipse is not used, add it to the list directly; if it is used, introduce a deviation
        if (ellipses_color in cluster_ellipses_color):

            r = ellipses_color[0]*random.uniform(1.2, 1.6)
            if r > 1:
                r = 1.0
            g = ellipses_color[1]*random.uniform(1.2, 1.6)
            if g > 1:
                g = 1.0
            b = ellipses_color[2]*random.uniform(1.2, 1.6)
            if b > 1:
                b = 1.0
            cluster_ellipses_color.append([r, g, b, 1])
        else:
            cluster_ellipses_color.append(group_edge_color[Groups.index(result)])


    fig = plt.figure(figsize=(5,5))
    axes = plt.gca()

    # Plot Scatter
    for row in range(df_sorted.shape[0]):
        x = None
        y = None
        if (Reduction == 'pca'):
            x = df_sorted['PCA 1'].values.tolist()[row]
            y = df_sorted['PCA 2'].values.tolist()[row]
        if (Reduction == 'umap'):
            x = df_sorted['UMAP 1'].values.tolist()[row]
            y = df_sorted['UMAP 2'].values.tolist()[row]
        group = df_sorted['Group'].values.tolist()[row]
        batch = df_sorted['Batch'].values.tolist()[row]
        cluster = df_sorted['Cluster Label'].values.tolist()[row]


        c = group_fillin_color[Groups.index(group)]
        edgecolors = cluster_ellipses_color[cluster_list.index(cluster)] 
        marker = batch_marker[Batches.index(batch)]
        plt.scatter(x = x, y = y,  marker = marker, color = np.array(c), edgecolors = np.array(edgecolors), linewidths = 1.2)


    # Plot ellipses
    cluster_label = df_sorted['Cluster Label'].values.tolist()
    for i in range(len(cluster_list)):
        index = [k for k,j in enumerate(cluster_label) if j == i]
        ellip = plot_point_cov((df_sorted.iloc[index, 3:5]).values, nstd=2, alpha=0.8, lw=1.5, ls = '--', edgecolor = cluster_ellipses_color[i], facecolor = 'none', label = 'Class {0}'.format(i+1)) 


    # Calculating and labeling metrics
    sample_purity = purity(df_sorted['Expected Label'].values.tolist(), cluster_label)
    normalized_sample_purity = (sample_purity - 1 / len(Groups)) / (1 - 1/len(Groups))
    batches = []
    for batch in (df_sorted['Batch'].values.tolist()):
        batches.append(Batches.index(batch))

    batch_purity = purity(batches, cluster_label)
    normalized_batch_purity = (batch_purity - 1 / len(Batches)) / (1 - 1/len(Batches))

    purity_score = 2 * normalized_sample_purity * (1 - normalized_batch_purity) / (normalized_sample_purity + 1 - normalized_batch_purity)

    plt.title('Purity Score = {:.3f}\nSample Purity = {:.3f}\nBatch Purity = {:.3f}'.format(purity_score, sample_purity, batch_purity), x=0.45, y=0.87, ha='left', va='center', color = 'black', size=16, family="Arial")

    if (Reduction == 'pca'):
        axes.set_xlabel('PC 1', fontsize=16)
        axes.set_ylabel('PC 2', fontsize=16)
        #PC_1_Ratio = "{:.1%}".format(PC1_2_Ratio[0])
        #PC_2_Ratio = "{:.1%}".format(PC1_2_Ratio[1])
        #axes.set_xlabel('PC 1({0})'.format(PC_1_Ratio), fontsize=16)
        #axes.set_ylabel('PC 2({0})'.format(PC_2_Ratio), fontsize=16)
    if (Reduction == 'umap'):
        axes.set_xlabel('UMAP1', fontsize=16)
        axes.set_ylabel('UMAP2', fontsize=16)
    plt.tick_params(axis='x', labelsize=14) 
    plt.tick_params(axis='y', labelsize=14)

    plt.tick_params(axis='x', width=2)
    plt.tick_params(axis='y', width=2)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2) 

    umap1 = None
    umap2 = None
    if (Reduction == 'pca'):
        umap1 = df_sorted['PCA 1'].values
        umap2 = df_sorted['PCA 2'].values
    if (Reduction == 'umap'):
        umap1 = df_sorted['UMAP 1'].values
        umap2 = df_sorted['UMAP 2'].values
    xlim_min = int(umap1[np.argmin(umap1)]) - 2
    xlim_max = int(umap1[np.argmax(umap1)]) + 2
    ylim_min = int(umap2[np.argmin(umap2)]) - 2
    ylim_max = int(umap2[np.argmax(umap2)]) + 2

    abs_xlim = max([abs(xlim_min), abs(xlim_max)])
    abs_ylim = max([abs(ylim_min), abs(ylim_max)])
    if abs_xlim%2 == 0:
        pass
    else:
        abs_xlim += 1

    if abs_ylim%2 == 0:
        pass
    else:
        abs_ylim += 1

    plt.xlim(-abs_xlim*1.1, abs_xlim*1.1)
    plt.ylim(-abs_ylim*1.1, abs_ylim*1.1)

    plt.xticks(np.linspace(-abs_xlim, abs_xlim, 5))
    plt.yticks(np.linspace(-abs_ylim, abs_ylim, 5))

    plt.subplots_adjust(left=0.145, right=0.99, bottom=0.12, top=0.99, wspace=None, hspace=0.2) 

    directory, file_name, file_extension = split_path(filepath)
    svgpath = os.path.join(directory, file_name+'.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 

    plt.show()
    plt.close()


    # Drawing Legends - Batch Legend
    fig_legend = plt.figure(figsize=(2.5,2.5))
    axes = plt.gca()

    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    s_list = []
    paths_batch_list = []
    sizes_batch_list = []
    facecolors_batch_list = []
    edgecolors_batch_list = []

    for i in Batches:
        for j in Groups:
            s_list.append(axes.scatter([100], [100], 
                                       marker = batch_marker[Batches.index(i)], 
                                       s=60, 
                                       color = group_fillin_color[Groups.index(j)], 
                                       edgecolors = group_edge_color[Groups.index(j)], 
                                       linewidths = 1.2))

            paths_batch_list.append(s_list[-1].get_paths()[0])
            sizes_batch_list.append(s_list[-1].get_sizes()[0])
            facecolors_batch_list.append(s_list[-1].get_facecolors()[0])
            edgecolors_batch_list.append(s_list[-1].get_edgecolors()[0])

    paths_batch_list = chunk_list(paths_batch_list, len(Groups))
    sizes_batch_list = chunk_list(sizes_batch_list, len(Groups))
    facecolors_batch_list = chunk_list(facecolors_batch_list, len(Groups))
    edgecolors_batch_list = chunk_list(edgecolors_batch_list, len(Groups))


    PC_batch_list = []
    for i in range(len(Batches)):
        PC_batch = mcol.PathCollection(paths_batch_list[i], sizes_batch_list[i], transOffset = axes.transData, facecolors = facecolors_batch_list[i], edgecolors = edgecolors_batch_list[i], linewidths = 1.2) 
        PC_batch_list.append(PC_batch)
        PC_batch.set_transform(mtransforms.IdentityTransform())


    paths_group_list = []
    sizes_group_list = []
    facecolors_group_list = []
    edgecolors_group_list = []

    for i in range(len(Groups)):
        for j in range(len(Batches)):

            paths_group_list.append(s_list[j*len(Groups)+i].get_paths()[0])
            sizes_group_list.append(s_list[j*len(Groups)+i].get_sizes()[0])
            facecolors_group_list.append(s_list[j*len(Groups)+i].get_facecolors()[0])
            edgecolors_group_list.append(s_list[j*len(Groups)+i].get_edgecolors()[0])

    paths_group_list = chunk_list(paths_group_list, len(Batches))
    sizes_group_list = chunk_list(sizes_group_list, len(Batches))
    facecolors_group_list = chunk_list(facecolors_group_list, len(Batches))
    edgecolors_group_list = chunk_list(edgecolors_group_list, len(Batches))


    PC_group_list = []
    for i in range(len(Groups)):
        PC_group = mcol.PathCollection(paths_group_list[i], sizes_group_list[i], transOffset = axes.transData, facecolors = facecolors_group_list[i], edgecolors = edgecolors_group_list[i], linewidths = 1.2) 
        PC_group_list.append(PC_group)
        PC_group.set_transform(mtransforms.IdentityTransform())


    l1 = axes.legend(PC_batch_list, Batches, handler_map={type(PC_batch_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_batch_list[0]), scatteryoffsets = [.5], handlelength = len(paths_batch_list[0]), fontsize=16, title = "Batch", loc = 'center')
    l1.get_title().set_fontsize(fontsize = 18)
    plt.gca().add_artist(l1)

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    svgpath = os.path.join(directory, 'Legend_Clustering_Batch.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 
    plt.show()
    plt.close()


    # Drawing Legends - Group Legend
    fig_legend = plt.figure(figsize=(2.5,2.5))
    axes = plt.gca()
       
    l2 = axes.legend(PC_group_list, Groups, handler_map={type(PC_group_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_group_list[0]), scatteryoffsets = [.5], handlelength = len(paths_group_list[0]), fontsize=16, title = "Group ", loc = 'center')
    l2.get_title().set_fontsize(fontsize = 18)
    plt.gca().add_artist(l2)

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    svgpath = os.path.join(directory, 'Legend_Clustering_Group.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True)
    plt.show()
    plt.close()


    return purity_score, sample_purity, batch_purity, sample_index_of_each_group, Compared_groups_label


# Draw the Purity Score version of the clustering diagram from the csv table of clustering results - Part 3 of the article
def Replot_Cluster_Result_From_File(filepath, 
                                    Groups = ['C2Y1E', 'C1Y2E'], 
                                    Batches = ['Batch1', 'Batch2', 'Batch3'],
                                    group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
                                                          [51/255, 157/255, 255/255, 1.0]],
                                    group_edge_color = [[139/255, 44/255, 60/255, 1.0],
                                                        [63/255, 109/255, 150/255, 1.0]],
                                    savefig = True):

    
    # 2 groups' fill colors: red, blue
    #group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
    #                      [51/255, 157/255, 255/255, 1.0]]
    # 2 groups' edge colors: dark red, dark blue
    #group_edge_color = [[139/255, 44/255, 60/255, 1.0],
    #                    [63/255, 109/255, 150/255, 1.0]]
    # Batch Marker
    batch_marker = ['o', '^', 's', 'd', '*', 'P', 'v', '<', 'X', '>', 'h', 'H']


    df = pd.read_csv(filepath)

    # If you specify not to generate images and CSV files, delete the clustering CSV file in the original path.
    if (savefig == False):
        os.remove(filepath)


    df_sorted = df.sort_values(by='Cluster Label', ascending=True)
    cluster_list = df_sorted['Cluster Label'].unique().tolist()
    cluster_ellipses_color = []
    for cluster in cluster_list:
        df_cluster = df_sorted[df_sorted['Cluster Label'] == cluster]
        group_list = df_cluster['Group'].values.tolist()
        count_elements = Counter(group_list)
        result = max(count_elements.elements(), key=count_elements.get)
        ellipses_color = group_edge_color[Groups.index(result)]
        if (ellipses_color in cluster_ellipses_color):
            r = ellipses_color[0]*random.uniform(1.2, 1.6)
            if r > 1:
                r = 1.0
            g = ellipses_color[1]*random.uniform(1.2, 1.6)
            if g > 1:
                g = 1.0
            b = ellipses_color[2]*random.uniform(1.2, 1.6)
            if b > 1:
                b = 1.0
            cluster_ellipses_color.append([r, g, b, 1])
        else:
            cluster_ellipses_color.append(group_edge_color[Groups.index(result)])


    # Before starting to draw, calculate the data to be output
    cluster_label = df_sorted['Cluster Label'].values.tolist()
    # Calculating and labeling metrics
    sample_purity = purity(df_sorted['Expected Label'].values.tolist(), cluster_label)
    normalized_sample_purity = (sample_purity - 1 / len(Groups)) / (1 - 1/len(Groups))
    batches = []
    for batch in (df_sorted['Batch'].values.tolist()):
        batches.append(Batches.index(batch))

    batch_purity = purity(batches, cluster_label)
    normalized_batch_purity = (batch_purity - 1 / len(Batches)) / (1 - 1/len(Batches))

    purity_score = 2 * normalized_sample_purity * (1 - normalized_batch_purity) / (normalized_sample_purity + 1 - normalized_batch_purity)



    if savefig:
        fig = plt.figure(figsize=(5,5))
        axes = plt.gca()

        # Scatter plot
        for row in range(df_sorted.shape[0]):
            x = None
            y = None
            if (Reduction == 'pca'):
                x = df_sorted['PCA 1'].values.tolist()[row]
                y = df_sorted['PCA 2'].values.tolist()[row]
            if (Reduction == 'umap'):
                x = df_sorted['UMAP 1'].values.tolist()[row]
                y = df_sorted['UMAP 2'].values.tolist()[row]
            group = df_sorted['Group'].values.tolist()[row]
            batch = df_sorted['Batch'].values.tolist()[row]
            cluster = df_sorted['Cluster Label'].values.tolist()[row]


            c = group_fillin_color[Groups.index(group)]
            edgecolors = cluster_ellipses_color[cluster_list.index(cluster)] 
            marker = batch_marker[Batches.index(batch)]
            plt.scatter(x = x, y = y,  marker = marker, color = np.array(c), edgecolors = np.array(edgecolors), linewidths = 1.2)


        # Draw ellipse
        #cluster_label = df_sorted['Cluster Label'].values.tolist()
        for i in range(len(cluster_list)):
            index = [k for k,j in enumerate(cluster_label) if j == i]
            ellip = plot_point_cov((df_sorted.iloc[index, 3:5]).values, nstd=2, alpha=0.8, lw=1.5, ls = '--', edgecolor = cluster_ellipses_color[i], facecolor = 'none', label = 'Class {0}'.format(i+1)) 


        ## Calculating and labeling metrics
        #sample_purity = purity(df_sorted['Expected Label'].values.tolist(), cluster_label)
        #normalized_sample_purity = (sample_purity - 1 / len(Groups)) / (1 - 1/len(Groups))
        #batches = []
        #for batch in (df_sorted['Batch'].values.tolist()):
        #    batches.append(Batches.index(batch))

        #batch_purity = purity(batches, cluster_label)
        #normalized_batch_purity = (batch_purity - 1 / len(Batches)) / (1 - 1/len(Batches))

        #purity_score = 2 * normalized_sample_purity * (1 - normalized_batch_purity) / (normalized_sample_purity + 1 - normalized_batch_purity)

        plt.title('Purity Score = {:.3f}\nSample Purity = {:.3f}\nBatch Purity = {:.3f}'.format(purity_score, sample_purity, batch_purity), x=0.45, y=0.87, ha='left', va='center', color = 'black', size=16, family="Arial")

        if (Reduction == 'pca'):
            axes.set_xlabel('PC 1', fontsize=16)
            axes.set_ylabel('PC 2', fontsize=16)
            #PC_1_Ratio = "{:.1%}".format(PC1_2_Ratio[0])
            #PC_2_Ratio = "{:.1%}".format(PC1_2_Ratio[1])
            #axes.set_xlabel('PC 1({0})'.format(PC_1_Ratio), fontsize=16)
            #axes.set_ylabel('PC 2({0})'.format(PC_2_Ratio), fontsize=16)
        if (Reduction == 'umap'):
            axes.set_xlabel('UMAP1', fontsize=16)
            axes.set_ylabel('UMAP2', fontsize=16)

        plt.tick_params(axis='x', labelsize=14) 
        plt.tick_params(axis='y', labelsize=14)

        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)

        umap1 = None
        umap2 = None
        if (Reduction == 'pca'):
            umap1 = df_sorted['PCA 1'].values
            umap2 = df_sorted['PCA 2'].values
        if (Reduction == 'umap'):
            umap1 = df_sorted['UMAP 1'].values
            umap2 = df_sorted['UMAP 2'].values

        xlim_min = int(umap1[np.argmin(umap1)]) - 2
        xlim_max = int(umap1[np.argmax(umap1)]) + 2
        ylim_min = int(umap2[np.argmin(umap2)]) - 2
        ylim_max = int(umap2[np.argmax(umap2)]) + 2

        abs_xlim = max([abs(xlim_min), abs(xlim_max)])
        abs_ylim = max([abs(ylim_min), abs(ylim_max)])
        if abs_xlim%2 == 0:
            pass
        else:
            abs_xlim += 1

        if abs_ylim%2 == 0:
            pass
        else:
            abs_ylim += 1

        plt.xlim(-abs_xlim*1.1, abs_xlim*1.1)
        plt.ylim(-abs_ylim*1.1, abs_ylim*1.1)

        plt.xticks(np.linspace(-abs_xlim, abs_xlim, 5))
        plt.yticks(np.linspace(-abs_ylim, abs_ylim, 5))

        plt.subplots_adjust(left=0.145, right=0.99, bottom=0.12, top=0.99, wspace=None, hspace=0.2) 

        directory, file_name, file_extension = split_path(filepath)
        svgpath = os.path.join(directory, file_name+'.svg')
        plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 

        plt.show()
        plt.close()


        # Drawing Legends - Batch Legend
        fig_legend = plt.figure(figsize=(2.5,2.5))
        axes = plt.gca()

        def chunk_list(lst, chunk_size):
            return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

        s_list = []
        paths_batch_list = []
        sizes_batch_list = []
        facecolors_batch_list = []
        edgecolors_batch_list = []

        for i in Batches:
            for j in Groups:
                s_list.append(axes.scatter([100], [100], 
                                           marker = batch_marker[Batches.index(i)], 
                                           s=60, 
                                           color = group_fillin_color[Groups.index(j)], 
                                           edgecolors = group_edge_color[Groups.index(j)], 
                                           linewidths = 1.2))

                paths_batch_list.append(s_list[-1].get_paths()[0])
                sizes_batch_list.append(s_list[-1].get_sizes()[0])
                facecolors_batch_list.append(s_list[-1].get_facecolors()[0])
                edgecolors_batch_list.append(s_list[-1].get_edgecolors()[0])

        paths_batch_list = chunk_list(paths_batch_list, len(Groups))
        sizes_batch_list = chunk_list(sizes_batch_list, len(Groups))
        facecolors_batch_list = chunk_list(facecolors_batch_list, len(Groups))
        edgecolors_batch_list = chunk_list(edgecolors_batch_list, len(Groups))


        PC_batch_list = []
        for i in range(len(Batches)):
            PC_batch = mcol.PathCollection(paths_batch_list[i], sizes_batch_list[i], transOffset = axes.transData, facecolors = facecolors_batch_list[i], edgecolors = edgecolors_batch_list[i], linewidths = 1.2) 
            PC_batch_list.append(PC_batch)
            PC_batch.set_transform(mtransforms.IdentityTransform())


        paths_group_list = []
        sizes_group_list = []
        facecolors_group_list = []
        edgecolors_group_list = []

        for i in range(len(Groups)):
            for j in range(len(Batches)):

                paths_group_list.append(s_list[j*len(Groups)+i].get_paths()[0])
                sizes_group_list.append(s_list[j*len(Groups)+i].get_sizes()[0])
                facecolors_group_list.append(s_list[j*len(Groups)+i].get_facecolors()[0])
                edgecolors_group_list.append(s_list[j*len(Groups)+i].get_edgecolors()[0])

        paths_group_list = chunk_list(paths_group_list, len(Batches))
        sizes_group_list = chunk_list(sizes_group_list, len(Batches))
        facecolors_group_list = chunk_list(facecolors_group_list, len(Batches))
        edgecolors_group_list = chunk_list(edgecolors_group_list, len(Batches))


        PC_group_list = []
        for i in range(len(Groups)):
            PC_group = mcol.PathCollection(paths_group_list[i], sizes_group_list[i], transOffset = axes.transData, facecolors = facecolors_group_list[i], edgecolors = edgecolors_group_list[i], linewidths = 1.2) 
            PC_group_list.append(PC_group)
            PC_group.set_transform(mtransforms.IdentityTransform())


        l1 = axes.legend(PC_batch_list, Batches, handler_map={type(PC_batch_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_batch_list[0]), scatteryoffsets = [.5], handlelength = len(paths_batch_list[0]), fontsize=16, title = "Batch", loc = 'center')
        l1.get_title().set_fontsize(fontsize = 18)
        plt.gca().add_artist(l1)

        plt.ylim(-5, 5)
        plt.xlim(-5, 5)

        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])

        svgpath = os.path.join(directory, 'Legend_Clustering_Batch.svg')
        plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()


        # Draw Group Legend
        fig_legend = plt.figure(figsize=(2.5,2.5))
        axes = plt.gca()
       
        l2 = axes.legend(PC_group_list, Groups, handler_map={type(PC_group_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_group_list[0]), scatteryoffsets = [.5], handlelength = len(paths_group_list[0]), fontsize=16, title = "Group ", loc = 'center')
        l2.get_title().set_fontsize(fontsize = 18)
        plt.gca().add_artist(l2)

        plt.ylim(-5, 5)
        plt.xlim(-5, 5)

        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])

        svgpath = os.path.join(directory, 'Legend_Clustering_Group.svg')
        plt.savefig(svgpath, dpi=600, format="svg", transparent=True)
        plt.show()
        plt.close()


    return purity_score, sample_purity, batch_purity



# Draw the ARI version of the cluster diagram - Part 2 of the article
# Plot clustering results
def Plot_ARI_Version_Cluster_Diagram(df_data, ari, 
                                       Groups = ['Hela-Yeast-Ecoli_50-10-40', 
                                                'Hela-Yeast-Ecoli_50-20-30',
                                                'Hela-Yeast-Ecoli_50-25-25', 
                                                'Hela-Yeast-Ecoli_50-30-20',
                                                'Hela-Yeast-Ecoli_50-40-10'],
                                       Batches = ['20230712','20230731', '20230904'],
                                       savefig = True, savefolder = './', savename = 'Cluster_Analysis'):

    np.random.seed(42)

    color_5 = [[244/255,172/255,140/255, 1.0],
                    [251/255,226/255,157/255, 1.0],
                    [211/255,159/255,235/255, 1.0],
                    [103/255,191/255,235/255, 1.0],
                    [144/255,216/255,144/255, 1.0]]

    edge_color_5 = [[233/255,91/255,27/255, 1.0],
                    [188/255,143/255,0/255, 1.0],
                    [161/255,78/255,224/255, 1.0],
                    [24/255,131/255,184/255, 1.0],
                    [57/255,161/255,57/255, 1.0]]

    # Fill Color
    basic_colors = np.vstack((color_5, np.random.rand(10, 4) ))
    basic_colors[:,3] = 1.0
    # Edge Color
    edge_colors = np.vstack((edge_color_5, np.random.rand(10, 4) ))
    edge_colors[:,3] = 1.0

    group_fillin_color = basic_colors.tolist()
    group_edge_color = edge_colors.tolist()
    batch_marker = ['o', '^', 's', 'd', '*', 'P', 'v', '<', 'X', '>', 'h', 'H']


    # Sort by clustering results in ascending order
    df_sorted = df_data.sort_values(by='Cluster Label', ascending=True)
    cluster_list = df_sorted['Cluster Label'].unique().tolist()

    cluster_ellipses_color = []
    for cluster in cluster_list:
        df_cluster = df_sorted[df_sorted['Cluster Label'] == cluster]
        group_list = df_cluster['Group'].values.tolist()
        count_elements = Counter(group_list)
        result = max(count_elements.elements(), key=count_elements.get)
        ellipses_color = group_edge_color[Groups.index(result)]

        if (ellipses_color in cluster_ellipses_color):
            r = ellipses_color[0]*random.uniform(1.2, 1.6)
            if r > 1:
                r = 1.0
            g = ellipses_color[1]*random.uniform(1.2, 1.6)
            if g > 1:
                g = 1.0
            b = ellipses_color[2]*random.uniform(1.2, 1.6)
            if b > 1:
                b = 1.0
            cluster_ellipses_color.append([r, g, b, 1])
        else:
            cluster_ellipses_color.append(group_edge_color[Groups.index(result)])


    fig = plt.figure(figsize=(5,5))
    axes = plt.gca()

    # Plotting scatter points
    for row in range(df_sorted.shape[0]):
        x = None
        y = None
        if (Reduction == 'pca'):
            x = df_sorted['PCA 1'].values.tolist()[row]
            y = df_sorted['PCA 2'].values.tolist()[row]
        if (Reduction == 'umap'):
            x = df_sorted['UMAP 1'].values.tolist()[row]
            y = df_sorted['UMAP 2'].values.tolist()[row]
        group = df_sorted['Group'].values.tolist()[row]
        batch = df_sorted['Batch'].values.tolist()[row]
        cluster = df_sorted['Cluster Label'].values.tolist()[row]


        c = group_fillin_color[Groups.index(group)]
        edgecolors = cluster_ellipses_color[cluster_list.index(cluster)] 
        marker = batch_marker[Batches.index(batch)]
        plt.scatter(x = x, y = y,  marker = marker, color = np.array(c), edgecolors = np.array(edgecolors), linewidths = 1.2)


    # Draw ellipse
    cluster_label = df_sorted['Cluster Label'].values.tolist()
    for i in range(len(cluster_list)):
        index = [k for k,j in enumerate(cluster_label) if j == i]
        ellip = plot_point_cov((df_sorted.iloc[index, 3:5]).values, nstd=2, alpha=0.8, lw=1.5, ls = '--', edgecolor = cluster_ellipses_color[i], facecolor = 'none', label = 'Class {0}'.format(i+1)) 

    # Mark ARI in the upper right corner
    plt.title('ARI = {:.3f}'.format(ari), x=0.8, y=0.9, ha='center', va='center', color = 'black', size=16, family="Arial")

    if (Reduction == 'pca'):
        axes.set_xlabel('PC 1', fontsize=16)
        axes.set_ylabel('PC 2', fontsize=16)
        #PC_1_Ratio = "{:.1%}".format(PC1_2_Ratio[0])
        #PC_2_Ratio = "{:.1%}".format(PC1_2_Ratio[1])
        #axes.set_xlabel('PC 1({0})'.format(PC_1_Ratio), fontsize=16)
        #axes.set_ylabel('PC 2({0})'.format(PC_2_Ratio), fontsize=16)
    if (Reduction == 'umap'):
        axes.set_xlabel('UMAP1', fontsize=16)
        axes.set_ylabel('UMAP2', fontsize=16)

    plt.tick_params(axis='x', labelsize=14) 
    plt.tick_params(axis='y', labelsize=14)

    plt.tick_params(axis='x', width=2)
    plt.tick_params(axis='y', width=2)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2) 

    umap1 = None
    umap2 = None
    if (Reduction == 'pca'):
        umap1 = df_sorted['PCA 1'].values
        umap2 = df_sorted['PCA 2'].values
    if (Reduction == 'umap'):
        umap1 = df_sorted['UMAP 1'].values
        umap2 = df_sorted['UMAP 2'].values
    xlim_min = int(umap1[np.argmin(umap1)]) - 2
    xlim_max = int(umap1[np.argmax(umap1)]) + 2
    ylim_min = int(umap2[np.argmin(umap2)]) - 2
    ylim_max = int(umap2[np.argmax(umap2)]) + 2

    abs_xlim = max([abs(xlim_min), abs(xlim_max)])
    abs_ylim = max([abs(ylim_min), abs(ylim_max)])
    if abs_xlim%2 == 0:
        pass
    else:
        abs_xlim += 1

    if abs_ylim%2 == 0:
        pass
    else:
        abs_ylim += 1

    plt.xlim(-abs_xlim*1.1, abs_xlim*1.1)
    plt.ylim(-abs_ylim*1.1, abs_ylim*1.1)

    plt.xticks(np.linspace(-abs_xlim, abs_xlim, 5))
    plt.yticks(np.linspace(-abs_ylim, abs_ylim, 5))

    plt.subplots_adjust(left=0.145, right=0.99, bottom=0.12, top=0.99, wspace=None, hspace=0.2) 

    filepath = "{0}{1}.svg".format(savefolder, savename)
    directory, file_name, file_extension = split_path(filepath)
    svgpath = os.path.join(directory, file_name+'.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 

    plt.show()
    plt.close()


    # Drawing Legends - Batch Legends
    fig_legend = plt.figure(figsize=(2.5,2.5))
    axes = plt.gca()

    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    s_list = []
    paths_batch_list = []
    sizes_batch_list = []
    facecolors_batch_list = []
    edgecolors_batch_list = []

    for i in Batches:
        for j in Groups:
            s_list.append(axes.scatter([100], [100], 
                                       marker = batch_marker[Batches.index(i)], 
                                       s=60, 
                                       color = group_fillin_color[Groups.index(j)], 
                                       edgecolors = group_edge_color[Groups.index(j)], 
                                       linewidths = 1.2))

            paths_batch_list.append(s_list[-1].get_paths()[0])
            sizes_batch_list.append(s_list[-1].get_sizes()[0])
            facecolors_batch_list.append(s_list[-1].get_facecolors()[0])
            edgecolors_batch_list.append(s_list[-1].get_edgecolors()[0])

    paths_batch_list = chunk_list(paths_batch_list, len(Groups))
    sizes_batch_list = chunk_list(sizes_batch_list, len(Groups))
    facecolors_batch_list = chunk_list(facecolors_batch_list, len(Groups))
    edgecolors_batch_list = chunk_list(edgecolors_batch_list, len(Groups))


    PC_batch_list = []
    for i in range(len(Batches)):
        PC_batch = mcol.PathCollection(paths_batch_list[i], sizes_batch_list[i], transOffset = axes.transData, facecolors = facecolors_batch_list[i], edgecolors = edgecolors_batch_list[i], linewidths = 1.2) 
        PC_batch_list.append(PC_batch)
        PC_batch.set_transform(mtransforms.IdentityTransform())


    paths_group_list = []
    sizes_group_list = []
    facecolors_group_list = []
    edgecolors_group_list = []

    for i in range(len(Groups)):
        for j in range(len(Batches)):

            paths_group_list.append(s_list[j*len(Groups)+i].get_paths()[0])
            sizes_group_list.append(s_list[j*len(Groups)+i].get_sizes()[0])
            facecolors_group_list.append(s_list[j*len(Groups)+i].get_facecolors()[0])
            edgecolors_group_list.append(s_list[j*len(Groups)+i].get_edgecolors()[0])

    paths_group_list = chunk_list(paths_group_list, len(Batches))
    sizes_group_list = chunk_list(sizes_group_list, len(Batches))
    facecolors_group_list = chunk_list(facecolors_group_list, len(Batches))
    edgecolors_group_list = chunk_list(edgecolors_group_list, len(Batches))


    PC_group_list = []
    for i in range(len(Groups)):
        PC_group = mcol.PathCollection(paths_group_list[i], sizes_group_list[i], transOffset = axes.transData, facecolors = facecolors_group_list[i], edgecolors = edgecolors_group_list[i], linewidths = 1.2) 
        PC_group_list.append(PC_group)
        PC_group.set_transform(mtransforms.IdentityTransform())


    l1 = axes.legend(PC_batch_list, Batches, handler_map={type(PC_batch_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_batch_list[0]), scatteryoffsets = [.5], handlelength = len(paths_batch_list[0]), fontsize=16, title = "Batch", loc = 'center')
    l1.get_title().set_fontsize(fontsize = 18)
    plt.gca().add_artist(l1)

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])


    svgpath = os.path.join(directory, 'Legend_Clustering_Batch.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True) 
    plt.show()
    plt.close()


    # Draw Group Legend
    fig_legend = plt.figure(figsize=(2.5,2.5))
    axes = plt.gca()
       
    l2 = axes.legend(PC_group_list, Groups, handler_map={type(PC_group_list[0]) : HandlerMultiPathCollection()}, scatterpoints=len(paths_group_list[0]), scatteryoffsets = [.5], handlelength = len(paths_group_list[0]), fontsize=16, title = "Group ", loc = 'center')
    l2.get_title().set_fontsize(fontsize = 18)
    plt.gca().add_artist(l2)

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])


    svgpath = os.path.join(directory, 'Legend_Clustering_Group.svg')
    plt.savefig(svgpath, dpi=600, format="svg", transparent=True)
    plt.show()
    plt.close()





class HandlerMultiPathCollection(HandlerPathCollection):
    """
    Handler for PathCollections, which are used by scatter
    """
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)(orig_handle.get_paths(), sizes=sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p



def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta


def split_path(file_path):
    directory, file_name_with_extension = os.path.split(file_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    return directory, file_name, file_extension



# Calculate P adjust
def correct_pvalues_for_multiple_testing(pvalues, correction_type = "Benjamini-Hochberg"):                
    """                                                                                                   
    consistent with R - print correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05, 0.069, 0.07, 0.071, 0.09, 0.1]) 
    """
    from numpy import array, empty                                                                        
    pvalues = array(pvalues) 
    n = int(pvalues.shape[0])                                                                           
    new_pvalues = empty(n)
    if correction_type == "Bonferroni":                                                                   
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":                                                            
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        for rank, vals in enumerate(values):                                                              
            pvalue, i = vals
            new_pvalues[i] = (n-rank) * pvalue                                                            
    elif correction_type == "Benjamini-Hochberg":                                                         
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        values.reverse()                                                                                  
        new_values = []
        for i, vals in enumerate(values):                                                                 
            rank = n - i
            pvalue, index = vals                                                                          
            new_values.append((n/rank) * pvalue)                                                          
        for i in range(0, int(n)-1):  
            if new_values[i] < new_values[i+1]:                                                           
                new_values[i+1] = new_values[i]                                                           
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]                                                                                                                  
    return new_pvalues


# The optimal critical value of the ROC curve
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    #print(optimal_threshold)
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


# Get the index of the nearest value in an array
def find_nearest(array, value):
    '''
    Get the index of the nearest value in an array
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Classification problem evaluation index
def evaluation(y_test, y_predict):
    accuracy=classification_report(y_test, y_predict,output_dict=True)['accuracy']
    s=classification_report(y_test, y_predict,output_dict=True)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    #kappa=cohen_kappa_score(y_test, y_predict)
    return accuracy,precision,recall,f1_score #, kappa



# CSV table conversion
def MethodSelectionTableConversion(csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Example/MethodSelection_DifferentialExpressionAnalysis.csv",
                                   SavePath = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Example/',
                                   Compared_groups_label = ['S4/S2', 'S5/S1']):


    # Read
    df_total = pd.read_csv(csv_path)
    # Sort in ascending order by No
    df_new = df_total.sort_values('No', ascending=True)

    # Filter out 4 df by p-value
    df_p1 = df_new[df_new[Compared_groups_label[0] + ' p-value'] == 0.001]
    df_p2 = df_new[df_new[Compared_groups_label[0] + ' p-value'] == 0.01]
    df_p3 = df_new[df_new[Compared_groups_label[0] + ' p-value'] == 0.05]
    df_p4 = df_new[df_new[Compared_groups_label[0] + ' p-value'] == 0.1]

    df_list = [df_p1, df_p2, df_p3, df_p4]

    # Loop through and sort the comparison groups for each df
    count = 0
    for sorted_df in df_list:
        for Compared_groups in Compared_groups_label:

            list_ari = sorted_df['ARI'].values.tolist()
            list_pauc = sorted_df[Compared_groups + ' pAUC'].values.tolist()
            list_f1_score = sorted_df[Compared_groups + ' F1-Score'].values.tolist()

            sorted_list_ari = sorted(list_ari)
            sorted_list_pauc = sorted(list_pauc)
            sorted_list_f1_score = sorted(list_f1_score)

            list_compared_group_rank_value = []
            for row in range(len(list_ari)):
                ARI_rank = sorted_list_ari.index(list_ari[row])
                pAUC_rank = sorted_list_pauc.index(list_pauc[row])
                F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

                list_compared_group_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


            Sorted_group_rank_value = sorted(list_compared_group_rank_value, reverse=True) 

            # Calculate the Rank of each comparison group
            Group_rank_list = []
            for value in list_compared_group_rank_value:
                index = Sorted_group_rank_value.index(value)
                Group_rank_list.append(index + 1)

            sorted_df[Compared_groups + ' Rank'] = Group_rank_list


        # Average Rank
        list_average_rank_value = []  

        list_ari = sorted_df['ARI'].values.tolist()
        list_pauc = sorted_df['Average pAUC'].values.tolist()
        list_f1_score = sorted_df['Average F1-Score'].values.tolist()

        sorted_list_ari = sorted(list_ari)
        sorted_list_pauc = sorted(list_pauc)
        sorted_list_f1_score = sorted(list_f1_score)

        for row in range(len(list_ari)):

            ARI_rank = sorted_list_ari.index(list_ari[row])
            pAUC_rank = sorted_list_pauc.index(list_pauc[row])
            F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

            list_average_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


        Sorted_rank_value = sorted(list_average_rank_value, reverse=True) 
        Rank_list = []

        for row in range(len(list_ari)):
            Rank_list.append(Sorted_rank_value.index(list_average_rank_value[row]) + 1)


        # Insert p-value column
        sorted_df.insert(loc=6, column='p-value', value=sorted_df[Compared_groups + ' p-value'].values.tolist())
        # Delete the p-value column and log2FC column of the comparison groups
        for Compared_groups in Compared_groups_label:
            sorted_df = sorted_df.drop(Compared_groups + ' p-value', axis=1)
            sorted_df = sorted_df.drop(Compared_groups + ' log2FC', axis=1)


        # Insert the Average Rank Column
        sorted_df.insert(loc=8, column='Average Rank', value=Rank_list)

        # Rename Rank to Total Rank
        sorted_df = sorted_df.rename(columns={'Rank': 'Total Rank'})


    
        if (count == 0):
            df_p1 = sorted_df
            #df_p1.to_csv(SavePath + 'df_p1.csv', index=False)
        elif (count == 1):
            df_p2 = sorted_df
            #df_p2.to_csv(SavePath + 'df_p2.csv', index=False)
        elif (count == 2):
            df_p3 = sorted_df
            #df_p3.to_csv(SavePath + 'df_p3.csv', index=False)
        elif (count == 3):
            df_p4 = sorted_df
            #df_p4.to_csv(SavePath + 'df_p4.csv', index=False)

        count += 1


    # Total Rank
    df_result = pd.concat([df_p1, df_p2, df_p3, df_p4])

    list_total_rank_value = []  

    list_ari = df_result['ARI'].values.tolist()
    list_pauc = df_result['Average pAUC'].values.tolist()
    list_f1_score = df_result['Average F1-Score'].values.tolist()

    sorted_list_ari = sorted(list_ari)
    sorted_list_pauc = sorted(list_pauc)
    sorted_list_f1_score = sorted(list_f1_score)

    for row in range(len(list_ari)):

        ARI_rank = sorted_list_ari.index(list_ari[row])
        pAUC_rank = sorted_list_pauc.index(list_pauc[row])
        F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

        list_total_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


    Sorted_rank_value = sorted(list_total_rank_value, reverse=True) 
    Total_Rank_list = []

    for row in range(len(list_ari)):
        Total_Rank_list.append(Sorted_rank_value.index(list_total_rank_value[row]) + 1)

    df_result['Total Rank'] = Total_Rank_list

    # Sort by Total Rank in ascending order
    df_result = df_result.sort_values('Total Rank', ascending=True)


    # Save csv
    df_result.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis-New.csv', index=False)



# When the user manually merges two tables (such as merging the KeepNA table with the UseCov table) and wants to re-rank it 
def RerankMethodSelectionTable(csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Example/MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv",
                               SavePath = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Example/',
                               Compared_groups_label = ['S4/S2', 'S5/S1']):


    # Read
    df_total = pd.read_csv(csv_path)
    # Sort in ascending order by No
    df_new = df_total.sort_values('No', ascending=True)

    # Filter out 4 df by p-value
    df_p1 = df_new[df_new['p-value'] == 0.001]
    df_p2 = df_new[df_new['p-value'] == 0.01]
    df_p3 = df_new[df_new['p-value'] == 0.05]
    df_p4 = df_new[df_new['p-value'] == 0.1]

    df_list = [df_p1, df_p2, df_p3, df_p4]

    # Loop through and sort the comparison groups for each df
    count = 0
    for sorted_df in df_list:
        for Compared_groups in Compared_groups_label:

            list_ari = sorted_df['ARI'].values.tolist()
            list_pauc = sorted_df[Compared_groups + ' pAUC'].values.tolist()
            list_f1_score = sorted_df[Compared_groups + ' F1-Score'].values.tolist()

            sorted_list_ari = sorted(list_ari)
            sorted_list_pauc = sorted(list_pauc)
            sorted_list_f1_score = sorted(list_f1_score)

            list_compared_group_rank_value = []
            for row in range(len(list_ari)):
                ARI_rank = sorted_list_ari.index(list_ari[row])
                pAUC_rank = sorted_list_pauc.index(list_pauc[row])
                F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

                list_compared_group_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


            Sorted_group_rank_value = sorted(list_compared_group_rank_value, reverse=True) 

            # Calculate the Rank of each comparison group
            Group_rank_list = []
            for value in list_compared_group_rank_value:
                index = Sorted_group_rank_value.index(value)
                Group_rank_list.append(index + 1)

            sorted_df[Compared_groups + ' Rank'] = Group_rank_list


        # Average Rank
        list_average_rank_value = []  

        list_ari = sorted_df['ARI'].values.tolist()
        list_pauc = sorted_df['Average pAUC'].values.tolist()
        list_f1_score = sorted_df['Average F1-Score'].values.tolist()

        sorted_list_ari = sorted(list_ari)
        sorted_list_pauc = sorted(list_pauc)
        sorted_list_f1_score = sorted(list_f1_score)

        for row in range(len(list_ari)):

            ARI_rank = sorted_list_ari.index(list_ari[row])
            pAUC_rank = sorted_list_pauc.index(list_pauc[row])
            F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

            list_average_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


        Sorted_rank_value = sorted(list_average_rank_value, reverse=True)
        Rank_list = []

        for row in range(len(list_ari)):
            Rank_list.append(Sorted_rank_value.index(list_average_rank_value[row]) + 1)


        sorted_df['Average Rank'] = Rank_list

    
        if (count == 0):
            df_p1 = sorted_df
        elif (count == 1):
            df_p2 = sorted_df
        elif (count == 2):
            df_p3 = sorted_df
        elif (count == 3):
            df_p4 = sorted_df

        count += 1


    df_result = pd.concat([df_p1, df_p2, df_p3, df_p4])

    list_total_rank_value = []  

    list_ari = df_result['ARI'].values.tolist()
    list_pauc = df_result['Average pAUC'].values.tolist()
    list_f1_score = df_result['Average F1-Score'].values.tolist()

    sorted_list_ari = sorted(list_ari)
    sorted_list_pauc = sorted(list_pauc)
    sorted_list_f1_score = sorted(list_f1_score)

    for row in range(len(list_ari)):

        ARI_rank = sorted_list_ari.index(list_ari[row])
        pAUC_rank = sorted_list_pauc.index(list_pauc[row])
        F1Score_rank = sorted_list_f1_score.index(list_f1_score[row])

        list_total_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)


    Sorted_rank_value = sorted(list_total_rank_value, reverse=True) 
    Total_Rank_list = []

    for row in range(len(list_ari)):
        Total_Rank_list.append(Sorted_rank_value.index(list_total_rank_value[row]) + 1)

    df_result['Total Rank'] = Total_Rank_list

    # Sort by Total Rank in ascending order
    df_result = df_result.sort_values('Total Rank', ascending=True)


    # Save csv
    df_result.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis_Reranked.csv', index=False)





# Draw hierarchical clustering diagram
def Hierarchical_Clustering(TopMethodsFilePath = "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\第四部分\DIANN\MethodSelection_DIANN_MCF7.csv", 
                            PathWay = 'GO', SR = 'NoSR', Comparison = 'T/C',
                            Result_Folder = 'E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/第四部分/DIANN/MCF7_3_Batches_MethodSelection_Result_1Y2E_不区分聚类簇/',
                            savefolder = 'E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/第四部分/DIANN/汇总图/1Y2E/'):

    # User input parameters
    df_top1 = pd.read_csv(TopMethodsFilePath) # Path to the csv file containing the top 1% method combinations
    df_top1 = df_top1.drop_duplicates() 

    GO = PathWay # 'Reactome'
    SR = SR
    Top_1_num = 12
    Comparison = Comparison
    Result_Folder = Result_Folder
    savefolder = savefolder


    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
    Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']


    df_SR = df_top1[df_top1['Sparsity Reduction'] == SR]
    
    df_GO_list = []
    X_label = []
    dict_ID_Description = {}  # Used to record ID and description
    for row in range(df_SR.shape[0]):
        GO_filename = Result_Folder + '{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(GO, SR, 
                                                                              df_SR['Missing Value Imputation'].values[row], 
                                                                              df_SR['Normalization'].values[row],
                                                                              df_SR['Batch Correction'].values[row],
                                                                              df_SR['Statistical Test'].values[row],
                                                                              Comparison.replace('/', '_vs_'))

        X_label.append('{0}-{1}-{2}-{3}'.format(str(Fill_NaN_methods.index(df_SR['Missing Value Imputation'].values[row])+1), 
                                                str(Normalization.index(df_SR['Normalization'].values[row])+1), 
                                                str(Batch_correction.index(df_SR['Batch Correction'].values[row])+1),
                                                str(Difference_analysis.index(df_SR['Statistical Test'].values[row])+1)))

        index_col = 0
        if (GO == 'GO'):
            index_col=1
        if (GO == 'KEGG'):
            index_col=0
        if (GO == 'Reactome'):
            index_col=0

        df_GO = pd.read_csv(GO_filename, index_col=index_col)
        df_GO_screened = df_GO[['Count', 'CountUp', 'CountDown']]
        df_GO_screened.columns = ['Count'+str(row+1), 'CountUp'+str(row+1), 'CountDown'+str(row+1)]
        df_GO_list.append(df_GO_screened)

        ID_list = df_GO.index.tolist()
        Description_list = df_GO['Description'].values.tolist()
        for ID in ID_list:
            dict_ID_Description[ID] = Description_list[ID_list.index(ID)]


    df_GO = pd.concat(df_GO_list, axis=1)

    # Calculate the proportion of missing values ​​for each row
    missing_ratio = df_GO.isnull().mean(axis=1)
    # Find rows with high missingness
    rows_with_high_missing = missing_ratio[missing_ratio > ((df_GO.shape[1]/3-3)/(df_GO.shape[1]/3))].index
    df_GO_Screened = df_GO.drop(rows_with_high_missing)

    df_GO.fillna(0, inplace=True)
    df_GO_Screened.fillna(0, inplace=True)


    Top_1_num = int(df_GO.shape[1]/3)

    X = []
    X_CountUp = []
    X_CountDown = []
    X_Screened = []
    X_CountUp_Screened = []
    X_CountDown_Screened = []
    for count in range(Top_1_num):
        X.append(df_GO.values[:, count*3].tolist())
        X_CountUp.append(df_GO.values[:, count*3+1].tolist())
        X_CountDown.append(df_GO.values[:, count*3+2].tolist())

        X_Screened.append(df_GO_Screened.values[:, count*3].tolist())
        X_CountUp_Screened.append(df_GO_Screened.values[:, count*3+1].tolist())
        X_CountDown_Screened.append(df_GO_Screened.values[:, count*3+2].tolist())

    X = np.array(X)
    X_CountUp = np.array(X_CountUp)
    X_CountDown = np.array(X_CountDown)

    X_Z_Score = (X_CountUp - X_CountDown)/np.sqrt(X)

    X_Screened = np.array(X_Screened)
    X_CountUp_Screened = np.array(X_CountUp_Screened)
    X_CountDown_Screened = np.array(X_CountDown_Screened)

    X_Z_Score_Screened = (X_CountUp_Screened - X_CountDown_Screened)/np.sqrt(X_Screened)


    # Y uses the filtered pathway data
    Y = X_Screened.T
    Y_Z_Score = X_Z_Score_Screened.T
    Y_label = df_GO_Screened.index.tolist()


    def jaccard_distance_1(x: np.ndarray, y: np.ndarray):
        x = np.sign(x)
        y = np.sign(y)
        intersect = np.nansum(x == y)
        union = intersect + 2 * np.nansum(x != y) + np.sum(np.isnan(x) ^ np.isnan(y))
        return 1 - intersect / union

    def jaccard_distance_2(x: np.ndarray, y: np.ndarray):
        x = np.sign(x)
        y = np.sign(y)
        intersect = np.nansum(x * y >= 0)
        union = intersect + 2 * np.nansum(x * y < 0) + np.sum(np.isnan(x) ^ np.isnan(y))
        return 1 - intersect / union


    def dice_distance_1(x: np.ndarray, y: np.ndarray):
        x = np.sign(x)
        y = np.sign(y)
        intersect = np.nansum(x == y)
        a = np.sum(np.isnan(x))
        b = np.sum(np.isnan(y))
        print(a+b)
        return 1 - 2 * intersect / (a + b)

    def dice_distance_2(x: np.ndarray, y: np.ndarray):
        x = np.sign(x)
        y = np.sign(y)
        intersect = np.nansum(x * y >= 0)    
        a = np.sum(np.isnan(x))
        b = np.sum(np.isnan(y))
        return 1 - 2 * intersect / (a + b)


    distance_matrix = np.zeros((X_Z_Score.shape[0], X_Z_Score.shape[0]))
    for i in range(X_Z_Score.shape[0]):
        for j in range(i, X_Z_Score.shape[0]):
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - jaccard_distance_2(X_Z_Score[i], X_Z_Score[j])

    distance_matrix[np.isinf(distance_matrix)] = np.max(distance_matrix[np.isfinite(distance_matrix)])
    distance_matrix = np.nan_to_num(distance_matrix)

    # Calculate the Jaccard distance between pairs of points - X
    #distance_matrix = np.zeros((X.shape[0], X.shape[0]))
    #for i in range(X.shape[0]):
    #    for j in range(i, X.shape[0]):
    #        distance_matrix[i, j] = distance_matrix[j, i] = 1 - jaccard_score(X[i], X[j], average = 'micro')


    Z = linkage(distance_matrix, method='single')

    # Data used to draw clusters on the X-axis
    X_Cluster_Init_Height = df_GO_Screened.shape[0] 
    Z[:,2] = Z[:,2]*(df_GO.shape[0]*0.03/(Z[:,2].max())) + X_Cluster_Init_Height

    X_clusters = {}
    X_clusters_height = {}
    X_label_index_after_clustering = []
    for i in range(Z.shape[0]):
        if (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            X_clusters[str(Top_1_num+i)] = [int(Z[i, 0]), int(Z[i, 1])]
            X_clusters_height[str(Top_1_num+i)] = Z[i, 2]

        elif (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] > (Top_1_num-1)):
            X_clusters[str(Top_1_num+i)] = [int(Z[i, 0])] + X_clusters[str(int(Z[i, 1]))]
            X_clusters_height[str(Top_1_num+i)] = Z[i, 2]

        elif (Z[i, 0] > (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            X_clusters[str(Top_1_num+i)] = X_clusters[str(int(Z[i, 0]))] + [int(Z[i, 1])]
            X_clusters_height[str(Top_1_num+i)] = Z[i, 2]

        else:
            X_clusters[str(Top_1_num+i)] = X_clusters[str(int(Z[i, 0]))] + X_clusters[str(int(Z[i, 1]))]
            #X_clusters_height[str(Top_1_num+i)] = max(X_clusters_height[str(int(Z[i, 0]))], X_clusters_height[str(int(Z[i, 1]))])
            X_clusters_height[str(Top_1_num+i)] = Z[i, 2]

    X_label_index_after_clustering = X_clusters[str(Z.shape[0] + Top_1_num - 1)]  # X-axis label
    X_label_after_clustering = []
    for i in X_label_index_after_clustering:
        X_label_after_clustering.append(X_label[i])


    X_clusters_center = {}
    for i in range(Z.shape[0]):
        if (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            X_clusters_center[str(Top_1_num+i)] = (X_label_index_after_clustering.index(int(Z[i, 0])) + X_label_index_after_clustering.index(int(Z[i, 1])))/2
        elif (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] > (Top_1_num-1)):
            X_clusters_center[str(Top_1_num+i)] = (X_label_index_after_clustering.index(int(Z[i, 0])) + X_clusters_center[str(int(Z[i, 1]))])/2
        elif (Z[i, 0] > (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            X_clusters_center[str(Top_1_num+i)] = (X_clusters_center[str(int(Z[i, 0]))] + X_label_index_after_clustering.index(int(Z[i, 1])))/2
        else:
            X_clusters_center[str(Top_1_num+i)] = (X_clusters_center[str(int(Z[i, 0]))] + X_clusters_center[str(int(Z[i, 1]))])/2



    distance_matrix = np.zeros((Y_Z_Score.shape[0], Y_Z_Score.shape[0]))
    for i in range(Y_Z_Score.shape[0]):
        for j in range(i, Y_Z_Score.shape[0]):
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - jaccard_distance_2(Y_Z_Score[i], Y_Z_Score[j])

    distance_matrix[np.isinf(distance_matrix)] = np.max(distance_matrix[np.isfinite(distance_matrix)])
    distance_matrix = np.nan_to_num(distance_matrix)

    # Calculate the Jaccard distance between pairs of points - Y
    #distance_matrix = np.zeros((Y.shape[0], Y.shape[0]))
    #for i in range(Y.shape[0]):
    #    for j in range(i, Y.shape[0]):
    #        distance_matrix[i, j] = distance_matrix[j, i] = 1 - jaccard_score(Y[i], Y[j], average = 'micro')
 

    Z2 = linkage(distance_matrix, method='single')

    # Data used to draw clusters on the Y-axis
    Y_Cluster_Init_Height = -1
    Z2[:,2] = Y_Cluster_Init_Height - Z2[:,2]*(df_GO.shape[1]*0.05/(Z2[:,2].max()))

    Y_clusters = {}
    Y_clusters_height = {}
    Y_label_index_after_clustering = []
    for i in range(Z2.shape[0]):
        if (Z2[i, 0] <= (Y.shape[0]-1)) & (Z2[i, 1] <= (Y.shape[0]-1)):
            Y_clusters[str(Y.shape[0]+i)] = [int(Z2[i, 0]), int(Z2[i, 1])]
            Y_clusters_height[str(Y.shape[0]+i)] = Z2[i, 2]

        elif (Z2[i, 0] <= (Y.shape[0]-1)) & (Z2[i, 1] > (Y.shape[0]-1)):
            Y_clusters[str(Y.shape[0]+i)] = [int(Z2[i, 0])] + Y_clusters[str(int(Z2[i, 1]))]
            Y_clusters_height[str(Y.shape[0]+i)] = Z2[i, 2]

        elif (Z2[i, 0] > (Y.shape[0]-1)) & (Z2[i, 1] <= (Y.shape[0]-1)):
            Y_clusters[str(Y.shape[0]+i)] = Y_clusters[str(int(Z2[i, 0]))] + [int(Z2[i, 1])]
            Y_clusters_height[str(Y.shape[0]+i)] = Z2[i, 2]

        else:
            Y_clusters[str(Y.shape[0]+i)] = Y_clusters[str(int(Z2[i, 0]))] + Y_clusters[str(int(Z2[i, 1]))]
            Y_clusters_height[str(Y.shape[0]+i)] = Z2[i, 2]

    Y_label_index_after_clustering = Y_clusters[str(Z2.shape[0] + Y.shape[0] - 1)] 
    Y_label_after_clustering = []
    for i in Y_label_index_after_clustering:
        Y_label_after_clustering.append(Y_label[i])


    Y_clusters_center = {}
    for i in range(Z2.shape[0]):
        if (Z2[i, 0] <= (Y.shape[0]-1)) & (Z2[i, 1] <= (Y.shape[0]-1)):
            Y_clusters_center[str(Y.shape[0]+i)] = (Y_label_index_after_clustering.index(int(Z2[i, 0])) + Y_label_index_after_clustering.index(int(Z2[i, 1])))/2
        elif (Z2[i, 0] <= (Y.shape[0]-1)) & (Z2[i, 1] > (Y.shape[0]-1)):
            Y_clusters_center[str(Y.shape[0]+i)] = (Y_label_index_after_clustering.index(int(Z2[i, 0])) + Y_clusters_center[str(int(Z2[i, 1]))])/2
        elif (Z2[i, 0] > (Y.shape[0]-1)) & (Z2[i, 1] <= (Y.shape[0]-1)):
            Y_clusters_center[str(Y.shape[0]+i)] = (Y_clusters_center[str(int(Z2[i, 0]))] + Y_label_index_after_clustering.index(int(Z2[i, 1])))/2
        else:
            Y_clusters_center[str(Y.shape[0]+i)] = (Y_clusters_center[str(int(Z2[i, 0]))] + Y_clusters_center[str(int(Z2[i, 1]))])/2



    if (GO == 'GO'):
        plt.figure(figsize=(10, 16))
    if (GO == 'KEGG'):
        plt.figure(figsize=(10, 8))
    if (GO == 'Reactome'):
        plt.figure(figsize=(10, 12))

    # Draw horizontal dendrogram
    for i in range(Z.shape[0]):
        A = []
        A_up = []
        B = []
        B_up= []
        if (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            A = [X_label_index_after_clustering.index(int(Z[i, 0])), X_Cluster_Init_Height]
            A_up = [X_label_index_after_clustering.index(int(Z[i, 0])), Z[i, 2]]
            B = [X_label_index_after_clustering.index(int(Z[i, 1])), X_Cluster_Init_Height]
            B_up = [X_label_index_after_clustering.index(int(Z[i, 1])), Z[i, 2]]

        elif (Z[i, 0] <= (Top_1_num-1)) & (Z[i, 1] > (Top_1_num-1)):
            A = [X_label_index_after_clustering.index(int(Z[i, 0])), X_Cluster_Init_Height]
            A_up = [X_label_index_after_clustering.index(int(Z[i, 0])), Z[i, 2]]
            B = [X_clusters_center[str(int(Z[i, 1]))], X_clusters_height[str(int(Z[i, 1]))]]
            B_up = [X_clusters_center[str(int(Z[i, 1]))], Z[i, 2]]
        
        elif (Z[i, 0] > (Top_1_num-1)) & (Z[i, 1] <= (Top_1_num-1)):
            A = [X_clusters_center[str(int(Z[i, 0]))], X_clusters_height[str(int(Z[i, 0]))]]
            A_up = [X_clusters_center[str(int(Z[i, 0]))], Z[i, 2]]
            B = [X_label_index_after_clustering.index(int(Z[i, 1])), X_Cluster_Init_Height]
            B_up = [X_label_index_after_clustering.index(int(Z[i, 1])), Z[i, 2]]

        else:
            A = [X_clusters_center[str(int(Z[i, 0]))], X_clusters_height[str(int(Z[i, 0]))]]
            A_up = [X_clusters_center[str(int(Z[i, 0]))], Z[i, 2]]
            B = [X_clusters_center[str(int(Z[i, 1]))], X_clusters_height[str(int(Z[i, 1]))]]
            B_up = [X_clusters_center[str(int(Z[i, 1]))], Z[i, 2]]
        
        plt.plot([A[0], A_up[0]], [A[1], A_up[1]], color = 'gray')
        plt.plot([A_up[0], B_up[0]], [A_up[1], B_up[1]], color = 'gray')
        plt.plot([B_up[0], B[0]], [B_up[1], B[1]], color = 'gray')


    # Draw vertical dendrogram
    for i in range(Z2.shape[0]):
        A = []
        A_up = []
        B = []
        B_up= []
        if (Z2[i, 0] <= (len(Y_label)-1)) & (Z2[i, 1] <= (len(Y_label)-1)):
            A = [Y_label_index_after_clustering.index(int(Z2[i, 0])), Y_Cluster_Init_Height]
            A_up = [Y_label_index_after_clustering.index(int(Z2[i, 0])), Z2[i, 2]]
            B = [Y_label_index_after_clustering.index(int(Z2[i, 1])), Y_Cluster_Init_Height]
            B_up = [Y_label_index_after_clustering.index(int(Z2[i, 1])), Z2[i, 2]]

        elif (Z2[i, 0] <= (len(Y_label)-1)) & (Z2[i, 1] > (len(Y_label)-1)):
            A = [Y_label_index_after_clustering.index(int(Z2[i, 0])), Y_Cluster_Init_Height]
            A_up = [Y_label_index_after_clustering.index(int(Z2[i, 0])), Z2[i, 2]]
            B = [Y_clusters_center[str(int(Z2[i, 1]))], Y_clusters_height[str(int(Z2[i, 1]))]]
            B_up = [Y_clusters_center[str(int(Z2[i, 1]))], Z2[i, 2]]
        
        elif (Z2[i, 0] > (len(Y_label)-1)) & (Z2[i, 1] <= (len(Y_label)-1)):
            A = [Y_clusters_center[str(int(Z2[i, 0]))], Y_clusters_height[str(int(Z2[i, 0]))]]
            A_up = [Y_clusters_center[str(int(Z2[i, 0]))], Z2[i, 2]]
            B = [Y_label_index_after_clustering.index(int(Z2[i, 1])), Y_Cluster_Init_Height]
            B_up = [Y_label_index_after_clustering.index(int(Z2[i, 1])), Z2[i, 2]]

        else:
            A = [Y_clusters_center[str(int(Z2[i, 0]))], Y_clusters_height[str(int(Z2[i, 0]))]]
            A_up = [Y_clusters_center[str(int(Z2[i, 0]))], Z2[i, 2]]
            B = [Y_clusters_center[str(int(Z2[i, 1]))], Y_clusters_height[str(int(Z2[i, 1]))]]
            B_up = [Y_clusters_center[str(int(Z2[i, 1]))], Z2[i, 2]]
        
        plt.plot([A[1], A_up[1]], [A[0], A_up[0]], color = 'gray')
        plt.plot([A_up[1], B_up[1]], [A_up[0], B_up[0]], color = 'gray')
        plt.plot([B_up[1], B[1]], [B_up[0], B[0]], color = 'gray')

    # Draw Scatter
    count_list = []
    Z_score_list = []
    for i in X_label_index_after_clustering:
        for j in Y_label_index_after_clustering:

            count = X_Screened[i][j]
            if (count != 0):
                count_up = X_CountUp_Screened[i][j]
                count_down = X_CountDown_Screened[i][j]
                Z_score = (count_up-count_down)/math.sqrt(count)

                count_list.append(count)
                Z_score_list.append(Z_score)


    min_count = min(count_list)
    max_count = max(count_list)
    min_Z_score = -2 # min(Z_score_list)
    max_Z_score = 2 # max(Z_score_list)

    min_scatter_size = 10
    max_sactter_size = 60

    color_Z_score_max = [157/255, 48/255, 238/255, 1.0]  # Purple
    color_Z_score_zero = [211/255, 211/255, 211/255, 1.0]  # Light Gray
    color_Z_score_min = [53/255, 131/255, 99/255, 1.0]  # Green


    ID_description = []
    for ID in list(reversed(Y_label_after_clustering)):
        ID_description.append(dict_ID_Description[ID])

    out_data = {'Description': ID_description}
    out_z_score_data = {}
    for i in X_label_index_after_clustering:
        column_data = []
        column_z_score_data = []
        for j in Y_label_index_after_clustering:

            count = X_Screened[i][j]
            Z_score = 0
            if (count != 0):
                count_up = X_CountUp_Screened[i][j]
                count_down = X_CountDown_Screened[i][j]
                Z_score = (count_up-count_down)/math.sqrt(count)

                s = min_scatter_size + (math.sqrt(count)-math.sqrt(min_count))*(max_sactter_size-min_scatter_size)/(math.sqrt(max_count)-math.sqrt(min_count))

                c = []
                if (Z_score >= 0):
                    if (Z_score <= max_Z_score):
                        c = [color_Z_score_zero[0] + Z_score*(color_Z_score_max[0] - color_Z_score_zero[0])/max_Z_score,
                             color_Z_score_zero[1] + Z_score*(color_Z_score_max[1] - color_Z_score_zero[1])/max_Z_score,
                             color_Z_score_zero[2] + Z_score*(color_Z_score_max[2] - color_Z_score_zero[2])/max_Z_score,
                             1]
                    else:
                        c = color_Z_score_max
                else:
                    if (Z_score >= min_Z_score):
                        c = [color_Z_score_zero[0] + Z_score*(color_Z_score_min[0] - color_Z_score_zero[0])/min_Z_score,
                             color_Z_score_zero[1] + Z_score*(color_Z_score_min[1] - color_Z_score_zero[1])/min_Z_score,
                             color_Z_score_zero[2] + Z_score*(color_Z_score_min[2] - color_Z_score_zero[2])/min_Z_score,
                             1]
                    else:
                        c = color_Z_score_min

                plt.scatter(X_label_index_after_clustering.index(i), Y_label_index_after_clustering.index(j), s = s, color = c)
            else:
                count = np.nan
                Z_score = np.nan

            column_data.append(count)
            column_z_score_data.append(Z_score)

        out_data['Count ' + X_label_after_clustering[X_label_index_after_clustering.index(i)]] = list(reversed(column_data))
        out_z_score_data['Z-score ' + X_label_after_clustering[X_label_index_after_clustering.index(i)]] = list(reversed(column_z_score_data))



    ax = plt.gca()

    plt.xlim(Z2[:,2].min(), Top_1_num-0.5)
    plt.xticks(list(range(Top_1_num)))
    ax.set_xticklabels(X_label_after_clustering, rotation=90, fontsize=12)


    # Add descriptions
    Y_label_add_description = []
    for label in Y_label_after_clustering:
        Label = label
    
        Description = dict_ID_Description[label]

        New_Description = Description
        if (len(Description) > 50):
            New_Description = Description[0:50] + '...'

        if (GO == 'Reactome'):

            Label = Label + '  '*(13-len(Label))

            if (len(Description) > 45):
                New_Description = Description[0:45] + '...'

        Y_label_add_description.append(Label + ' ' + New_Description)

    plt.ylim(-0.5, Z[:,2].max())
    plt.yticks(list(range(len(Y_label))))
    ax.yaxis.tick_right()
    #ax.set_yticklabels(Y_label_after_clustering, rotation=0, fontsize=10)
    ax.set_yticklabels(Y_label_add_description, rotation=0, fontsize=10)

    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tick_params(axis='both', which='both', bottom=False, right=False, labelbottom=True, labelright=True)


    if (GO == 'GO'):
        plt.subplots_adjust(left=0.01, right=0.5, bottom=0.05, top=0.99, wspace=0.1)
    if (GO == 'KEGG'):
        plt.subplots_adjust(left=0.01, right=0.6, bottom=0.12, top=0.99, wspace=0.1)
    if (GO == 'Reactome'):
        plt.subplots_adjust(left=0.01, right=0.5, bottom=0.07, top=0.99, wspace=0.1)

    out_data.update(out_z_score_data)
    df_result = pd.DataFrame(out_data, index=list(reversed(Y_label_after_clustering)))
    df_result.index.names = ['ID']
    df_result.to_csv(savefolder + "{0}_{1}.csv".format(GO, SR), index=True)

    plt.savefig(savefolder + "{0}_{1}.svg".format(GO, SR), dpi=600, format="svg", transparent=True)
    plt.show()
    plt.close()



    # Draw color legend
    fig, ax = plt.subplots(figsize=(2.5, 5))
    list_of_values = np.arange(min_Z_score, max_Z_score + 0.01, 0.01)
    axes = plt.gca()
    for value in list_of_values:

        Z_score = value
        color = (0,0,0)
        if value >=0:
            if (value <= max_Z_score):
                c = [color_Z_score_zero[0] + Z_score*(color_Z_score_max[0] - color_Z_score_zero[0])/max_Z_score,
                        color_Z_score_zero[1] + Z_score*(color_Z_score_max[1] - color_Z_score_zero[1])/max_Z_score,
                        color_Z_score_zero[2] + Z_score*(color_Z_score_max[2] - color_Z_score_zero[2])/max_Z_score,
                        1]
            else:
                c = color_Z_score_max
        else:
            if (value >= min_Z_score):
                c = [color_Z_score_zero[0] + Z_score*(color_Z_score_min[0] - color_Z_score_zero[0])/min_Z_score,
                        color_Z_score_zero[1] + Z_score*(color_Z_score_min[1] - color_Z_score_zero[1])/min_Z_score,
                        color_Z_score_zero[2] + Z_score*(color_Z_score_min[2] - color_Z_score_zero[2])/min_Z_score,
                        1]
            else:
                c = color_Z_score_min


        rect = patches.Rectangle((-1, value), width=1, height=2/((max_Z_score-min_Z_score)*100),
                                    color = c, fill=True)

        
        axes.add_patch(rect)


    plt.xlim(-2, 1.5)
    plt.ylim(-1, 1) 

    plt.xticks([])
    axes.yaxis.set_ticks_position('right')
    axes.spines['right'].set_position(('data', 0)) 

    plt.yticks([min_Z_score, min_Z_score/2, 0, max_Z_score/2, max_Z_score], [format(min_Z_score, '.2f'), format(min_Z_score/2, '.2f'), '0.00', format(max_Z_score/2, '.2f'), format(max_Z_score, '.2f')])
    plt.tick_params(labelsize=14) 

    yticks = axes.get_yticklabels()
 
    for label in yticks:
        label.set_va('center') 

    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False) 
    axes.spines['left'].set_visible(False)

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    plt.savefig(savefolder + 'Legend_{0}_Z-score.svg'.format(GO), dpi=600, format="svg", transparent=True)

    plt.show()
    plt.close()



    # Plot legend for scatter point sizes
    fig_legend = plt.figure(figsize=(2.5,5))
    axes = plt.gca()

    count_list = [int(min_count), 
                  int(min_count + (max_count-min_count)*1/4), 
                  int(min_count + (max_count-min_count)*2/4), 
                  int(min_count + (max_count-min_count)*3/4),
                  int(max_count)]

    labels = []
    for i in count_list:
        labels.append(str(i))
        scatter_size = min_scatter_size + (math.sqrt(i)-math.sqrt(min_count))*(max_sactter_size-min_scatter_size)/(math.sqrt(max_count)-math.sqrt(min_count))

        parts = axes.scatter([100], [100], color = [157/255, 48/255, 238/255, 1.0], s = scatter_size)


    axes.legend(labels = labels, title='Count', title_fontsize=18, fontsize=16, 
                loc = 'center',
                borderpad=1, 
                labelspacing=1.0, 
                markerfirst=True, markerscale=1.0) 

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    plt.savefig(savefolder + 'Legend_{0}_{1}_Count.svg'.format(GO, SR), dpi=600, format="svg", transparent=True)
    plt.show()
    plt.close()

    # Return X-axis method combination
    return X_label_after_clustering



# Draw a histogram of the indicators ('Purity Score', 'Sample Purity', 'Batch Purity') of the X-axis method combination after hierarchical clustering
def Plot_PurityScore_Histogram(GO, SR, X_label_after_clustering, DifferentialExpressionAnalysis_CSV_Path, savefolder,
                               Groups = 2, Batches = 3):

    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
    Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']

    # Load the differential analysis result file
    df = pd.read_csv(DifferentialExpressionAnalysis_CSV_Path)

    Purity_Score_List = []
    Sample_Purity_List = []
    Batch_Purity_List = []

    for X_label in X_label_after_clustering:
        # method combination
        i = SR
        j = Fill_NaN_methods[int(X_label.split('-')[0]) - 1]
        k = Normalization[int(X_label.split('-')[1]) - 1]
        l = Batch_correction[int(X_label.split('-')[2]) - 1]
        m = Difference_analysis[int(X_label.split('-')[3]) - 1]

        # Purity Score, Sample Purity, Batch Purity of this method combination
        df_screened = df[(df['Sparsity Reduction'] == i) & (df['Missing Value Imputation'] == j) & (df['Normalization'] == k) & (df['Batch Correction'] == l)]
        Purity_Score = df_screened['Purity Score'].values[0]
        Sample_Purity = df_screened['Sample Purity'].values[0]
        Batch_Purity = df_screened['Batch Purity'].values[0]

        Purity_Score_List.append(Purity_Score)
        Sample_Purity_List.append(Sample_Purity)
        Batch_Purity_List.append(Batch_Purity)


    PlotLabel = ['Purity Score', 'Sample Purity', 'Batch Purity']
    PlotData = {'Purity Score': Purity_Score_List,
                'Sample Purity': Sample_Purity_List,
                'Batch Purity': Batch_Purity_List}

    for label in PlotLabel:

        # Draw bar chart
        fig, ax = plt.subplots(1, 1, figsize=(5,2.5))

        # Color
        color_list = {'Purity Score': [24/255, 131/255, 184/255, 1.0],
                      'Sample Purity': [141/255, 158/255, 38/255, 1.0],
                      'Batch Purity': [176/255, 32/255, 63/255, 1.0]}

        bar1 = plt.bar(list(range(len(PlotData[label]))), PlotData[label], width = 0.75, color = color_list[label])

        plt.tick_params(labelsize=14) 
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        
        plt.xlim(-1, len(X_label_after_clustering)) 
        
        Y_min = 0
        if (label == 'Purity Score'):
            Y_min = 0
        if (label == 'Sample Purity'):
            Y_min = 1/Groups
        if (label == 'Batch Purity'):
            Y_min = 1/Batches
        plt.ylim(Y_min, 1) 
        plt.yticks(np.linspace(Y_min, 1, 3)) 

        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2) 

        plt.xticks(list(range(len(X_label_after_clustering))))
        axes.set_xticklabels(X_label_after_clustering, rotation=90, fontsize=12)
        axes.set_yticklabels(['{:.2f}'.format(Y_min), '{:.2f}'.format((Y_min+1)/2), '1.00'], rotation=0, fontsize=14)

        plt.ylabel(label, fontsize=16) 

        plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)
        plt.subplots_adjust(left=0.18, right=0.99, bottom=0.40, top=0.90, wspace=0.05)

        plt.savefig(savefolder + '{0}_{1}_{2}.svg'.format(label.replace(' ', ''), GO, SR), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()




# Draw a histogram of the indicators ('ARI') of the X-axis method combination after hierarchical clustering
def Plot_ARI_Histogram(GO, SR, X_label_after_clustering, DifferentialExpressionAnalysis_CSV_Path, savefolder,
                               Groups = 2, Batches = 3):

    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
    Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']

    # Load the differential analysis result file
    df = pd.read_csv(DifferentialExpressionAnalysis_CSV_Path)

    ARI_List = []

    for X_label in X_label_after_clustering:
        # method combination
        i = SR
        j = Fill_NaN_methods[int(X_label.split('-')[0]) - 1]
        k = Normalization[int(X_label.split('-')[1]) - 1]
        l = Batch_correction[int(X_label.split('-')[2]) - 1]
        m = Difference_analysis[int(X_label.split('-')[3]) - 1]

        # ARI of this method combination
        df_screened = df[(df['Sparsity Reduction'] == i) & (df['Missing Value Imputation'] == j) & (df['Normalization'] == k) & (df['Batch Correction'] == l)]
        ARI = df_screened['ARI'].values[0]

        ARI_List.append(ARI)


    PlotLabel = ['ARI']
    PlotData = {'ARI': ARI_List}

    for label in PlotLabel:

        # Draw bar chart
        fig, ax = plt.subplots(1, 1, figsize=(5,2.5))

        # Color
        color_list = {'ARI': [24/255, 131/255, 184/255, 1.0]}

        bar1 = plt.bar(list(range(len(PlotData[label]))), PlotData[label], width = 0.75, color = color_list[label])

        plt.tick_params(labelsize=14) 
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        
        plt.xlim(-1, len(X_label_after_clustering)) 
        
        Y_min = 0
        if (label == 'ARI'):
            Y_min = 0

        plt.ylim(Y_min, 1) 
        plt.yticks(np.linspace(Y_min, 1, 3)) 

        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2) 

        plt.xticks(list(range(len(X_label_after_clustering))))
        axes.set_xticklabels(X_label_after_clustering, rotation=90, fontsize=12)
        axes.set_yticklabels(['{:.2f}'.format(Y_min), '{:.2f}'.format((Y_min+1)/2), '1.00'], rotation=0, fontsize=14)

        plt.ylabel(label, fontsize=16) 

        plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)
        plt.subplots_adjust(left=0.18, right=0.99, bottom=0.40, top=0.90, wspace=0.05)

        plt.savefig(savefolder + '{0}_{1}_{2}.svg'.format(label, GO, SR), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()


