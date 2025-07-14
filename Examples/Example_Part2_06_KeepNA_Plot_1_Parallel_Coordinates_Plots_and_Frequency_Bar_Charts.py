

import numpy as np
import pandas as pd
import svgutils.transform as st

import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["font.sans-serif"] = ["Arial"] 
matplotlib.rcParams["axes.unicode_minus"] = False



# # Draw parallel coordinates and stacked bar charts of the frequency of occurrence of various methods
def KeepNA_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts(
    # Table after the merger of KeepNA and UseCov
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates/Results/'):
    


    #pValue_List = [0.001, 0.01, 0.05, 0.1]
    pValue_List = [0.05]
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Compared_groups_label = ['S4/S2', 'S5/S1']

    Use_Given_PValue_and_FC = True 
    ARI_or_Purity = 'ARI'


    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute', 'KeepNA']
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN']
    BatchCorrection = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']
    Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']



    # Colors
    colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890', '#fbe29d', '#c8d961']
    colors2 = ['#fb8072', '#1965b0', '#7bafde', '#882e72', '#b17ba6', '#ff7f00', '#fdb462', '#e7298a']





    # （1）Parallel Coordinates Plot
    for P_value in pValue_List:
        for SR_method in SR_methods:

            # Read csv file
            df_result = pd.read_csv(result_csv_path) 
            # Filter by p-value
            df_p = df_result[df_result['p-value'] == P_value]
            # Filter by SR
            df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]

            df_screened = df_p_SR.copy(deep = True)
            df_screened = df_screened.sort_values('No', ascending=True)


            for Compared_groups in Compared_groups_label:

                # If the report uses the given FC and P values, add Precision to the plot
                columns = None
                if Use_Given_PValue_and_FC:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                                ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Precision', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                                Compared_groups + ' Rank']
                else:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                                ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                                Compared_groups + ' Rank']

                # KeepNA 
                filtered_df = df_screened[df_screened['Missing Value Imputation'].isin(['KeepNA'])]
                filtered_df = filtered_df.sort_values(Compared_groups + ' Rank', ascending=True)
                KeepNA_No_list = filtered_df['No'].values.tolist()

                # 'Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'
                filtered_df = df_screened[df_screened['Missing Value Imputation'].isin(['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'])]
                filtered_df = filtered_df.sort_values(Compared_groups + ' Rank', ascending=True)
                UseCovariate_No_list = filtered_df['No'].values.tolist()


                df_plot = df_screened[columns].copy(deep = True) 
                value_list = []
                for  i in df_plot['Missing Value Imputation'].values.tolist():
                    value_list.append(Imputation.index(i))
                df_plot['Missing Value Imputation'] = value_list

                value_list = []
                for  i in df_plot['Normalization'].values.tolist():
                    value_list.append(Normalization.index(i))
                df_plot['Normalization'] = value_list

                value_list = []
                for  i in df_plot['Batch Correction'].values.tolist():
                    value_list.append(BatchCorrection.index(i))
                df_plot['Batch Correction'] = value_list

                value_list = []
                for  i in df_plot['Statistical Test'].values.tolist():
                    value_list.append(StatisticalTest.index(i))
                df_plot['Statistical Test'] = value_list


                df_plot.astype('float')

                current_rank_list = df_plot[Compared_groups + ' Rank'].values.tolist()

                sorted_current_rank_list = sorted(current_rank_list)
                total_num = len(current_rank_list)

                new_rank_list = []
                for rank in current_rank_list:
                    rank_ratio = sorted_current_rank_list.index(rank)/total_num

                    #if rank_ratio == 0:
                    #    new_rank_list.append(1)
                    #elif rank_ratio <= 0.01:
                    #    new_rank_list.append(int(700*rank_ratio/0.01))
                    #elif rank_ratio <= 0.02:
                    #    new_rank_list.append(int(700*1+700*(rank_ratio-0.01)/0.01))
                    #elif rank_ratio <= 0.04:
                    #    new_rank_list.append(int(700*2+700*(rank_ratio-0.02)/0.02))
                    #elif rank_ratio <= 0.06:
                    #    new_rank_list.append(int(700*3+700*(rank_ratio-0.04)/0.02))
                    #elif rank_ratio <= 0.08:
                    #    new_rank_list.append(int(700*4+700*(rank_ratio-0.06)/0.02))
                    #elif rank_ratio <= 0.10:
                    #    new_rank_list.append(int(700*5+700*(rank_ratio-0.08)/0.02))
                    #else:
                    #    new_rank_list.append(int(700*6+700*(rank_ratio-0.10)/0.9))

                    if rank_ratio == 0:
                        new_rank_list.append(1)
                    elif rank_ratio <= 0.01:
                        new_rank_list.append(int((5020/6)*rank_ratio/0.01))
                    elif rank_ratio <= 0.02:
                        new_rank_list.append(int((5020/6)*1+(5020/6)*(rank_ratio-0.01)/0.01))
                    elif rank_ratio <= 0.03:
                        new_rank_list.append(int((5020/6)*2+(5020/6)*(rank_ratio-0.02)/0.01))
                    elif rank_ratio <= 0.04:
                        new_rank_list.append(int((5020/6)*3+(5020/6)*(rank_ratio-0.03)/0.01))
                    elif rank_ratio <= 0.05:
                        new_rank_list.append(int((5020/6)*4+(5020/6)*(rank_ratio-0.04)/0.01))
                    else:
                        new_rank_list.append(int((5020/6)*5+(5020/6)*(rank_ratio-0.05)/0.95))


                df_plot[Compared_groups + ' Rank'] = new_rank_list

                max_rank_value = 5020 
                min_rank_value = 1 
                df_plot[Compared_groups + ' Rank'] = max_rank_value - df_plot[Compared_groups + ' Rank'] + min_rank_value


                ynames = None
                if Use_Given_PValue_and_FC:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                                ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
                else:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                                ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']

                ys = np.array(df_plot.values.tolist())

                if Use_Given_PValue_and_FC:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 5020]], axis=0)
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)
                    ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
                else:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 5020]], axis=0)
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 0]], axis=0)

                    # Enlarge the gap between the last few columns
                    ys[:, -5:-1] = 1/(1.5 - (ys[:, -5:-1] ** 4))


                noise_upper_bound = 1.05
                noise_lower_bound = 0.95
                noise = np.random.normal(loc=0, scale=np.ones((ys.shape[0], 4)) * 0.06, size=ys[:, :4].shape)
                ys[:, :4] = ys[:, :4] + noise 


                ymins = ys.min(axis=0)
                ymaxs = ys.max(axis=0)
                dys = ymaxs - ymins
                ymins -= dys * 0.02 
                ymaxs += dys * 0.02

                zs = np.zeros_like(ys)
                zs[:, 0] = ys[:, 0]
                zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

                for plot_index in range(2):

                    fig, host = plt.subplots(figsize=(12, 5))

                    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
                    for i, ax in enumerate(axes):
                
                        if i == 0:
                            ax.set_yticks(range(len(Imputation)))
                            ax.set_yticklabels(Imputation, fontsize=10, rotation = 250, ha = 'right', va = 'top')

                        if i>=5:
                            ax.set_ylim(ymins[i], ymaxs[i])
                            
                        elif i == 4:
                            ax.set_ylim(ymins[i], ymaxs[i])
                            
                        else:
                            ax.set_ylim(ymins[i], ymaxs[i])

                        ax.spines["top"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)
                        ax.spines['right'].set_linewidth(2) 
                        ax.tick_params(axis="y", width=2, labelsize=10) 

                        if plot_index == 0:
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)

                        if ax != host:

                            ax.spines["left"].set_visible(False)

                            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
                            ax.tick_params(axis="y", width=2, labelsize=10, direction='in' )

                            if i == 1:
                                ax.set_yticks(range(len(Normalization)))
                                ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 2:
                                ax.set_yticks(range(len(BatchCorrection)))
                                ax.set_yticklabels(BatchCorrection, fontsize=10, rotation = 250, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 3:
                                ax.set_yticks(range(len(StatisticalTest)))
                                ax.set_yticklabels(StatisticalTest, fontsize=10, rotation = 250, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 4:
                                ax.set_yticks(np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5))
                                ytick_label = np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5).tolist()
                            
                                str_list = []
                                for data in ytick_label:
                                    str_list.append('%.3f' % data)

                                ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)


                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)
                            #elif i == 9:
                            elif (i == (len(columns)-1)):

                            

                                ax.set_yticks([0, (5020/6)*1, (5020/6)*2, (5020/6)*3, (5020/6)*4, (5020/6)*5, 5020])
                                str_list = ['100%', '5%', '4%', '3%', '2%', '1%', '0%']


                                ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            else:
                                ax.set_yticks(np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5))
                                ytick_label = np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5).tolist()
                                float_list = []
                                for label in ytick_label:
                                    float_list.append(np.sqrt(np.sqrt(1.5 - 1/round(label,14))))


                                str_list = []
                                for data in float_list:
                                    str_list.append('%.3f' % data)
                                ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            

                    host.set_xlim(0, ys.shape[1] - 1)
                    host.set_xticks(range(ys.shape[1]))
                    host.set_xticklabels(ynames, fontsize=12, rotation = 270, ha = 'left', va = 'top') 
                    host.tick_params(axis="x", which="major", pad=0, width=2, labelsize=13) 
                    host.tick_params(axis="y", width=2, labelsize=10, direction='out') 
                    host.spines["right"].set_visible(False)
                    host.spines['left'].set_linewidth(2) 
                    host.xaxis.tick_top()
                    color_1 = (255,255,255) 
                    color_2 = (24,131,184) 

                    if plot_index == 0:
                        host.set_xticks([])
                        host.set_yticks([])
                        host.spines['right'].set_visible(False)
                        host.spines['left'].set_visible(False)

                    
                    for tuceng in range(3):
                        for j in range(ys.shape[0] - 2):
                            verts = list(
                                zip(
                                    [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                                    np.repeat(zs[j, :], 3)[1:-1],
                                )
                            )
                            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                            path = Path(verts, codes)


                            edgecolor = '#e95b1b'
                            alpha = 0.85
                            linewidth = 0.95


                            # No of current method combination
                            current_No = df_screened['No'].values.tolist()[j]
                            current_No_rank = None

                            if (current_No in KeepNA_No_list):

                                current_No_rank = KeepNA_No_list.index(current_No)/(len(KeepNA_No_list))

                                # Top3 Red Bold
                                if (KeepNA_No_list.index(current_No) <= 2):
                                    if (tuceng != 2):
                                        continue
                                    edgecolor = '#b0203f'
                                    linewidth = 2.0
                                    alpha = 1.0
                                
                                # Top 1% Orange
                                elif (KeepNA_No_list.index(current_No) <= len(UseCovariate_No_list)*0.01):
                                #elif ( current_No_rank < 0.01):
                                    if (tuceng != 1):
                                        continue
                                    edgecolor = '#e95b1b'
                                    alpha = 0.85
                                    linewidth = 0.95

                                # Other gray
                                else:
                                    if (tuceng != 0):
                                        continue
                                    edgecolor = '#d9d8d2'
                                    alpha = 0.2
                                    linewidth = 0.2


                            if (current_No in UseCovariate_No_list):

                                current_No_rank = UseCovariate_No_list.index(current_No)/(len(UseCovariate_No_list))

                                # Top3 Purple Bold
                                if (UseCovariate_No_list.index(current_No) <= 2):
                                    if (tuceng != 2):
                                        continue
                                    edgecolor = '#a14ee0'
                                    linewidth = 2.0
                                    alpha = 1.0
                                
                                # Top 1% Blue
                                elif ( current_No_rank < 0.01):
                                    if (tuceng != 1):
                                        continue
                                    edgecolor = '#1883b8'
                                    alpha = 0.85
                                    linewidth = 0.95

                                # Other gray
                                else:
                                    if (tuceng != 0):
                                        continue
                                    edgecolor = '#d9d8d2'
                                    alpha = 0.2
                                    linewidth = 0.2


                            #if (current_No in BC_and_Scan_No_list):

                            #    current_No_rank = BC_and_Scan_No_list.index(current_No)/(len(BC_and_Scan_No_list))

                            #    # Top 1% Green
                            #    if ( current_No_rank < 0.01):
                            #        if (tuceng != 1):
                            #            continue
                            #        edgecolor = '#39a139'
                            #        alpha = 0.85
                            #        linewidth = 0.95

                            #    # Other gray
                            #    else:
                            #        if (tuceng != 0):
                            #            continue
                            #        edgecolor = '#d9d8d2'
                            #        alpha = 0.2
                            #        linewidth = 0.2

                        
                
                            if plot_index == 0:

                                if current_No_rank > 0.01:
                                    patch = patches.PathPatch(
                                        path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                    )
                                    #legend_handles[j] = patch
                                    host.add_patch(patch)

                            else:
                                if current_No_rank <= 0.01:
                                    patch = patches.PathPatch(
                                        path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                    )
                                    #legend_handles[j] = patch
                                    host.add_patch(patch)
                
                    if plot_index == 0:
                        host.set_rasterized(True)


                    plt.subplots_adjust(left=0.044, right=0.98, bottom=0.07, top=0.98, wspace=0.05)

                    if plot_index == 0:
                        svg1 = st.from_mpl(fig, savefig_kw=dict(transparent=True))
                    if plot_index == 1:
                        svg2 = st.from_mpl(fig, savefig_kw=dict(transparent=True))

                
                    plt.close()

                svg1.append(svg2)
                svg1.save(savefolder + 'StatisticalMetrics_PValue{0}_{1}_{2}_vs_{3}.svg'.format(str(P_value)[2:], SR_method, Compared_groups.split('/')[0], Compared_groups.split('/')[1]).replace('PValue1', 'PValue10'))




            columns = None
            if Use_Given_PValue_and_FC:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1-Score',
                            'Average Rank']
            else:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Recall', 'Average F1-Score',
                            'Average Rank']

            # KeepNA
            filtered_df = df_screened[df_screened['Missing Value Imputation'].isin(['KeepNA'])]
            filtered_df = filtered_df.sort_values(Compared_groups + ' Rank', ascending=True)
            KeepNA_No_list = filtered_df['No'].values.tolist()

            # 'Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'
            filtered_df = df_screened[df_screened['Missing Value Imputation'].isin(['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'])]
            filtered_df = filtered_df.sort_values(Compared_groups + ' Rank', ascending=True)
            UseCovariate_No_list = filtered_df['No'].values.tolist()


            df_plot = df_screened[columns].copy(deep = True) 

            value_list = []
            for  i in df_plot['Missing Value Imputation'].values.tolist():
                value_list.append(Imputation.index(i))
            df_plot['Missing Value Imputation'] = value_list

            value_list = []
            for  i in df_plot['Normalization'].values.tolist():
                value_list.append(Normalization.index(i))
            df_plot['Normalization'] = value_list

            value_list = []
            for  i in df_plot['Batch Correction'].values.tolist():
                value_list.append(BatchCorrection.index(i))
            df_plot['Batch Correction'] = value_list

            value_list = []
            for  i in df_plot['Statistical Test'].values.tolist():
                value_list.append(StatisticalTest.index(i))
            df_plot['Statistical Test'] = value_list


            df_plot.astype('float')

            current_rank_list = df_plot['Average Rank'].values.tolist()

            sorted_current_rank_list = sorted(current_rank_list)
            total_num = len(current_rank_list)

            new_rank_list = []
            for rank in current_rank_list:
                rank_ratio = sorted_current_rank_list.index(rank)/total_num

                #if rank_ratio == 0:
                #    new_rank_list.append(1)
                #elif rank_ratio <= 0.01:
                #    new_rank_list.append(int(700*rank_ratio/0.01))
                #elif rank_ratio <= 0.02:
                #    new_rank_list.append(int(700*1+700*(rank_ratio-0.01)/0.01))
                #elif rank_ratio <= 0.04:
                #    new_rank_list.append(int(700*2+700*(rank_ratio-0.02)/0.02))
                #elif rank_ratio <= 0.06:
                #    new_rank_list.append(int(700*3+700*(rank_ratio-0.04)/0.02))
                #elif rank_ratio <= 0.08:
                #    new_rank_list.append(int(700*4+700*(rank_ratio-0.06)/0.02))
                #elif rank_ratio <= 0.10:
                #    new_rank_list.append(int(700*5+700*(rank_ratio-0.08)/0.02))
                #else:
                #    new_rank_list.append(int(700*6+700*(rank_ratio-0.10)/0.9))

                if rank_ratio == 0:
                    new_rank_list.append(1)
                elif rank_ratio <= 0.01:
                    new_rank_list.append(int((5020/6)*rank_ratio/0.01))
                elif rank_ratio <= 0.02:
                    new_rank_list.append(int((5020/6)*1+(5020/6)*(rank_ratio-0.01)/0.01))
                elif rank_ratio <= 0.03:
                    new_rank_list.append(int((5020/6)*2+(5020/6)*(rank_ratio-0.02)/0.01))
                elif rank_ratio <= 0.04:
                    new_rank_list.append(int((5020/6)*3+(5020/6)*(rank_ratio-0.03)/0.01))
                elif rank_ratio <= 0.05:
                    new_rank_list.append(int((5020/6)*4+(5020/6)*(rank_ratio-0.04)/0.01))
                else:
                    new_rank_list.append(int((5020/6)*5+(5020/6)*(rank_ratio-0.05)/0.95))

            df_plot['Average Rank'] = new_rank_list

            max_rank_value = 5020
            min_rank_value = 1
            df_plot['Average Rank'] = max_rank_value - df_plot['Average Rank'] + min_rank_value

            ynames = None
            if Use_Given_PValue_and_FC:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
            else:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']
            ys = np.array(df_plot.values.tolist())

            if Use_Given_PValue_and_FC:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 5020]], axis=0)
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)

                ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
            else:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 5020]], axis=0)
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 0]], axis=0)

                ys[:, -5:-1] = 1/(1.5 - (ys[:, -5:-1] ** 4))

            noise_upper_bound = 1.05
            noise_lower_bound = 0.95
            noise = np.random.normal(loc=0, scale=np.ones((ys.shape[0], 4)) * 0.06, size=ys[:, :4].shape)
            ys[:, :4] = ys[:, :4] + noise 


            ymins = ys.min(axis=0)
            ymaxs = ys.max(axis=0)
            dys = ymaxs - ymins
            ymins -= dys * 0.02 
            ymaxs += dys * 0.02

            zs = np.zeros_like(ys)
            zs[:, 0] = ys[:, 0]
            zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

            for plot_index in range(2):

                fig, host = plt.subplots(figsize=(12, 5))

                axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
                for i, ax in enumerate(axes):
                
                    if i == 0:
                        ax.set_yticks(range(len(Imputation)))
                        ax.set_yticklabels(Imputation, fontsize=10, rotation = 250, ha = 'right', va = 'top')


                    if i>=5:
                        ax.set_ylim(ymins[i], ymaxs[i])
                    elif i == 4:
                        ax.set_ylim(ymins[i], ymaxs[i])
                    else:
                        ax.set_ylim(ymins[i], ymaxs[i])

                    ax.spines["top"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.spines['right'].set_linewidth(2) 
                    ax.tick_params(axis="y", width=2, labelsize=10) 

                    if plot_index == 0:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)


                    if ax != host:


                        ax.spines["left"].set_visible(False)

                        ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
                        ax.tick_params(axis="y", width=2, labelsize=10, direction='in' )

                        if i == 1:
                            ax.set_yticks(range(len(Normalization)))
                            ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 2:
                            ax.set_yticks(range(len(BatchCorrection)))
                            ax.set_yticklabels(BatchCorrection, fontsize=10, rotation = 250, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 3:
                            ax.set_yticks(range(len(StatisticalTest)))
                            ax.set_yticklabels(StatisticalTest, fontsize=10, rotation = 250, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 4:
                            ax.set_yticks(np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5))
                            ytick_label = np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5).tolist()
                            
                            str_list = []
                            for data in ytick_label:
                                str_list.append('%.3f' % data)

                            ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)


                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                        #elif i == 9:
                        elif (i == (len(columns)-1)):
                            

                            ax.set_yticks([0, (5020/6)*1, (5020/6)*2, (5020/6)*3, (5020/6)*4, (5020/6)*5, 5020])
                            str_list = ['100%', '5%', '4%', '3%', '2%', '1%', '0%']

                            ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)


                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        else:

                            ax.set_yticks(np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5))
                            ytick_label = np.linspace(ymins[i] + dys[i] * 0.02, ymaxs[i] - dys[i] * 0.02, 5).tolist()
                            float_list = []
                            for label in ytick_label:
                                float_list.append(np.sqrt(np.sqrt(1.5 - 1/round(label,14))))

                            str_list = []
                            for data in float_list:
                                str_list.append('%.3f' % data)
                            ax.set_yticklabels(str_list, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                            

                host.set_xlim(0, ys.shape[1] - 1)
                host.set_xticks(range(ys.shape[1]))
                host.set_xticklabels(ynames, fontsize=12, rotation = 270, ha = 'left', va = 'top') 
                host.tick_params(axis="x", which="major", pad=0, width=2, labelsize=13) 
                host.tick_params(axis="y", width=2, labelsize=10, direction='out') 
                host.spines["right"].set_visible(False)
                host.spines['left'].set_linewidth(2) 
                host.xaxis.tick_top()


                if plot_index == 0:
                    host.set_xticks([])
                    host.set_yticks([])
                    host.spines['right'].set_visible(False)
                    host.spines['left'].set_visible(False)


                for tuceng in range(3):

                    for j in range(ys.shape[0] - 2):
                        verts = list(
                            zip(
                                [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                                np.repeat(zs[j, :], 3)[1:-1],
                            )
                        )
                        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                        path = Path(verts, codes)


                        edgecolor = '#e95b1b'
                        alpha = 0.85
                        linewidth = 0.95


                        # No of current method combination
                        current_No = df_screened['No'].values.tolist()[j]
                        current_No_rank = None

                        if (current_No in KeepNA_No_list):

                            current_No_rank = KeepNA_No_list.index(current_No)/(len(KeepNA_No_list))

                            # Top3 Red Bold
                            if (KeepNA_No_list.index(current_No) <= 2):
                                if (tuceng != 2):
                                    continue
                                edgecolor = '#b0203f'
                                linewidth = 2.0
                                alpha = 1.0
                                
                            # Top 1% Orange
                            #elif ( current_No_rank < 0.01):
                            elif (KeepNA_No_list.index(current_No) <= len(UseCovariate_No_list)*0.01):
                                if (tuceng != 1):
                                    continue
                                edgecolor = '#e95b1b'
                                alpha = 0.85
                                linewidth = 0.95

                            # Other gray
                            else:
                                if (tuceng != 0):
                                    continue
                                edgecolor = '#d9d8d2'
                                alpha = 0.2
                                linewidth = 0.2


                        if (current_No in UseCovariate_No_list):

                            current_No_rank = UseCovariate_No_list.index(current_No)/(len(UseCovariate_No_list))

                            # Top3 Purple Bold
                            if (UseCovariate_No_list.index(current_No) <= 2):
                                if (tuceng != 2):
                                    continue
                                edgecolor = '#a14ee0'
                                linewidth = 2.0
                                alpha = 1.0
                                
                            # Top 1% Blue
                            elif ( current_No_rank < 0.01):
                                if (tuceng != 1):
                                    continue
                                edgecolor = '#1883b8'
                                alpha = 0.85
                                linewidth = 0.95

                            # Other gray
                            else:
                                if (tuceng != 0):
                                    continue
                                edgecolor = '#d9d8d2'
                                alpha = 0.2
                                linewidth = 0.2



                        if plot_index == 0:

                            if current_No_rank > 0.05:
                                patch = patches.PathPatch(
                                    path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                )
                                #legend_handles[j] = patch
                                host.add_patch(patch)

                        else:
                            if current_No_rank <= 0.05:
                                patch = patches.PathPatch(
                                    path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                )
                                #legend_handles[j] = patch
                                host.add_patch(patch)
                
                if plot_index == 0:
                    host.set_rasterized(True)


                plt.subplots_adjust(left=0.044, right=0.98, bottom=0.07, top=0.98, wspace=0.05)

                if plot_index == 0:
                    svg1 = st.from_mpl(fig, savefig_kw=dict(transparent=True))
                if plot_index == 1:
                    svg2 = st.from_mpl(fig, savefig_kw=dict(transparent=True))

                
                plt.close()


            svg1.append(svg2)
            svg1.save(savefolder + 'StatisticalMetrics_PValue{0}_{1}_Average.svg'.format(str(P_value)[2:], SR_method).replace('PValue1', 'PValue10'))




    #（2）Stacked column chart

    Top_Percent_list = [0.01, 0.05]


    for P_value in pValue_List:
        for SR_method in SR_methods:
            for Top_Percent in Top_Percent_list:
                for Compared_groups in Compared_groups_label:

                    # Read csv file
                    df_result = pd.read_csv(result_csv_path, index_col=0) 
                    # Filter by p-value

                    df_p = df_result[df_result['p-value'] == P_value]
                    # Filter by SR
                    df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]

                    df_screened = df_p_SR.copy(deep = True)
                    # Sort by Rank of each comparison group in ascending order
                    df_sorted = df_screened.sort_values(Compared_groups + ' Rank', ascending=True)
                    # Filter out the top n% method combinations
                    df_sorted_top = df_sorted.head(int(Top_Percent*df_sorted.shape[0])) 

                    # Count the frequency of each missing value imputation method
                    Imputation_Frequency = []
                    for i in Imputation:
                        num_count = df_sorted_top[df_sorted_top['Missing Value Imputation'] == i].shape[0]
                        Imputation_Frequency.append(num_count/df_sorted_top.shape[0])
                    # Sort by frequency from highest to lowest
                    sorted_with_index = sorted(zip(Imputation_Frequency, range(len(Imputation_Frequency))), key=lambda x: x[0], reverse=True)
                    sorted_list, original_indexes = zip(*sorted_with_index)
                    # Convert the index to a list
                    original_indexes = list(original_indexes)

                    Imputation_Frequency_Sorted = sorted_list
                    Imputation_Frequency_Sorted_Colors = []
                    for j in original_indexes:
                        Imputation_Frequency_Sorted_Colors.append(colors2[j])


                    # Count the frequency of each normalization method
                    Normalization_Frequency = []
                    for i in Normalization:
                        num_count = df_sorted_top[df_sorted_top['Normalization'] == i].shape[0]
                        Normalization_Frequency.append(num_count/df_sorted_top.shape[0])
                    
                    sorted_with_index = sorted(zip(Normalization_Frequency, range(len(Normalization_Frequency))), key=lambda x: x[0], reverse=True)
                    sorted_list, original_indexes = zip(*sorted_with_index)
                    
                    original_indexes = list(original_indexes)

                    Normalization_Frequency_Sorted = sorted_list
                    Normalization_Frequency_Sorted_Colors = []
                    for j in original_indexes:
                        Normalization_Frequency_Sorted_Colors.append(colors2[j])


                    # Count the frequency of each batch correction method
                    BatchCorrection_Frequency = []
                    for i in BatchCorrection:
                        num_count = df_sorted_top[df_sorted_top['Batch Correction'] == i].shape[0] 
                        BatchCorrection_Frequency.append(num_count/df_sorted_top.shape[0])
                    
                    sorted_with_index = sorted(zip(BatchCorrection_Frequency, range(len(BatchCorrection_Frequency))), key=lambda x: x[0], reverse=True)
                    sorted_list, original_indexes = zip(*sorted_with_index)
                    
                    original_indexes = list(original_indexes)

                    BatchCorrection_Frequency_Sorted = sorted_list
                    BatchCorrection_Frequency_Sorted_Colors = []
                    for j in original_indexes:
                        BatchCorrection_Frequency_Sorted_Colors.append(colors2[j])


                    # Count the frequency of each Statistical Test method
                    StatisticalTest_Frequency = []
                    for i in StatisticalTest:
                        num_count = df_sorted_top[df_sorted_top['Statistical Test'] == i].shape[0] 
                        StatisticalTest_Frequency.append(num_count/df_sorted_top.shape[0])
                    
                    sorted_with_index = sorted(zip(StatisticalTest_Frequency, range(len(StatisticalTest_Frequency))), key=lambda x: x[0], reverse=True)
                    sorted_list, original_indexes = zip(*sorted_with_index)
                    
                    original_indexes = list(original_indexes)

                    StatisticalTest_Frequency_Sorted = sorted_list
                    StatisticalTest_Frequency_Sorted_Colors = []
                    for j in original_indexes:
                        StatisticalTest_Frequency_Sorted_Colors.append(colors2[j])


                    # Draw stacked column chart
                    fig, ax = plt.subplots(1, 1, figsize=(1.8,5))
                    categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                    value = 0
                    count = 0
                    for i in Imputation_Frequency_Sorted:
                        plt.bar([0], [i], width = 0.8, bottom = [value], color = Imputation_Frequency_Sorted_Colors[count]) 
                        #if (value < 0.9999):
                        #    ax.text(0, value + i/2, Imputation[colors.index(Imputation_Frequency_Sorted_Colors[count])], ha = 'center', va = 'center', fontsize=6)
                        value += i
                        count += 1

                    value = 0
                    count = 0
                    for i in Normalization_Frequency_Sorted:
                        plt.bar([1], [i], width = 0.8, bottom = [value], color = Normalization_Frequency_Sorted_Colors[count]) 
                        #if (value < 0.9999):
                        #    ax.text(1, value + i/2, Normalization[colors.index(Normalization_Frequency_Sorted_Colors[count])], ha = 'center', va = 'center', fontsize=6)
                        value += i
                        count += 1

                    value = 0
                    count = 0
                    for i in BatchCorrection_Frequency_Sorted:
                        plt.bar([2], [i], width = 0.8, bottom = [value], color = BatchCorrection_Frequency_Sorted_Colors[count]) 
                        #if (value < 0.9999):
                        #    ax.text(2, value + i/2, BatchCorrection[colors.index(BatchCorrection_Frequency_Sorted_Colors[count])], ha = 'center', va = 'center', fontsize=6)
                        value += i
                        count += 1

                    value = 0
                    count = 0
                    for i in StatisticalTest_Frequency_Sorted:
                        plt.bar([3], [i], width = 0.8, bottom = [value], color = StatisticalTest_Frequency_Sorted_Colors[count]) 
                        #if (value < 0.9999):
                        #    ax.text(3, value + i/2, StatisticalTest[colors.index(StatisticalTest_Frequency_Sorted_Colors[count])], ha = 'center', va = 'center', fontsize=6)
                        value += i
                        count += 1


                    plt.ylim(0, 1) 
                    plt.yticks(np.linspace(0, 1, 5)) 
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14) 

                    ax.set_xticks(np.arange(len(categories)))
                    ax.set_xticklabels(categories, rotation = 90)

                    plt.tick_params(labelsize=14) 

                    plt.tick_params(axis='x', width=2)
                    plt.tick_params(axis='y', width=2)

                    axes = plt.gca()

                    plt.ylabel('Frequency', fontsize=16)
                    axes.spines['left'].set_bounds((0, 1)) 
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(True)
                    axes.spines['left'].set_visible(True)
                    axes.spines['bottom'].set_linewidth(2) 
                    axes.spines['left'].set_linewidth(2) 
                    plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                    plt.subplots_adjust(left=0.53, right=0.98, bottom=0.33, top=0.95, wspace=0.1)

                    plt.savefig(savefolder + 'MethodProportion_Top{0}_{1}_PValue{2}_{3}.svg'.format(str(Top_Percent)[2:], SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_'), str(Top_Percent)).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                    #plt.show()
                    plt.close()


    # Generate Legend

    Steps = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

    Dict_Steps = {'Imputation': ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute', 'KeepNA'],
                  'Normalization': ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                  'Batch Correction': ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama', 'limma-NC', 'Combat-P-NC', 'Combat-NP-NC'],
                  'Statistical Test': ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']}


    #colors2 = ['#fb8072', '#1965b0', '#7bafde', '#882e72', '#b17ba6', '#ff7f00', '#fdb462', '#e7298a']

    for step in Steps:
        methods = Dict_Steps[step]

        fig_legend = plt.figure(figsize=(2.5, 4))
        axes = plt.gca()

        for method in methods:
            axes.bar([0], [1], width = 0.85, bottom = [100], color = colors2[methods.index(method)]) 


        axes.legend(labels = methods, title = step, title_fontsize=18, fontsize=16, 
                    loc = 'center',
                    labelspacing=1.0, 
                    markerfirst=True, markerscale=2.0) 

        plt.ylim(-5, 5)
        plt.xlim(-5, 5)

        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])

        plt.savefig(savefolder + 'Legend_{0}.svg'.format(step), dpi=600, format="svg", transparent=True) 
        #plt.show()
        plt.close()



if __name__ == '__main__': 
    KeepNA_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts(
    
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates/Results/')


