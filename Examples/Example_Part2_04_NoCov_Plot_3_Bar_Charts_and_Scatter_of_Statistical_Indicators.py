

import math
import numpy as np
import pandas as pd

import matplotlib as matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["font.sans-serif"] = ["Arial"] 
matplotlib.rcParams["axes.unicode_minus"] = False


# Draw bar charts and scatter of statistical indicators
def NoCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators(
    
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-NC.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_NoCovariates/Results/'):

    
    #pValue_List = [0.001, 0.01, 0.05, 0.1]
    pValue_List = [0.05]
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Compared_groups_label = ['S4/S2', 'S5/S1']

    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN']
    BatchCorrection = ['NoBC', 'limma-NC', 'Combat-P-NC', 'Combat-NP-NC', 'Scanorama']

    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']


    #colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890', '#fbe29d', '#c8d961']
    marker_list = ['o', '^', 's', 'd', '*', 'v', '<', '>']

    # Read Data
    df_result = pd.read_csv(result_csv_path, index_col=0) 


    for pValue in pValue_List:
        for SR in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]
                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)
                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method

                df_p_SR_Imputation = df_p_SR[df_p_SR['Missing Value Imputation'] == top1_Imputation_method]  

                fig, ax = plt.subplots(len(Normalization), len(BatchCorrection), figsize=(5,5))


                # Count the maximum and minimum values ​​of the subgraph data
                max_value = None
                min_value = None
                for BC in BatchCorrection:
                    for Norm in Normalization:
                        for ST in StatisticalTest:
                            filtered_df = df_p_SR_Imputation[df_p_SR_Imputation['Batch Correction'] == BC]
                            filtered_df = filtered_df[filtered_df['Normalization'] == Norm]
                            filtered_df = filtered_df[filtered_df['Statistical Test'] == ST]

                            F1_score = filtered_df[Compared_groups + ' F1-Score'].values.tolist()[0]
                            Accuracy = filtered_df[Compared_groups + ' Accuracy'].values.tolist()[0]
                            Recall = filtered_df[Compared_groups + ' Recall'].values.tolist()[0]
                            Precision = filtered_df[Compared_groups + ' Precision'].values.tolist()[0]

                            this_max = max([F1_score, Accuracy, Recall, Precision])
                            this_min = min([F1_score, Accuracy, Recall, Precision])

                            if max_value is None:
                                max_value = this_max
                                min_value = this_min
                            else:
                                if (this_max > max_value):
                                    max_value = this_max
                                if (this_min < min_value):
                                    min_value = this_min

                
                new_max_value = round(math.ceil(max_value/0.2)*0.2, 1)
                new_min_value = round(math.floor(min_value/0.2)*0.2, 1)


                for BC in BatchCorrection:
                    for Norm in Normalization:

                        plot_index = (len(BatchCorrection) - BatchCorrection.index(BC) -1)*len(Normalization) + Normalization.index(Norm) + 1
                        plt.subplot(len(BatchCorrection), len(Normalization), plot_index)

                        # Statistics of drawing data for each subgraph
                        F1_score_list = []
                        Accuracy_list = []
                        Recall_list = []
                        Precision_list = []
                        for ST in StatisticalTest:
                            filtered_df = df_p_SR_Imputation[df_p_SR_Imputation['Batch Correction'] == BC]
                            filtered_df = filtered_df[filtered_df['Normalization'] == Norm]
                            filtered_df = filtered_df[filtered_df['Statistical Test'] == ST]

                            F1_score = filtered_df[Compared_groups + ' F1-Score'].values.tolist()[0]
                            Accuracy = filtered_df[Compared_groups + ' Accuracy'].values.tolist()[0]
                            Recall = filtered_df[Compared_groups + ' Recall'].values.tolist()[0]
                            Precision = filtered_df[Compared_groups + ' Precision'].values.tolist()[0]

                            F1_score_list.append(F1_score)
                            Accuracy_list.append(Accuracy)
                            Recall_list.append(Recall)
                            Precision_list.append(Precision)

                        # Drawing subgraphs
                        count = 0
                        for F1_score in F1_score_list:
                            Init_color = '#1965b0'

                            # Color transparency range 0.3-1
                            def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                                return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                            # If ARI is negative, it is not displayed
                            if (F1_score >= 0):
                                Init_color_alpha = map_to_range(F1_score)*map_to_range(F1_score)
                            else:
                                Init_color_alpha = 0

                            plt.bar([count], [F1_score], width = 0.80, bottom = 0, color = Init_color, alpha = Init_color_alpha) # F1-score
                            count += 1

                        # Accuracy
                        plt.plot(list(range(len(StatisticalTest))), Accuracy_list, marker = marker_list[0], markersize = 2, linewidth = 0.5, color = 'black')
                        # Recall
                        plt.plot(list(range(len(StatisticalTest))), Recall_list, marker = marker_list[1], markersize = 2, linewidth = 0.5, color = 'black')
                        # Precision
                        plt.plot(list(range(len(StatisticalTest))), Precision_list, marker = marker_list[2], markersize = 2, linewidth = 0.5, color = 'black')


                        plt.tick_params(labelsize=10) 
                        plt.tick_params(axis='x', width=2)
                        plt.tick_params(axis='y', width=2)

                        #plt.ylim(-0.05, 1.05) 
                        #plt.yticks(np.linspace(0, 1, 3)) 
                        plt.ylim(-0.05 + new_min_value, 0.05 + new_max_value)
                        plt.yticks(np.linspace(new_min_value, new_max_value, 3)) 

                        axes = plt.gca()
                        axes.spines['top'].set_visible(False) 
                        axes.spines['right'].set_visible(False)
                        axes.spines['bottom'].set_visible(True)
                        axes.spines['left'].set_visible(False)
                        axes.spines['bottom'].set_linewidth(2) 
                        axes.spines['left'].set_linewidth(2) 

                        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 

                        if (BatchCorrection.index(BC) == 4):
                            title_name = Norm

                            if title_name == 'TRQN':
                                title_name = '      TRQN      '
                            if title_name == 'QN':
                                title_name = '        QN        '
                            if title_name == 'Sum':
                                title_name = '       Sum       '
                            if title_name == 'Median':
                                title_name = '     Median     '
                            if title_name == 'Unnormalized':
                                title_name = 'Unnormalized'

                            bbox = plt.title(title_name, horizontalalignment='center', verticalalignment='center', fontsize=8, pad = 15, bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1))
                    

                        if (Normalization.index(Norm) == 0):

                            axes.spines['left'].set_visible(True) 
                            plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 


                        if (Normalization.index(Norm) == 4):
                            axes.yaxis.set_label_position('right') 

                            y_label = BC
                    
                            if y_label == 'NoBC':
                                y_label = '      NoBC      '
                            if y_label == 'limma':
                                y_label = '      limma      '
                            if y_label == 'Combat-P':
                                y_label = '   Combat-P   '
                            if y_label == 'Combat-NP':
                                y_label = '  Combat-NP  '
                            if y_label == 'Scanorama':
                                y_label = '  Scanorama  '
                            if y_label == 'limma-NC':
                                y_label = '   limma-NC   '
                            if y_label == 'Combat-P-NC':
                                y_label = 'Combat-P-NC'
                            if y_label == 'Combat-NP-NC':
                                y_label = 'Combat-NP-NC'

                            fontsize=8
                            if y_label == 'Combat-NP-NC':
                                fontsize=7
                            plt.ylabel(y_label, rotation = 270, horizontalalignment='center', verticalalignment='center', fontsize=fontsize, labelpad=15, 
                                        bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1)) 

                plt.subplots_adjust(left=0.08, right=0.90, bottom=0.01, top=0.90, wspace=0.05)

                plt.savefig(savefolder + 'StatisticalTest_Metrics_{0}_{1}_PValue{2}_{3}.svg'.format(SR, top1_Imputation_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 
                #plt.show()
                plt.close()


    # Draw Legend

    fig_legend = plt.figure(figsize=(2.5, 2.5))
    axes = plt.gca()

    for i in range(3):
        parts = axes.scatter([100], [100], color = 'black', edgecolor='white', marker = marker_list[i], s = 4, alpha=1)
            
                    
    axes.legend(labels = ['Accuracy', 'Recall', 'Precision'], title='Metrics', title_fontsize=18, fontsize=16, 
                loc = 'center',
                labelspacing=1.0,  
                markerfirst=True, markerscale=6.0) 

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    axes.spines['top'].set_visible(False) 
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    plt.savefig(savefolder + 'Legend_StatisticalTest_Metrics.svg', dpi=600, format="svg", transparent=True) 
    #plt.show()
    plt.close()




if __name__ == '__main__': 
    NoCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators(
        
        result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-NC.csv",
        savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_NoCovariates/Results/')

