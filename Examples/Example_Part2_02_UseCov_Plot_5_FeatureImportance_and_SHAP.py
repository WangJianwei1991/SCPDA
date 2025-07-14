

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRanker

import shap
import math
import random


# Plot feature importance graphs and SHAP related graphs
def UseCov_Plot_5_FeatureImportance_and_SHAP(
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/',

    pValue_List = [0.001, 0.01, 0.05, 0.1],
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'],
    Compared_groups_label = ['S5/S1', 'S4/S2'],


    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
    BatchCorrection = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']):
    


    for P_value in pValue_List:
        for SR_method in SR_methods:


            # Read csv file
            df_result = pd.read_csv(result_csv_path, index_col=0) 
            # Filter by p-value
            df_p = df_result[df_result['p-value'] == P_value]
            # Filter by SR
            df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]

            for Compared_groups in Compared_groups_label:

                # Sort by each comparison group's Rank in ascending order
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)

                # Linearly transform the Rank data to 0-1
                df_p_SR[Compared_groups + ' Score'] = 1.0 - (df_p_SR[Compared_groups + ' Rank'] - df_p_SR[Compared_groups + ' Rank'].min()) / (df_p_SR[Compared_groups + ' Rank'].max() - df_p_SR[Compared_groups + ' Rank'].min())

                columns_to_extract = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test', Compared_groups + ' Score']
                data = df_p_SR[columns_to_extract]


                encoder = OrdinalEncoder()
                encoded_data = encoder.fit_transform(data.drop(Compared_groups + ' Score', axis=1))

                seed = 1994
                rng = np.random.default_rng(seed)
                n_query_groups = 1
                qid = rng.integers(0, 1, size=encoded_data.shape[0])

                sorted_idx = np.argsort(qid)
                X = encoded_data[sorted_idx, :]
                y = data[Compared_groups + ' Score'].values[sorted_idx]
                qid = qid[sorted_idx]

                # Training the ranking model
                model = XGBRanker(objective='rank:pairwise')
                model.fit(X, y, qid=qid)
                feature_importances = model.feature_importances_ 

                # Draw importance bar chart
                #fig, ax = plt.subplots(1, 1, figsize=(3.1,5))
                fig, ax = plt.subplots(1, 1, figsize=(2.5, 5))
                categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                feature_importances_x100 = [item * 100 for item in feature_importances]

                # Save CSV
                df_save = pd.DataFrame({'Step': categories, 'Importance': feature_importances_x100})
                df_save.to_csv(savefolder + 'FeatureImportance_{0}_PValue{1}_{2}.csv'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), index=False)

                # Y-axis range
                y_min_ = min(feature_importances_x100)
                y_max_ = max(feature_importances_x100)
                y_min = 0
                y_max = math.ceil(y_max_/20)*20 + 10

                plt.ylim(0, y_max)
                yticks = list(range(0, y_max, 20))
                plt.yticks(yticks)

                yticklabels = []
                for i in yticks:
                    yticklabels.append(str(i) + '%')
                ax.set_yticklabels(yticklabels, fontsize=14)

                color_list = ['#1965b0', '#1965b0', '#1965b0', '#1965b0']

                # Color transparency range 
                def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                    return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                for i in range(4):
                    color_alpha = map_to_range(feature_importances_x100[i], in_min=y_min_, in_max=y_max_, out_min=0.3, out_max=1)
                    plt.bar([i], [feature_importances_x100[i]], width = 0.8, bottom = 0, color = color_list[i], alpha = color_alpha) 
                    # Label number
                    plt.text(i, feature_importances_x100[i] + 1, '{0}%'.format(str(round(feature_importances_x100[i],1))), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
            
                #plt.ylim(0, 1) 
                #plt.yticks(np.linspace(0, 1, 5)) 
                #ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14)  # , rotation = 270, ha = 'right', va = 'top'

                ax.set_xticks(np.arange(len(categories)))
                ax.set_xticklabels(categories, rotation = 90, fontsize=14)

                plt.tick_params(axis="y", labelsize=14) 

                #plt.tick_params(labelsize=14) 

                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes = plt.gca()

                plt.ylabel('Importance', fontsize=16)
                # Move the Y-axis title to the specified position (e.g., x=0.5, y=0.5)
                ax.yaxis.set_label_coords(-0.5, 0.5)

                #axes.spines['left'].set_bounds((0, 1)) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                plt.subplots_adjust(left=1/2.5, right=1 - 0.05/2.5, bottom=0.36, top=0.99, wspace=0.1)
                #plt.subplots_adjust(left=0.20, right=0.99, bottom=0.36, top=0.99, wspace=0.1)

                plt.savefig(savefolder + 'FeatureImportance_{0}_PValue{1}_{2}.svg'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()



                #    >>>>SHAP<<<<

                encoded_data_top5 = encoded_data[0:(int(encoded_data.shape[0]*0.25)), :]  # encoded_data takes the first 25%
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(encoded_data_top5)
                #print(shap_values)

                # Draw a histogram of the SHAP value at each step (the horizontal axis is the step, and the vertical axis is the average of the absolute value of SHAP at each step)
                abs_shap_values = abs(shap_values)
                y_for_plot = [np.mean(abs_shap_values[:, 0]), np.mean(abs_shap_values[:, 1]), np.mean(abs_shap_values[:, 2]), np.mean(abs_shap_values[:, 3])]
                categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                bar_colors = ['#1965b0', '#1965b0', '#1965b0', '#1965b0']

                # Steps are sorted from largest to smallest
                sorted_with_index = sorted(zip(y_for_plot, range(len(y_for_plot))), key=lambda x: x[0], reverse=True)
                sorted_y_for_plot, original_indexes = zip(*sorted_with_index)
                
                original_indexes = list(original_indexes)
                sorted_categories = []
                sorted_bar_colors = []
                for j in original_indexes:
                    sorted_categories.append(categories[j])
                    sorted_bar_colors.append(bar_colors[j])


                # Save CSV
                df_save = pd.DataFrame({'Step': sorted_categories, 'Mean |SHAP Value|': sorted_y_for_plot})
                df_save.to_csv(savefolder + 'SHAPSummary_{0}_PValue{1}_{2}.csv'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), index=False)



                fig, ax = plt.subplots(1, 1, figsize=(2.5,5))


                # Color transparency range 
                def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                    return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                for i in range(4):
                    color_alpha = map_to_range(sorted_y_for_plot[i], in_min=min(sorted_y_for_plot), in_max=max(sorted_y_for_plot), out_min=0.3, out_max=1)
                    plt.bar([i], [sorted_y_for_plot[i]], width = 0.8, bottom = 0, color = bar_colors[i], alpha = color_alpha) 
                    # Label number
                    plt.text(i, sorted_y_for_plot[i] + 0.05, '{0}'.format(str(round(sorted_y_for_plot[i],2))), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 

    
                #plt.bar([0,1,2,3], sorted_y_for_plot, width = 0.8, bottom = 0, color = sorted_bar_colors) 
                y_max = math.ceil(max(sorted_y_for_plot))
                plt.ylim(0, y_max*1.2) 
                plt.yticks(np.linspace(0, y_max, 5)) 
                #ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14)  # , rotation = 270, ha = 'right', va = 'top'

                ax.set_xticks(np.arange(len(sorted_categories)))
                ax.set_xticklabels(sorted_categories, rotation = 90, fontsize=14)

                plt.tick_params(axis="y", labelsize=14) 

                #plt.tick_params(labelsize=14) 

                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes = plt.gca()

                plt.ylabel('Mean |SHAP Value|', fontsize=16)
                # Move the Y-axis title to the specified position (e.g., x=0.5, y=0.5)
                ax.yaxis.set_label_coords(-0.5, 0.5)

                #axes.spines['left'].set_bounds((0, 1)) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                plt.subplots_adjust(left=1/2.5, right=1 - 0.05/2.5, bottom=0.36, top=0.99, wspace=0.1)
                #plt.subplots_adjust(left=0.20, right=0.99, bottom=0.36, top=0.99, wspace=0.1)


                plt.savefig(savefolder + 'SHAPSummary_{0}_PValue{1}_{2}.svg'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()



                # For each step, draw a box plot of the SHAP values ​​of different methods, with the horizontal axis being the method and the vertical axis being the SHAP value, 
                # and sort them according to the frequency of occurrence of each method.
                categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                categories_name_in_df = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                methods_of_categories = {'Imputation': ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                         'Normalization': ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                         'Batch Correction': ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                         'Statistical Test': ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']}

                linecolor = ['#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0']

                # Top 25% Method Combinations
                df_p_SR_top5 = df_p_SR.head(int(df_p_SR.shape[0]*0.25))

                for step in categories:
                    # Count the number of times each method appears in the top 25% method combinations in this step
                    methods = methods_of_categories[step]  # List of methods for this step
                    methods_shap_values = [[]]*len(methods)  # List of SHAP values ​​for each method
                    methods_shap_values_count = [0]*len(methods)  # Number of SHAP values ​​for each method

                    for row in range(shap_values.shape[0]):
            
                        # the method used for this row of data in this step
                        method_of_this_step_and_row = df_p_SR_top5[categories_name_in_df[categories.index(step)]].values.tolist()[row]
                        # Take out the SHAP value and store it in methods_shap_values
                        methods_shap_values[methods_of_categories[step].index(method_of_this_step_and_row)] = methods_shap_values[methods_of_categories[step].index(method_of_this_step_and_row)] + [shap_values[row, categories.index(step)]]
                        # The number of SHAP values ​​of this method +1
                        methods_shap_values_count[methods_of_categories[step].index(method_of_this_step_and_row)] += 1

                    # Draw box plot based on the frequency of each method
                    sorted_with_index = sorted(zip(methods_shap_values_count, range(len(methods_shap_values_count))), key=lambda x: x[0], reverse=True)
                    sorted_methods_shap_values_count, original_indexes = zip(*sorted_with_index)
                    
                    original_indexes = list(original_indexes)
                    sorted_methods = []
                    sorted_methods_shap_values = []
                    sorted_linecolor = []
                    for j in original_indexes:
                        sorted_methods.append(methods[j])
                        sorted_methods_shap_values.append(methods_shap_values[j])
                        sorted_linecolor.append(linecolor[j])


                    # Draw box plot

                    fig, ax = plt.subplots(1, 1, figsize=(5,5))
                    axes = plt.gca()

        
                    # Remove embedded empty lists
                    filtered_list = [item for item in sorted_methods_shap_values if item]
                    y_min = math.floor(min(min(sublist) for sublist in filtered_list )) - 1
                    y_max = math.ceil(max(max(sublist) for sublist in filtered_list )) + 1

                    
                    df_list = []

                    x_label_num_count = 0

                    for j in range(len(sorted_methods)):

                        df_list.append(pd.DataFrame({'Step': [sorted_methods[j]]*len(sorted_methods_shap_values[j]),
                                                     'SHAP': sorted_methods_shap_values[j]}))

                        if (sorted_methods_shap_values[j] == []):
                            continue

                        if (sorted_methods_shap_values_count[j] < 10):
                            continue

                        x_label_num_count += 1

                        

                        b = axes.boxplot(sorted_methods_shap_values[j],
                                positions=[j+1], # position of the box
                                widths=0.5, 
                                meanline=False,
                                showmeans=False,
                                meanprops={'color': sorted_linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                                medianprops = {'color': sorted_linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                                showcaps = True,  
                                capprops = dict(color = sorted_linecolor[j]),
                                showfliers = False, 
                                patch_artist=True, 
                                boxprops = {'color':sorted_linecolor[j], 'facecolor':'#a3c1df', 'linewidth':'1.5'},
                                whiskerprops = {'color':sorted_linecolor[j], 'linewidth':'1.5'},
                                zorder=0
                                )

                        # Scatter plot
                        for i, d in enumerate([sorted_methods_shap_values[j]]):
                            np.random.seed(42) 
                            x = np.random.normal(j+1, 0.10, size=len(d))
                            plt.scatter(x, d, alpha=0.7, color='#4e4e4e', linewidths=0, s = 10, zorder=1) 

                        #plt.scatter([j+1]*len(sorted_methods_shap_values[j]), sorted_methods_shap_values[j], color=sorted_linecolor[j], s = 10) 

                        # Label the median
                        median = b['medians'][0].get_ydata()
                        add = ''
                        if (median[0] > 0):
                            add = '+'
                        #plt.text(j+1, y_max, r'$\it{Median}\rm{=' + '{0}'.format("{:.2f}".format(median[0])) + '}$', ha='center', va='top', rotation=90, size=12, family="Arial") 
                        plt.text(j+1, y_max + (y_max-y_min)*0.10, r'${' + add + '{0}'.format("{:.2f}".format(median[0])) + '}$', ha='center', va='top', rotation=90, size=12, family="Arial") 

                        # Label the number
                        plt.text(j+1, y_min - (y_max-y_min)*0.1, r'$\it{n}\rm{=' + '{0}'.format(str(sorted_methods_shap_values_count[j])) + '}$', ha='center', va='bottom', rotation=90, size=12, family="Arial") 

                    # Save CSV
                    df_total = pd.concat(df_list)
                    df_total.to_csv(savefolder + 'SHAP_{0}_{1}_PValue{2}_{3}.csv'.format(step, SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_'), step).replace('PValue1', 'PValue10'), index=False)

        
                    plt.ylim(y_min - (y_max-y_min)*0.15, y_max + (y_max-y_min)*0.15) 
                    yticks = list(range(y_min, y_max + 1, 2))
                    plt.yticks(yticks) 
                    yticklabels = []
                    for i in yticks:
                        if i > 0:
                            yticklabels.append('+' + str(i))
                        elif i ==0:
                            yticklabels.append(str(i))
                        elif i < 0:
                            yticklabels.append(r'${' + '-' + str(abs(i)) + '}$')

                    ax.set_yticklabels(yticklabels, fontsize=14)

                    ax.set_xticks(np.arange(x_label_num_count) + 1)
                    ax.set_xticklabels(sorted_methods[0:x_label_num_count], rotation = 90, fontsize=14)

                    #plt.tick_params(axis="y", labelsize=14) 

                    #plt.tick_params(labelsize=14) 

                    plt.tick_params(axis='x', width=2)
                    plt.tick_params(axis='y', width=2)

                    axes = plt.gca()

                    plt.ylabel('SHAP Value', fontsize=16)
                    # Move the Y-axis title to the specified position (e.g., x=0.5, y=0.5)
                    ax.yaxis.set_label_coords(-0.1835, 0.5)

                    axes.spines['left'].set_bounds((y_min - (y_max-y_min)*0.15, y_max + (y_max-y_min)*0.15)) 
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(True)
                    axes.spines['left'].set_visible(True)
                    axes.spines['bottom'].set_linewidth(2) 
                    axes.spines['left'].set_linewidth(2) 
                    plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                    #plt.subplots_adjust(left=0.86/3.1 + 0.1, right=0.98, bottom=0.36, top=0.95, wspace=0.1)
                    plt.subplots_adjust(left=0.20, right=0.99, bottom=0.36, top=0.99, wspace=0.1)

                    plt.savefig(savefolder + 'SHAP_{0}_{1}_PValue{2}_{3}.svg'.format(step, SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_'), step).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                    #plt.show()
                    plt.close()




                # Draw a bar chart with the step combinations on the horizontal axis and the mean absolute SHAP interaction value (Mean |SHAP Interaction Value|) on the vertical axis. 
                # The step combinations are sorted from large to small.

                shap_interaction = explainer.shap_interaction_values(encoded_data_top5)
                categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                # Step combination name
                #step_step_name_list = ['2-1', '3-1', '3-2', '4-1', '4-2', '4-3']
                step_step_name_list = ['Normalization\nImputation', 'Batch Correction\nImputation', 'Batch Correction\nNormalization', 'Statistical Test\nImputation', 'Statistical Test\nNormalization', 'Statistical Test\nBatch Correction']
                step_step_index_list = [[1,0], [2,0], [2,1], [3,0], [3,1], [3,2]]
                abs_shap_interaction_values = abs(shap_interaction)

                bar_colors = ['#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0']

                y_for_plot = []
                for index in step_step_index_list:
                    data = np.mean(abs_shap_interaction_values[:, index[0], index[1]])
                    y_for_plot.append(data)

                # The step combinations are sorted from large to small
                sorted_with_index = sorted(zip(y_for_plot, range(len(y_for_plot))), key=lambda x: x[0], reverse=True)
                sorted_y_for_plot, original_indexes = zip(*sorted_with_index)

                original_indexes = list(original_indexes)
                sorted_step_step_name_list = []
                sorted_step_step_index_list = []
                sorted_bar_colors = []
                for j in original_indexes:
                    sorted_step_step_index_list.append(step_step_index_list[j])
                    sorted_step_step_name_list.append(step_step_name_list[j])
                    sorted_bar_colors.append(bar_colors[j])

                # Save CSV
                df_save = pd.DataFrame({'Combination': sorted_step_step_name_list,
                                        'Mean |SHAP Interaction Value|': sorted_y_for_plot})

                df_save.to_csv(savefolder + 'SHAPInteractionSummary_{0}_PValue{1}_{2}.csv'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), index=False)


                fig, ax = plt.subplots(1, 1, figsize=(5,5))


                # Color transparency range 
                def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                    return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                for i in range(6):
                    color_alpha = map_to_range(sorted_y_for_plot[i], in_min=min(sorted_y_for_plot), in_max=max(sorted_y_for_plot), out_min=0.3, out_max=1)
                    plt.bar([i], [sorted_y_for_plot[i]], width = 0.8, bottom = 0, color = bar_colors[i], alpha = color_alpha) 
                    # Label number
                    plt.text(i, sorted_y_for_plot[i] + 0.05*max(sorted_y_for_plot), '{0}'.format(str(round(sorted_y_for_plot[i],2))), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 

    
                #plt.bar(list(range(6)), sorted_y_for_plot, width = 0.8, bottom = 0, color = sorted_bar_colors) 
                #y_max = math.ceil(max(sorted_y_for_plot))
                y_max = max(sorted_y_for_plot)
                plt.ylim(0, y_max*1.3) 

                yticks = list(np.arange(0, y_max*1.2, 0.2))
                plt.yticks(yticks) 

                #plt.yticks(np.linspace(0, y_max, 5)) 
                #ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14)  # , rotation = 270, ha = 'right', va = 'top'

                ax.set_xticks(np.arange(len(sorted_step_step_name_list)))
                #ax.set_xticklabels(sorted_step_step_name_list)
                ax.set_xticklabels(sorted_step_step_name_list, rotation = 90, fontsize=14)

                plt.tick_params(axis="y", labelsize=14) 

                #plt.tick_params(labelsize=14) 

                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes = plt.gca()

                plt.ylabel('Mean |SHAP Interaction Value|', fontsize=14)

                ax.yaxis.set_label_coords(-0.1835, 0.5)

                #axes.spines['left'].set_bounds((0, 1)) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                #plt.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.95, wspace=0.1)
                plt.subplots_adjust(left=0.20, right=0.99, bottom=0.36, top=0.99, wspace=0.1)

                plt.savefig(savefolder + 'SHAPInteractionSummary_{0}_PValue{1}_{2}.svg'.format(SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()




                # For each step combination, draw a box plot of the SHAP interaction values ​​of different method combinations, 
                # with the horizontal axis method combination and the vertical axis SHAP interaction value, sorted by the frequency of occurrence of each method combination
                count = 0
                for step_step_index in sorted_step_step_index_list:

                    
                    First_Step_Step_Index = step_step_index
                    
                    Step_Step_Name = sorted_step_step_name_list[count]

                    count += 1

                    categories = ['Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                    categories_name_in_df = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test']

                    methods_of_categories = {'Imputation': ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                             'Normalization': ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                             'Batch Correction': ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                             'Statistical Test': ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']}

                    linecolor = ['#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0', '#1965b0']


                    # SHAP interaction values ​​for different method combinations
                    categorie_1 = categories[First_Step_Step_Index[0]]
                    categorie_2 = categories[First_Step_Step_Index[1]]

                    step_1_methods = methods_of_categories[categorie_1]
                    step_2_methods = methods_of_categories[categorie_2]


                    method_method_list = []
                    method_method_index_list = []
                    method_method_shap_interaction_values_list = [[]]*len(step_1_methods)*len(step_2_methods)  
                    method_method_shap_interaction_values_count = [0]*len(step_1_methods)*len(step_2_methods) 
    
                    for method1 in step_1_methods:
                        for method2 in step_2_methods:
                            method_method_list.append([method1, method2])
                            method_method_index_list.append([step_1_methods.index(method1), step_2_methods.index(method2)])


                    shap_interaction_values = shap_interaction[:, First_Step_Step_Index[0], First_Step_Step_Index[1]]  

                    for row in range(df_p_SR_top5.shape[0]):

                        method1_of_this_row = df_p_SR_top5[categories_name_in_df[First_Step_Step_Index[0]]].values.tolist()[row]
                        method2_of_this_row = df_p_SR_top5[categories_name_in_df[First_Step_Step_Index[1]]].values.tolist()[row]

                        shap_interaction_value = shap_interaction_values[row]


                        index = step_1_methods.index(method1_of_this_row)*len(step_2_methods) + step_2_methods.index(method2_of_this_row)
                        method_method_shap_interaction_values_list[index] = method_method_shap_interaction_values_list[index] + [shap_interaction_value]
                        method_method_shap_interaction_values_count[index] += 1


                    # Sort by frequency of method combinations
                    sorted_with_index = sorted(zip(method_method_shap_interaction_values_count, range(len(method_method_shap_interaction_values_count))), key=lambda x: x[0], reverse=True)
                    sorted_method_method_shap_interaction_values_count, original_indexes = zip(*sorted_with_index)
                    
                    original_indexes = list(original_indexes)
                    sorted_method_method_list = []
                    sorted_method_method_index_list = []
                    sorted_method_method_shap_interaction_values_list = []
                    for j in original_indexes:
                        sorted_method_method_list.append(method_method_list[j])
                        sorted_method_method_index_list.append(method_method_index_list[j])
                        sorted_method_method_shap_interaction_values_list.append(method_method_shap_interaction_values_list[j])

    
                    # Draw box plot

                    fig, ax = plt.subplots(1, 1, figsize=(5,5))
                    axes = plt.gca()


                    # Remove embedded empty lists
                    filtered_list = [item for item in sorted_method_method_shap_interaction_values_list if item]
                    y_min = math.floor(min(min(sublist) for sublist in filtered_list )) - 0
                    y_max = math.ceil(max(max(sublist) for sublist in filtered_list )) + 0

                    # Number of existing method combinations
                    current_nums = len(filtered_list)
                    # The maximum number of drawing method combinations is 6
                    max_plot_num = 6
                    if current_nums < max_plot_num:
                        max_plot_num = current_nums


                    df_list = []

                    x_label_num_count = 0

                    for j in range(len(sorted_method_method_list)):

                        df_list.append(pd.DataFrame({'Combination': [sorted_method_method_list[j][0] + '\n' + sorted_method_method_list[j][1]]*len(sorted_method_method_shap_interaction_values_list[j]),
                                                     'SHAP Interaction Value': sorted_method_method_shap_interaction_values_list[j]}))


                        if j >= max_plot_num:
                            continue

                        if (sorted_method_method_shap_interaction_values_count[j] < 10):
                            continue

                        x_label_num_count += 1

                        #if (sorted_method_method_shap_interaction_values_list[j] == []):
                        #    continue

                        
                        b = axes.boxplot(sorted_method_method_shap_interaction_values_list[j],
                                positions=[j+1], # position of the box
                                widths=0.5, 
                                meanline=False,
                                showmeans=False,
                                meanprops={'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                                medianprops = {'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                                showcaps = True,  
                                capprops = dict(color = linecolor[j]),
                                showfliers = False, 
                                patch_artist=True, 
                                boxprops = {'color':linecolor[j], 'facecolor':'#a3c1df', 'linewidth':'1.5'},
                                whiskerprops = {'color':linecolor[j], 'linewidth':'1.5'},
                                zorder = 0
                                )


                        # Scatter plot
                        for i, d in enumerate([sorted_method_method_shap_interaction_values_list[j]]):
                            np.random.seed(42) 
                            x = np.random.normal(j+1, 0.10, size=len(d))
                            plt.scatter(x, d, alpha=0.7, color='#4e4e4e', linewidths=0, s = 10, zorder=1)  


                        # Label the median
                        median = b['medians'][0].get_ydata()
                        add = ''
                        if (median[0] > 0):
                            add = '+'
                        #plt.text(j+1, y_max, r'$\it{Median}\rm{=' + '{0}'.format("{:.2f}".format(median[0])) + '}$', ha='center', va='top', rotation=90, size=12, family="Arial") 
                        plt.text(j+1, y_max + (y_max-y_min)*0.10, r'${' + add + '{0}'.format("{:.2f}".format(median[0])) + '}$', ha='center', va='top', rotation=90, size=12, family="Arial") 

                        # Label the number
                        plt.text(j+1, y_min - (y_max-y_min)*0.05, r'$\it{n}\rm{=' + '{0}'.format(str(sorted_method_method_shap_interaction_values_count[j])) + '}$', ha='center', va='bottom', rotation=90, size=12, family="Arial") 

        
                    # save CSV
                    df_total = pd.concat(df_list)
                    df_total.to_csv(savefolder + 'SHAPInteraction_{0}_{1}_{2}_PValue{3}_{4}.csv'.format(categories[step_step_index[0]], categories[step_step_index[1]], SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), index=False)


                    plt.ylim(y_min - (y_max-y_min)*0.10, y_max + (y_max-y_min)*0.10) 
                    yticks = np.linspace(y_min, y_max, 5)
                    plt.yticks(yticks) 
                    yticklabels = []
                    for i in yticks:
                        if i > 0:
                            yticklabels.append('+' + str(i))
                        elif i ==0:
                            yticklabels.append(str(i))
                        elif i < 0:
                            yticklabels.append(r'${' + '-' + str(abs(i)) + '}$')

                    ax.set_yticklabels(yticklabels, fontsize=14)

                    x_labels = []
                    for item in sorted_method_method_list:
                        x_labels.append(item[0] + '\n' + item[1])

                    ax.set_xticks(np.arange(x_label_num_count) + 1)
                    ax.set_xticklabels(x_labels[0:x_label_num_count], rotation = 90, fontsize=14)

                    #plt.tick_params(axis="y", labelsize=14) 

                    plt.tick_params(axis='x', width=2)
                    plt.tick_params(axis='y', width=2)

                    axes = plt.gca()

                    plt.ylabel('SHAP Interaction Value', fontsize=16)

                    ax.yaxis.set_label_coords(-0.1835, 0.5)

                    axes.spines['left'].set_bounds((y_min - (y_max-y_min)*0.10, y_max + (y_max-y_min)*0.10)) 
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(True)
                    axes.spines['left'].set_visible(True)
                    axes.spines['bottom'].set_linewidth(2) 
                    axes.spines['left'].set_linewidth(2) 
                    plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

                    #plt.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.97, wspace=0.1)
                    plt.subplots_adjust(left=0.20, right=0.99, bottom=0.36, top=0.99, wspace=0.1)

                    plt.savefig(savefolder + 'SHAPInteraction_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(categories[step_step_index[0]], categories[step_step_index[1]], SR_method, str(P_value)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                    #plt.show()
                    plt.close()


if __name__ == '__main__': 
    UseCov_Plot_5_FeatureImportance_and_SHAP(
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/')



