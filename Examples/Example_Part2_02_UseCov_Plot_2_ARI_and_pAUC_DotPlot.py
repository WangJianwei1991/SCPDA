

import os
import numpy as np
import pandas as pd

import matplotlib as matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["font.sans-serif"] = ["Arial"] 
matplotlib.rcParams["axes.unicode_minus"] = False



# Draw dot plot for ARI and pAUC
def UseCov_Plot_2_ARI_and_pAUC_DotPlot(
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/',


    pValue_List = [0.001, 0.01, 0.05, 0.10],
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'],
    Compared_groups_label = ['S4/S2', 'S5/S1'],

    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
    BatchCorrection = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']):



    # Read data
    df_result = pd.read_csv(result_csv_path, index_col=0) 


    # Such as: Imputation_Normalization_ARI_SR75_NoBC.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method

                df_p_SR_BC = df_p_SR[df_p_SR['Batch Correction'] == top1_BatchCorrection_method]  

    

                # Plotting data
                ARI_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,5))

                # Draw a dashed border around the best method
                best_x_index = Normalization.index(top1_Normalization_method)
                x_length = len(Normalization)
                best_y_index = Imputation.index(top1_Imputation_method)
                y_length = len(Imputation)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in Imputation:
                    for j in Normalization:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Missing Value Imputation'] == i]
                        filtered_df = filtered_df[filtered_df['Normalization'] == j]
                        ARI_data = filtered_df['ARI'].values.tolist()[0]
                        ARI_Data.append(ARI_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (ARI_data >= 0):
                            scatter_color_alpha = map_to_range(ARI_data)*map_to_range(ARI_data)
                            scatter_size = Init_scatter_size*ARI_data*ARI_data
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(Normalization)-1) + 0.7)
            
                plt.xticks(list(range(0, len(Normalization), 1)), Normalization)

                plt.ylim(-0.7, (len(Imputation)-1) + 0.7)
                plt.yticks(list(range(0, len(Imputation), 1)), Imputation)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.28, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Imputation_Normalization_ARI_{0}_{1}_PValue{2}_{3}.svg'.format(SR_method, top1_BatchCorrection_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/10))

        scatter_color_alpha = map_to_range((i+1)/10)*map_to_range((i+1)/10)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='ARI', title_fontsize=18, fontsize=16, 
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


    plt.savefig(savefolder + 'Legend_Imputation_Normalization_ARI.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()




    # Such as: Imputation_BatchCorrection_ARI_SR75_Sum.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method

                df_p_SR_Norma = df_p_SR[df_p_SR['Normalization'] == top1_Normalization_method]  



                # Plotting data
                ARI_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,5))

                # Draw a dashed border around the best method
                best_x_index = BatchCorrection.index(top1_BatchCorrection_method)
                x_length = len(BatchCorrection)
                best_y_index = Imputation.index(top1_Imputation_method)
                y_length = len(Imputation)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in Imputation:
                    for j in BatchCorrection:
                        filtered_df = df_p_SR_Norma[df_p_SR_Norma['Missing Value Imputation'] == i]
                        filtered_df = filtered_df[filtered_df['Batch Correction'] == j]
                        ARI_data = filtered_df['ARI'].values.tolist()[0]
                        ARI_Data.append(ARI_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (ARI_data >= 0):
                            scatter_color_alpha = map_to_range(ARI_data)*map_to_range(ARI_data)
                            scatter_size = Init_scatter_size*ARI_data*ARI_data
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(BatchCorrection)-1) + 0.7)
            
                plt.xticks(list(range(0, len(BatchCorrection), 1)), BatchCorrection)

                plt.ylim(-0.7, (len(Imputation)-1) + 0.7)
                plt.yticks(list(range(0, len(Imputation), 1)), Imputation)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.28, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Imputation_BatchCorrection_ARI_{0}_{1}_PValue{2}_{3}.svg'.format(SR_method, top1_Normalization_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/10))

        scatter_color_alpha = map_to_range((i+1)/10)*map_to_range((i+1)/10)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='ARI', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Imputation_BatchCorrection_ARI.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()






    # Such as: Normalization_BatchCorrection_ARI_SR75_RowMean.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method

                df_p_SR_Impu = df_p_SR[df_p_SR['Missing Value Imputation'] == top1_Imputation_method]  

    


                # Plotting data
                ARI_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,5*0.8))

                # Draw a dashed border around the best method
                best_x_index = BatchCorrection.index(top1_BatchCorrection_method)
                x_length = len(BatchCorrection)
                best_y_index = Normalization.index(top1_Normalization_method)
                y_length = len(Normalization)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')


                for i in Normalization:
                    for j in BatchCorrection:
                        filtered_df = df_p_SR_Impu[df_p_SR_Impu['Normalization'] == i]
                        filtered_df = filtered_df[filtered_df['Batch Correction'] == j]
                        ARI_data = filtered_df['ARI'].values.tolist()[0]
                        ARI_Data.append(ARI_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (ARI_data >= 0):
                            scatter_color_alpha = map_to_range(ARI_data)*map_to_range(ARI_data)
                            scatter_size = Init_scatter_size*ARI_data*ARI_data
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(BatchCorrection)-1) + 0.7)
            
                plt.xticks(list(range(0, len(BatchCorrection), 1)), BatchCorrection)

                plt.ylim(-0.7, (len(Normalization)-1) + 0.7)
                plt.yticks(list(range(0, len(Normalization), 1)), Normalization)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.33, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Normalization_BatchCorrection_ARI_{0}_{1}_PValue{2}_{3}.svg'.format(SR_method, top1_Imputation_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/10))

        scatter_color_alpha = map_to_range((i+1)/10)*map_to_range((i+1)/10)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='ARI', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Normalization_BatchCorrection_ARI.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()








    # pAUC Dot Plot
    # Such as: Imputation_Normalization_pAUC_SR75_Combat-P_DESeq2_S4_vs_S2.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method

                df_p_SR_BC = df_p_SR[df_p_SR['Batch Correction'] == top1_BatchCorrection_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Statistical Test'] == top1_StatisticalTest_method]  


                #  Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,5))

                # Draw a dashed border around the best method
                best_x_index = Normalization.index(top1_Normalization_method)
                x_length = len(Normalization)
                best_y_index = Imputation.index(top1_Imputation_method)
                y_length = len(Imputation)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in Imputation:
                    for j in Normalization:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Missing Value Imputation'] == i]
                        filtered_df = filtered_df[filtered_df['Normalization'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(Normalization)-1) + 0.7)
            
                plt.xticks(list(range(0, len(Normalization), 1)), Normalization)

                plt.ylim(-0.7, (len(Imputation)-1) + 0.7)
                plt.yticks(list(range(0, len(Imputation), 1)), Imputation)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.28, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Imputation_Normalization_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()



    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Imputation_Normalization_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()




    # Such as: Imputation_BatchCorrection_pAUC_SR90_Sum_DESeq2_S5_vs_S1.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method


                df_p_SR_BC = df_p_SR[df_p_SR['Normalization'] == top1_Normalization_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Statistical Test'] == top1_StatisticalTest_method]  


                # Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,5))

                # Draw a dashed border around the best method
                best_x_index = BatchCorrection.index(top1_BatchCorrection_method)
                x_length = len(BatchCorrection)
                best_y_index = Imputation.index(top1_Imputation_method)
                y_length = len(Imputation)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')


                for i in Imputation:
                    for j in BatchCorrection:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Missing Value Imputation'] == i]
                        filtered_df = filtered_df[filtered_df['Batch Correction'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(BatchCorrection)-1) + 0.7)
            
                plt.xticks(list(range(0, len(BatchCorrection), 1)), BatchCorrection)

                plt.ylim(-0.7, (len(Imputation)-1) + 0.7)
                plt.yticks(list(range(0, len(Imputation), 1)), Imputation)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.28, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Imputation_BatchCorrection_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_Normalization_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Imputation_BatchCorrection_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()



    # Such as: Normalization_BatchCorrection_pAUC_SR90_SoftImpute_DESeq2_S4_vs_S2.svg
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method


                df_p_SR_BC = df_p_SR[df_p_SR['Missing Value Imputation'] == top1_Imputation_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Statistical Test'] == top1_StatisticalTest_method]  


                # Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(4,4))

                # Draw a dashed border around the best method
                best_x_index = BatchCorrection.index(top1_BatchCorrection_method)
                x_length = len(BatchCorrection)
                best_y_index = Normalization.index(top1_Normalization_method)
                y_length = len(Normalization)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in Normalization:
                    for j in BatchCorrection:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Normalization'] == i]
                        filtered_df = filtered_df[filtered_df['Batch Correction'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(BatchCorrection)-1) + 0.7)
            
                plt.xticks(list(range(0, len(BatchCorrection), 1)), BatchCorrection)

                plt.ylim(-0.7, (len(Normalization)-1) + 0.7)
                plt.yticks(list(range(0, len(Normalization), 1)), Normalization)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.33, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Normalization_BatchCorrection_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_Imputation_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Normalization_BatchCorrection_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()



    # Imputation_StatisticalTest  Normalization_StatisticalTest  BatchCorrection_StatisticalTest
    # Imputation_StatisticalTest

    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method


                df_p_SR_BC = df_p_SR[df_p_SR['Normalization'] == top1_Normalization_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Batch Correction'] == top1_BatchCorrection_method]  


                # Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(5,5))

                # Draw a dashed border around the best method
                best_x_index = StatisticalTest.index(top1_StatisticalTest_method)
                x_length = len(StatisticalTest)
                best_y_index = Imputation.index(top1_Imputation_method)
                y_length = len(Imputation)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')


                for i in Imputation:
                    for j in StatisticalTest:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Missing Value Imputation'] == i]
                        filtered_df = filtered_df[filtered_df['Statistical Test'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(StatisticalTest)-1) + 0.7)
            
                plt.xticks(list(range(0, len(StatisticalTest), 1)), StatisticalTest)

                plt.ylim(-0.7, (len(Imputation)-1) + 0.7)
                plt.yticks(list(range(0, len(Imputation), 1)), Imputation)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.33, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Imputation_StatisticalTest_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_Normalization_method, top1_BatchCorrection_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Imputation_StatisticalTest_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()



    # Normalization_StatisticalTest
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method


                df_p_SR_BC = df_p_SR[df_p_SR['Missing Value Imputation'] == top1_Imputation_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Batch Correction'] == top1_BatchCorrection_method]  


                # Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(5,4))

                # Draw a dashed border around the best method
                best_x_index = StatisticalTest.index(top1_StatisticalTest_method)
                x_length = len(StatisticalTest)
                best_y_index = Normalization.index(top1_Normalization_method)
                y_length = len(Normalization)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in Normalization:
                    for j in StatisticalTest:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Normalization'] == i]
                        filtered_df = filtered_df[filtered_df['Statistical Test'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(StatisticalTest)-1) + 0.7)
            
                plt.xticks(list(range(0, len(StatisticalTest), 1)), StatisticalTest)

                plt.ylim(-0.7, (len(Normalization)-1) + 0.7)
                plt.yticks(list(range(0, len(Normalization), 1)), Normalization)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.33, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'Normalization_StatisticalTest_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_Imputation_method, top1_BatchCorrection_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_Normalization_StatisticalTest_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()




    # BatchCorrection_StatisticalTest
    for pValue in pValue_List:
        for SR_method in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]  # Filter by p-value

                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR_method]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)  # Sort by the rank of the comparison groups in ascending order

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test method


                df_p_SR_BC = df_p_SR[df_p_SR['Missing Value Imputation'] == top1_Imputation_method]  
                df_p_SR_BC = df_p_SR_BC[df_p_SR_BC['Normalization'] == top1_Normalization_method]  


                # Plotting data
                pAUC_Data = []

                fig, ax = plt.subplots(1, 1, figsize=(5,4))

                # Draw a dashed border around the best method
                best_x_index = StatisticalTest.index(top1_StatisticalTest_method)
                x_length = len(StatisticalTest)
                best_y_index = BatchCorrection.index(top1_BatchCorrection_method)
                y_length = len(BatchCorrection)
                x1 = [best_x_index-0.5, best_x_index-0.5, best_x_index+0.5, best_x_index+0.5, best_x_index-0.5]
                y1 = [0-0.7, y_length+0.7, y_length+0.7, 0-0.7, 0-0.7]
                x2 = [0-0.7, x_length+0.7, x_length+0.7, 0-0.7, 0-0.7]
                y2 = [best_y_index-0.5, best_y_index-0.5, best_y_index+0.5, best_y_index+0.5, best_y_index-0.5]
                plt.plot(x1, y1, color='gray', linestyle='--')
                plt.plot(x2, y2, color='gray', linestyle='--')

                for i in BatchCorrection:
                    for j in StatisticalTest:
                        filtered_df = df_p_SR_BC[df_p_SR_BC['Batch Correction'] == i]
                        filtered_df = filtered_df[filtered_df['Statistical Test'] == j]
                        pAUC_data = filtered_df[Compared_groups + ' pAUC'].values.tolist()[0]
                        pAUC_Data.append(pAUC_data)

                        Init_scatter_color = '#1965b0'
                        Init_scatter_size = 800

                        # Color transparency range 0.3-1
                        def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
                            return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

                        # If ARI is negative, it is not displayed
                        if (pAUC_data >= 0):
                            scatter_color_alpha = map_to_range(pAUC_data)*map_to_range(pAUC_data)
                            scatter_size = Init_scatter_size*pAUC_data*pAUC_data*100
                        else:
                            scatter_color_alpha = 0
                            scatter_size = 0

                        # Scatter
                        plt.scatter([j], [i], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)
                plt.xticks(rotation=90) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.xlim(-0.7, (len(StatisticalTest)-1) + 0.7)
            
                plt.xticks(list(range(0, len(StatisticalTest), 1)), StatisticalTest)

                plt.ylim(-0.7, (len(BatchCorrection)-1) + 0.7)
                plt.yticks(list(range(0, len(BatchCorrection), 1)), BatchCorrection)

                #plt.subplots_adjust(left=0.33, right=1, bottom=0.33, top=1, wspace=0.05)
                plt.subplots_adjust(left=0.37, right=1, bottom=0.37, top=1, wspace=0.05)

                save_folder = '{0}_PValue{1}_{2}/'.format(SR_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')
                if not os.path.exists(savefolder + save_folder):
                    os.makedirs(savefolder + save_folder)
                else:
                    print(f"The directory {savefolder + save_folder} already exists")

                plt.savefig(savefolder + save_folder + 'BatchCorrection_StatisticalTest_pAUC_{0}_{1}_{2}_PValue{3}_{4}.svg'.format(SR_method, top1_Imputation_method, top1_Normalization_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True) 


                #plt.show()
                plt.close()


    # Legend
    fig_legend = plt.figure(figsize=(2.5,5))
 
    axes = plt.gca()

    def map_to_range(value, in_min=0, in_max=0.1, out_min=0.3, out_max=1):
        return (out_max - out_min) * (value - in_min) / (in_max - in_min) + out_min

    labels = []
    for i in range(10):

        labels.append('{:.3f}'.format((i+1)/100))

        scatter_color_alpha = map_to_range((i+1)/100)*map_to_range((i+1)/100)

        Init_scatter_size = 800
        scatter_size = Init_scatter_size*(i+1)*(i+1)/100

        parts = axes.scatter([100], [100], color = '#1965b0', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

    axes.legend(labels = labels, title='pAUC', title_fontsize=18, fontsize=16, 
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

    plt.savefig(savefolder + 'Legend_BatchCorrection_StatisticalTest_pAUC.svg', dpi=600, format="svg", transparent=True) 

    #plt.show()
    plt.close()



if __name__ == '__main__': 
    UseCov_Plot_2_ARI_and_pAUC_DotPlot(
        result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
        savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/')

