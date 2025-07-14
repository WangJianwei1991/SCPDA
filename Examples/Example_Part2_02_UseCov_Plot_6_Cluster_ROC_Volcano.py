

import os
import numpy as np
import pandas as pd
import additional_plot_methods
import matplotlib as matplotlib
import matplotlib.pyplot as plt

from SCPDA import SCPDA

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["font.sans-serif"] = ["Arial"] 
matplotlib.rcParams["axes.unicode_minus"] = False



# Draw clustering, ROC and volcano plots
def UseCov_Plot_6_Cluster_ROC_Volcano(
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/',

    Software = 'DIANN',
    Report_Folder = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/",
    Report_Name = 'three_mix_report.pg_matrix.tsv',

    samples_csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Samples_Template.csv",
    composition_csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Composition_Template.csv",

    pValue_List = [0.001, 0.01, 0.05, 0.1],
    FC_For_Groups = [1.2, 1.3],
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'],
    Compared_groups_label = ['S4/S2', 'S5/S1']

    ):
    


     # >>>Parameters
    Reduction = 'pca'  # Dimensionality reduction method
    additional_plot_methods.Reduction = Reduction
    UseCovariates = True  # Whether to use covariates when correcting for batch effects
    # Whether to use the given FC and P value
    Use_Given_PValue_and_FC = True



    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN']
    BatchCorrection = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']


    # Read CSV
    df_result = pd.read_csv(result_csv_path, index_col=0) 



    for pValue in pValue_List:
        for SR in SR_methods:
            for Compared_groups in Compared_groups_label:

                df_p = df_result[df_result['p-value'] == pValue]
                df_p_SR = df_p[df_p['Sparsity Reduction'] == SR]
                df_p_SR = df_p_SR.sort_values(Compared_groups + ' Rank', ascending=True)

                top1_Imputation_method = df_p_SR['Missing Value Imputation'].values.tolist()[0] # Best Missing Value Imputation Method
                top1_Normalization_method = df_p_SR['Normalization'].values.tolist()[0] # Best Normalization Method
                top1_BatchCorrection_method = df_p_SR['Batch Correction'].values.tolist()[0] # Best Batch Correction Method
                top1_StatisticalTest_method = df_p_SR['Statistical Test'].values.tolist()[0] # Best Statistical Test Method

                # Create SCPDA object
                a = SCPDA(samples_csv_path = samples_csv_path,
                          composition_csv_path = composition_csv_path)

                # Using the given p-value and FC
                if (Use_Given_PValue_and_FC):
                    a.Use_Given_PValue_and_FC = True
                    a.Given_PValue = pValue
                    a.Given_FC = FC_For_Groups[Compared_groups_label.index(Compared_groups)]

                df_all, dict_species = None, None
                if (Software == 'DIANN'):
                    df_all, dict_species = a.Get_Protein_Groups_Data_From_DIANN_Report(FolderPath = Report_Folder, 
                                                                                       FileName = Report_Name, 
                                                                                       SaveResult = False)
                elif (Software == 'Spectronaut'):
                    df_all, dict_species = a.Get_Protein_Groups_Data_From_Spectronaut_Report(FolderPath = Report_Folder, 
                                                                                       FileName = Report_Name, 
                                                                                       SaveResult = False)
                elif (Software == 'PEAKS'):
                    df_all, dict_species = a.Get_Protein_Groups_Data_From_PeaksStudio_Report(FolderPath = Report_Folder, 
                                                                                       FileName = Report_Name, 
                                                                                       SaveResult = False)



                df = a.Sparsity_Reduction(df_all, method = SR)
                df = a.Missing_Data_Imputation(df, method = top1_Imputation_method)
                df = a.Data_Normalization(df, method = top1_Normalization_method)
                # Apply log2(x+1) to each element in the DataFrame
                df = df.apply(lambda x: np.log2(x+1))

                df = a.Batch_Correction(df, method = top1_BatchCorrection_method, UseCovariates = True)
                df, ari = a.Cluster_Analysis(df, savefig = True, savefolder = savefolder, 
                                             savename= 'Clustering_{0}_{1}_{2}_{3}'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method))


                if (top1_StatisticalTest_method == 'edgeR-QLF') | (top1_StatisticalTest_method == 'edgeR-LRT') | (top1_StatisticalTest_method == 'Limma-voom') | (top1_StatisticalTest_method == 'limma-voom'):
                    df = df.apply(lambda x: np.power(2, x)-1)
                if (top1_StatisticalTest_method == 'DESeq2') | (top1_StatisticalTest_method == 'DESeq2-parametric'):
                    df = df.apply(lambda x: np.power(2, x)-1)
                    if df.values.max() > 10000:
                        pass
                    else:
                        df = df.apply(lambda x: x*10000)

                a.FilterProteinMatrix = True 


                list_pauc_, list_pvalue_, list_log2fc_, list_TP_, list_TN_, list_FP_, list_FN_, list_accuracy_, list_precision_, list_recall_, list_f1_score_, df_list, list_overall_label_true_data_, list_overall_label_predict_data_ = a.Difference_Analysis(
                                        df, dict_species, method = top1_StatisticalTest_method,
                                        Compared_groups_label = [Compared_groups],
                                        title_methods = '{0}_{1}_{2}_{3}_{4}_PValue{5}'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:]).replace('PValue1', 'PValue10'),
                                        savefig = True,
                                        savefolder = savefolder)


                PR_Filenames = [savefolder + 'PR_{0}_{1}_{2}_{3}_{4}_PValue{5}_{6}.svg'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'),
                                savefolder + 'PR_{0}_{1}_{2}_{3}_{4}_PValue{5}_{6}.csv'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'),
                                savefolder + 'Legend_PR.svg']
            
                for filepath_to_delete in PR_Filenames:
                    if os.path.exists(filepath_to_delete):
                        os.remove(filepath_to_delete)


                Old_ROC_Filenames = [savefolder + 'ROC_{0}_{1}_{2}_{3}_{4}_PValue{5}_{6}.svg'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10'),
                                     savefolder + 'ROC_{0}_{1}_{2}_{3}_{4}_PValue{5}_{6}.csv'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, str(pValue)[2:], Compared_groups.replace('/', '_vs_')).replace('PValue1', 'PValue10')]
            
                New_ROC_Filenames = [savefolder + 'ROC_{0}_{1}_{2}_{3}_{4}_{5}.svg'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, Compared_groups.replace('/', '_vs_')),
                                     savefolder + 'ROC_{0}_{1}_{2}_{3}_{4}_{5}.csv'.format(SR, top1_Imputation_method, top1_Normalization_method, top1_BatchCorrection_method, top1_StatisticalTest_method, Compared_groups.replace('/', '_vs_'))]
            
                for old_name in Old_ROC_Filenames:
                    if os.path.exists(old_name):
                        new_name = New_ROC_Filenames[Old_ROC_Filenames.index(old_name)]

                        if os.path.exists(new_name):
                            os.remove(new_name)
                        os.rename(old_name, new_name)


if __name__ == '__main__': 
    UseCov_Plot_6_Cluster_ROC_Volcano(
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Second_Run_Results\MethodSelection_DifferentialExpressionAnalysis.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Results/',
    Software = 'DIANN',
    Report_Folder = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/",
    Report_Name = 'three_mix_report.pg_matrix.tsv',

    samples_csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Samples_Template.csv",
    composition_csv_path = "E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_UsePCA_UseCovariates/Composition_Template.csv")



