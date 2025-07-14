

import math
import numpy as np
import copy
import random
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt



# Draw differential protein bar charts and correlation coefficient box plots
def KeepNA_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot(
    
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates/Results/'):
    

    #pValue_List = [0.001, 0.01, 0.05, 0.1]
    pValue_List = [0.05]
    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    Compared_groups_label = ['S5/S1', 'S4/S2']


    Imputation = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute', 'KeepNA']
    Normalization = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN']
    BatchCorrection = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    StatisticalTest = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']

    colors = ['#8bd2ca', '#fe708a', '#67b7fd', '#d29fea']

    # Read csv file
    df_total = pd.read_csv(result_csv_path) 


    # Draw differential protein bar charts
    for p in pValue_List:
        for Compared_groups in Compared_groups_label:

            # Filter by p-value
            df_p = df_total[df_total['p-value'] == p]
            df_result = df_p.copy(deep = True)


            fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))
            
            # Data
            TN_list = []
            TP_list = []
            FP_list = []
            FN_list = []
            No_list = []

            Methods_index_list = []

            for SR in SR_methods:
                df_screened = df_result[df_result['Sparsity Reduction'] == SR]
                # Sort in ascending order by Rank
                df_screened2 = df_screened.sort_values(Compared_groups + ' Rank', ascending=True)
                TN_list.append(df_screened2[Compared_groups + ' TN'].values.tolist()[0])
                TP_list.append(df_screened2[Compared_groups + ' TP'].values.tolist()[0])
                FP_list.append(df_screened2[Compared_groups + ' FP'].values.tolist()[0]*2)
                FN_list.append(df_screened2[Compared_groups + ' FN'].values.tolist()[0]*2)

                No_list.append(df_screened2['No'].values.tolist()[0])

                setp1 = df_screened2['Missing Value Imputation'].values.tolist()[0]
                setp2 = df_screened2['Normalization'].values.tolist()[0]
                setp3 = df_screened2['Batch Correction'].values.tolist()[0]
                setp4 = df_screened2['Statistical Test'].values.tolist()[0]
                Methods_index = '[{0}-{1}-{2}-{3}]'.format(str(Imputation.index(setp1) + 1), str(Normalization.index(setp2) + 1), str(BatchCorrection.index(setp3) + 1), str(StatisticalTest.index(setp4) + 1))
                Methods_index_list.append(Methods_index)


            def autolabel(rects, flag = '+'):
                for rect in rects:
                    height = rect.get_height()
                    if height > 0:
                        if height > 450:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.5*height, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                    elif height == 0:
                        if (flag == '+'):
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial")
                    else:
                        if height > -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100 + 0.05*height, '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.05*height, '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 

            def autolabel2(rects, bottom):
                count = 0
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if bottom[count] < 300:
                            bottom[count] = 300
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    else:
                        if bottom[count] == 0:
                            bottom[count] = 0
                        elif (bottom[count] > -400) & (height > -400):
                            bottom[count] = -400
                        elif (bottom[count] > -400) & (height <= -400):
                            pass
                        if height > -200:
                            bottom[count] += -240

                        if height < -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., bottom[count], '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial")
                    count+=1

            # TP
            bar_x_TP = []
            bar_y_TP = []
            for index in range(len(SR_methods)):
                bar_x_TP.append(index+1)
                bar_y_TP.append(TP_list[index])

            bar1 = plt.bar(bar_x_TP, bar_y_TP, width = 0.75, color = colors[0]) 
            autolabel(bar1) 

            # FP
            bar_x_FP = []
            bar_y_FP = []
            for index in range(len(SR_methods)):
                bar_x_FP.append(index+1)
                bar_y_FP.append( - FP_list[index])

            bar2 = plt.bar(bar_x_FP, bar_y_FP, width = 0.75, color = colors[1]) 
            #autolabel(bar2, flag = '-') 

        


            # TN
            bar_x_TN = []
            bar_y_TN = []
            for index in range(len(SR_methods)):
                bar_x_TN.append(index+1)
                bar_y_TN.append(TN_list[index])

            bar3 = plt.bar(bar_x_TN, bar_y_TN, bottom = bar_y_TP, width = 0.75, color = colors[2]) 
            autolabel2(bar3, bottom = bar_y_TP) 

            # FN
            bar_x_FN = []
            bar_y_FN = []
            for index in range(len(SR_methods)):
                bar_x_FN.append(index+1)
                bar_y_FN.append( - FN_list[index])

            bar4_bottom = copy.copy(bar_y_FP)
            bar4 = plt.bar(bar_x_FN, bar_y_FN, bottom = bar4_bottom, width = 0.75, color = colors[3]) 
            #autolabel2(bar4, bottom = bar4_bottom) 


            axes = plt.gca()

            plt.tick_params(labelsize=14) 
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(True)
            axes.spines['left'].set_visible(True)
            axes.spines['bottom'].set_linewidth(2) 
            axes.spines['left'].set_linewidth(2) 

            y_max = max(bar_y_TP) + max(bar_y_TN)
            y_min = min(bar_y_FP) + min(bar_y_FN)

            y_max = math.ceil(y_max/1000)*1000 + 1000
            y_min = math.ceil(-y_min/1000)*1000 + 1000
            plt.ylim(-y_min-1000, y_max)
            yticks = list(range(-y_min, y_max, 1000))
            plt.yticks(yticks)
            yticklabels = []
            for ytick in yticks:
                if ytick < 0:
                    yticklabels.append(str(int(-ytick/2)))
                else:
                    yticklabels.append(str(ytick))
            axes.set_yticklabels(yticklabels, fontsize=14)

            #y_max = 4000
            #y_min = -3000
            #plt.ylim(y_min, y_max)
            #plt.yticks(list(range(-2000, y_max, 1000)))
            #axes.set_yticklabels(['1000', '500', '0', '1000', '2000', '3000'], fontsize=14)


            # Label FP arrows and text
            for i in range(len(bar_x_FP)):

                xytext_y = bar_y_FP[i] + bar_y_FN[i]
                if abs(xytext_y) < 600:
                    xytext_y = -1200
                else:
                    xytext_y -= 600

                plt.annotate(str(int(FP_list[i]/2)), xy=(bar_x_FP[i] + 0.2,bar_y_FP[i]), xytext=(bar_x_FP[i] + 0.2, xytext_y),
                             arrowprops=dict(facecolor=colors[1], edgecolor=colors[1], width = 1, headwidth = 5, headlength = 6, shrink = 0),
                             horizontalalignment='center', verticalalignment='top', rotation=90, fontsize=11.5)


            # Label FN arrows and text
                xytext_y -= 600

                plt.annotate(str(int(FN_list[i]/2)), xy=(bar_x_FN[i] - 0.2, bar_y_FP[i] + bar_y_FN[i]), xytext=(bar_x_FN[i] - 0.2, xytext_y),
                             arrowprops=dict(facecolor=colors[3], edgecolor=colors[3], width = 1, headwidth = 5, headlength = 6, shrink=0.05),
                             horizontalalignment='center', verticalalignment='top', rotation=90 ,fontsize=11.5)


            plt.xlim(0.35, 0.5+len(SR_methods))
            plt.xticks(list(range(1, 1+len(SR_methods))))
            xticklabels = []
            for SR in SR_methods:
                xticklabels.append(Methods_index_list[SR_methods.index(SR)] + ' ' + SR)

            axes.set_xticklabels(xticklabels, fontsize=14, rotation = 90, ha = 'center', va = 'top')
            plt.ylabel('# Proteins', y=0.5, fontsize=16) 
            plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)


            plt.subplots_adjust(left=0.28, right=0.98, bottom=0.33, top=0.95, wspace=0.05, hspace=0.1) 

            plt.savefig(savefolder + 'SparsityReduction_DifferentialProteins_PValue{0}_{1}_vs_{2}.svg'.format(str(p)[2:], Compared_groups.split('/')[0], Compared_groups.split('/')[1]).replace('PValue1', 'PValue10'), dpi=600, format="svg", transparent=True)
            #plt.show()
            plt.close()




    
    pValue_List = [0.001, 0.01, 0.05, 0.1]
    for SR in SR_methods:
        for Compared_groups in Compared_groups_label:

            # Filter by SR
            df_sr = df_total[df_total['Sparsity Reduction'] == SR]
            df_result = df_sr.copy(deep = True)


            fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))
            
            # Data
            TN_list = []
            TP_list = []
            FP_list = []
            FN_list = []
            No_list = []
            
            Methods_index_list = []

            for p in pValue_List:
                df_screened = df_result[df_result['p-value'] == p]
                # Sort in ascending order by Rank
                df_screened2 = df_screened.sort_values(Compared_groups + ' Rank', ascending=True)
                TN_list.append(df_screened2[Compared_groups + ' TN'].values.tolist()[0])
                TP_list.append(df_screened2[Compared_groups + ' TP'].values.tolist()[0])
                FP_list.append(df_screened2[Compared_groups + ' FP'].values.tolist()[0]*2)
                FN_list.append(df_screened2[Compared_groups + ' FN'].values.tolist()[0]*2)

                No_list.append(df_screened2['No'].values.tolist()[0])

                setp1 = df_screened2['Missing Value Imputation'].values.tolist()[0]
                setp2 = df_screened2['Normalization'].values.tolist()[0]
                setp3 = df_screened2['Batch Correction'].values.tolist()[0]
                setp4 = df_screened2['Statistical Test'].values.tolist()[0]
                Methods_index = '[{0}-{1}-{2}-{3}]'.format(str(Imputation.index(setp1) + 1), str(Normalization.index(setp2) + 1), str(BatchCorrection.index(setp3) + 1), str(StatisticalTest.index(setp4) + 1))
                Methods_index_list.append(Methods_index)


            def autolabel(rects, flag = '+'):
                for rect in rects:
                    height = rect.get_height()
                    if height > 0:
                        if height > 450:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.5*height, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                    elif height == 0:
                        if (flag == '+'):
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial")
                    else:
                        if height > -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100 + 0.05*height, '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.05*height, '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 

            def autolabel2(rects, bottom):
                count = 0
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if bottom[count] < 300:
                            bottom[count] = 300
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    else:
                        if bottom[count] == 0:
                            bottom[count] = 0
                        elif (bottom[count] > -400) & (height > -400):
                            bottom[count] = -400
                        elif (bottom[count] > -400) & (height <= -400):
                            pass
                        if height > -200:
                            bottom[count] += -240

                        if height < -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., bottom[count], '%s' % int(-height/2), ha='center', va='top', rotation=90, size=11.5, family="Arial")
                    count+=1

            # TP
            bar_x_TP = []
            bar_y_TP = []
            for index in range(len(pValue_List)):
                bar_x_TP.append(index+1)
                bar_y_TP.append(TP_list[index])

            bar1 = plt.bar(bar_x_TP, bar_y_TP, width = 0.75, color = colors[0]) 
            autolabel(bar1) 

            # FP
            bar_x_FP = []
            bar_y_FP = []
            for index in range(len(pValue_List)):
                bar_x_FP.append(index+1)
                bar_y_FP.append( - FP_list[index])

            bar2 = plt.bar(bar_x_FP, bar_y_FP, width = 0.75, color = colors[1]) 
            #autolabel(bar2, flag = '-') 

            # TN
            bar_x_TN = []
            bar_y_TN = []
            for index in range(len(pValue_List)):
                bar_x_TN.append(index+1)
                bar_y_TN.append(TN_list[index])

            bar3 = plt.bar(bar_x_TN, bar_y_TN, bottom = bar_y_TP, width = 0.75, color = colors[2]) 
            autolabel2(bar3, bottom = bar_y_TP) 

            # FN
            bar_x_FN = []
            bar_y_FN = []
            for index in range(len(pValue_List)):
                bar_x_FN.append(index+1)
                bar_y_FN.append( - FN_list[index])

            bar4 = plt.bar(bar_x_FN, bar_y_FN, bottom = bar_y_FP, width = 0.75, color = colors[3]) 
            #autolabel2(bar4, bottom = bar_y_FP) 


            axes = plt.gca()

            plt.tick_params(labelsize=14) 
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(True)
            axes.spines['left'].set_visible(True)
            axes.spines['bottom'].set_linewidth(2) 
            axes.spines['left'].set_linewidth(2) 

            y_max = max(bar_y_TP) + max(bar_y_TN)
            y_min = min(bar_y_FP) + min(bar_y_FN)

            y_max = math.ceil(y_max/1000)*1000 + 1000
            y_min = math.ceil(-y_min/1000)*1000 + 1000
            plt.ylim(-y_min-1000, y_max)
            yticks = list(range(-y_min, y_max, 1000))
            plt.yticks(yticks)
            yticklabels = []
            for ytick in yticks:
                if ytick < 0:
                    yticklabels.append(str(int(-ytick/2)))
                else:
                    yticklabels.append(str(ytick))
            axes.set_yticklabels(yticklabels, fontsize=14)

            #y_max = 4000
            #y_min = -3000
            #plt.ylim(y_min, y_max)
            #plt.yticks(list(range(-2000, y_max, 1000)))
            #axes.set_yticklabels(['1000', '500', '0', '1000', '2000', '3000'], fontsize=14)

            # Label FP arrows and text
            for i in range(len(bar_x_FP)):

                xytext_y = bar_y_FP[i] + bar_y_FN[i]
                if abs(xytext_y) < 600:
                    xytext_y = -1200
                else:
                    xytext_y -= 600

                plt.annotate(str(int(FP_list[i]/2)), xy=(bar_x_FP[i] + 0.2,bar_y_FP[i]), xytext=(bar_x_FP[i] + 0.2, xytext_y),
                             arrowprops=dict(facecolor=colors[1], edgecolor=colors[1], width = 1, headwidth = 5, headlength = 6, shrink = 0),
                             horizontalalignment='center', verticalalignment='top', rotation=90, fontsize=11.5)


            # Label FN arrows and text
                xytext_y -= 600

                plt.annotate(str(int(FN_list[i]/2)), xy=(bar_x_FN[i] - 0.2, bar_y_FP[i] + bar_y_FN[i]), xytext=(bar_x_FN[i] - 0.2, xytext_y),
                             arrowprops=dict(facecolor=colors[3], edgecolor=colors[3], width = 1, headwidth = 5, headlength = 6, shrink=0.05),
                             horizontalalignment='center', verticalalignment='top', rotation=90 ,fontsize=11.5)

            plt.xlim(0.35, 0.5+len(pValue_List))
            plt.xticks(list(range(1, 1+len(pValue_List))))
            xticklabels = []
            blank_list = [' ', '   ', '   ', '     ']
            for p in pValue_List:
                xticklabels.append(Methods_index_list[pValue_List.index(p)] + blank_list[pValue_List.index(p)] + str(p))

            axes.set_xticklabels(xticklabels, fontsize=14, rotation = 90, ha = 'center', va = 'top')
            plt.ylabel('# Proteins', y=0.5, fontsize=16) 
            plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)


            plt.subplots_adjust(left=0.28, right=0.98, bottom=0.33, top=0.95, wspace=0.05, hspace=0.1) 

            plt.savefig(savefolder + 'PValue_DifferentialProteins_{0}_{1}_vs_{2}.svg'.format(SR, Compared_groups.split('/')[0], Compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)
            #plt.show()
            plt.close()





if __name__ == '__main__': 
    KeepNA_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot(
    
    result_csv_path = "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv",
    savefolder = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates/Results/')






