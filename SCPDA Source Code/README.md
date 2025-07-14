#SCPDA

##1. Introduction
This tool is designed for single-cell proteomics benchmark data analysis.


##2. Development environment installation
The tool is mainly written in Python, but some functions are implemented in R. Therefore, users need to configure the development environment of Python and R languages ​​at the same time and install the required toolkits.
###Python
In this work, the author used Anaconda to create a virtual environment containing **Python 3.6.13**  
Anaconda Download Address: **https://www.anaconda.com/**  
Once Anaconda is installed, open **Anaconda Prompt** and create your virtual environment using the following command:  

	conda create -n your_env python=3.6.13


###Python libraries
Use pip to install the following Python libraries in your virtual environment:  

	activate your_env
	pip install matplotlib==3.2.2
	pip install pandas==1.1.5
	pip install numpy==1.19.3
	pip install scipy==1.5.4
	pip install scanorama==1.7.4
	pip install svgutils==0.1.0
	pip install fancyimpute==0.7.0
	pip install rpy2==3.4.5
	pip install scikit-learn==0.24.2
    pip install xgboost==1.5.2
    pip install shap==0.41.0


###R
In this work, **R for Windows 4.3.1** is installed in the default path ('C:\Program Files\R\R-4.3.1').  
If the user's R language is installed in another path, modify the 32nd line of code in the SCPDA.py file and replace the R installation path with the user's own path.  

**R for Windows 4.3.1** download address: https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/base/old/4.3.1  
R and RStudio installation tutorial: https://rstudio-education.github.io/hopr/starting.html  
R package installation tutorial: https://rstudio-education.github.io/hopr/packages2.html  

###R packages
Install the following R packages:  

**Package**			**Version**  
MBQN				2.12.0  
limma				3.56.2  
edgeR				3.42.4  
Seurat				4.3.0.1  
sva					3.48.0  
statmod				1.5.0  
DESeq2				1.40.2  
readr				2.1.4  
clusterProfiler		4.8.3  
ReactomePA			1.44.0  
org.Hs.eg.db		3.17.0  
KEGG.db				1.0  

The installation method of the **KEGG.db** package can be found in the following link:  
https://mp.weixin.qq.com/s/PwrdQAkG3pTlwMB6Mj8wXQ

###Development environment installation time
About 1-2 hours.  


##3. Examples
First, open **Anaconda Prompt** and activate your virtual environment where Python 3.16.13 is located.  
Then, use the **cd** command to enter the directory where the source code is located.  
Finally, enter the command to run the corresponding task.  

###3.1 Example - Generate sample information template file
**Description:**  
According to the software's report file (DIA-NN, Spectronaut, PEAKS, MaxQuant), generate a sample information template file, which contains 3 columns (Run, Group, Batch). The data of 'Run' column is automatically obtained by the program, and the user only needs to complete the sample group and batch information.

**Command:**  

	python SCPDA.py --Task Generate_Samples_Template --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SavePath "../your_save_folder/" 


**Parameter Description:**  
Task - Task name  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
A few seconds (Spectronaut takes slightly longer).  

**Output:**  
Samples\_Template.csv  

###3.2 Example - Generate sample composition information template file
**Description:**  
According to the software's report file (DIA-NN, Spectronaut, PEAKS, MaxQuant) and sample information file, generate a sample composition information template file. The table contains a 'Organism' column and group columns. The data of 'Organism' column is automatically obtained by the program, users need to supplement the data of each group column.  
For example, the program automatically obtains the data of the 'Organism' column as 'ECOLI', 'HUMAN', 'YEAST'. If the user's 'Group1' is composed of 20% ECOLI protein, 50% HUMAN protein, and 30% YEAST protein, the corresponding data of the 'Group1' column should be filled in as 20, 50, 30. If the user's 'Group2' is composed of 100% HUAMN protein, then the data in the 'Group2' column should be 0, 100, 0.  

**Command:**  

	python SCPDA.py --Task Generate_Composition_Template --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--SavePath "../your_save_folder/" 


**Parameter Description:**  
Task - Task name  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
A few seconds (Spectronaut takes slightly longer).  

**Output:**  
Composition\_Template.csv  

###3.3 Example - Evaluation of identification number, data completeness and quantitative precision
**Description:**  
According to the software's report file (DIA-NN, Spectronaut, PEAKS, MaxQuant), sample information file and sample composition information file, generate graphs and data files of identification results.  

**Command:**  

	python SCPDA.py --Task IdentificationResult --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Level Protein ^
	--SavePath "../your_save_folder/" 


**Parameter Description:**  
Task - Task name  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Level - Report type, optional: Protein, Peptide  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About one minute (Spectronaut and Peptide report take slightly longer to execute).  

**Output:**  
Protein\_Matrix.csv  
RunIdentifications\_Proteins.svg  
RunIdentifications\_Proteins.csv  
DataCompleteness\_Proteins.svg  
DataCompleteness\_Proteins.csv  
CumulativeIdentifications\_Proteins.svg  
CumulativeIdentifications\_Proteins.csv  
CVDistribution\_Proteins.svg  
CVDistribution\_Proteins.csv  

###3.4 Example - Evaluation of Quantitative Accuracy
**Description:**  
According to the software's report file (DIA-NN, Spectronaut, PEAKS, MaxQuant), sample information file and sample composition information file, generate graphs and data files of quantitative accuracy results.  

**Command:**  

	python SCPDA.py --Task FoldChange --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Level Protein ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
	--SavePath "../your_save_folder/" 


**Parameter Description:**  
Task - Task name  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Level - Report type, optional: Protein, Peptide  
Comparison - The groups to be compared. Like: S5/S1 S4/S2  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About one minute (Spectronaut and Peptide report take slightly longer to execute).  

**Output:**  
Protein\_Matrix.csv  
FoldChange\_Proteins.svg  
FoldChange\_Proteins.csv  
FoldChange\_Proteins\_Group1\_vs\_Group2.svg  
...  
Legend_Organisms.svg  

###3.5 Example - Evaluation of Identification Error Rate
**Description:**  
According to the software's report file (DIA-NN, Spectronaut, PEAKS, MaxQuant), sample information file and sample composition information file, evaluate the software's identification error rate.  
Note: The report file must contain samples composed of a single species.  

**Command:**  

	python SCPDA.py --Task Entrapment --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Level Protein ^
	--SavePath "../your_save_folder/" 


**Parameter Description:**  
Task - Task name  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Level - Report type, optional: Protein, Peptide  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
Less than one minute (Spectronaut and Peptide report take slightly longer to execute).  

**Output:**  
Protein\_Matrix.csv  
EntrapmentIdentifications\_Proteins.svg  
EntrapmentIdentifications\_Proteins.csv  
EntrapmentDataCompleteness\_Proteins.svg  
EntrapmentDataCompleteness\_Proteins.csv  
EntrapmentQuantityRank\_Proteins\_{Organism}.svg  
EntrapmentQuantityRank\_Proteins.csv  

###3.6 Example - Comparison of results from different software/library construction methods
**Description:**  
Evaluate the differences in results from different software/library construction methods.  
This function can compare up to 4 software/library construction methods. As shown below, if you want to compare 3 software/library construction methods, fill in 3 DatasetName and DatasetFolder.  

**Command:**  

	python SCPDA.py --Task ResultComparison ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Level Protein ^
	--DatasetName1 "DIAN-NN" ^
	--DatasetName2 "Spectronaut" ^
	--DatasetName3 "PEAKS" ^
	--DatasetFolder1 "../DIA-NN_report_folder/" ^
	--DatasetFolder2 "../Spectronaut_report_folder/" ^
	--DatasetFolder3 "../PEAKS_report_folder/" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
	--SavePath "../your_save_folder/" 

**Parameter Description:**  
Task - Task name   
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Level - Report type, optional: Protein, Peptide  
DatasetName1 (or DatasetName2, DatasetName3, DatasetName4) - Name of software or library construction method  
DatasetFolder1 (or DatasetFolder2, DatasetFolder3, DatasetFolder4) - The folder path where Dataset1 is located  
Comparison - The groups to be compared. Like: S5/S1 S4/S2  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About one minute (Peptide data execution takes slightly longer).  

**Output:**  
Comparison\_Identifications\_Proteins.svg  
Legend\_Datasets.svg  
Comparison\_OverlapIdentifications\_Proteins.svg  
Comparison\_DataCompleteness\_Proteins.svg  
Comparison\_FoldChange\_Proteins\_Group1\_vs\_Group2.svg  
Comparison\_FoldChange\_Proteins\_Group3\_vs\_Group4.svg  
...  
Comparison\_CVDistribution\_Proteins\_Group1.svg  
Comparison\_CVDistribution\_Proteins\_Group2.svg  
Comparison\_CVDistribution\_Proteins\_Group3.svg  
Comparison\_CVDistribution\_Proteins\_Group4.svg  
...  

###3.7 Example - Evaluating the performance of differential expression analysis with different combinations of analytical methods (AutoFC)
**Description:**  
By combining sparsity reduction methods ('NoSR', 'SR66', 'SR75', 'SR90'), missing value imputation methods ('Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'), data normalization methods ('Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'), batch effect correction methods ('NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama') and statistical test methods ('t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'), evaluate the performance of each method combination for finding differentially expressed proteins.The p-value and FC used to find differentially expressed proteins are automatically determined by the program.  

**Command:**  

	python SCPDA.py --Task MethodSelectionAutoFC ^ 
    --Reduction PCA ^ 
    --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
	--ClusteringEvaluation ARI ^
    --OutputMethodSelectionFigures False ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
Reduction - The dimension reduction method used for cluster analysis. The default is PCA, and UMAP can also be selected  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ClusteringEvaluation - The metric used to evaluate clustering performance, optional: ARI or PurityScore  
OutputMethodSelectionFigures - Whether to output graphs and tables. If False is selected, only one file, MethodSelection\_DifferentialExpressionAnalysis.csv, will be output  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 1-2 days. 

**Output:**  
MethodSelection\_DifferentialExpressionAnalysis.csv  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Legend\_Volcano.svg  
Legend\_PR.svg  
Legend\_Clustering\_Batch.svg  
Legend\_Clustering\_Group.svg  

###3.8 Example - Evaluating the performance of differential expression analysis with different combinations of analytical methods (UseCov)
**Description:**  
By combining sparsity reduction methods ('NoSR', 'SR66', 'SR75', 'SR90'), missing value imputation methods ('Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'), data normalization methods ('Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'), batch effect correction methods ('NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama') and statistical test methods ('t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'), evaluate the performance of each method combination for finding differentially expressed proteins.This mode iterates over 4 p-values ​​[0.001, 0.01, 0.05, 0.1] and uses the FC specified by the user. This mode use covariates when running the batch effect correction algorithm.  

**Command:**  

	python SCPDA.py --Task MethodSelection ^ 
    --Type UseCov --Reduction PCA ^ 
    --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--ClusteringEvaluation ARI ^
    --OutputMethodSelectionFigures False ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
Type - Selectable analysis task types. For this taks is UseCov. Users can choose UseCov, NoCov or KeepNA  
Reduction - The dimension reduction method used for cluster analysis. The default is PCA, and UMAP can also be selected  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
ClusteringEvaluation - The metric used to evaluate clustering performance, optional: ARI or PurityScore  
OutputMethodSelectionFigures - Whether to output graphs and tables. If False is selected, only one file, MethodSelection\_DifferentialExpressionAnalysis.csv, will be output  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 1-2 days. 

**Output:**  
MethodSelection\_DifferentialExpressionAnalysis.csv  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Legend\_Volcano.svg  
Legend\_PR.svg  
Legend\_Clustering\_Batch.svg  
Legend\_Clustering\_Group.svg  

###3.9 Example - Evaluating the performance of differential expression analysis with different combinations of analytical methods (NoCov)
**Description:**  
By combining sparsity reduction methods ('NoSR', 'SR66', 'SR75', 'SR90'), missing value imputation methods ('Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'), data normalization methods ('Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'), batch effect correction methods ('NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama') and statistical test methods ('t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'), evaluate the performance of each method combination for finding differentially expressed proteins.This mode iterates over 4 p-values ​​[0.001, 0.01, 0.05, 0.1] and uses the FC specified by the user. This mode does not use covariates when running the batch effect correction algorithm.  

**Command:**  

	python SCPDA.py --Task MethodSelection ^ 
    --Type NoCov --Reduction PCA ^ 
    --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--ClusteringEvaluation ARI ^
    --OutputMethodSelectionFigures False ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
Type - Selectable analysis task types. For this taks is NoCov. Users can choose UseCov, NoCov or KeepNA  
Reduction - The dimension reduction method used for cluster analysis. The default is PCA, and UMAP can also be selected  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
ClusteringEvaluation - The metric used to evaluate clustering performance, optional: ARI or PurityScore  
OutputMethodSelectionFigures - Whether to output graphs and tables. If False is selected, only one file, MethodSelection\_DifferentialExpressionAnalysis.csv, will be output  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 1-2 days. 

**Output:**  
MethodSelection\_DifferentialExpressionAnalysis.csv  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Legend\_Volcano.svg  
Legend\_PR.svg  
Legend\_Clustering\_Batch.svg  
Legend\_Clustering\_Group.svg  

###3.10 Example - Evaluating the performance of differential expression analysis with different combinations of analytical methods (KeepNA)
**Description:**  
By combining sparsity reduction methods ('NoSR', 'SR66', 'SR75', 'SR90'), missing value imputation methods ('Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'), data normalization methods ('Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'), batch effect correction methods ('NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama') and statistical test methods ('t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'), evaluate the performance of each method combination for finding differentially expressed proteins.This mode iterates over 4 p-values ​​[0.001, 0.01, 0.05, 0.1] and uses the FC specified by the user. Furthermore, it does not perform missing value filling algorithms (keeping missing values), and uses covariates when running the batch effect correction algorithm.  

**Command:**  

	python SCPDA.py --Task MethodSelection ^ 
    --Type KeepNA --Reduction PCA ^ 
    --Software DIANN ^
	--ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--ClusteringEvaluation ARI ^
    --OutputMethodSelectionFigures False ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
Type - Selectable analysis task types. For this taks is KeepNA. Users can choose UseCov, NoCov or KeepNA  
Reduction - The dimension reduction method used for cluster analysis. The default is PCA, and UMAP can also be selected  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
ClusteringEvaluation - The metric used to evaluate clustering performance, optional: ARI or PurityScore  
OutputMethodSelectionFigures - Whether to output graphs and tables. If False is selected, only one file, MethodSelection\_DifferentialExpressionAnalysis.csv, will be output.  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 1-2 days. 

**Output:**  
MethodSelection\_DifferentialExpressionAnalysis.csv  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
PR\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Legend\_Volcano.svg  
Legend\_PR.svg  
Legend\_Clustering\_Batch.svg  
Legend\_Clustering\_Group.svg  

###3.11 Example - Visualize the analysis results of Example 3.8 (UseCov)
**Description:**  
Visualize the metrics of the analysis results of Example 3.8 (UseCov).  

**Command:**  

	python SCPDA.py --Task PlotForUseCov ^
	--ResultCSVPath "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis.csv" ^
	--Software DIANN ^ 
    --ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ResultCSVPath - Path to the file **MethodSelection_DifferentialExpressionAnalysis.csv**  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 15 minutes. 

**Output:**  
SHAP\_Statistical Test\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAP\_Statistical Test\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAP\_Batch Correction\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAP\_Batch Correction\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAP\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAP\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAP\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAP\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv 

SHAPSummary\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPSummary\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv    

FeatureImportance\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
FeatureImportance\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv   

SHAPInteraction\_Normalization\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Normalization\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAPInteraction\_Statistical Test\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Statistical Test\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  

SHAPInteraction\_Batch Correction\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Batch Correction\_Imputation\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAPInteraction\_Batch Correction\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Batch Correction\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  

SHAPInteraction\_Statistical Test\_Batch Correction\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Statistical Test\_Batch Correction\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SHAPInteraction\_Statistical Test\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteraction\_Statistical Test\_Normalization\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  

SHAPInteractionSummary\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
SHAPInteractionSummary\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  

Correlation\_Comparison\_Group.svg  
Correlation\_Comparison\_Group.csv  
Correlation\_Comparison\_PValue.svg  
Correlation\_Comparison\_PValue.csv  
Correlation\_Comparison\_SR.svg  
Correlation\_Comparison\_SR.csv  

MethodProportion\_Top01\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
MethodProportion\_Top01\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
MethodProportion\_Top05\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
MethodProportion\_Top05\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  

PValue\_DifferentialProteins\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.svg  
PValue\_DifferentialProteins\_{Sparsity Reduction}\_{PValue}\_Group1\_vs\_Group2.csv  
SparsityReduction\_DifferentialProteins\_{PValue}\_Group1\_vs\_Group2.svg  
SparsityReduction\_DifferentialProteins\_{PValue}\_Group1\_vs\_Group2.csv  

StatisticalMetrics\_{PValue}\_{Sparsity Reduction}\_Group1\_vs\_Group2.svg  
StatisticalMetrics\_{PValue}\_{Sparsity Reduction}\_Average.svg  

StatisticalTest\_Metrics\_{Sparsity Reduction}\_{Imputation}\_{PValue}\_Group1\_vs\_Group2.svg  

Imputation\_BatchCorrection\_ARI\_{Sparsity Reduction}\_{Normalization}\_{PValue}\_Group1\_vs\_Group2.svg  
Imputation\_BatchCorrection\_pAUC\_{Sparsity Reduction}\_{Normalization}\_{StatisticalTest}\_{PValue}\_Group1\_vs\_Group2.svg  

Imputation\_Normalization\_ARI\_{Sparsity Reduction}\_{BatchCorrection}\_{PValue}\_Group1\_vs\_Group2.svg  
Imputation\_Normalization\_pAUC\_{Sparsity Reduction}\_{BatchCorrection}\_{StatisticalTest}\_{PValue}\_Group1\_vs\_Group2.svg   

Normalization\_BatchCorrection\_ARI\_{Sparsity Reduction}\_{Imputation}\_{PValue}\_Group1\_vs\_Group2.svg  
Normalization\_BatchCorrection\_pAUC\_{Sparsity Reduction}\_{Imputation}\_{StatisticalTest}\_{PValue}\_Group1\_vs\_Group2.svg   

BatchCorrection\_StatisticalTest\_pAUC\_{Sparsity Reduction}\_{Imputation}\_{Normalization}\_{PValue}\_Group1\_vs\_Group2.svg   
Imputation\_StatisticalTest\_pAUC\_{Sparsity Reduction}\_{Normalization}\_{BatchCorrection}\_{PValue}\_Group1\_vs\_Group2.svg   
Imputation\_Normalization\_pAUC\_{Sparsity Reduction}\_{BatchCorrection}\_{StatisticalTest}\_{PValue}\_Group1\_vs\_Group2.svg   


Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  

ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_{PValue}\_Group1\_vs\_Group2.svg  
ROC\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_{PValue}\_Group1\_vs\_Group2.csv  
  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_{PValue}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_{PValue}\_Group1\_vs\_Group2.csv  


Legend\_Imputation.svg  
Legend\_Normalization.svg  
Legend\_BatchCorrection.svg  
Legend\_Statistical Test.svg  
Legend\_StatisticalTest\_Metrics.svg  

Legend\_Imputation\_BatchCorrection\_ARI.svg  
Legend\_Imputation\_Normalization\_ARI.svg  
Legend\_Normalization\_BatchCorrection\_ARI.svg  
Legend\_BatchCorrection\_StatisticalTest\_pAUC.svg  
Legend\_Imputation\_BatchCorrection\_pAUC.svg  
Legend\_Imputation\_Normalization\_pAUC.svg  
Legend\_Imputation\_StatisticalTest\_pAUC.svg  
Legend\_Normalization\_BatchCorrection\_pAUC.svg  
Legend\_Normalization\_StatisticalTest\_pAUC.svg  

Legend\_Volcano.svg  
Legend\_Clustering\_Batch.svg  
Legend\_Clustering\_Group.svg  

###3.12 Example - Visualize the analysis results of Example 3.9 (NoCov)  
**Description:**  
Visualize the metrics of the analysis results of Example 3.9 (NoCov).  

**Command:**  

	python SCPDA.py --Task PlotForNoCov ^
	--ResultCSVPath "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis.csv" ^
    --ResultCSVPath2 "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis-NC.csv" ^
    --ResultCSVPath3 "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv" ^
	--Software DIANN ^ 
    --ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ResultCSVPath - Path to the NoCov's original table  
ResultCSVPath2 - Path to the new NoCov's table with '-NC' appended to the batch effect correction method ('limma', 'Combat-P', 'Combat-NP') name  
ResultCSVPath3 - Path to the new table formed by merging the data in the NoCov table that does not use covariates with the UseCov table  
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 15 minutes. 

**Output:**  
The output is the same as Example 3.11 (except that the SHAP-related graph is not included).  

###3.13 Example - Visualize the analysis results of Example 3.10 (KeepNA)  
**Description:**  
Visualize the metrics of the analysis results of Example 3.10 (KeepNA).  

**Command:**  

	python SCPDA.py --Task PlotForKeepNA ^
	--ResultCSVPath "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis.csv" ^
	--Software DIANN ^ 
    --ReportPath "../your_path_to_report/DIAnn_report.pg_matrix.tsv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--CompositionPath "../your_path_to_composition/Composition.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
    --ComparisonFC 1.2 1.3 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ResultCSVPath - Path to the table after KeepNA and UseCov merge    
Software - Software to which this report belongs. Optional software: DIANN(or DIA-NN), Spectronaut, PEAKS, MaxQuant  
ReportPath - Path to the report file  
SamplesPath - Path to the sample information file  
CompositionPath - Path to the sample composition information file  
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
ComparisonFC - Fold Change (FC) applied to each comparison group  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About 15 minutes. 

**Output:**  
The output is the same as Example 3.11 (except that the SHAP-related graph is not included).  

###3.14 Example - MethodSelection Table Conversion
**Description:**  
Convert the original table (MethodSelection\_DifferentialExpressionAnalysis.csv) to a new table containing TotalRank, AverageRank (usually used for UseCov, NoCov, KeepNA tables).  

**Command:**  

	python SCPDA.py --Task MethodSelectionTableConversion ^
	--ResultCSVPath "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ResultCSVPath - Path to the original table   
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About a few seconds. 

**Output:**  
a new table containing TotalRank, AverageRank.  


###3.15 Example - MethodSelection Table Rerank  
**Description:**  
When the user manually merges two tables (such as merging the KeepNA table with the UseCov table) and wants to re-rank it.  

**Command:**  

	python SCPDA.py --Task MethodSelectionTableRerank ^
	--ResultCSVPath "../your_path_to_result/MethodSelection_DifferentialExpressionAnalysis.csv" ^
	--Comparison Group1/Group2 Group3/Group4 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ResultCSVPath - Path to the table you want to rerank   
Comparison - The groups to be compared. Like: S4/S2 S5/S1  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About a few seconds. 

**Output:**  
Reranked table. 


###3.16 Example - Selection of the high-performing method combinations
Based on the SHAP explanations, a strategy was established for the recommendations of high-performing method combinations.  

**Command:**  

	python SCPDA.py --Task BeamSearch ^
	--SHAPFolder "../your_SHAP_folder/" ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
SHAPFolder - Path to the folder containing SHAP data  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About a few seconds. 

**Output:**  
top\_paths\_VaR95.csv  
dot\_VaR95.txt  


###3.17 Example - Search for differentially proteins and related pathways
Search for differentially proteins and related pathways based on the protein matrix csv file, method selection csv file, and sample information csv file provided by the user.  
The method selection file is a user-defined file, which contains five columns: 'Sparsity Reduction', 'Missing Value Imputation', 'Normalization', 'Batch Correction', and 'Statistical Test'. Users need to fill in the method combination they want to try.  

**Command:**  

	python SCPDA.py --Task DifferentialProtein ^
	--ProteinMatrixPath "../your_path_to_protein_matrix/Protein_Matrix.csv" ^
	--MethodSelectionPath "../your_path_to_methods/MethodSelection.csv" ^
	--SamplesPath "../your_path_to_samples/Samples.csv" ^
	--Comparison Group1/Group2 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
ProteinMatrixPath - Path to the file **Protein_Matrix.csv**. This file can be obtained from Example 3.3  
MethodSelectionPath - Path to method selection csv file  
SamplesPath - Path to the sample information file  
Comparison - The groups to be compared. Like: T/C  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About a few hours (depending on how many method combinations you want to try). 

**Output:**  
MethodSelection\_DifferentialExpressionAnalysis.csv  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.svg  
Clustering\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}.csv  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.svg  
Volcano\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
GO\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
KEGG\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv
Reactome\_{Sparsity Reduction}\_{Missing Value Imputation}\_{Normalization}\_{Batch Correction}\_{Statistical Test}\_Group1\_vs\_Group2.csv  
...  
Legend\_Clustering\_Group.svg  
Legend\_Clustering\_Batch.svg  



###3.18 Example - Draw hierarchical cluster diagram of pathways and methods
According to the method selection file and pathway analysis results (GO, Reactome), draw hierarchical clustering diagram of methods and pathways. And according to the method arrangement on the X-axis of the hierarchical clustering diagram, draw the Purity Score, Sample Purity, Batch Purity histogram.  

**Command:**  

	python SCPDA.py --Task HierarchicalClustering ^
	--MethodSelectionPath "../your_path_to_methods/MethodSelection.csv" ^
	--PathwayResultPath "../your_pathway_result_folder/" ^
	--Comparison Group1/Group2 ... ^
	--SavePath "../your_save_folder/"

**Parameter Description:**  
Task - Task name  
MethodSelectionPath - Path to method selection csv file  
PathwayResultPath - The path to the folder where the pathway analysis results are saved  
Comparison - The groups to be compared. Like: T/C  
SavePath - Path to the folder where the result files will be saved  

**Command Execution Time:**  
About one minute. 

**Output:**  
GO\_{Sparsity Reduction}.svg  
GO\_{Sparsity Reduction}.csv  
Reactome\_{Sparsity Reduction}.svg  
Reactome\_{Sparsity Reduction}.csv  
SamplePurity\_GO\_{Sparsity Reduction}.svg  
SamplePurity\_Reactome\_{Sparsity Reduction}.svg  
BatchPurity\_GO\_{Sparsity Reduction}.svg  
BatchPurity\_Reactome\_{Sparsity Reduction}.svg  
PurityScore\_GO\_{Sparsity Reduction}.svg  
PurityScore\_Reactome\_{Sparsity Reduction}.svg  
or  
ARI\_GO\_{Sparsity Reduction}.svg  
ARI\_Reactome\_{Sparsity Reduction}.svg  
Legend\_GO\_Z-score.svg  
Legend\_GO\_{Sparsity Reduction}\_Count.svg  
Legend\_Reactome\_Z-score.svg  
Legend\_Reactome\_{Sparsity Reduction}\_Count.svg  

