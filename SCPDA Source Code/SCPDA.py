import os
import time
import re
import copy
import venn
import math
import pandas as pd
import numpy as np
import scipy
from scipy.stats import gaussian_kde, ttest_ind, levene, rankdata, ranksums, pearsonr, spearmanr
from scipy.interpolate import interp1d

import statistics
from statistics import mean
import scanorama
import svgutils.transform as st

import matplotlib as matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["font.sans-serif"] = ["Arial"] 
matplotlib.rcParams["axes.unicode_minus"] = False


#from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from fancyimpute import SimpleFill, IterativeSVD, SoftImpute, KNN

os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.1'  # Set the installation directory of R
import rpy2.robjects as robjects  
from rpy2.robjects import pandas2ri, globalenv
from rpy2.robjects.conversion import localconverter as lc

# Python调用R出现“UnicodeDecodeError: ‘utf-8‘ codec can‘t decode byte 0xb2” 问题
# https://blog.csdn.net/qq_44645101/article/details/127069531

import additional_plot_methods
from additional_plot_methods import *
import argparse



# Single Cell DIA Benchmark Tool
class SCPDA(object):
    
    def __init__(self, samples_csv_path = '',
                 composition_csv_path = ''):
        
        if (samples_csv_path != 'Generate_Template') & (samples_csv_path != 'DifferentialProtein') & (samples_csv_path != ''):


            # Sample information
            self.df_samples = pd.read_csv(samples_csv_path)
            # Species composition
            self.df_composition = pd.read_csv(composition_csv_path)

            # Run name
            self.run_name = self.df_samples['Run Name'].values.tolist()

            # Group name  
            # Such as: ['S1', 'S2', 'S3', 'S4', 'S5']
            self.group_name = self.df_composition.columns.tolist()[1:]

            # Number of groups  
            # Such as: 5
            self.groups = len(self.group_name)  

            # Number of samples  
            # Such as: 30
            self.total_samples = self.df_samples['Run Name'].unique().size  

            # Number of batches
            self.batches = self.df_samples['Batch'].unique().size
            # Batch name
            self.batch_name = self.df_samples['Batch'].unique().tolist() 
            # List of batch information for each sample
            self.sample_batch_list = self.df_samples['Batch'].values.tolist()
            # List of grouping information for each sample
            self.sample_group_list = self.df_samples['Group'].values.tolist()


            # The label of each sample (for plotting)
            labels = list(range(1, 1+self.total_samples, 1))
            self.sample_labels = [str(i) for i in labels]

            # Species Information  
            # Such as: ['HUMAN', 'YEAST', 'ECOLI']
            self.species = self.df_composition['Organism'].unique().tolist() 

            # Sample index for each group 
            # Such as: {'S1': [0, 1, 2, 3, 4, 5], 'S2': [6, 7, 8, 9, 10, 11], 'S3': [12, 13, 14, 15, 16, 17], 'S4': [18, 19, 20, 21, 22, 23, 24], 'S5': [25, 26, 27, 28, 29]}
            self.sample_index_of_each_group = {}
            # Run Name for each group
            self.run_name_of_each_group = {}

            # Sample labels for each group (for plotting) 
            # Such as: {'S1': [1, 2, 3, 4, 5, 6], 'S2': [1, 2, 3, 4, 5, 6], 'S3': [1, 2, 3, 4, 5, 6], 'S4': [1, 2, 3, 4, 5, 6], 'S5': [1, 2, 3, 4, 5, 6]}
            self.labels_of_each_group = {}
            for group in self.group_name:
                row_indices = self.df_samples[self.df_samples['Group'] == group].index
                sample_index = row_indices.values.tolist()
                self.sample_index_of_each_group.update({group:sample_index})
                self.run_name_of_each_group.update({group: self.df_samples.iloc[row_indices, 0].values.tolist()})

                labels = list(range(1, 1+len(sample_index), 1))
                str_labels = [str(i) for i in labels]
                self.labels_of_each_group.update({group:str_labels})


            # Sample serial numbers corresponding to different batches  
            # Such as: {'Batch1': [1,2,3,...], 'Batch2': [31,32,33,...], 'Batch3': [61,62,63,...]}
            self.sample_index_of_each_batch = {}
            for batch in self.batch_name:
                row_indices = self.df_samples[self.df_samples['Batch'] == batch].index
                sample_index = row_indices.values.tolist()
                self.sample_index_of_each_batch.update({batch:sample_index})



            # Species composition of each group 
            # Such as: {'S1': [50, 10, 40], 'S2': [50, 20, 30], 'S3': [50, 25, 25], 'S4': [50, 30, 20], 'S5': [50, 40, 10]}
            self.composition_of_each_group = {}
            for group in self.group_name:
                composition = self.df_composition[group].values.tolist()
                self.composition_of_each_group.update({group:composition})

            # FC thresholds for up- and down-regulated proteins
            self.FC_threshold = 1.0

            # Whether to screen the protein expression matrix so that the number of effective samples of proteins in the two comparison groups is greater than or equal to 2
            self.FilterProteinMatrix = False

            # When searching for differentially expressed proteins, use fixed FC and p-value
            self.Use_Given_PValue_and_FC = False
            self.Given_PValue = 0.05
            self.Given_FC = 1.5

            # Only fixed p-value is used, FC is automatically determined
            self.Only_Use_Given_PValue = False

            # Use p-value list
            self.Use_PValue_List = False
            self.PValue_List = [0.001, 0.01, 0.05, 0.1]

            self.FC_For_Groups = [1.5, 1.5]  # FC for different comparison groups


            # Clustering parameter
            self.FindClusters_resolution = '1.0'



    # Generate sample information template
    def Generate_Samples_Template(self, FolderPath, FileName, Protein_or_Peptide = 'Protein', Software = 'DIANN', SavePath = './'):

        if (Software == 'DIANN') | (Software == 'DIA-NN'):
            if Protein_or_Peptide == 'Protein':
                df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

                if ('Protein.Ids' in df.columns):
                    pass
                else:
                    df.insert(1, 'Protein.Ids', df['Protein.Group'].values)

                Column_Name = df.columns.tolist()
                Run_Name = {'Run Name' : Column_Name[5:]}
                Group = {'Group' : []}
                Batch = {'Batch' : []}

                df_data = pd.concat( [ pd.DataFrame(Run_Name), pd.DataFrame(Group), pd.DataFrame(Batch)] , axis=1 )
                df_data.to_csv(SavePath + 'Samples_Template.csv', index=False)


        if (Software == 'Spectronaut'):
            if Protein_or_Peptide == 'Protein':
                df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)
                #df_R_Label = df[['R.Label']].copy()
                #df_R_Label.drop_duplicates('R.Label', inplace = True)
                #sample_list = df_R_Label['R.Label'].values.tolist()
                #Run_Name = {'Run Name' : sample_list}
                #Group = {'Group' : []}
                #Batch = {'Batch' : []}

                df_R_Label = df[['R.FileName']].copy()
                df_R_Label.drop_duplicates('R.FileName', inplace = True)
                sample_list = df_R_Label['R.FileName'].values.tolist()
                Run_Name = {'Run Name' : sample_list}
                Group = {'Group' : []}
                Batch = {'Batch' : []}

                df_data = pd.concat( [ pd.DataFrame(Run_Name), pd.DataFrame(Group), pd.DataFrame(Batch)] , axis=1 )
                df_data.to_csv(SavePath + 'Samples_Template.csv', index=False)

        if (Software == 'Peaks') | (Software == 'PEAKS'):
            if Protein_or_Peptide == 'Protein':
                # Load Peaks proteins report
                df = pd.read_csv(FolderPath + FileName)

                # Filter columns containing intensity information 
                ColumnName = df.columns.tolist()
                start_index = ColumnName.index('PTM')
                end_index = ColumnName.index('Sample Profile (Ratio)')
                AreaSampleColumns = ColumnName[start_index+1:end_index]

                NewColumns = []
                for c in AreaSampleColumns:
                    NewColumns.append(c[:-5])

                Run_Name = {'Run Name' : NewColumns}
                Group = {'Group' : []}
                Batch = {'Batch' : []}

                df_data = pd.concat( [ pd.DataFrame(Run_Name), pd.DataFrame(Group), pd.DataFrame(Batch)] , axis=1 )
                df_data.to_csv(SavePath + 'Samples_Template.csv', index=False)

        if (Software == 'MaxQuant'):
            if Protein_or_Peptide == 'Protein':
                # Load MaxQuant proteinGroups report
                df = pd.read_csv(FolderPath + FileName, sep='\t')

                # Filter columns containing intensity information 
                search_str = 'LFQ intensity'
                LFQIntensityColumns = [i for i, col in enumerate(df.columns) if search_str in col]

                ColumnNames = df.iloc[:,LFQIntensityColumns].columns.tolist()

                NewColumns = []
                for c in ColumnNames:
                    NewColumns.append(c[14:])

                Run_Name = {'Run Name' : NewColumns}
                Group = {'Group' : []}
                Batch = {'Batch' : []}

                df_data = pd.concat( [ pd.DataFrame(Run_Name), pd.DataFrame(Group), pd.DataFrame(Batch)] , axis=1 )
                df_data.to_csv(SavePath + 'Samples_Template.csv', index=False)



    # Generate composition information template
    def Generate_Composition_Template(self, FolderPath, FileName, SamplesCSVPath, Protein_or_Peptide = 'Protein', Software = 'DIANN', SavePath = './'):

        df_samples = pd.read_csv(SamplesCSVPath)
        group_name = df_samples['Group'].unique().tolist() 


        if (Software == 'DIANN') | (Software == 'DIA-NN'):
            if Protein_or_Peptide == 'Protein':
                df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

                PG_Name = df['Protein.Names'].values.tolist()
                Species_List = []
                for PG in PG_Name:
                    if ';' in PG:
                        pass
                    else:
                        Species_List.append(PG.split('_')[1])
                Unique_Species = list(set(Species_List))
                Unique_Species.sort()

                data_list = [pd.DataFrame({'Organism' : Unique_Species})]
                for group in group_name:
                    data_list.append(pd.DataFrame({group : []}))

                df_data = pd.concat( data_list , axis=1 )
                df_data.to_csv(SavePath + 'Composition_Template.csv', index=False)


        if (Software == 'Spectronaut'):
            if Protein_or_Peptide == 'Protein':
                df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

                PG_Name = df['PG.ProteinNames'].values.tolist()
                Species_List = []
                for PG in PG_Name:
                    if ';' in PG:
                        pass
                    else:
                        Species_List.append(PG.split('_')[1])
                Unique_Species = list(set(Species_List))
                Unique_Species.sort()

                data_list = [pd.DataFrame({'Organism' : Unique_Species})]
                for group in group_name:
                    data_list.append(pd.DataFrame({group : []}))

                df_data = pd.concat( data_list , axis=1 )
                df_data.to_csv(SavePath + 'Composition_Template.csv', index=False)


        if (Software == 'Peaks') | (Software == 'PEAKS'):
            if Protein_or_Peptide == 'Protein':
                # Load Peaks proteins report
                df = pd.read_csv(FolderPath + FileName)

                PG_Name = df['Accession'].values.tolist()
                Species_List = []
                for PG in PG_Name:
                    if ';' in PG:
                        pass
                    else:
                        Species_List.append(PG.split('_')[1])
                Unique_Species = list(set(Species_List))
                Unique_Species.sort()

                data_list = [pd.DataFrame({'Organism' : Unique_Species})]
                for group in group_name:
                    data_list.append(pd.DataFrame({group : []}))

                df_data = pd.concat( data_list , axis=1 )
                df_data.to_csv(SavePath + 'Composition_Template.csv', index=False)


        if (Software == 'MaxQuant'):
            if Protein_or_Peptide == 'Protein':
                # Load MaxQuant proteinGroups report
                df = pd.read_csv(FolderPath + FileName, sep='\t')

                # Check if there are missing values ​​in the 'Taxonomy names' column
                if df['Taxonomy names'].isnull().any():
                    # There are missing values, use 'Fasta headers' column to count species
                    PG_Name = df['Fasta headers'].dropna().tolist()
                    Species_List = []
                    for PG in PG_Name:
                        if 'sp' in PG:
                            a = PG.split()[0]
                            b = a.split('_')[-1]

                            Species_List.append(b)
                    Unique_Species = list(set(Species_List))
                    Unique_Species.sort()
                else:
                    # There are no missing values, and the 'Taxonomy names' column is used to count species.
                    Taxonomy = df['Taxonomy names'].values.tolist()
                    Species_List = []
                    for i in Taxonomy:
                        if ';' in i:
                            pass
                        else:
                            Species_List.append(i)

                    Unique_Species = list(set(Species_List))
                    Unique_Species.sort()

                data_list = [pd.DataFrame({'Organism' : Unique_Species})]
                for group in group_name:
                    data_list.append(pd.DataFrame({group : []}))

                df_data = pd.concat( data_list , axis=1 )
                df_data.to_csv(SavePath + 'Composition_Template.csv', index=False)



    
    # Get protein expression matrix from DIANN report
    def Get_Protein_Groups_Data_From_DIANN_Report(self, FolderPath, FileName, SaveResult = False, SavePath = './', ResultFileName = 'DIANN_result.xlsx'):

        # Read DIA-NN report
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

        # If the dataframe does not contain Protein.Ids, insert a column  (Version = '1.9.2')
        if ('Protein.Ids' in df.columns):
            pass
        else:
            df.insert(1, 'Protein.Ids', df['Protein.Group'].values)

        dict_species = {} # used to store dataframe of different species
        df_unique_list = []
        for sepcies in self.species:
            df_species = df.loc[df['Protein.Names'].str.contains(sepcies)]
            # Delete useless columns
            df_species = df_species.drop([df.columns[1],df.columns[2],df.columns[3],df.columns[4]], axis=1).set_index('Protein.Group')
            # Delete all empty rows in the dataframe
            df_species = df_species.dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan
            
            # Reorder the expression matrix
            new_columns = self.run_name
            #df_species = df_species[new_columns]
            df_species = df_species.reindex(columns = new_columns)

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})

        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Protein']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)


        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)
           
        df_unique.drop(['Organism'], axis=1, inplace=True)
        return df_unique, dict_species



    # Get peptide expression matrix from DIANN report
    def Get_Peptide_Data_From_DIANN_Report(self, FolderPath, FileName, SaveResult = False, SavePath = './', ResultFileName = 'DIANN_result.xlsx'):


        # Read DIA-NN report
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

        dict_species = {} # used to store dataframe of different species
        df_unique_list = []
        for sepcies in self.species:
            df_species = df.loc[df['Protein.Names'].str.contains(sepcies, na = False)]
            # Delete useless columns
            df_species = df_species.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[8],df.columns[9]], axis=1).set_index('Stripped.Sequence')
            # Delete all empty rows in the dataframe
            df_species = df_species.dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan

            # Reorder the expression matrix
            new_columns = self.run_name
            #df_species = df_species[new_columns]
            df_species = df_species.reindex(columns = new_columns)

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})

        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Aggregate by Stripped.Sequence (group sum)
        df_unique = df_unique.groupby('Stripped.Sequence').sum()
        df_unique[df_unique == 0] = np.nan
        # Modify the name of the index column
        df_unique.index.names = ['Peptide']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Add 'Protein' column
        Protein_data = []
        for i in df_unique.index.tolist():
            filtered_df = df[df['Stripped.Sequence'] == i]
            Protein_data.append(filtered_df['Protein.Group'].values[0])
        df_unique.insert(1, 'Protein', Protein_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Peptide_Matrix.csv', index=True)
           
        df_unique.drop(['Organism', 'Protein'], axis=1, inplace=True)
        return df_unique, dict_species


    # Use DIANN's main report (report.tsv)
    def Get_Protein_Groups_Data_From_DIANN_Main_Report(self, FolderPath, FileName, SaveResult = False, SavePath = './', ResultFileName = 'Protein_Matrix.csv'):

        dict_species = {}  # used to store dataframe of different species
        df_unique_list = [] 

        # Sample list
        sample_list = []

        # Read DIA-NN report
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

        # Filter data that meets the following conditions: Q.Value < 0.01, Lib.Q.Value < 0.01, Lib.PG.Q.Value < 0.01, PG.Q.Value < 0.05
        filtered_df = df[(df['Q.Value'] < 0.01) & (df['Lib.Q.Value'] < 0.01) & (df['Lib.PG.Q.Value'] < 0.01) & (df['PG.Q.Value'] < 0.05)]

        # File.Name
        df_Run = filtered_df[['File.Name']].copy()
        df_Run.drop_duplicates('File.Name', inplace = True)
        sample_list = df_Run['File.Name'].values.tolist()
        

        # Filter by species
        for sepcies in self.species:

            # Dictionary for species
            df_species_report = {}

            # Collect protein information for each sample
            for sample in sample_list:

                df_species = filtered_df.loc[(filtered_df['Protein.Names'].str.contains(sepcies)) & (filtered_df['File.Name'] == sample)]

                df_species_2 = df_species[['Protein.Group','PG.MaxLFQ']].copy()
                df_species_2.drop_duplicates('Protein.Group', inplace = True)  
                keys = df_species_2['Protein.Group']
                values = df_species_2['PG.MaxLFQ']
                df_species_dict = dict(zip(keys, values))
                df_species_report[sample] = df_species_dict

            # Delete all empty rows in the dataframe
            df_species = pd.DataFrame(df_species_report).dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan


            # Reorder the expression matrix
            new_columns = self.run_name
            #df_species = df_species[new_columns]
            df_species = df_species.reindex(columns = new_columns)

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)
            df_species = df_species.sort_index()

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})


        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Protein']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)

        df_unique.drop(['Organism'], axis=1, inplace=True)

        return df_unique, dict_species



    def Get_Peptide_Data_From_DIANN_Main_Report(self, FolderPath, FileName, SaveResult = False, SavePath = './', ResultFileName = 'Peptide_Matrix.csv'):

        dict_species = {}  # used to store dataframe of different species
        df_unique_list = [] 

        # Sample list
        sample_list = []

        # Read DIA-NN report
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)

        # Filter column
        df = df[['File.Name', 'Run', 'Protein.Group', 'Protein.Names', 'Stripped.Sequence', 'Precursor.Id', 'Precursor.Normalised', 'Q.Value', 'Lib.Q.Value', 'Lib.PG.Q.Value', 'PG.Q.Value']]

        # Filter data that meets the following conditions: Q.Value < 0.01, Lib.Q.Value < 0.01, Lib.PG.Q.Value < 0.01, PG.Q.Value < 0.05
        df = df[(df['Q.Value'] < 0.01) & (df['Lib.Q.Value'] < 0.01) & (df['Lib.PG.Q.Value'] < 0.01) & (df['PG.Q.Value'] < 0.05)]

        df = df[['File.Name', 'Run', 'Protein.Group', 'Protein.Names', 'Stripped.Sequence', 'Precursor.Id', 'Precursor.Normalised']]

        # Get peptide list
        peptide_list = df['Stripped.Sequence'].unique()

        df_peptide_list = []

        count = 0
        for peptide in peptide_list:
            count += 1
            if (count % 100 == 0):
               print(count)

            # Precursor corresponding to this peptide
            df_peptide = df[df['Stripped.Sequence'] == peptide]
            precursor_list = df_peptide['Precursor.Id'].unique()

            if len(precursor_list) == 1:
                df_peptide_list.append(df_peptide)

            if (len(precursor_list) == 2) | (len(precursor_list) == 3):
                # Add data with the same file name
                filenames = df_peptide['File.Name'].unique()
                for filename in filenames:
                    df_temp = df_peptide[df_peptide['File.Name'] == filename]
                    if df_temp.shape[0] == 1:
                        df_peptide_list.append(df_temp)
                    else:
                        # Sum
                        sum_of_column = df_temp['Precursor.Normalised'].sum()
                        df_temp.loc[:, 'Precursor.Normalised'] = sum_of_column
                        # Keep the first row
                        first_row = df_temp.head(1)
                        df_peptide_list.append(first_row)

            if len(precursor_list) > 3:

                precursor_average_value = []
                top_3_precursor = []
                for precursor in precursor_list:
                    df_precursor = df_peptide[(df_peptide['Precursor.Id'] == precursor)]
                    average = df_precursor['Precursor.Normalised'].mean()
                    precursor_average_value.append(average)

                sorted_precursor_average_value = sorted(enumerate(precursor_average_value), key=lambda x: x[1], reverse=True)
                sorted_indices = [x[0] for x in sorted_precursor_average_value]
                dropped_index = sorted_indices[3:]
                dropped_precursor = []
                for index in dropped_index:
                    dropped_precursor.append(precursor_list[index])

                # Only keep the top 3 precursor
                df_peptide = df_peptide[~df_peptide['Precursor.Id'].isin(dropped_precursor)]


                filenames = df_peptide['File.Name'].unique()
                for filename in filenames:
                    df_temp = df_peptide[df_peptide['File.Name'] == filename]
                    if df_temp.shape[0] == 1:
                        df_peptide_list.append(df_temp)
                    else:
                        # Sum
                        sum_of_column = df_temp['Precursor.Normalised'].sum()
                        df_temp.loc[:, 'Precursor.Normalised'] = sum_of_column
                        # Keep the first row
                        first_row = df_temp.head(1)
                        df_peptide_list.append(first_row)


        df_new = pd.concat(df_peptide_list, axis=0)
        #df_new.to_csv(SavePath + 'df_new.csv', index=True)

        filtered_df = df_new

        # File.Name
        df_Run = filtered_df[['File.Name']].copy()
        df_Run.drop_duplicates('File.Name', inplace = True)
        sample_list = df_Run['File.Name'].values.tolist()

        # Filter by species
        for sepcies in self.species:

            # Dictionary for species
            df_species_report = {}

            # Collect protein information for each sample
            for sample in sample_list:

                df_species = filtered_df.loc[(filtered_df['Protein.Names'].str.contains(sepcies)) & (filtered_df['File.Name'] == sample)]

                df_species_2 = df_species[['Stripped.Sequence','Precursor.Normalised']].copy()
                df_species_2.drop_duplicates('Stripped.Sequence', inplace = True)
                keys = df_species_2['Stripped.Sequence']
                values = df_species_2['Precursor.Normalised']
                df_species_dict = dict(zip(keys, values))
                df_species_report[sample] = df_species_dict

            # Delete all empty rows in the dataframe
            df_species = pd.DataFrame(df_species_report).dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan


            # Reorder the expression matrix
            new_columns = self.run_name
            df_species = df_species.reindex(columns = new_columns)


            #df_species = df_species[new_columns]

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)
            df_species = df_species.sort_index()

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})


        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Peptide']

        df_unique = df_unique.sort_index()

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Add 'Protein' column
        keys = df['Stripped.Sequence']
        values = df['Protein.Group']
        dict_Peptide_Protein = dict(zip(keys, values))

        Protein_data = []
        for i in df_unique.index.tolist():
            #filtered_df = df[df['PEP.StrippedSequence'] == i]
            Protein_data.append(dict_Peptide_Protein[i])
        df_unique.insert(1, 'Protein', Protein_data)
        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Peptide_Matrix.csv', index=True)

        df_unique.drop(['Organism', 'Protein'], axis=1, inplace=True)
        return df_unique, dict_species


    # Get protein expression matrix from Spectronaut report
    def Get_Protein_Groups_Data_From_Spectronaut_Report(self, FolderPath, FileName, 
                                                        SaveResult = False, 
                                                        SavePath = './',
                                                        ResultFileName = 'Spectronaut_result.xlsx'):
        
        
        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Sample list
        sample_list = []
        
        # Load the file and obtain the sample information
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)
        #df_R_Label = df[['R.Label']].copy()
        #df_R_Label.drop_duplicates('R.Label', inplace = True)
        #sample_list = df_R_Label['R.Label'].values.tolist()

        #df = df.dropna(subset=['PG.Quantity'])  # Delete rows where PG.Quantity is NaN

        df_R_Label = df[['R.FileName']].copy()
        df_R_Label.drop_duplicates('R.FileName', inplace = True)
        sample_list = df_R_Label['R.FileName'].values.tolist()


    
        
        for sepcies in self.species:

            # Dictionary for species
            df_species_report = {}

            # Collect protein information for each sample
            for sample in sample_list:

                #df_species = df.loc[(df['PG.ProteinNames'].str.contains(sepcies)) & (df['R.Label'] == sample)]
                df_species = df.loc[(df['PG.ProteinNames'].str.contains(sepcies)) & (df['R.FileName'] == sample)]

                df_species_2 = df_species[['PG.ProteinAccessions','PG.Quantity']].copy()
                df_species_2.drop_duplicates('PG.ProteinAccessions', inplace = True)  
                keys = df_species_2['PG.ProteinAccessions']
                values = df_species_2['PG.Quantity']
                df_species_dict = dict(zip(keys, values))
                df_species_report[sample] = df_species_dict

            # Delete all empty rows in the dataframe
            df_species = pd.DataFrame(df_species_report).dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan


            # Reorder the expression matrix
            new_columns = self.run_name
            df_species = df_species[new_columns]

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})


        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Protein']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)

        df_unique.drop(['Organism'], axis=1, inplace=True)
        return df_unique, dict_species


    # Get peptide expression matrix from Spectronaut report
    def Get_Peptide_Data_From_Spectronaut_Report(self, FolderPath, FileName, 
                                      SaveResult = False, 
                                      SavePath = './',
                                      ResultFileName = 'Spectronaut_result.xlsx'):
        
        
        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Sample list
        sample_list = []

        # Load the file and obtain the sample information
        df = pd.read_csv(FolderPath + FileName, sep = '\t', low_memory=False)
        #df_R_Label = df[['R.Label']].copy()
        #df_R_Label.drop_duplicates('R.Label', inplace = True)
        #sample_list = df_R_Label['R.Label'].values.tolist()

        #df = df.dropna(subset=['PEP.Quantity'])  # Delete rows where PG.Quantity is NaN

        df_R_Label = df[['R.FileName']].copy()
        df_R_Label.drop_duplicates('R.FileName', inplace = True)
        sample_list = df_R_Label['R.FileName'].values.tolist()
    

        for sepcies in self.species:

            # Dictionary for species
            df_species_report = {}

            # Collect protein information for each sample
            for sample in sample_list:

                df_species = df.loc[(df['PG.ProteinNames'].str.contains(sepcies, na = False)) & (df['R.FileName'] == sample)]

                df_species_2 = df_species[['PEP.StrippedSequence','PEP.Quantity']].copy()
                df_species_2.drop_duplicates('PEP.StrippedSequence', inplace = True)  
                keys = df_species_2['PEP.StrippedSequence']
                values = df_species_2['PEP.Quantity']
                df_species_dict = dict(zip(keys, values))
                df_species_report[sample] = df_species_dict

            # Delete all empty rows in the dataframe
            df_species = pd.DataFrame(df_species_report).dropna(axis=0, how='all')
            df_species[df_species == 0] = np.nan

            # Reorder the expression matrix
            new_columns = self.run_name
            df_species = df_species[new_columns]

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})

        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Peptide']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Add 'Protein' column
        keys = df['PEP.StrippedSequence']
        values = df['PG.ProteinAccessions']
        dict_Peptide_Protein = dict(zip(keys, values))

        Protein_data = []
        for i in df_unique.index.tolist():
            #filtered_df = df[df['PEP.StrippedSequence'] == i]
            Protein_data.append(dict_Peptide_Protein[i])
        df_unique.insert(1, 'Protein', Protein_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Peptide_Matrix.csv', index=True)
        
        df_unique.drop(['Organism', 'Protein'], axis=1, inplace=True)
        return df_unique, dict_species




    ## Get protein expression matrix from PEAKS Studio report
    ## lfq.dia.proteins.csv
    #def Get_Protein_Groups_Data_From_PeaksStudio_Report(self, FolderPath, FileName, 
    #                                                  SaveResult = False, 
    #                                                  SavePath = './',
    #                                                  ResultFileName = 'PeaksStudio_result.xlsx'):

    #    dict_species = {} # used to store dataframe of different species
    #    df_unique_list = []

    #    # Load Peaks proteins report
    #    df = pd.read_csv(FolderPath + FileName)

    #    # Filter columns containing intensity information 
    #    ColumnName = df.columns.tolist()
    #    start_index = ColumnName.index('PTM')
    #    end_index = ColumnName.index('Sample Profile (Ratio)')
    #    AreaSampleColumns = ColumnName[start_index+1:end_index]

    #    # Filter rows where the value in the column 'Top' is 'True'
    #    filtered_df = df[df['Top'] == True]
    #    df = filtered_df

    #    # Remove duplicates based on the value of the 'Protein Group' column
    #    #df = df.drop_duplicates(subset='Protein Group')
    #    df = df.drop_duplicates(subset='Accession')

    #    for sepcies in self.species:

    #        df_species = df.loc[df['Accession'].str.contains(sepcies)]

    #        index = ['Accession'] + AreaSampleColumns

    #        df_species = df_species.loc[:,index]
    #        df_species.rename(columns = {"Accession":"Protein Group"}, inplace=True)
    #        df_species['Protein Group'] = df_species['Protein Group'].astype(str).str.split("|").str[0]
    #        df_species = df_species.set_index('Protein Group')

    #        # Delete rows with all 0
    #        df_species = df_species[df_species.ne(0).any(axis=1)]
    #        df_species[df_species == 0] = np.nan

    #        # Reorder the expression matrix
    #        new_columns = []
    #        for name in self.run_name:
    #            new_columns.append(name + ' Area')
    #        #new_columns = self.run_name
    #        df_species = df_species[new_columns]
    #        df_species.columns = self.run_name

    #        # After reordering, there may be rows with all missing values ​​that need to be deleted
    #        missing_rows = df_species.isna().all(axis=1)
    #        df_species = df_species.drop(df_species[missing_rows].index)

    #        df_unique_list.append(df_species)
    #        dict_species.update({sepcies:df_species})


    #    # Merge the above dataframes
    #    df_unique = pd.concat(df_unique_list, axis=0)
    #    # Remove duplicate rows
    #    df_unique = df_unique.drop_duplicates()

    #    # Modify the name of the index column
    #    df_unique.index.names = ['Protein']

    #    # Replace 0 with nan
    #    df_unique[df_unique == 0] = np.nan

    #    # Add 'Organism' column
    #    Organism_data = []
    #    for i in df_unique.index.tolist():
    #        Organism = ''
    #        for sepcies in self.species:
    #            if (i in dict_species[sepcies].index.tolist()):
    #                if (Organism == ''):
    #                    Organism += sepcies
    #                else:
    #                    Organism += (';' + sepcies)

    #        Organism_data.append(Organism)
    #    df_unique.insert(0, 'Organism', Organism_data)

        
    #    # Save protein expression matrix to csv file
    #    if SaveResult:

    #        df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)

    #    df_unique.drop(['Organism'], axis=1, inplace=True)
    #    return df_unique, dict_species



    # Get protein expression matrix from PEAKS Studio report
    # lfq.dia.proteins.csv sl.proteins.csv  dia_db.proteins.csv
    def Get_Protein_Groups_Data_From_PeaksStudio_Report(self, FolderPath, FileName, 
                                                        SaveResult = False, 
                                                        SavePath = './',
                                                        ResultFileName = 'PeaksStudio_result.xlsx'):

        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Load Peaks proteins report
        df = pd.read_csv(FolderPath + FileName)

        # Filter columns containing intensity information 
        ColumnName = df.columns.tolist()
        start_index = None
        end_index = None
        
        if ('sl.proteins' in FileName) | ('dia_db.proteins' in FileName):
            start_index = ColumnName.index('#Peptides')
            end_index = ColumnName.index('#Unique')
        else:
            #  ('lfq.dia.proteins' in FileName)
            start_index = ColumnName.index('PTM')
            end_index = ColumnName.index('Sample Profile (Ratio)')


        AreaSampleColumns = ColumnName[start_index+1:end_index]

        # Filter rows where the value in the column 'Top' is 'True'
        filtered_df = df[df['Top'] == True]
        df = filtered_df

        # Remove duplicates based on the value of the 'Protein Group' column
        df = df.drop_duplicates(subset='Protein Group')


        for sepcies in self.species:

            df_species = df.loc[df['Accession'].str.contains(sepcies)]

            index = ['Accession'] + AreaSampleColumns

            df_species = df_species.loc[:,index]
            df_species.rename(columns = {"Accession":"Protein Group"}, inplace=True)
            df_species['Protein Group'] = df_species['Protein Group'].astype(str).str.split("|").str[0]
            df_species = df_species.set_index('Protein Group')

            # Delete rows with all 0
            df_species = df_species[df_species.ne(0).any(axis=1)]
            df_species[df_species == 0] = np.nan

            # Reorder the expression matrix
            new_columns = []
            for name in self.run_name:
                if ('sl.proteins' in FileName) | ('dia_db.proteins' in FileName):
                    new_columns.append('Area ' + name)
                else:
                    new_columns.append(name + ' Area')
            #new_columns = self.run_name
            #df_species = df_species[new_columns]
            df_species = df_species.reindex(columns = new_columns)

            df_species.columns = self.run_name

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})


        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Protein']

        # Replace 0 with nan
        df_unique[df_unique == 0] = np.nan

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)

        df_unique.drop(['Organism'], axis=1, inplace=True)
        return df_unique, dict_species



    # Get peptide expression matrix from PEAKS Studio report
    def Get_Peptide_Data_From_PeaksStudio_Report(self, FolderPath, FileName, 
                                               SaveResult = False,
                                               SavePath = './',
                                               ResultFileName = 'Peaks_result.xlsx'):


        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Load Peaks proteins report
        df = pd.read_csv(FolderPath + FileName)

        # Remove duplicates based on the value of the 'Peptide' column
        df = df.drop_duplicates(subset='Peptide')

        # Filter columns containing intensity information 
        ColumnName = df.columns.tolist()
        start_index = None
        end_index = None
        
        if ('sl.peptides' in FileName) | ('dia_db.peptides' in FileName):
            start_index = ColumnName.index('1/k0 End')
            end_index = int(ColumnName.index('1/k0 End') + (ColumnName.index('Frame') + 1 - ColumnName.index('1/k0 End'))/2)
        else:
            #  ('lfq.dia.peptides' in FileName)
            start_index = ColumnName.index('1/K0')
            end_index = ColumnName.index('Sample Profile (Ratio)')

        
        AreaSampleColumns = ColumnName[start_index+1:end_index]

        dict_Peptide_Protein = {}

        for sepcies in self.species:

            df_species = df.loc[df['Accession'].str.contains(sepcies, na=False)]

            index = ['Peptide'] + AreaSampleColumns

            Protein_list = df_species['Accession'].values.tolist()
            df_species = df_species.loc[:,index]
            Peptide_list = df_species['Peptide'].values.tolist()
            temp = []
            for i in Peptide_list:
                temp.append(re.sub(r"\(.*?\)", "", i))

                #filtered_df = df_species[df_species['Peptide'] == i]
                #protein_info = filtered_df['Accession'].values[0]
                protein_info = Protein_list[Peptide_list.index(i)]
                protein_info_list = []
                if ':' in protein_info:
                    protein_info_list = protein_info.split(':')
                else:
                    protein_info_list = [protein_info]
                Protein = ''
                for item in protein_info_list:
                    if (Protein == ''):
                        Protein += item.split('|')[0]
                    else:
                        Protein += ';' + item.split('|')[0]

                dict_Peptide_Protein[re.sub(r"\(.*?\)", "", i)] = Protein

            Peptide_list = temp
            df_species['Peptide'] = Peptide_list
            df_species = df_species.set_index('Peptide')

            # Delete rows with all 0
            df_species = df_species[df_species.ne(0).any(axis=1)]
            df_species[df_species == 0] = np.nan

            # Reorder the expression matrix
            new_columns = []
            for name in self.run_name:
                #if ('sl.peptides' in FileName) | ('dia_db.peptides' in FileName):
                #    new_columns.append(name + ' Area')
                #else:
                #    new_columns.append('Area ' + name)

                new_columns.append('Area ' + name)
            #new_columns = self.run_name
            #df_species = df_species[new_columns]
            df_species = df_species.reindex(columns = new_columns)

            df_species.columns = self.run_name

            # After reordering, there may be rows with all missing values ​​that need to be deleted
            missing_rows = df_species.isna().all(axis=1)
            df_species = df_species.drop(df_species[missing_rows].index)

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})


        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Peptide']


        # Merge duplicate peptides
        df_unique = df_unique.groupby(df_unique.index).sum()

        # Replace 0 with nan
        df_unique[df_unique == 0] = np.nan


        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Add 'Protein' column
        Protein_data = []
        for i in df_unique.index.tolist():
            Protein_data.append(dict_Peptide_Protein[i])
        df_unique.insert(1, 'Protein', Protein_data)

        
        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Peptide_Matrix.csv', index=True)

        df_unique.drop(['Organism', 'Protein'], axis=1, inplace=True)

        

        return df_unique, dict_species



    
    # Get protein expression matrix from MaxQuant report
    def Get_Protein_Groups_Data_From_MaxQuant_Report(self, FolderPath, FileName, 
                                                     SaveResult = False, 
                                                     SavePath = './',
                                                     ResultFileName = 'MaxQuant_result.xlsx'):


        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Load MaxQuant proteinGroups report
        df = pd.read_csv(FolderPath + FileName, sep='\t')

        # Delete rows with a value of '+' in the 'Potential consistent' or 'Reverse' columns
        df = df[~(df['Potential contaminant'] == '+')]
        df = df[~(df['Reverse'] == '+')]

        # Filter columns containing intensity information 
        search_str = 'LFQ intensity'
        LFQIntensityColumns = [i for i, col in enumerate(df.columns) if search_str in col]

        for sepcies in self.species:
            # Dataframe of species
            if df['Taxonomy names'].isnull().any():
                df_species = df.loc[df['Fasta headers'].str.contains(sepcies)]
            else:
                df_species = df.loc[df['Taxonomy names'].str.contains(sepcies)]

            index = [0]
            index.extend(LFQIntensityColumns)

            df_species = df_species.iloc[:,index]
            df_species.rename(columns = {"Protein IDs":"Protein Group"}, inplace=True)
            # Correct incorrect values in the "Protein Group" column, such as 'sp|O15523|DDX3Y_HUMAN;sp|O00571|DDX3X_HUMAN;P06634;P24784;sp|Q9NQI0|DDX4_HUMAN'
            PG_list = df_species["Protein Group"].values.tolist()
            new_PG_list = []
            for PG in PG_list:
                new_PG = PG.replace('sp|', '')
                for i in self.species:
                    new_PG = re.sub("\|\w+{0}".format(i), '', new_PG)

                new_PG_list.append(new_PG)
            df_species["Protein Group"] = new_PG_list
            df_species = df_species.set_index('Protein Group')

            # Change the 0 value in dataframe to NaN
            df_species[df_species == 0] = pd.NA

            # Delete all empty rows in the dataframe
            df_species = df_species.dropna(axis=0, how='all')

            # Reorder the expression matrix
            new_columns = []
            for name in self.run_name:
                new_columns.append('LFQ intensity ' + name)
            #new_columns = self.run_name
            df_species = df_species[new_columns]
            df_species.columns = self.run_name

            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})

        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Protein']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Protein_Matrix.csv', index=True)

        df_unique.drop(['Organism'], axis=1, inplace=True)
        return df_unique, dict_species



    # Get peptide expression matrix from MaxQuant report
    def Get_Peptide_Data_From_MaxQuant_Report(self, FolderPath, FileName, 
                                              SaveResult = False, 
                                              SavePath = './',
                                              ResultFileName = 'MaxQuant_result.xlsx'):

        
        dict_species = {} # used to store dataframe of different species
        df_unique_list = []

        # Load MaxQuant peptides report
        df = pd.read_csv(FolderPath + FileName, sep='\t')

        # Delete rows with a value of '+' in the 'Potential consistent' or 'Reverse' columns
        df = df[~(df['Potential contaminant'] == '+')]
        df = df[~(df['Reverse'] == '+')]

        # Filter columns containing intensity information 
        search_str = 'LFQ intensity'
        LFQIntensityColumns = [i for i, col in enumerate(df.columns) if search_str in col]

        dict_Peptide_Protein = {}

        for sepcies in self.species:
            # Dataframe of species
            if df['Taxonomy names'].isnull().any():
                df_species = df.loc[df['Proteins'].str.contains(sepcies, na = False)]

                for row in range(df_species.shape[0]):
                    peptide_name = df_species['Sequence'].values[row]
                    protein_name = df_species['Proteins'].values[row]
                    protein_name_list = []
                    if (';' in protein_name):
                        protein_name_list = protein_name.split(';')
                    else:
                        protein_name_list = [protein_name]
                    protein_name = ''
                    for item in protein_name_list:
                        if (protein_name == ''):
                            if ('|' in item):
                                protein_name += item.split('|')[1]
                        else:
                            if ('|' in item):
                                protein_name += (';' + item.split('|')[1])

                    dict_Peptide_Protein[peptide_name] = protein_name

            else:
                df_species = df.loc[df['Taxonomy names'].str.contains(sepcies, na = False)]

                for row in range(df_species.shape[0]):
                    peptide_name = df_species['Sequence'].values[row]
                    protein_name = df_species['Proteins'].values[row]
                    dict_Peptide_Protein[peptide_name] = protein_name

            

            index = [0]
            index.extend(LFQIntensityColumns)

            df_species = df_species.iloc[:,index]
            df_species = df_species.set_index('Sequence')

            # Change the 0 value in dataframe to NaN
            df_species[df_species == 0] = pd.NA

            # Delete all empty rows in the dataframe
            df_species = df_species.dropna(axis=0, how='all')

            # Reorder the expression matrix
            new_columns = []
            for name in self.run_name:
                new_columns.append('LFQ intensity ' + name)
            #new_columns = self.run_name
            df_species = df_species[new_columns]
            df_species.columns = self.run_name


            df_unique_list.append(df_species)
            dict_species.update({sepcies:df_species})

        # Merge the above dataframes
        df_unique = pd.concat(df_unique_list, axis=0)
        # Remove duplicate rows
        df_unique = df_unique.drop_duplicates()

        # Modify the name of the index column
        df_unique.index.names = ['Peptide']

        # Add 'Organism' column
        Organism_data = []
        for i in df_unique.index.tolist():
            Organism = ''
            for sepcies in self.species:
                if (i in dict_species[sepcies].index.tolist()):
                    if (Organism == ''):
                        Organism += sepcies
                    else:
                        Organism += (';' + sepcies)

            Organism_data.append(Organism)
        df_unique.insert(0, 'Organism', Organism_data)

        # Add 'Protein' column
        Protein_data = []
        for i in df_unique.index.tolist():
            Protein_data.append(dict_Peptide_Protein[i])
        df_unique.insert(1, 'Protein', Protein_data)

        # Save protein expression matrix to csv file
        if SaveResult:

            df_unique.to_csv(SavePath + 'Peptide_Matrix.csv', index=True)

        df_unique.drop(['Organism', 'Protein'], axis=1, inplace=True)
        return df_unique, dict_species




    # Draw a bar chart of sample identification numbers and a line chart of data completeness
    def Plot_Identification_Result_of_Each_Sample(self, df_all, 
                                                  software = 'DIA-NN', Protein_or_Peptide = 'Protein',
                                                  c = ['#8bd2cb', '#68b6fc', '#ff6681', '#7d7d73'],
                                                  savefig = True, 
                                                  savefolder = './'):
    

        groups = self.groups
        total_samples = self.total_samples
        sample_index_of_each_group = list(self.sample_index_of_each_group.values())
        group_name = list(self.sample_index_of_each_group.keys())
        labels_of_each_group = list(self.labels_of_each_group.values())

        # Four lists used to store calculation results
        shared_by_all_runs = [0]*total_samples  
        shared_by_at_least_two_runs = [0]*total_samples  
        only_identify_by_this_run = [0]*total_samples  
        nan_nums_of_all_runs = [0]*total_samples  


        # Loop through each protein or peptide
        index = df_all.index.tolist()
        for i in range(groups):
            for j in range(len(index)):
                df = df_all.iloc[j:j+1, np.r_[sample_index_of_each_group[i]]]
                # shared_by_all_runs, 0 missing values
                if df.isnull().sum().sum() == 0:
                    count = 0
                    for k in df.isnull().values[0]:
                        sample_index = len(sum(sample_index_of_each_group[:i], [])) + count
                        shared_by_all_runs[sample_index] += 1
                        count += 1
                
                # shared_by_at_least_two_runs, 1 - (each_group_samples - 2) missing values
                elif (df.isnull().sum().sum() >=1) & (df.isnull().sum().sum() <= (len(sample_index_of_each_group[i])-2)):
                    count = 0
                    for k in df.isnull().values[0]:
                        sample_index = len(sum(sample_index_of_each_group[:i], [])) + count
                        if k == True:
                            shared_by_at_least_two_runs[sample_index] += 0
                        elif k == False:
                            shared_by_at_least_two_runs[sample_index] += 1
                        count += 1

                # only_identify_by_this_run, unique protein/peptide
                elif df.isnull().sum().sum() == (len(sample_index_of_each_group[i])-1):
                    count = 0
                    for k in df.isnull().values[0]:
                        sample_index = len(sum(sample_index_of_each_group[:i], [])) + count
                        if k == True:
                            only_identify_by_this_run[sample_index] += 0
                        elif k == False:
                            only_identify_by_this_run[sample_index] += 1
                        
                        count += 1

                # Number of missing values in each sample
                count = 0
                for k in df.isnull().values[0]:
                    sample_index = len(sum(sample_index_of_each_group[:i], [])) + count
                    if k == True:
                        nan_nums_of_all_runs[sample_index] += 1
                    count += 1

        # Statistics of the maximum number of identifications and missing values
        identification_max = np.max(np.array(shared_by_all_runs) + np.array(shared_by_at_least_two_runs) + np.array(only_identify_by_this_run))
        if identification_max>0:
            identification_max = math.ceil(identification_max/500)*500
        missing_max = np.max(np.array(nan_nums_of_all_runs))
        if missing_max>0:
            missing_max = math.ceil(missing_max/500)*500

        # Draw histogram
        fig, ax = plt.subplots(1, groups, figsize=(9,5)) 

        gap = 0.06

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if int(height) != 0:
                    if height/identification_max < gap:
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.48*height + gap*identification_max, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial")
                    else:
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.48*height, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial") 

        def autolabel2(rects, bottom):
            count = 0
            for rect in rects:
                height = rect.get_height()
                if int(height) != 0:
                    if (bottom[count]/identification_max < gap) & (height/identification_max < gap) & (int(bottom[count]) != 0):
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.48*height + gap*2*identification_max, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial")
                    elif (bottom[count]/identification_max < gap) & (height/identification_max >= gap)  & (int(bottom[count]) != 0):
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.48*height + gap*identification_max, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial")
                    else:
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.48*height + bottom[count], '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial")
                count+=1

        def autolabel3(rects, bottom):
            count = 0
            for rect in rects:
                height = rect.get_height()
                if int(height) != 0:
                    if (bottom[count]/identification_max < gap*2) & (height/identification_max < gap)  & (int(bottom[count]) != 0):
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.1*height + gap*3*identification_max, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    elif (bottom[count]/identification_max < gap*2) & (height/identification_max >= gap) & (int(bottom[count]) != 0):
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.1*height + gap*2*identification_max, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    else:
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.1*height + bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")

                if (int(height) == 0) & (bottom[count] == 0):
                    plt.text(rect.get_x()+1.1*rect.get_width()/2., gap*identification_max*0.48, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")

                if (int(height) == 0) & (bottom[count] != 0):
                    plt.text(rect.get_x()+1.1*rect.get_width()/2., bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")

                count+=1

        for i in range(groups):
            plt.subplot(1,groups,i+1)

            # shared_by_all_runs
            sample_index_begin = len(sum(sample_index_of_each_group[:i], []))
            sample_index_end = len(sum(sample_index_of_each_group[:(i+1)], []))

            bar1 = plt.bar(labels_of_each_group[i], shared_by_all_runs[sample_index_begin:sample_index_end], width = 0.8, color = c[0]) 
            autolabel(bar1) 

            # shared_by_at_least_two_runs
            bar2 = plt.bar(labels_of_each_group[i], shared_by_at_least_two_runs[sample_index_begin:sample_index_end], bottom = shared_by_all_runs[sample_index_begin:sample_index_end], width = 0.8, color = c[1])
            autolabel2(bar2, bottom = shared_by_all_runs[sample_index_begin:sample_index_end])

            # only_identify_by_this_run
            bar3 = plt.bar(labels_of_each_group[i], only_identify_by_this_run[sample_index_begin:sample_index_end], bottom =  [x + y for x, y in zip(shared_by_all_runs[sample_index_begin:sample_index_end], shared_by_at_least_two_runs[sample_index_begin:sample_index_end])], width = 0.8, color = c[2])
            autolabel3(bar3, bottom = [x + y for x, y in zip(shared_by_all_runs[sample_index_begin:sample_index_end], shared_by_at_least_two_runs[sample_index_begin:sample_index_end])])
        
            #plt.ylim(0, identification_max + int(500*identification_max/3500))
            plt.ylim(0, identification_max*1.2)
            plt.yticks(np.linspace(0, identification_max, 6))
            

            # Draw rectanglea and label group_name
            delta_y = 200
            if identification_max >2000:
                delta_y = int(400*identification_max/3500)
            rectangle = plt.broken_barh([(-0.5,len(sample_index_of_each_group[i]))], (identification_max*1.1, identification_max*0.10), color = '#dadada')
            

            rx, ry = -0.5, identification_max*1.1
            cx = rx + len(sample_index_of_each_group[i])/2
            cy = ry + int(identification_max*0.10)/2

            plt.text(cx, cy/1.01, group_name[i], size=18, ha='center', va='center') 

            axes = plt.gca()

            plt.tick_params(labelsize=14)
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            if i ==0:
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                axes.spines['left'].set_bounds((0, identification_max*1.06)) 
                ylabel = 'Proteins'
                if 'Peptide' in Protein_or_Peptide:
                    ylabel = 'Peptides'
                plt.ylabel('# ' + ylabel, y=0.45, fontsize=16) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

            if i != 0:
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(False)
                axes.spines['bottom'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=False, labelbottom=True, labelleft=False)
                

        if identification_max >= 10000:
            plt.suptitle('Run', x=0.555, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            plt.subplots_adjust(left=0.11, right=1, bottom=0.12, top=1, wspace=0.05, hspace=0.1) 
        elif identification_max < 1000:
            plt.suptitle('Run', x=0.5425, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            plt.subplots_adjust(left=0.085, right=1, bottom=0.12, top=1, wspace=0.05, hspace=0.1) 
        else:
            plt.suptitle('Run', x=0.55, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            plt.subplots_adjust(left=0.1, right=1, bottom=0.12, top=1, wspace=0.05, hspace=0.1) 


        # Save data
        if savefig:
            df_data = pd.DataFrame()

            run_info = []
            group_info = []
            run_of_each_group = []
            for i in range(len(group_name)):
                group_info += [group_name[i]]*len(sample_index_of_each_group[i])
                run_of_each_group +=   self.labels_of_each_group[group_name[i]] 
                run_info += self.run_name_of_each_group[group_name[i]]

            df_data['Run Name'] = run_info
            df_data['Group'] = group_info
            df_data['Run'] = run_of_each_group
            df_data['Full'] = shared_by_all_runs
            df_data['Shared'] = shared_by_at_least_two_runs
            df_data['Unique'] = only_identify_by_this_run
            df_data.to_csv(savefolder + 'RunIdentifications_{0}s.csv'.format(Protein_or_Peptide), index=False)

        if savefig:
            plt.savefig(savefolder + 'RunIdentifications_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()

        


        # Data Completeness
        x_list = [] 
        y_list = [] 

        missing_ratio = df_all.isnull().sum(axis=1) / df_all.shape[1]
        missing_ratio = missing_ratio.values

        for i in range(101):
            
            indices = np.where(missing_ratio <= i/100)
            
            if len(x_list) >= 1:
                if y_list[-1] == len(indices[0]):
                    pass
                else:
                    x_list.append(i)
                    y_list.append(len(indices[0]))

            if len(x_list) == 0:
                x_list.append(i)
                y_list.append(len(indices[0]))

            if len(indices[0]) == df_all.shape[0]:
                break


        # Draw a line chart
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))

        plt.subplot(1,1,1)

        x = np.array(x_list)
        x = 100-x
        y = np.array(y_list)

        plt.plot(x, y , linewidth=2)

        # Vertical dashed line
        plt.axvline( x = 66, linestyle='--', color='gray')
        plt.axvline( x = 75, linestyle='--', color='gray')
        plt.axvline( x = 90, linestyle='--', color='gray')

        def find_closest_index(lst, target):
            closest = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
            return closest

        # Mark the value
        # 100%
        plt.text(x[0]-1, 0.85*y[0], '%s' % int(y[0]), ha='left', va='center', rotation=90, size=12.5, family="Arial")  

        # 90%
        index_90 = find_closest_index(x, 90)
        plt.text(90-1, 0.9*y[index_90], '%s' % int(y[index_90]), ha='left', va='center', rotation=90, size=12.5, family="Arial")

        # 75%
        index_75 = find_closest_index(x, 75)
        plt.text(75-1, 0.92*y[index_75], '%s' % int(y[index_75]), ha='left', va='center', rotation=90, size=12.5, family="Arial")

        # 66%
        index_66 = find_closest_index(x, 66)
        plt.text(66-1, 0.92*y[index_66], '%s' % int(y[index_66]), ha='left', va='center', rotation=90, size=12.5, family="Arial")

        # Minimum
        plt.text(x[-1]-1, 0.92*y[-1], '%s' % int(y[-1]), ha='right', va='center', rotation=90, size=12.5, family="Arial") 

        plt.tick_params(labelsize=14)
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        plt.xlabel('Data Completeness (%)', y=0.5, fontsize=16)
        ylabel = 'Proteins'
        if 'Peptide' in Protein_or_Peptide:
            ylabel = 'Peptides'
        plt.ylabel('# ' + ylabel, y=0.5, fontsize=16)


        plt.xlim(-2, 103 ) 
        xticks = [0, 25, 50, 66, 75, 90, 100]
        plt.xticks(xticks, ['0', '25', '50', '66', '75', '90', '100'])

        plt.ylim(0, math.ceil(max(y_list)/500)*500) 
        plt.yticks(np.linspace(0, math.ceil(max(y_list)/500)*500, 6)) 
        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_linewidth(2) 
        axes.spines['left'].set_linewidth(2) 
        axes.invert_xaxis() 

        if math.ceil(max(y_list)/500)*500 >= 10000:
            plt.subplots_adjust(left=0.22, right=1, bottom=0.13, top=0.97, wspace=0.05)
        elif math.ceil(max(y_list)/500)*500 < 1000:
            plt.subplots_adjust(left=0.17, right=1, bottom=0.13, top=0.97, wspace=0.05)
        else:
            plt.subplots_adjust(left=0.19, right=1, bottom=0.13, top=0.97, wspace=0.05)

        # Save data
        if savefig:
            df_data = pd.DataFrame()
            df_data['Data Completeness (%)'] = x.tolist()
            df_data['# {0}s'.format(Protein_or_Peptide)] = y.tolist()
            df_data.to_csv(savefolder + 'DataCompleteness_{0}s.csv'.format(Protein_or_Peptide), index=False)

            plt.savefig(savefolder + 'DataCompleteness_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True)
        plt.show()
        plt.close()

        
        return shared_by_all_runs, shared_by_at_least_two_runs, only_identify_by_this_run, nan_nums_of_all_runs





    # Draw a bar chart of the cumulative identification results
    def Plot_Cumulative_Identification_Quantity(self, df_all, 
                                                software = 'DIA-NN', 
                                                Protein_or_Peptide = 'Protein', 
                                                label_size = 14,
                                                savefig = True, 
                                                savefolder = './'):
    

        groups = self.groups
        total_samples = self.total_samples
        sample_index_of_each_group = list(self.sample_index_of_each_group.values())
        labels = self.sample_labels
        group_name = list(self.sample_index_of_each_group.keys())

        # Two lists used to store the identification quantity of each sample
        shared_by_all_runs = [0]*total_samples 
        total_by_all_runs = [0]*total_samples 

        # Two lists used to store the names of accumulated/shared proteins/peptides in all samples
        list_shared = [] 
        list_total = [] 

        # Loop through each group
        for i in range(groups):
            for j in range(len(sample_index_of_each_group[i])):
                # Count each sample
                sample_index = len(sum(sample_index_of_each_group[:i], [])) + j
                df = df_all.iloc[:, sample_index:sample_index+1]

                if (sample_index) == 0:
                    # First sample
                    list_shared = df.dropna(axis=0, how='any').index.tolist()
                    list_total = copy.deepcopy(list_shared)
                    shared_by_all_runs[sample_index] = len(list_shared)
                    total_by_all_runs[sample_index] = len(list_total)
                else:
                    # Not first sample
                    # List of proteins/peptides of this sample
                    list_PG = df.dropna(axis=0, how='any').index.tolist()

                    # detect shared proteins/peptides
                    temp = copy.deepcopy(list_shared)
                    #count = 0
                    for k in temp:
                        if k in list_PG:
                            pass # This protein/peptide is shared
                        else:
                            # This protein/peptide is not shared, remove it from list_shared
                            list_shared.remove(k)
                            #count += 1

                    # detect accumulated proteins/peptides
                    temp = copy.deepcopy(list_total)
                    #count2 = 0
                    for k in list_PG:
                        if k in temp:
                            pass # This protein/peptide is already in list_total
                        else:
                            # This protein/peptide is not in list_total
                            list_total.append(k)
                            #count2 += 1

                    shared_by_all_runs[sample_index] = len(list_shared)
                    total_by_all_runs[sample_index] = len(list_total)

        y_lim = np.max(np.array(total_by_all_runs))
        if y_lim>0:
            y_lim = math.ceil(y_lim/500)*500

        # Draw a scatter and line chart
        plt.figure(figsize=(9, 4.5))
        x = list(range(1, 1 + total_samples))
        y1 = shared_by_all_runs  
        y2 = total_by_all_runs  
        delta_y = np.array(y2) - np.array(y1)
        delta_y = delta_y.tolist()
 
        c = ['#8bd2cb', '#68b6fc', '#ff6681']

        # Draw a bar chart
        bar1 = plt.bar(x, y1, width = 0.8, color = c[0]) 
        bar2 = plt.bar(x, delta_y, width = 0.8, bottom = y1, color = c[1]) 

        plt.text(x[0]+0.05, y1[0] + int(60*y_lim/4000), '%s' % int(y1[0]), ha='center', va='bottom', size=12.5, rotation=90, family="Arial") 
        plt.text(x[-1]+0.05, y1[-1] + int(60*y_lim/4000), '%s' % int(y1[-1]), ha='center', va='bottom', size=12.5, rotation=90, family="Arial")
        plt.text(x[-1]+0.05, y2[-1] + int(60*y_lim/4000), '%s' % int(y2[-1]), ha='center', va='bottom', size=12.5, rotation=90, family="Arial")
        plt.xlabel('Cumulative Runs', fontsize=16)  # Cumulative Runs

        ylabel = 'Proteins'
        if 'Peptide' in Protein_or_Peptide:
            ylabel = 'Peptides'
        plt.ylabel('# {0}'.format(ylabel), fontsize=16)

        plt.xlim(0, 1 + total_samples) 
        x = np.linspace(1, total_samples, total_samples)
        plt.xticks(x, labels)
        plt.tick_params(axis='x', labelsize=14) 
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        if y_lim < 10000:
            plt.ylim(0, y_lim*1.13) 
        else:
            plt.ylim(0, y_lim*1.2) 
        plt.yticks(np.linspace(0, y_lim, 6)) 
        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        #axes.spines['left'].set_bounds((0, y_lim)) 
        axes.spines['bottom'].set_linewidth(2) 
        axes.spines['left'].set_linewidth(2) 


        if y_lim >= 10000:
            plt.subplots_adjust(left=0.11, right=1, bottom=0.13, top=1, wspace=0.05)
        elif y_lim < 1000:
            plt.subplots_adjust(left=0.085, right=1, bottom=0.13, top=1, wspace=0.05)
        else:
            plt.subplots_adjust(left=0.1, right=1, bottom=0.13, top=1, wspace=0.05)

        # Save data
        if savefig:

            run_info = []
            for i in range(len(group_name)):
                run_info += self.run_name_of_each_group[group_name[i]]

            df_data = pd.DataFrame()
            df_data['Run Name'] = run_info
            df_data['Cumulative Runs'] = x
            df_data['Shared'] = shared_by_all_runs
            df_data['Total'] = total_by_all_runs
            df_data.to_csv(savefolder + 'CumulativeIdentifications_{0}s.csv'.format(Protein_or_Peptide), index=False)

            plt.savefig(savefolder + 'CumulativeIdentifications_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()

        return shared_by_all_runs, total_by_all_runs





    # The function to make dataset for venn diagram
    def prepare_subsets(self, df_all_list):


        subsets = []
        for df_all in df_all_list:
            temp = set()
            for i in range(df_all.shape[0]):
                # Screen protein/peptide
                df = df_all.iloc[i:i+1, :]
                # If the percentage of missing values for this protein <=50% samples, keep this protein
                if df.isnull().sum().sum() <= int(df_all.shape[1]/2):
                    PG_name = df_all.index.tolist()[i]
                    if ';' in PG_name:
                        temp.add(PG_name.split(';')[0])
                    else:
                        temp.add(PG_name)

            subsets.append(copy.deepcopy(temp))

        return subsets


    def Plot_Veen_Diagram(self, venn_num, subsets, names, Protein_or_Peptide = 'Protein', 
                          savefig = True, savefolder = './'):


        labels = venn.get_labels(subsets, fill=['number'])
        if venn_num == 2:
            fig, ax = venn.venn2(labels, names, fontsize=14)
        if venn_num == 3:
            fig, ax = venn.venn3(labels, names, fontsize=14, legend_fontsize=18)
        if venn_num == 4:
            fig, ax = venn.venn4(labels, names, fontsize=14, legend_fontsize=18)
        if venn_num == 5:
            fig, ax = venn.venn5(labels, names, fontsize=14)
        if venn_num == 6:
            fig, ax = venn.venn6(labels, names, fontsize=14)
        
        
        if savefig:
            plt.savefig(savefolder + 'Comparison_OverlapIdentifications_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True)
        plt.show()
        plt.close()





    # Draw CV distribution graph
    def plot_CV(self, df_all, 
                software = 'DIA-NN', Protein_or_Peptide = 'Protein',
                x_lim = 1.1, linecolor = '#3f6d96', fillin_color = '#cde7fe',
                savefig = True, savefolder = './'):


        groups = self.groups
        total_samples = self.total_samples
        sample_index_of_each_group = list(self.sample_index_of_each_group.values())
        group_name = list(self.sample_index_of_each_group.keys())
    
        PG_list = []
        cv_list = []
        for j in range(groups):
            temp_PG_list = []
            temp_list = []
            for i in range(df_all.shape[0]):

                sample_index_begin = len(sum(sample_index_of_each_group[:j], []))
                sample_index_end = len(sum(sample_index_of_each_group[:(j+1)], []))

                # Process the i-th protein/peptide of group j
                df = df_all.iloc[i:i+1, np.r_[sample_index_of_each_group[j]]]
                # Calculating CV value requires at least 3 data points, ignoring missing values and 0
                # Only count protein/peptide with a proportion of missing values <=50% in each group
                if df.isnull().sum().sum() <= len(sample_index_of_each_group[j])/2:
                    df = df.dropna(axis=1)  # Delete columns containing missing values
                    df_list = df.values[0].tolist() 
                    # Delete 0 value
                    if 0 in df_list:
                        df_list.remove(0)
                    if len(df_list) >= 3:
                        std = np.std(df_list, ddof=1)  # Calculate standard deviation
                        mean = np.mean(df_list)  # Calculate the average value
                        cv = std / mean  # Calculate CV
                        if cv != np.nan:
                            temp_list.append(cv)
                            temp_PG_list.append(df_all.index.values[i])

            cv_list.append(temp_list)
            PG_list.append(temp_PG_list)

        
        fig, ax = plt.subplots(groups, 1, figsize=(4.5,4.5))
        count = 0
        for data in cv_list:
            if data != []:
                count += 1
                # Build a "density" function based on the dataset
                # When you give a value from the X axis to this function, it returns the according value on the Y axis
                density = gaussian_kde(data)
                density.covariance_factor = lambda : .25
                density._compute_covariance()

                # Create a vector of 200 values going from min to max:
                xs = np.linspace(min(data), max(data), 200)

                plt.subplot(groups, 1, count)

                # Make the chart
                # We're actually building a line chart where x values are set all along the axis and y value are
                # the corresponding values from the density function
                plt.plot(xs, density(xs), color = linecolor, lw = 1)
                # Draw a vertical dashed line representing the median
                median_ = np.median(data)
                y_max = max(density(xs))
                plt.axvline(x=median_, ymin=0, ymax=y_max, c=linecolor, ls="--", lw=1.5, label=None)
                plt.text(median_ + 0.01, y_max*0.3, '{:.1%}'.format(median_), ha='left', va='center', color = linecolor, size=14, family="Arial")
                plt.fill_between(xs, 0, density(xs), facecolor = fillin_color) 

                plt.xlim(-0.05, x_lim)
                plt.xticks(np.linspace(0, 1, 6))  
                plt.ylim(-0.1, 0.5+y_max)  

                plt.tick_params(axis='x', labelsize=14) 
                plt.tick_params(axis='y', labelsize=14)
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes = plt.gca()
        
                axes.yaxis.set_label_position('right')
                axes.set_ylabel(group_name[count-1], fontsize=16, rotation=270)

                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                if count == groups:
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(True)
                    axes.spines['left'].set_visible(False)
                    axes.spines['bottom'].set_position(('data', 0)) 
                    axes.spines['left'].set_position(('data', 0))
                    plt.tick_params(axis='both', which='both', bottom=True, left=False, labelbottom=True, labelleft=False) 
                    plt.xlabel(Protein_or_Peptide + ' CV', fontsize=16)
            
                else:
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(False)
                    axes.spines['left'].set_visible(False)
                    axes.spines['bottom'].set_position(('data', 0)) 
                    axes.spines['left'].set_position(('data', 0))
                    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 
        

        plt.subplots_adjust(left=0, right=0.97, bottom=0.13, top=0.98, wspace=None, hspace=0.2) 

        # Save data
        if savefig:
            df_data = pd.DataFrame()
            for i in range(groups):

                add = pd.DataFrame(cv_list[i], columns=[group_name[i] + ' CV'], index = PG_list[i])
                is_duplicate = add.index.duplicated()
                add = add[~is_duplicate]
                df_data = pd.concat([df_data, add], axis=1)

            df_data.index.names = [Protein_or_Peptide]
            df_data.to_csv(savefolder + 'CVDistribution_{0}s.csv'.format(Protein_or_Peptide), index=True)

            plt.savefig(savefolder + 'CVDistribution_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True)
        plt.show()
        plt.close()



    
    # Used to compare the quantitative results of different software/library construction methods
    def Comparison_of_Quantitative_Accuracy_From_FoldChange_CSV(self, 
                                                                fc_path = ['DIA-NN_FC.csv', 'SP_FC.csv', 'Peaks_FC.csv', 'MaxQuant_FC.csv'],
                                                                Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                                                                Compared_softwares = ['DIA-NN', 'Peaks', 'Spectronaut', 'MaxQuant'], 
                                                                linecolor = ['#E95B1B', '#B0203F', '#A14EE0', '#1883B8', '#39A139'],
                                                                fillin_color = ['#F4AC8C', '#ED9DAE', '#D39FEB', '#67BFEB', '#90D890'],
                                                                Protein_or_Peptide = 'Protein',
                                                                ylabel_size = 14,
                                                       
                                                                scatter_plot_box_position_list = [19, 19, 19, 19],
                                                                scatter_plot__xlim = [[-2, 20], [-2, 20], [-2, 20], [-2, 20]],
                                                                scatter_plot__xticks = [np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7)],
                                                                scatter_plot__xticklabels = [['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16']],
                                                                scatter_plot__ylim = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]],
                                                                scatter_plot__yticks = [np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5)],

                                                                savefig = True, savefolder = './'):

        # Read FC files
        FC_df_list = []
        for csv_path in fc_path:
            df = pd.read_csv(csv_path)
            FC_df_list.append(df)


        # Number of software/library construction methods compared
        software_num = len(fc_path)
        

        groups = self.groups
        total_samples = self.total_samples
        sample_index_of_each_group = list(self.sample_index_of_each_group.values())
        group_name = list(self.sample_index_of_each_group.keys())

        Compared_groups_index = []
        for item in Compared_groups_label:
            item_A = item.split('/')[0]
            item_B = item.split('/')[1]
            Compared_groups_index.append([group_name.index(item_A), group_name.index(item_B)])

        # Actual protein/peptide fold change
        Compared_groups_true_fc = {}
        for species in self.species:
            true_fc_list = []
            for compared_index in Compared_groups_index:
                group_name_A = group_name[compared_index[0]]
                group_name_B = group_name[compared_index[1]]
                true_fc = (self.df_composition.loc[self.species.index(species), group_name_A])/(self.df_composition.loc[self.species.index(species), group_name_B])
                true_fc_list.append(true_fc)

            Compared_groups_true_fc[species] = true_fc_list


        for compared_groups in Compared_groups_label:

            # Draw violin diagram
            fig, ax = plt.subplots(1, len(self.species), figsize=(6.5,4.5))

            y_label_for_export = []

            # Evaluate the y-axis scale range (data distribution range)
            y_data_range = []
            for species in self.species:
                y_data_range.append(Compared_groups_true_fc[species][Compared_groups_label.index(compared_groups)])
            log2_y_data_range_top = math.ceil(np.log2(max(y_data_range)) + 0.5)
            log2_y_data_range_bottom = math.floor(np.log2(min(y_data_range)) - 0.5)
            log2_y_top = log2_y_data_range_top +1
            log2_y_bottom = log2_y_data_range_bottom -2
            log2_y_top = min([abs(log2_y_top), abs(log2_y_bottom)])
            log2_y_bottom = -log2_y_top

            # Record drawing data
            violinplot_csv_data = []

            comparision_names_list = []  # Record the name of the comparison groups
            comparision_p_value = []  # t-test p-value
            comparision_Mean_Difference = []  # Mean Difference
            comparision_cohen_d = []  # cohen_d
            comparision_ci = []  # 95% confidence range

            for i in range(len(self.species)):
                plt.subplot(1, len(self.species), i+1)
                axes = plt.gca()

                # Screening for shared proteins/peptides
                shared_PG_names = None 
                for software_index in range(software_num):

                    df_software = FC_df_list[software_index]
                    filtered_df = df_software[df_software['Organism'] == self.species[i]]
                    filtered_df = filtered_df.dropna(subset=[compared_groups + ' log2FC'])
                    if shared_PG_names is None:
                        shared_PG_names = filtered_df[Protein_or_Peptide].values.tolist()
                    else:
                        set1 = set(shared_PG_names)
                        set2 = set(filtered_df[Protein_or_Peptide].values.tolist())
                        common_elements = set1 & set2
                        shared_PG_names = list(common_elements)



                data = []
                true_fc = []

                
                data_add_to_csv = []
                for software_index in range(software_num):

                    df_software = FC_df_list[software_index]
                    filtered_df = df_software[df_software['Organism'] == self.species[i]]
                    filtered_df = filtered_df.dropna(subset=[compared_groups + ' log2FC'])

                    # Filter rows for shared proteins/peptides
                    filtered_df = filtered_df[filtered_df[Protein_or_Peptide].isin(shared_PG_names)]
                    # Sort by protein/peptide name in ascending order to obtain a uniform row index
                    filtered_df.sort_values(by=Protein_or_Peptide, ascending=True, inplace=True)

                    data_add = filtered_df[compared_groups + ' log2FC'].values.tolist()

                    float_list = [num for num in data_add if not math.isnan(num)]

                    if len(float_list) == 0:
                        float_list = [10000]*10000

                    data.append(float_list)

                    # Add to drawing data
                    if (software_index == 0):
                        df_to_add = filtered_df[['Organism', Protein_or_Peptide]]
                        df_to_add[Compared_softwares[software_index] + ' ' + compared_groups + ' log2FC'] = filtered_df[compared_groups + ' log2FC'].values.tolist()
                        df_to_add.reset_index(drop=True, inplace=True)
                        data_add_to_csv.append(df_to_add)
                    else:
                        df_to_add = pd.DataFrame({Compared_softwares[software_index] + ' ' + compared_groups + ' log2FC' : filtered_df[compared_groups + ' log2FC'].values.tolist()})
                        #df_to_add[Compared_softwares[software_index] + ' ' + compared_groups + ' log2FC'] = filtered_df[compared_groups + ' log2FC'].values.tolist()
                        data_add_to_csv.append(df_to_add)

                df_temp = pd.concat(data_add_to_csv, axis=1)
                violinplot_csv_data = violinplot_csv_data + [df_temp]  #copy.deepcopy(data_add_to_csv)


                position = list(range(1, 1+software_num))

                violinplot_data = data


                
                # Draw a violin plot
                parts = axes.violinplot(violinplot_data, position, widths=0.8,
                                    showmeans=False, showmedians=False, showextrema=False)


                # Modify the colors of the violin diagram
                count = 0
                for pc in parts['bodies']:
                    pc.set_facecolor(fillin_color[count])
                    pc.set_edgecolor('white')
                    pc.set_alpha(1)

                    count += 1

                
                max_log2fc = None
                # Draw box plot
                for j in range(software_num):
                    b = axes.boxplot(violinplot_data[j],
                            positions=[j+1], # position of the box
                            widths=0.06*2, 
                            meanline=False,
                            showmeans=False,
                            meanprops={'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                            medianprops = {'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                            showcaps = False,  
                            showfliers = False, 
                            patch_artist=True, 
                            boxprops = {'color':linecolor[j], 'facecolor':'white', 'linewidth':'1.5'},
                            whiskerprops = {'color':linecolor[j], 'linewidth':'1.5'} 
                            )

                    if max_log2fc is None:
                        max_log2fc = b['whiskers'][1].get_ydata()[1] 
                    else:
                        if (b['whiskers'][1].get_ydata()[1] > max_log2fc):
                            max_log2fc = b['whiskers'][1].get_ydata()[1]



                # p-value of t-test
                p_value_and_position = []  # p value, starting position of the line segment, ending position of the line segment, height of the line segment
                #comparision_names_list = []
                #comparision_p_value = []
                #max_log2fc = max(max(sublist) for sublist in violinplot_data)
                def T_Test(data1, data2):
                    t_statistic, p_value = ttest_ind(data1, data2, equal_var=False)
                    return p_value

                from numpy import std, mean, sqrt
                # Mean Difference
                def MeanDifference(x, y):
                    return mean(x) - mean(y)

                
                #correct if the population S.D. is expected to be equal for the two groups.
                def cohen_d(x,y):
                    nx = len(x)
                    ny = len(y)
                    dof = nx + ny - 2
                    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

                def CI(x,y):
                    nx = len(x)
                    ny = len(y)
                    dof = nx + ny - 2
                    d = (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)
                    SE = sqrt((nx+ny)/(nx*ny) + d*d/(2*(nx+ny)))
                    ci = [d-1.96*SE, d+1.96*SE]
                    return ci


                if len(violinplot_data) == 2:
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[1]), 1, 2, log2_y_top -(log2_y_top/4)*0.5])
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[1]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[1]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[1]))
                    # comparision_cohen_d
                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[1]))
                    # CI
                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[1]))

                if len(violinplot_data) == 3:
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[2]), 1, 3, log2_y_top -(log2_y_top/4)*0.5])
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[1]), 1, 1.95, log2_y_top -(log2_y_top/4)*1.0])
                    p_value_and_position.append([T_Test(violinplot_data[1], violinplot_data[2]), 2.05, 3, log2_y_top -(log2_y_top/4)*1.0])

                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[2]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[1]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[1], Compared_softwares[2]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[2]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[1]))
                    comparision_p_value.append(T_Test(violinplot_data[1], violinplot_data[2]))

                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[2]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[1]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[1], violinplot_data[2]))

                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[2]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[1]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[1], violinplot_data[2]))

                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[2]))
                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[1]))
                    comparision_ci.append(CI(violinplot_data[1], violinplot_data[2]))
                    
                    count = 0
                    for data in p_value_and_position:
                        p_value = data[0]
                        abs_d = abs(comparision_cohen_d[i*len(p_value_and_position): (i+1)*len(p_value_and_position)][count])
                        if (abs_d <= 0.2) & (count == 0):
                            p_value_and_position[1][3] += (log2_y_top/4)*0.5
                            p_value_and_position[2][3] += (log2_y_top/4)*0.5

                        count+=1

                if len(violinplot_data) == 4:
                    
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[3]), 1, 4, log2_y_top -(log2_y_top/4)*0.5])
                    p_value_and_position.append([T_Test(violinplot_data[1], violinplot_data[3]), 2, 4, log2_y_top -(log2_y_top/4)*1.0])
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[2]), 1, 3, log2_y_top -(log2_y_top/4)*1.5])
                    p_value_and_position.append([T_Test(violinplot_data[0], violinplot_data[1]), 1, 1.95, log2_y_top -(log2_y_top/4)*2.0])
                    p_value_and_position.append([T_Test(violinplot_data[1], violinplot_data[2]), 2.05, 2.95, log2_y_top -(log2_y_top/4)*2.0])
                    p_value_and_position.append([T_Test(violinplot_data[2], violinplot_data[3]), 3.05, 4, log2_y_top -(log2_y_top/4)*2.0])

                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[3]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[1], Compared_softwares[3]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[2]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[0], Compared_softwares[1]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[1], Compared_softwares[2]))
                    comparision_names_list.append('{0} {1} {2}'.format(self.species[i], Compared_softwares[2], Compared_softwares[3]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[3]))
                    comparision_p_value.append(T_Test(violinplot_data[1], violinplot_data[3]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[2]))
                    comparision_p_value.append(T_Test(violinplot_data[0], violinplot_data[1]))
                    comparision_p_value.append(T_Test(violinplot_data[1], violinplot_data[2]))
                    comparision_p_value.append(T_Test(violinplot_data[2], violinplot_data[3]))

                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[3]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[1], violinplot_data[3]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[2]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[0], violinplot_data[1]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[1], violinplot_data[2]))
                    comparision_Mean_Difference.append(MeanDifference(violinplot_data[2], violinplot_data[3]))

                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[3]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[1], violinplot_data[3]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[2]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[0], violinplot_data[1]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[1], violinplot_data[2]))
                    comparision_cohen_d.append(cohen_d(violinplot_data[2], violinplot_data[3]))

                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[3]))
                    comparision_ci.append(CI(violinplot_data[1], violinplot_data[3]))
                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[2]))
                    comparision_ci.append(CI(violinplot_data[0], violinplot_data[1]))
                    comparision_ci.append(CI(violinplot_data[1], violinplot_data[2]))
                    comparision_ci.append(CI(violinplot_data[2], violinplot_data[3]))

                    count = 0
                    for data in p_value_and_position:
                        p_value = data[0]
                        abs_d = abs(comparision_cohen_d[i*len(p_value_and_position): (i+1)*len(p_value_and_position)][count])
                        if (abs_d <= 0.2) & (count == 0):
                            p_value_and_position[1][3] += (log2_y_top/4)*0.5
                            p_value_and_position[2][3] += (log2_y_top/4)*0.5
                            p_value_and_position[3][3] += (log2_y_top/4)*0.5
                            p_value_and_position[4][3] += (log2_y_top/4)*0.5
                            p_value_and_position[5][3] += (log2_y_top/4)*0.5
                        if (abs_d <= 0.2) & (count == 1):
                            p_value_and_position[2][3] += (log2_y_top/4)*0.5
                            p_value_and_position[3][3] += (log2_y_top/4)*0.5
                            p_value_and_position[4][3] += (log2_y_top/4)*0.5
                            p_value_and_position[5][3] += (log2_y_top/4)*0.5
                        if (abs_d <= 0.2) & (count == 2):
                            p_value_and_position[3][3] += (log2_y_top/4)*0.5
                            p_value_and_position[4][3] += (log2_y_top/4)*0.5
                            p_value_and_position[5][3] += (log2_y_top/4)*0.5

                        count+=1


                count = 0
                for data_ in p_value_and_position:
                    abs_d = abs(comparision_cohen_d[i*len(p_value_and_position): (i+1)*len(p_value_and_position)][count])
                    if abs_d > 0.2:
                        p_value = data_[0]
                        # Draw black line
                        if p_value >= 0.05:
                            pass
                        else:
                            plt.plot([data_[1], data_[2]], [data_[3], data_[3]], color='black')
                            plt.plot([data_[1], data_[1]], [data_[3]-(log2_y_top/4)*0.12, data_[3]], color='black')
                            plt.plot([data_[2], data_[2]], [data_[3]-(log2_y_top/4)*0.12, data_[3]], color='black')
                        # Annotating p-values
                        if p_value >= 0.05:
                            pass
                        elif p_value <0.001:
                            #plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, "${p}$" + '=' + format(data_[0], '.1E'), horizontalalignment='center', fontsize=7, color='black')
                            if (data_[2] - data_[1] < 1.1):
                                plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, format(data_[0], '.0E'), horizontalalignment='center', fontsize=12, color='black')
                            else:
                                plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, format(data_[0], '.0E'), horizontalalignment='center', fontsize=12, color='black')
                        else:
                            #plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, "${p}$" + '={0}'.format(str(round(data_[0], 3))), horizontalalignment='center', fontsize=7, color='black')
                            if (data_[2] - data_[1] < 1.1):
                                plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, '{0}'.format(str(round(data_[0], 3))), horizontalalignment='center', fontsize=12, color='black')
                            else:
                                plt.text((data_[1] + data_[2])/2, data_[3]+(log2_y_top/4)*0.1, '{0}'.format(str(round(data_[0], 3))), horizontalalignment='center', fontsize=12, color='black')

                    count += 1

                # Draw the rectangle and species name on top of the image
                rectangle = plt.broken_barh([(0.5, 0.5 + software_num)], (log2_y_top*1.06, log2_y_top*0.25), color = '#dadada')

                rx, ry = 0.5, log2_y_top*1.06
                cx = rx + software_num/2
                cy = ry + log2_y_top*0.25/2

                plt.text(cx, cy/1.01, self.species[i], size=18, ha='center', va='center')  


                # Draw horizontal dashed line
                true_fc = Compared_groups_true_fc[self.species[i]][Compared_groups_label.index(compared_groups)]
                dashed_line_y = np.log2(true_fc)  # np.log2
                plt.axhline(y = dashed_line_y, color = 'black', linestyle = '--')


                plt.ylim(log2_y_bottom, log2_y_top*1.31)
                plt.yticks(np.array([log2_y_bottom, log2_y_bottom/2, 0, log2_y_top/2, log2_y_top]), ["{:.1f}".format(log2_y_bottom), "{:.1f}".format(log2_y_bottom/2), '0.0', "{:.1f}".format(log2_y_top/2), "{:.1f}".format(log2_y_top)])
                #plt.yticks(np.array([-4, -2, 0, 2, 4]), ['-4.0', '-2.0', '0.0', '2.0', '4.0'])

                plt.xlim(0.5, 0.5+software_num)

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(False)
                axes.spines['left'].set_visible(False)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 

                if i==0:
                    axes.spines['left'].set_visible(True)
                    plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 
                    plt.ylabel('log$_{2}$FC', fontsize=16)  # Log2FC 

                    plt.xticks([])
                    axes.spines['bottom'].set_visible(False)
                    axes.spines['left'].set_bounds((log2_y_bottom, log2_y_top))

                    plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True)
                else:
                    plt.xticks([])
                    axes.spines['bottom'].set_visible(False)

                    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


                # The number of shared proteins
                shared_PG_numbers = len(shared_PG_names)
                plt.text((software_num+1)/2, log2_y_bottom*0.99, r'$\it{n}\rm{=' + '{0}'.format(str(shared_PG_numbers)) + '}$', ha='center', va='center', size=12, family="Arial") 


                ## Mark the number of proteins/peptides used for the drawing
                #count = 0
                #for index in position:
                #    num = len(violinplot_data[count])
                #    if num == 10000:
                #        num = 0
                #    plt.text(index, -4.5, r'$\it{n}\rm{=' + '{0}'.format(str(num)) + '}$', ha='center', va='bottom', rotation=90, size=12, family="Arial") 
                #    count += 1
                

                # Calculate and mark the difference between the detected fold change and the true fold change
                delta_fc_list = []
                abs_delta_fc_list = []
                for software_index in range(software_num):
                    delta_fc = np.median(violinplot_data[software_index]) - np.log2(true_fc)

                    abs_delta_fc_list.append(abs(delta_fc))
                    delta_fc_list.append(delta_fc)

                min_index = abs_delta_fc_list.index(min(abs_delta_fc_list)) 

                # Annotate the fc difference for each species
                for software_index in range(software_num):
                    if len(violinplot_data[software_index]) != 10000:
                        if software_index == min_index:
                            plt.text(position[software_index], log2_y_bottom*0.8, '{:+.2f}'.format(delta_fc_list[software_index]), ha='left', va='center', size=11, color = 'black', family="Arial", rotation=90) 
                        else:
                            plt.text(position[software_index], log2_y_bottom*0.8, '{:+.2f}'.format(delta_fc_list[software_index]), ha='left', va='center', size=11, color = '#aeaeae', family="Arial", rotation=90) 

            # Save csv
            result_data = pd.concat(violinplot_csv_data, axis=0)
            result_data.to_csv(savefolder + 'Comparison_FoldChange_{0}s_{1}_vs_{2}.csv'.format(Protein_or_Peptide, compared_groups.split('/')[0], compared_groups.split('/')[1]), index=False) 

            comparision_a = []
            comparision_b = []
            comparision_c = []
            for name in comparision_names_list:
                comparision_a.append(name.split(' ')[0])
                comparision_b.append(name.split(' ')[1])
                comparision_c.append(name.split(' ')[2])

            CI_list = []
            for ci in comparision_ci:
                CI_list.append(str(ci))

            result_p = pd.DataFrame({'Organism': comparision_a,
                                     'Comparision A': comparision_b,
                                     'Comparision B': comparision_c,
                                     'p-value': comparision_p_value,
                                     'Mean Difference': comparision_Mean_Difference,
                                     "Cohen's d": comparision_cohen_d,
                                     '95% CI': CI_list})
            result_p.to_csv(savefolder + 'Comparison_FoldChange_Stats_{0}s_{1}_vs_{2}.csv'.format(Protein_or_Peptide, compared_groups.split('/')[0], compared_groups.split('/')[1]), index=False) 


            plt.subplots_adjust(left=0.12, right=1, bottom=0.02, top=1, wspace=0.1)

            if savefig:
                plt.savefig(savefolder + 'Comparison_FoldChange_{0}s_{1}_vs_{2}.svg'.format(Protein_or_Peptide, compared_groups.split('/')[0], compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)

            plt.show()
            plt.close()






    # Quantitative accuracy violin plot + scatter plot
    def Comparison_of_Quantitative_Accuracy_1_Software(self, 
                                                        dict_species_software_1,
                                                        Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                                                        Compared_softwares = ['Spectronaut'], 
                                                        linecolor = ['#3f6d96', '#689893', '#8b2c3c', '#ff9900', '#996633', '#660066', '#006600', '#ff3300'],
                                                        fillin_color = ['#cde7fe', '#d9f0ee', '#ffcdd5', '#ffff66', '#ffcc99', '#cc00cc', '#00cc00', '#ff9933'],
                                                        Protein_or_Peptide = 'Protein',
                                                        ylabel_size = 14,

                                                        scatter_plot_box_position_list = [19, 19, 19, 19],
                                                        scatter_plot__xlim = [[-2, 20], [-2, 20], [-2, 20], [-2, 20]],
                                                        scatter_plot__xticks = [np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7)],
                                                        scatter_plot__xticklabels = [['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16']],
                                                        scatter_plot__ylim = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]],
                                                        scatter_plot__yticks = [np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5)],

                                                        savefig = True, savefolder = './'):


        groups = self.groups
        total_samples = self.total_samples
        sample_index_of_each_group = list(self.sample_index_of_each_group.values())
        group_name = list(self.sample_index_of_each_group.keys())

        Compared_groups_index = []
        for item in Compared_groups_label:
            item_A = item.split('/')[0]
            item_B = item.split('/')[1]
            Compared_groups_index.append([group_name.index(item_A), group_name.index(item_B)])

        # Count the actual protein/peptide change fold
        Compared_groups_true_fc = {}
        for species in self.species:
            true_fc_list = []
            for compared_index in Compared_groups_index:
                # True FC
                group_name_A = group_name[compared_index[0]]
                group_name_B = group_name[compared_index[1]]
                true_fc = (self.df_composition.loc[self.species.index(species), group_name_A])/(self.df_composition.loc[self.species.index(species), group_name_B])
                true_fc_list.append(true_fc)

            Compared_groups_true_fc[species] = true_fc_list


        # Delete protein/peptide data from multiple species

        lists = []
        for species in self.species:
            lists.append(dict_species_software_1[species].index.tolist())

        common_elements = list(set.intersection(*map(set, lists)))  # proteins/peptides from multiple species

        for species in self.species:
            dict_species_software_1[species] = dict_species_software_1[species].drop(index=common_elements)


        # Calculate the ratio of average proteins/peptides intensity between groups A and B
        def calculate_two_groups_fc(df, group_A_index, group_B_index):

            # 3 lists, used to store the A/B ratio, average value of group A, and average value of group B
            fc_list = []
            group_A_list = []
            group_B_list = []

            PG_name_list = []

            for i in range(df.shape[0]):

                df_group_A = df.iloc[i:i+1, group_A_index]
                df_group_B = df.iloc[i:i+1, group_B_index]

                
                # Compare protein groups/peptides with missing values less than 50% in two groups
                if (df_group_A.isnull().sum().sum() <= len(group_A_index)/2) & (df_group_B.isnull().sum().sum() <= len(group_B_index)/2):

                    # Delete 0 and missing values
                    def move_0_and_nan(df):
                        df = df.dropna(axis=1) 
                        df_list = df.values[0].tolist() 
                        if 0 in df_list:
                            df_list.remove(0)
                        return df_list

                    df_group_A_list = move_0_and_nan(df_group_A)
                    df_group_B_list = move_0_and_nan(df_group_B)

                    fc_list.append(mean(df_group_A_list)/mean(df_group_B_list))
                    group_A_list.append(mean(df_group_A_list))
                    group_B_list.append(mean(df_group_B_list))

                    PG_name_list.append(df.index.tolist()[i])

            return fc_list, group_A_list, group_B_list, PG_name_list

        # Calculate the FC values of the Human, Ecoli, and Yeast protein groups for all comparison groups

        fc_list_software_1 = {}

        for species in self.species:
            temp = []
            for i in Compared_groups_index:
                fc_list, group_A_list, group_B_list, PG_name_list = calculate_two_groups_fc(dict_species_software_1[species], sample_index_of_each_group[i[0]], sample_index_of_each_group[i[1]])
                temp.append([fc_list, group_B_list, PG_name_list])

            fc_list_software_1[species] = temp


        # Draw violin diagram

        label = []
        y_label = []
        species = []
        data = []  
        group_B_data = [] 
        true_fc = [] 
        PG_name_data = [] 
        y_label_for_export = []

        # Traverse the drawing information of each species
        for i in self.species:
            for j in range(len(Compared_groups_index)):
                label.append(Compared_groups_label[j]) 

                y_label.append('log$_{2}$(' + Compared_groups_label[j].replace('_','\_') + ')')
                y_label_for_export.append('Log2(%s)-(%s)'%(Compared_groups_label[j], i))
                species.append(i)
                data.append([fc_list_software_1[i][j][0]])
                group_B_data.append([fc_list_software_1[i][j][1]])
                true_fc.append(Compared_groups_true_fc[i][j])

                PG_name_data.append([fc_list_software_1[i][j][2]])


        # Save data
        if savefig:
            df_data = pd.DataFrame()
            
            for k in range(len(Compared_softwares)):

                for i in range(len(self.species)):

                    df_temp = pd.DataFrame()
                    for j in range(len(Compared_groups_label)):
                    
                        # Protein name
                        PG_data = PG_name_data[i*len(Compared_groups_label) + j][k]

                        plot_data = data[i*len(Compared_groups_label) + j][k]
                        log2_plot_data = np.log2(plot_data)

                        plot_B_data = group_B_data[i*len(Compared_groups_label) + j][k]
                        log2_plot_B_data = np.log2(plot_B_data)
                        
                        df_compared_group_data = pd.DataFrame({Compared_groups_label[j] + ' log2FC': log2_plot_data,
                                                               'log2' +Compared_groups_label[j].split('/')[1]: log2_plot_B_data},
                                                              index = PG_data)

                        # Before merging df, deduplicate the index to avoid merging errors
                        is_duplicate = df_compared_group_data.index.duplicated()
                        df_compared_group_data = df_compared_group_data[~is_duplicate]
                        
                        df_temp = pd.concat( [df_temp, df_compared_group_data], axis=1 )


                    df_temp.insert(0, Protein_or_Peptide, df_temp.index.tolist()) 
                    df_temp.reset_index(drop=True, inplace=True) 
                    df_temp.insert(0, 'Organism', [self.species[i]]*df_temp.shape[0])

                    df_data = pd.concat( [df_data, df_temp], axis=0 ) 


            df_data.to_csv(savefolder + 'FoldChange_{0}s.csv'.format(Protein_or_Peptide), index=False)


        for i in range(len(self.species)*len(Compared_groups_index)):
            # There may be a situation where the data of the two groups being compared is empty
            for index in range(len(data[i])):
                if data[i][index] == []:
                    data[i][index] = [10000]*99999


        
        if savefig:
            # Legend
            fig_legend = plt.figure(figsize=(2,2))
 
            axes = plt.gca()
            parts = axes.violinplot([[10000]]*len(self.species), list(range(1, len(self.species)+1)), widths=0.8,
                                    showmeans=False, showmedians=False, showextrema=False)
            count = 0
            for pc in parts['bodies']:
                pc.set_facecolor(fillin_color[count])
                pc.set_edgecolor('white')
                pc.set_alpha(1)

                count += 1

            max_label_length = len(max(self.species, key=len))
            fontsize = 16 - 4.5*int(max_label_length/15)
            axes.legend(labels = self.species, title='Organism', title_fontsize=18, fontsize = fontsize, 
                        loc = 'center',
                        markerfirst=True, markerscale=2.0) 

            plt.ylim(-5, 5)
            plt.xlim(-5, 5)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

            plt.xticks([])
            plt.yticks([])

            plt.savefig(savefolder + 'Legend_Organisms.svg', dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()


        # Violin Plot
        for software_index in range(len(Compared_softwares)):

            box_lenend_info = []

            fig, ax = plt.subplots(1, len(Compared_groups_label), figsize=(6.5,4.5), gridspec_kw={'width_ratios': [1]*len(Compared_groups_index)}) 
            
            # Traverse comparison groups
            for i in range(1*len(Compared_groups_index)):
                plt.subplot(1, len(Compared_groups_index), i+1)
                axes = plt.gca()

                position =  list(range(1, len(self.species)+1)) 
                violinplot_data = [] 
                for species_index in range(len(self.species)):
                    violinplot_data.append(np.log2(data[species_index*len(Compared_groups_index) + i][software_index])) 


                # Draw a gray square
                rectangle = plt.broken_barh([(0.5, 0.5 + len(self.species))], (5.25, 1), color = '#dadada')

                rx, ry = 0.5, 5.25
                cx = rx + len(self.species)/2
                cy = ry + 1/2

                plt.text(cx, cy/1.01, Compared_groups_label[i], size=18, ha='center', va='center')  
            
                # Draw violin plot
                parts = axes.violinplot(violinplot_data, position, widths=0.8,
                                    showmeans=False, showmedians=False, showextrema=False)

                # Modify the colors of the violin diagram
                count = 0
                for pc in parts['bodies']:
                    pc.set_facecolor(fillin_color[count])
                    pc.set_edgecolor('white')
                    pc.set_alpha(1)

                    count += 1


                # Draw box plot
                for j in range(len(self.species)):
                    b = axes.boxplot(violinplot_data[j],
                            positions=[j+1], # position of the box
                            widths=0.06*2, 
                            meanline=False,
                            showmeans=False,
                            meanprops={'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                            medianprops = {'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                            showcaps = False,  
                            showfliers = False, 
                            patch_artist=True, 
                            boxprops = {'color':linecolor[j], 'facecolor':'white', 'linewidth':'1.5'},
                            whiskerprops = {'color':linecolor[j], 'linewidth':'1.5'} 
                            )

                    box_lenend_info.append(b['boxes'][0])

                # Draw horizontal dashed line
                dashed_line_y = 0
                for species_index in range(len(self.species)):
                    dashed_line_y = np.log2(true_fc[species_index*len(Compared_groups_index) + i])  # np.log2

                    plt.axhline(y = dashed_line_y, xmin = species_index/len(self.species), xmax = (species_index+1)/len(self.species), color = linecolor[species_index], linestyle = '--')


                plt.ylim(-5, 6.25)
                plt.yticks(np.array([-5, -2.5, 0, 2.5, 5]), ['-5.0', '-2.5', '0.0', '2.5', '5.0'])

                plt.xlim(0.5, 0.5+len(self.species))

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(False)
                axes.spines['left'].set_visible(False)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 

                if i==0:
                    axes.spines['left'].set_visible(True)
                    plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 
                    plt.ylabel('log$_{2}$FC', fontsize=16)  # Log2FC 

                    plt.xticks([])
                    axes.spines['bottom'].set_visible(False)
                    axes.spines['left'].set_bounds((-5, 5))

                    plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True)
                else:
                    plt.xticks([])
                    axes.spines['bottom'].set_visible(False)

                    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


                # Mark the number of proteins/peptides used for the drawing
                count = 0
                for index in position:
                    num = len(violinplot_data[count])
                    if num == 99999:
                        num = 0
                    plt.text(index, -4.5, r'$\it{n}\rm{=' + '{0}'.format(str(num)) + '}$', ha='center', va='bottom', rotation=90, size=12, family="Arial") 
                    count += 1

                # Calculate and mark the difference between the detected fold change and the true fold change
                delta_fc_list = []
                abs_delta_fc_list = []
                for species_index in range(len(self.species)):
                    delta_fc = np.median(violinplot_data[species_index]) - np.log2(true_fc[species_index*len(Compared_groups_index) + i])

                    abs_delta_fc_list.append(abs(delta_fc))
                    delta_fc_list.append(delta_fc)

                min_index = abs_delta_fc_list.index(min(abs_delta_fc_list)) 

                # Annotate the fc difference for each species
                for species_index in range(len(self.species)):
                    if len(violinplot_data[species_index]) != 99999:
                        if species_index == min_index:
                            plt.text(position[species_index], 4.5, '{:+.2f}'.format(delta_fc_list[species_index]), ha='left', va='center', size=11, color = 'black', family="Arial", rotation=90) 
                        else:
                            plt.text(position[species_index], 4.5, '{:+.2f}'.format(delta_fc_list[species_index]), ha='left', va='center', size=11, color = 'black', family="Arial", rotation=90)  # #aeaeae

        
            plt.tight_layout() 
            plt.subplots_adjust(left=0.12, bottom=0.03, right=1, top=1, wspace=0.1, hspace=0.2) 
            if savefig:
                plt.savefig(savefolder + 'FoldChange_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 

            plt.show()
            plt.close()
        

        # Draw the scatter plot of quantitative results
        for software_index in range(len(Compared_softwares)):

            df_data = pd.DataFrame()
            scatter_lenend_info = []

            for i in range(len(Compared_groups_index)):
                fig, ax = plt.subplots(1, 1, figsize=(5,5)) 

                plt.subplot(1, 1, 1)
                axes = plt.gca()

                violinplot_data = [] 
                for species_index in range(len(self.species)):
                    violinplot_data.append(np.log2(data[species_index*len(Compared_groups_index) + i][software_index])) 

                # Draw horizontal dashed line
                dashed_line_y = 0
                for species_index in range(len(self.species)):
                    dashed_line_y = np.log2(true_fc[species_index*len(Compared_groups_index) + i])  # np.log2

                    plt.axhline(y = dashed_line_y, color = linecolor[species_index], linestyle = '--')

                # Box plot
                for j in range(len(self.species)):

                    bplot1 = axes.boxplot(violinplot_data[j],
                            positions=[19 + 2*j], 
                            widths=0.7, 
                            meanline=False,
                            showmeans=False,
                            meanprops={'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'},
                            medianprops = {'color': linecolor[j], 'ls': '-', 'linewidth': '1.5'}, 
                            showcaps = True,  
                            showfliers = False,  
                            patch_artist=True,  
                            capprops = {'color':linecolor[j], 'linewidth':'1.5'}, 
                            boxprops = {'color':linecolor[j], 'facecolor':'white', 'linewidth':'1.5'},  
                            whiskerprops = {'color':linecolor[j], 'linewidth':'1.5'} 
                            )

                    for patch in bplot1['boxes']:
                        patch.set_facecolor(fillin_color[j])

                    log2_data = violinplot_data[j]
                    lower_whisker = [item.get_ydata()[1] for item in bplot1['whiskers']][0]
                    upper_whisker = [item.get_ydata()[1] for item in bplot1['whiskers']][1]
                    good_data = log2_data[(log2_data >= lower_whisker) & (log2_data <= upper_whisker)]

                    log2_group_B_data = np.log2(group_B_data[j*len(Compared_groups_index)+i][software_index])

                    # Scatter plot
                    if len(violinplot_data[j]) != 99999:
                        s = axes.scatter(log2_group_B_data, log2_data, c = fillin_color[j], edgecolors = linecolor[j], marker = '.', s = 4, alpha = 0.7, rasterized=True) 
                        scatter_lenend_info.append(s)

                    # Save Data
                        x_label = 'Log2(%s)'%(Compared_groups_label[i].split('/')[1])
                        df_data = pd.concat( [ df_data  ,  pd.DataFrame(log2_group_B_data, columns=[x_label]), pd.DataFrame(log2_data, columns=[y_label_for_export[j*len(Compared_groups_index)+i]]) ] , axis=1 )
                    else:
                        # Scatter data is empty
                        s = axes.scatter([], [], c = fillin_color[j], edgecolors = linecolor[j], marker = '.', s = 4, alpha = 0.7, rasterized=True) 
                        scatter_lenend_info.append(s)

                        x_label = 'Log2(%s)'%(Compared_groups_label[i].split('/')[1])
                        df_data = pd.concat( [ df_data  ,  pd.DataFrame([], columns=[x_label]), pd.DataFrame([], columns=[y_label_for_export[j*len(Compared_groups_index)+i]]) ] , axis=1 )
                

                plt.xlim(-2, 20+2*(len(self.species)-1)) 
                plt.xticks(np.linspace(-2, 16, 7))  
                axes.set_xticklabels(['-2','1','4','7','10', '13', '16'])
                plt.ylim(-5, 5) 
                plt.yticks(np.linspace(-5, 5, 5))  

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True) 

                plt.xlabel('log$_{2}$' + Compared_groups_label[i].split('/')[1], fontsize = 16)
                plt.ylabel(y_label[i], fontsize = 16)

                plt.subplots_adjust(left=0.16, right=1, bottom=0.12, top=0.97, wspace=0.05, hspace=0.1) 
                if savefig:
                    plt.savefig(savefolder + 'FoldChange_{0}s_{1}_vs_{2}.svg'.format(Protein_or_Peptide, Compared_groups_label[i].split('/')[0], Compared_groups_label[i].split('/')[1]), dpi=600, format="svg", transparent=True)  # , bbox_inches='tight'
                plt.show()
                plt.close()




    # Entrapment Identifications
    def MBR_Error_Rate(self, 
                       dict_species, 
                       species_columns = {'ECOLI':[36,37,38,39,40,41], 
                                          'HUMAN':[33,34,35,42,43,44], 
                                          'YEAST':[30,31,32,45,46,47]},

                       software = 'DIA-NN', Protein_or_Peptide = 'Protein', 
                       Use_blank_list = False,
                       Blank_list_path = None, dict_peptide_to_protein = None,
                       savefig = True, savefolder = './'):

        
        trap_group = [] 
        trap_group_columns = [] 
        trap_name = [] 
        target_name = [] 

        df_blank_list = None
        if Use_blank_list:
            df_blank_list = pd.read_csv(Blank_list_path)

        for group in self.group_name:
            if 0 in self.composition_of_each_group[group]:
                trap_group.append(group)
                trap_group_columns.append(self.sample_index_of_each_group[group])

                trap_list = []
                target_list = []
                count = 0
                for i in self.composition_of_each_group[group]:
                    if i == 0:
                        #trap_list.append(self.species[self.composition_of_each_group[group].index(i)])
                        trap_list.append(self.species[count])
                    else:
                        #target_list.append(self.species[self.composition_of_each_group[group].index(i)])
                        target_list.append(self.species[count])
                    count += 1

                trap_name.append(trap_list)
                target_name.append(target_list)

        # Store the number of target identifications, number of trap identifications, and identification error rates of samples of different species
        dict_traget_num = {}
        dict_trap_num = {}

        dict_target_num_composition = {}  # Composition of target identifications per single-species sample, sorted by species in the samples template file
        dict_trap_num_composition = {}  # Composition of trap identifications per single-species sample, sorted by species in the samples template file

        dict_error_rate = {}

        Run_Name = [] 
        Group = []
        Run = []

        
        for i in range(len(trap_group)):

            # Calculate the expression matrix of target species and trap species in each group
            df_target = []
            df_trap = []

            #target_num_composition_of_this_group = {}
            #trap_num_composition_of_this_group = {}

            for trap in trap_name[i]:
                df_of_this_trap = dict_species[trap].iloc[:,trap_group_columns[i]]

                df_trap.append(df_of_this_trap)

                #trap_num_composition_of_this_group[trap] = df_of_this_trap.count().values

            for target in target_name[i]:
                df_of_this_target = dict_species[target].iloc[:,trap_group_columns[i]]
                df_target.append(df_of_this_target)

                #target_num_composition_of_this_group[target] = df_of_this_target.count().values

            #dict_target_num_composition[trap_group[i]] = target_num_composition_of_this_group
            #dict_trap_num_composition[trap_group[i]] = trap_num_composition_of_this_group


            df_trap = pd.concat(df_trap, axis=0) 
            if (df_target != []):
                df_target = pd.concat(df_target, axis=0) 
            else:
                df_target = pd.DataFrame(columns=df_trap.columns, dtype=float)

            Run_Name.append(df_trap.columns.tolist())
            Group.append([trap_group[i]]*len(trap_group_columns[i]))
            Run.append(list(range(1, len(trap_group_columns[i])+1, 1)))


            # Delete all empty rows
            df_trap = df_trap.dropna(axis=0, how='all')
            df_target = df_target.dropna(axis=0, how='all')
            # Remove duplicate rows
            df_trap = df_trap.drop_duplicates()
            df_target = df_target.drop_duplicates()
            # Remove proteins/peptides common to the target species and other species
            A = df_trap.index.tolist()
            B = df_target.index.tolist()
            delete_index = [x for x in A if x in B]
            df_trap = df_trap.drop(delete_index)

            # identification results
            trap_num_composition_of_this_group = {}
            target_num_composition_of_this_group = {}

            
            target_add_df = []
            trap_drop_name_list = []
            for trap in trap_name[i]:
                common_index = df_trap.index.intersection(dict_species[trap].index)
                #common_data_df_trap = df_trap.loc[common_index]

                target_add_name = []
                if (trap_group[i] != 'Blank') & (Use_blank_list):  # (trap_group[i] != 'Blank') & 
                    new_common_list = []
                    for original_name in common_index:
                        # If the protein/peptide of the trap species is in the blank list, it will be classified as the target protein/peptide
                        if original_name in df_blank_list[Protein_or_Peptide].values.tolist():
                            target_add_name.append(original_name)
                            trap_drop_name_list.append(original_name)

                        else:
                            # If it is a peptide and the peptide is not in the blank list, but the protein corresponding to the peptide is in the blank list, it will be classified as the target peptide
                            if (Protein_or_Peptide == 'Peptide'):
                                protein_of_this_peptide = dict_peptide_to_protein.get(original_name) 
                                if (protein_of_this_peptide in df_blank_list['Protein'].values.tolist()):
                                    target_add_name.append(original_name)
                                    trap_drop_name_list.append(original_name)
                                else:
                                    new_common_list.append(original_name)
                            else:
                                new_common_list.append(original_name)

                    common_index = new_common_list

                    target_add_df.append( df_trap.loc[target_add_name] )
                    


                # Counting proteins from trap species
                common_data_df_trap = df_trap.loc[common_index]
                trap_num_composition_of_this_group[trap] = common_data_df_trap.count().values

            for target in target_name[i]:
                common_index = df_target.index.intersection(dict_species[target].index)
                common_data_df_target = df_target.loc[common_index]

                if (trap_group[i] != 'Blank') & (Use_blank_list) & (target_add_df != []): 
                    # If the newly added protein/peptide is in df_target
                    df_target = pd.concat([df_target] + target_add_df, axis=0) 
                    
                    common_data_df_target = df_target#.loc[common_index]

                # Counting proteins from target species
                target_num_composition_of_this_group[target] = common_data_df_target.count().values

            dict_trap_num_composition[trap_group[i]] = trap_num_composition_of_this_group
            dict_target_num_composition[trap_group[i]] = target_num_composition_of_this_group

            ## df_trap after deletion
            if (trap_group[i] != 'Blank') & (Use_blank_list) & (trap_drop_name_list != []): 

                df_trap = df_trap.drop(index=trap_drop_name_list)  


            # Number of identifications of target species in samples
            traget_num = df_target.count().values 
            # Number of identifications of trap species in samples
            trap_num = df_trap.count().values 
            # Identification error rate of samples
            error_rate = trap_num/(traget_num + trap_num)

            dict_traget_num[trap_group[i]] = traget_num
            dict_trap_num[trap_group[i]] = trap_num
            dict_error_rate[trap_group[i]] = error_rate


        # Combine the number of target species identifications and the number of trap species identifications
        traget_num = []
        trap_num = []


        for trap in trap_group:
            traget_num.append(dict_traget_num[trap].tolist())
            trap_num.append(dict_trap_num[trap].tolist())


        traget_num = np.array(sum(traget_num,[]))
        trap_num = np.array(sum(trap_num,[]))


        y_lim_max = max(traget_num)
        if y_lim_max>0:
            if (Protein_or_Peptide == 'Protein'):
                y_lim_max = math.ceil(y_lim_max/500)*500
            else:
                y_lim_max = math.ceil(y_lim_max/2000)*2000

        y_lim_min = max(trap_num)
        if y_lim_min>0:
            if (Protein_or_Peptide == 'Protein'):
                #y_lim_min = (math.ceil(y_lim_min/500)*500)*1.3
                y_lim_min = (math.ceil(y_lim_min/500)*500)
            else:
                y_lim_min = (math.ceil(y_lim_min/2000)*2000)

        ylim = None
        ylim = [-y_lim_min*1.5, y_lim_max]
        #if y_lim_min/y_lim_max < 0.15:
        #    if (Protein_or_Peptide == 'Protein'):
        #        ylim = [-0.15*y_lim_max, y_lim_max]
        #    else:
        #        ylim = [-0.3*y_lim_max, y_lim_max]
        #elif (y_lim_min/y_lim_max > 0.5) & (max(trap_num)/max(traget_num) < 0.2):
        #    ylim = [-0.3*y_lim_max, y_lim_max]
        #else:
        #    if (Protein_or_Peptide == 'Protein'):
        #        ylim = [-y_lim_min*1.1, y_lim_max]
        #    else:
        #        ylim = [-y_lim_min*1.5, y_lim_max]

        # Save Data
        if savefig:
            df_data = pd.DataFrame()

            Run_Name = sum(Run_Name,[])
            Group = sum(Group,[])
            Run = sum(Run,[])

            df_data['Run Name'] = Run_Name
            df_data['Group'] = Group
            df_data['Run'] = Run
            df_data['Target'] = traget_num
            df_data['Entrapment'] = trap_num

            #if Use_blank_list:
            #    df_data['Target Average By Group'] = traget_average_num_by_group
            #    df_data['Entrapment Average By Group'] = trap_average_num_by_group
            #    df_data['Target Average By Run'] = traget_average_num_by_run
            #    df_data['Entrapment Average By Run'] = trap_average_num_by_run

            Species_Name = self.species
            Species_Composition = [[]]*len(self.species)
            for trap_group_name in trap_group:
                for trap_species in Species_Name:
                    if (trap_species in dict_trap_num_composition[trap_group_name]):
                        data_to_add = dict_trap_num_composition[trap_group_name][trap_species].tolist()
                        Species_Composition[Species_Name.index(trap_species)] = Species_Composition[Species_Name.index(trap_species)] + data_to_add
                    else:
                        data_to_add = dict_target_num_composition[trap_group_name][trap_species].tolist()

                        #data_to_add = [0]*len(trap_group_columns[trap_group.index(trap_group_name)])
                        Species_Composition[Species_Name.index(trap_species)] = Species_Composition[Species_Name.index(trap_species)] + data_to_add
                        
            for species in Species_Name:
                df_data[species] = Species_Composition[Species_Name.index(species)]

            if Use_blank_list:
                for row in range(df_data.shape[0]):
                    if df_data['Group'].values.tolist()[row] == 'Blank':
                       df_data.iat[row, 3] = df_data['Entrapment'].values.tolist()[row]
                       df_data.iat[row, 4] = 0

                # Calculate the Target Average and Entrapment Average of each group
                Group_List = df_data['Group'].unique() 
                Group_Target_Average = []
                Group_Entrapment_Average = []
                df_mean_info = []
                for group in Group_List:
                    # Calculate the mean number of Target and Entrapment for the group
                    df_this_group = df_data[df_data['Group'] == group]
                    mean_value_target = np.mean(df_this_group['Target'].values)
                    mean_value_entrapment = np.mean(df_this_group['Entrapment'].values)
                    Group_Target_Average.append(mean_value_target)
                    Group_Entrapment_Average.append(mean_value_entrapment)
                    # Add data to new df
                    df_to_add = pd.DataFrame({'Run Name': [' '],
                                              'Group': [group + ' Average'],
                                              'Run': [' '],
                                              'Target': [int(mean_value_target)],
                                              'Entrapment': [int(mean_value_entrapment)],
                                              'ECOLI': [' '],
                                              'YEAST': [' '],
                                              'HUMAN': [' ']})
                    df_to_add.columns = df_data.columns.tolist()
                    df_mean_info.append(df_to_add)

                df_data = pd.concat([df_data] + df_mean_info, axis = 0)

            df_data.to_csv(savefolder + 'EntrapmentIdentifications_{0}s.csv'.format(Protein_or_Peptide), index=False)


        groups = len(trap_group)
        xlabel = trap_group
        index_list = []
        labels = []


        for i in range(groups):
            if i==0:
                index_list.append([0, len(trap_group_columns[i])])
                strings = list(map(str, Run[0: len(trap_group_columns[i])]))
                labels.append(strings)

            else:
                begin_index = index_list[i-1][1]
                index_list.append([begin_index, begin_index + len(trap_group_columns[i])])
                #strings = list(map(str, Run[0: len(trap_group_columns[i])]))
                strings = list(map(str, Run[index_list[i][0]: index_list[i][1]]))

                labels.append(strings)


        gap = 0.1

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if int(height) != 0:
                    if (int(height)/ylim[1]) < gap:
                        plt.text(rect.get_x()+rect.get_width()/2. + 0.05, 0.48*ylim[1]*gap, '%s' % int(height), ha='center', va='center', rotation=90, size=12.5, family="Arial") 
                    else:
                        plt.text(rect.get_x()+rect.get_width()/2. + 0.05, 0.48*height, '%s' % int(height), ha='center', va='center', rotation=90, size=12.5, family="Arial") 
                else:
                    plt.text(rect.get_x()+rect.get_width()/2. + 0.05, 0.48*ylim[1]*gap, '%s' % int(height), ha='center', va='center', rotation=90, size=12.5, family="Arial") 

        def autolabel2(rects):
            for rect in rects:
                height = rect.get_height()
                if int(height) != 0:
                    if (int(height)/ylim[0]) < gap:
                        plt.text(rect.get_x()+rect.get_width()/2. + 0.05, ylim[0]*gap, '%s' % int(-1*height), ha='center', va='top', rotation=90, size=12.5, family="Arial")
                    else:
                        plt.text(rect.get_x()+rect.get_width()/2. + 0.05, 1.05*height, '%s' % int(-1*height), ha='center', va='top', rotation=90, size=12.5, family="Arial") 
                else:
                    plt.text(rect.get_x()+rect.get_width()/2. + 0.05, ylim[0]*gap, '%s' % int(-1*height), ha='center', va='top', rotation=90, size=12.5, family="Arial")


        # The identification number bar chart only needs Human + blank
        subplots_num = None

        if ('Yeast' in trap_group) | ('YEAST' in trap_group):
            subplots_num = groups-1
        else:
            subplots_num = groups

        if ('E.coli' in trap_group) | ('E. coli' in trap_group) | ('ECOLI' in trap_group):
            subplots_num = subplots_num-1

        fig, ax = plt.subplots(1, subplots_num, figsize=(3.5,5))

        count = 1
        for index in range(groups):
            if (trap_group[index] == 'Yeast') | (trap_group[index] == 'E. coli'):
                continue
            plt.subplot(1, subplots_num, count)
            count += 1

            bar1 = plt.bar(labels[index], traget_num[index_list[index][0] : index_list[index][1]], width = 0.8, color = '#8bd2cb') 
            autolabel(bar1)
            bar2 = plt.bar(labels[index], -1*trap_num[index_list[index][0] : index_list[index][1]], width = 0.8, color = '#ffffff')   # '#ff6681'
            autolabel2(bar2)

            # Color of the trap species in the Bar Chart
            trap_color_list = {'ECOLI': '#ff6681',
                               'YEAST': '#ff99ab',
                               'HUMAN': '#ffe5ea'}

            current_group_name = trap_group[index]
            current_group_trap_composition = dict_trap_num_composition[current_group_name]
            keys_list = list(current_group_trap_composition.keys())
            x_trap_list = []
            y_trap_list = []
            y_trap_color_list = []
            for keys in keys_list:
                x = list(range(len(current_group_trap_composition[keys])))
                x_trap_list.append(x)

                y = None
                if (y_trap_list == []):
                    y = -1*current_group_trap_composition[keys]
                    y_trap_list.append(y)
                else:
                    y = y_trap_list[-1] - current_group_trap_composition[keys]
                    y_trap_list.append(y)

                y_trap_color_list.append(trap_color_list[keys])


            #trap_num = len(y_trap_list)
            for i in range(len(y_trap_list)):
                #index = -1 - i
                plt.bar(x_trap_list[-1 - i], y_trap_list[-1 - i], width = 0.8, color = y_trap_color_list[-1 - i]) 

            plt.ylim(ylim[0], ylim[1] + (ylim[1]-ylim[0])*0.15) 

            ## Mark the scale value in the negative direction of the Y axis
            #below_zero_tick_num = math.floor(-ylim[0]/(ylim[1]/5))
            #ticks_ = []
            #labels_ = []
            #for j in range(below_zero_tick_num):
            #    ticks_.append(-(below_zero_tick_num-j)*(ylim[1]/5))
            #    labels_.append(str(int((below_zero_tick_num-j)*(ylim[1]/5))))
            #if (below_zero_tick_num == 0):
            #    ticks_.append(-(ylim[1]/5))
            #    labels_.append(str(int(ylim[1]/5)))

            #for j in range(6):
            #    ticks_.append(j*(ylim[1]/5))
            #    labels_.append(str(int(j*(ylim[1]/5))))

            #if index ==0:
            #    plt.yticks(ticks_, labels_) 

            if (Protein_or_Peptide == 'Protein'):
                ticks_ = list(range(-y_lim_min, y_lim_max+1, 500))
                labels_ = [str(abs(x)) for x in ticks_]
                plt.yticks(ticks_, labels_) 
            else:
                ticks_ = list(range(-y_lim_min, y_lim_max+1, 2000))
                labels_ = [str(abs(x)) for x in ticks_]
                plt.yticks(ticks_, labels_) 

            plt.tick_params(labelsize=14) 

            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)


            # Draw a gray square
            rectangle = plt.broken_barh([(-0.5, len(trap_group_columns[index]))], (ylim[1] + (ylim[1]-ylim[0])*0.05, (ylim[1]-ylim[0])*0.10), color = '#dadada')

            rx, ry = -0.5, ylim[1] + (ylim[1]-ylim[0])*0.05
            cx = rx + len(trap_group_columns[index])/2
            cy = ry + (ylim[1]-ylim[0])*0.10/2

            plt.text(cx, cy/1.01, trap_group[index], size=18, ha='center', va='center')  


            axes = plt.gca()
            if index ==0:
                plt.ylabel('# {0}s'.format(Protein_or_Peptide), fontsize=16)
                axes.spines['left'].set_bounds((ylim[0], ylim[1])) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)
            else:
                axes.spines['left'].set_bounds((ylim[0], ylim[1])) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(False)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 
                plt.tick_params(axis='both', which='both', bottom=True, left=False, labelbottom=True, labelleft=False)

        if ylim[1] < 10000:
            plt.suptitle('Run', x=0.62535, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            #plt.subplots_adjust(left=0.135, right=1, bottom=0.12, top=1, wspace=0.1, hspace=0.1)
            plt.subplots_adjust(left=0.135*6.5/3.5, right=1, bottom=0.12, top=1, wspace=0.1, hspace=0.1)

        else:
            plt.suptitle('Run', x=0.6393, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            #plt.subplots_adjust(left=0.15, right=1, bottom=0.12, top=1, wspace=0.1, hspace=0.1)
            plt.subplots_adjust(left=0.15*6.5/3.5, right=1, bottom=0.12, top=1, wspace=0.1, hspace=0.1)

        if savefig & (Use_blank_list == False):
            plt.savefig(savefolder + 'EntrapmentIdentifications_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()



        # Entrapment Data Completeness
        def ReturnDataCompleteness(df_all):
            x_list = []  
            y_list = [] 

            missing_ratio = df_all.isnull().sum(axis=1) / df_all.shape[1]
            missing_ratio = missing_ratio.values

            for i in range(101):
            
                indices = np.where(missing_ratio <= i/100)
            
                if len(x_list) >= 1:
                    if y_list[-1] == len(indices[0]):
                        pass
                    else:
                        x_list.append(i)
                        y_list.append(len(indices[0]))

                if len(x_list) == 0:
                    x_list.append(i)
                    y_list.append(len(indices[0]))

                if len(indices[0]) == df_all.shape[0]:
                    break

            return x_list, y_list

        subplots_num = None
        if ('Blank' in trap_group) | ('blank' in trap_group):
            subplots_num = groups - 1
        else:
            subplots_num = groups

        if ('Yeast' in trap_group) | ('YEAST' in trap_group):
            subplots_num = subplots_num-1

        if ('E.coli' in trap_group) | ('E. coli' in trap_group) | ('ECOLI' in trap_group):
            subplots_num = subplots_num-1


        fig, ax = plt.subplots(1, subplots_num, figsize=(3.5,5))

        df_data_list = []
        y_lim_max = 0
        for i in range(subplots_num):
            plt.subplot(1, subplots_num, i+1)

            df_target = []
            df_trap = []
            for trap in trap_name[i]:
                df_trap.append(dict_species[trap].iloc[:,trap_group_columns[i]])

            for target in target_name[i]:
                df_target.append(dict_species[target].iloc[:,trap_group_columns[i]])

                if (dict_species[target].iloc[:,trap_group_columns[i]].shape[0]) > y_lim_max:
                    y_lim_max = (dict_species[target].iloc[:,trap_group_columns[i]].shape[0])

            df_trap = pd.concat(df_trap, axis=0) 

            if (df_target != []):
                df_target = pd.concat(df_target, axis=0) 
            else:
                df_target = pd.DataFrame(columns=df_trap.columns, dtype=float)

            #df_target = pd.concat(df_target, axis=0) 


            # Delete all empty rows
            df_trap = df_trap.dropna(axis=0, how='all')
            df_target = df_target.dropna(axis=0, how='all')

            # Remove duplicate rows
            df_trap = df_trap.drop_duplicates()
            df_target = df_target.drop_duplicates()


            # If df_trap contains proteins/peptides from blank_list, take them out and put them into df_target
            if Use_blank_list:

                df_trap_index_list = df_trap.index.tolist()
                df_blank_index_list = df_blank_list[Protein_or_Peptide].values.tolist()
                common_index_list = [item for item in df_trap_index_list if item in df_blank_index_list]

                if (common_index_list != []):
                    
                    row_to_add_list = []
                    for trap_drop_name in common_index_list:
                        if trap_drop_name in df_trap.index.tolist():
                            row_index_in_trap = (df_trap.index.tolist()).index(trap_drop_name)
                            # Take out the row and add it to df_target
                            row_to_add = df_trap.iloc[[row_index_in_trap]]
                            row_to_add_list.append(row_to_add)
                            # Delete the row from df_trap
                            df_trap.index.name = Protein_or_Peptide
                            df_trap = df_trap.reset_index(drop=False)
                            df_trap = df_trap.drop(row_index_in_trap)
                            df_trap = df_trap.set_index(Protein_or_Peptide)
                            #df_trap.index.name = Protein_or_Peptide

                    if row_to_add_list != []:
                        df_target = pd.concat([df_target] + row_to_add_list, axis=0)


                    

            
            x_list_target, y_list_target = ReturnDataCompleteness(df_target)
            x_list_trap, y_list_trap = ReturnDataCompleteness(df_trap)

            plt.plot(100-np.array(x_list_target), np.array(y_list_target) , color = '#8bd2cb', linewidth=2)
            plt.plot(100-np.array(x_list_trap), np.array(y_list_trap) , color = '#ff6681', linewidth=2)

            axvline_ymax = (math.ceil(y_lim_max/500)*500*1.02)/(math.ceil(y_lim_max/500)*500 + 0.17*math.ceil(y_lim_max/500)*500)
            plt.axvline( x = 66, ymax = axvline_ymax, linestyle='--', color='gray')
            plt.axvline( x = 75, ymax = axvline_ymax, linestyle='--', color='gray')
            plt.axvline( x = 90, ymax = axvline_ymax, linestyle='--', color='gray')

            plt.tick_params(labelsize=14) 
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            ylabel = 'Proteins'
            if 'Peptide' in Protein_or_Peptide:
                ylabel = 'Peptides'
            
            plt.xlim(-2, 103 ) 
            xticks = [0, 25, 50, 66, 75, 90, 100]
            plt.xticks(xticks, ['0', '25', '50', '', '75', '', '100'])

            plt.ylim(-0.02*math.ceil(y_lim_max/500)*500, math.ceil(y_lim_max/500)*500 + 0.15*math.ceil(y_lim_max/500)*500) 
            plt.yticks(np.linspace(0, math.ceil(y_lim_max/500)*500, 6)) 

            rectangle = plt.broken_barh([(-2, 103)], (math.ceil(y_lim_max/500)*500 + 0.05*math.ceil(y_lim_max/500)*500, 0.10*math.ceil(y_lim_max/500)*500), color = '#dadada')

            rx, ry = -2, math.ceil(y_lim_max/500)*500 + 0.05*math.ceil(y_lim_max/500)*500
            cx = rx + 103/2
            cy = ry + 0.10*math.ceil(y_lim_max/500)*500/2

            plt.text(cx, cy/1.01, trap_group[i], size=18, ha='center', va='center')  


            axes = plt.gca()

            if i == 0:
                plt.ylabel('# ' + ylabel, y=0.5, fontsize=16)
                axes.spines['left'].set_bounds((-0.02*math.ceil(y_lim_max/500)*500, math.ceil(y_lim_max/500)*500)) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
            else:
                axes.spines['left'].set_bounds((0, math.ceil(y_lim_max/500)*500)) 
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(False)
                plt.tick_params(axis='both', which='both', bottom=True, left=False, labelbottom=True, labelleft=False)

            axes.spines['bottom'].set_linewidth(2) 
            axes.spines['left'].set_linewidth(2) 
            axes.invert_xaxis() 

            df_temp = pd.DataFrame(index = 100-np.array(x_list_target))
            df_temp['{0} Target'.format(trap_group[i])] = y_list_target

            df_temp2 = pd.DataFrame(index = 100-np.array(x_list_trap))
            df_temp2['{0} Entrapment'.format(trap_group[i])] = y_list_trap

            df_data_list.append(df_temp)
            df_data_list.append(df_temp2)


        if (math.ceil(y_lim_max/500)*500) < 10000:
            plt.suptitle('Data Completeness (%)', x=0.62535, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            #plt.subplots_adjust(left=0.135, right=0.99, bottom=0.12, top=1, wspace=0.1)
            plt.subplots_adjust(left=0.135*6.5/3.5, right=1 - 6.5*0.01/3.5, bottom=0.12, top=1, wspace=0.1)
        else:
            plt.suptitle('Data Completeness (%)', x=0.6393, y=0.03, horizontalalignment='center',verticalalignment='center', fontproperties='Arial', fontsize=16)
            #plt.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=1, wspace=0.1)
            plt.subplots_adjust(left=0.15*6.5/3.5, right=1 - 6.5*0.01/3.5, bottom=0.12, top=1, wspace=0.1)

        # Save data
        if savefig:
            df_data = pd.concat(df_data_list, axis=1)
            df_data.reset_index(inplace=True)
            df_data.rename(columns={df_data.columns[0]: 'Data Completeness (%)'}, inplace=True)
            df_data.to_csv(savefolder + 'EntrapmentDataCompleteness_{0}s.csv'.format(Protein_or_Peptide), index=False)

        if savefig & (Use_blank_list == False):
            plt.savefig(savefolder + 'EntrapmentDataCompleteness_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()

        if Use_blank_list:
            return 

        # Entrapment Quantity Rank
        
        df_data_list = []
        for i in range(groups):

            if (trap_group[i] == 'Blank') | (trap_group[i] == 'blank'):
                continue
            if (trap_group[i] == 'YEAST') | (trap_group[i] == 'Yeast'):
                continue
            if (trap_group[i] == 'ECOLI') | (trap_group[i] == 'E. coli') | (trap_group[i] == 'E.coli'):
                continue
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            plt.subplot(1, 1, 1)


            df_target = []
            df_trap = []
            for trap in trap_name[i]:
                df_trap.append(dict_species[trap].iloc[:,trap_group_columns[i]])

            for target in target_name[i]:
                df_target.append(dict_species[target].iloc[:,trap_group_columns[i]])

            df_trap = pd.concat(df_trap, axis=0) 

            if (df_target != []):
                df_target = pd.concat(df_target, axis=0) 
            else:
                df_target = pd.DataFrame(columns=df_trap.columns, dtype=float)
            #df_target = pd.concat(df_target, axis=0) 


            # Delete all empty rows
            df_trap = df_trap.dropna(axis=0, how='all')
            df_target = df_target.dropna(axis=0, how='all')
            # Remove duplicate rows
            df_trap = df_trap.drop_duplicates()
            df_target = df_target.drop_duplicates()

            # Merge df of target and trap species
            df_target_with_marker = pd.concat([df_target, pd.DataFrame({'Entrapment': ['False']*df_target.shape[0]}, index = df_target.index)], axis=1) 
            df_trap_with_marker = pd.concat([df_trap, pd.DataFrame({'Entrapment': ['True']*df_trap.shape[0]}, index = df_trap.index)], axis=1) 
            df_combined = pd.concat([df_target_with_marker, df_trap_with_marker], axis=0) 

            # The average value of all rows
            df_trap_mean = pd.DataFrame(df_trap.mean(axis=1), columns = ['average'])
            df_target_mean = pd.DataFrame(df_target.mean(axis=1), columns = ['average'])

            columns_to_select = df_combined.columns.tolist()[:-1]
            df_combined_mean = pd.DataFrame(df_combined[columns_to_select].mean(axis=1), columns = ['average'])
            df_combined_mean['Entrapment'] = df_combined['Entrapment'].values.tolist()
            # Sort from high to low
            df_trap_sorted = df_trap_mean.sort_values(by='average', ascending=False)
            df_target_sorted = df_target_mean.sort_values(by='average', ascending=False)

            df_combined_sorted = df_combined_mean.sort_values(by='average', ascending=False)
            df_combined_sorted['Rank'] = list(range(1, 1 + df_combined_sorted.shape[0]))
            # Do log10 processing
            df_trap_sorted['log'] = np.log10(df_trap_sorted['average'])
            df_target_sorted['log'] = np.log10(df_target_sorted['average'])

            df_combined_sorted['log'] = np.log10(df_combined_sorted['average'])


            # Draw target species scatter plots
            axes = plt.gca()
            x_data = df_combined_sorted[df_combined_sorted['Entrapment'] == 'False']['Rank'].values.tolist()
            y_data = df_combined_sorted[df_combined_sorted['Entrapment'] == 'False']['log'].values.tolist()
            axes.scatter(x_data, y_data, color = '#8bd2cb', edgecolor='#689893', marker = 'o', s = 11, rasterized=True)

            # Draw trap species scatter plots
            x_data2 = df_combined_sorted[df_combined_sorted['Entrapment'] == 'True']['Rank'].values.tolist()
            y_data2 = df_combined_sorted[df_combined_sorted['Entrapment'] == 'True']['log'].values.tolist()
            axes.scatter(x_data2, y_data2, color = '#ff6681', edgecolor='#8B2C3C', marker = 'D', s = 15, rasterized=True)

            y_lim_middle = int((max(y_data + y_data2) + min(y_data + y_data2))/2)
            y_lim_min = y_lim_middle - 4
            y_lim_max = y_lim_middle + 4
            
            plt.tick_params(labelsize=14)
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            # X-axis range and scale
            x_max = len(df_combined_sorted['log'].values.tolist())
            x_lim = math.ceil(x_max/500)*500
            plt.xlim(-0.05*x_lim, x_lim*1.15)
            plt.xticks(np.linspace(0, x_lim, 5)) 
            plt.xlabel('{0} Rank'.format(Protein_or_Peptide), fontsize=16)

            # Y-axis range and scale
            y_max = df_combined_sorted['log'].values.tolist()[0]
            #y_lim = math.ceil(y_max/2)*2
            y_lim = 8
            plt.ylim(-0.05*y_lim, y_lim*1.1)
            plt.yticks(np.linspace(0, y_lim, 5)) 
            plt.ylabel('log$_{10}$Quantity', fontsize=16)

            
            axes = plt.gca()
            axes.spines['bottom'].set_bounds((-0.05*x_lim, x_lim*1.15)) 
            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(True)
            axes.spines['left'].set_visible(True)
            axes.spines['bottom'].set_linewidth(2) 
            axes.spines['left'].set_linewidth(2) 
            plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)


            # Box plot
            linecolor = ['#689893', '#8B2C3C']
            fillin_color = ['#8bd2cb', '#ff6681']
            bplot1 = axes.boxplot(df_target_sorted['log'].values.tolist(),
                        positions=[x_lim*1.05],
                        widths=int(x_lim*0.02), 
                        meanline=False,
                        showmeans=False,
                        meanprops={'color': linecolor[0], 'ls': '-', 'linewidth': '1.5'},
                        medianprops = {'color': linecolor[0], 'ls': '-', 'linewidth': '1.5'},
                        showcaps = True,  
                        showfliers = False, 
                        patch_artist=True, 
                        boxprops = {'color':linecolor[0], 'facecolor':'white', 'linewidth':'1.5'},
                        whiskerprops = {'color':linecolor[0], 'linewidth':'1.5'}
                        )

            for patch in bplot1['boxes']:
                patch.set_facecolor(fillin_color[0])

            bplot2 = axes.boxplot(df_trap_sorted['log'].values.tolist(),
                        positions=[x_lim*1.10],
                        widths=int(x_lim*0.02), 
                        meanline=False,
                        showmeans=False,
                        meanprops={'color': linecolor[1], 'ls': '-', 'linewidth': '1.5'},
                        medianprops = {'color': linecolor[1], 'ls': '-', 'linewidth': '1.5'},
                        showcaps = True,  
                        showfliers = False, 
                        patch_artist=True, 
                        boxprops = {'color':linecolor[1], 'facecolor':'white', 'linewidth':'1.5'},
                        whiskerprops = {'color':linecolor[1], 'linewidth':'1.5'}
                        )

            for patch in bplot2['boxes']:
                patch.set_facecolor(fillin_color[1])

            plt.xticks(np.linspace(0, x_lim, 5))

            xticklabels = []
            temp = np.linspace(0, x_lim, 5).astype(int).tolist()
            for k in temp:
                xticklabels.append(str(k))
            axes.set_xticklabels(xticklabels)


            plt.subplots_adjust(left=0.115, right=0.99, bottom=0.12, top=1.0, wspace=0.1)

            
            df_data_temp = []
            df_data_temp.append(pd.DataFrame({'Entrapment': df_combined_sorted['Entrapment'].values.tolist()}, index = df_combined_sorted.index.tolist()))
            df_data_temp.append(pd.DataFrame({'log10Quantity': df_combined_sorted['log'].values.tolist()}, index = df_combined_sorted.index.tolist()))
            df_data_temp = pd.concat(df_data_temp, axis=1)
            df_data_temp.reset_index(inplace=True)
            df_data_temp.rename(columns={df_data_temp.columns[0]: Protein_or_Peptide}, inplace=True)

            df_data_temp.insert(0, 'Rank', df_combined_sorted['Rank'].values.tolist())
            df_data_temp.insert(0, 'Group', [trap_group[i]]*df_data_temp.shape[0])
            df_data_list.append(df_data_temp)

            if savefig:
                plt.savefig(savefolder + 'EntrapmentQuantityRank_{0}s_{1}.svg'.format(Protein_or_Peptide, trap_group[i]), dpi=600, format="svg", transparent=True) 

            plt.show()
            plt.close()

        # Save data
        if savefig:
            df_data = pd.concat(df_data_list, axis=0) 
            df_data.to_csv(savefolder + 'EntrapmentQuantityRank_{0}s.csv'.format(Protein_or_Peptide), index=False)



    # Generate the comparision results of different softwares/library construction methods
    def ResultComparison(self, 
                         dataset_path = ['DIA-NN.csv', 'SP.csv', 'Peaks.csv', 'MaxQuant.csv'], 
                         fc_path = ['DIA-NN_FC.csv', 'SP_FC.csv', 'Peaks_FC.csv', 'MaxQuant_FC.csv'],
                         dataset_name = ['DIA-NN', 'Spectronaut', 'Peaks', 'MaxQuant'],
                         Protein_or_Peptide = 'Protein',
                         Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                         savefig = True,
                         savefolder = './'):

        linecolor = ['#E95B1B', '#B0203F', '#A14EE0', '#1883B8', '#39A139']
        fillin_color = ['#F4AC8C', '#ED9DAE', '#D39FEB', '#67BFEB', '#90D890']

        if dataset_name == ['DIA-NN', 'PEAKS']:
            linecolor = ['#E95B1B', '#A14EE0']
            fillin_color = ['#F4AC8C', '#D39FEB']

        df_list = []
        for csv_path in dataset_path:
            df = pd.read_csv(csv_path, index_col=0) 
            if (Protein_or_Peptide == 'Protein'):
                df.drop(['Organism'], axis=1, inplace=True)
            if (Protein_or_Peptide == 'Peptide'):
                df.drop(['Organism', 'Protein'], axis=1, inplace=True)

            df_list.append(df)


        # >>>>>Comparison of sample identification numbers (mean standard deviation)<<<<<<
        identification_data = []
        average_identification_num = []  # The average number of identifications per dataset
        sd = []  # The standard deviation of each data set
        for df in df_list:
            valid_count_per_column = df.count().values.tolist()

            identification_data.append(valid_count_per_column)
            average_identification_num.append(statistics.mean(valid_count_per_column))
            sd.append(statistics.stdev(valid_count_per_column))




        # Bar Chart
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 4.2))
        #fig, ax = plt.subplots(1, 1, figsize=(3.5, 4.5))
        plt.subplot(1, 1, 1)

        index = np.arange(len(average_identification_num))
        width = 0.7 
        bars_ = plt.bar(index, average_identification_num, width, color=fillin_color[:len(dataset_path)])




        # Add error bars
        def add_errorbars(bars, errors):
            count = 0
            for bar, err_high in zip(bars, errors):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()

                if (len(dataset_path) == 4) & (math.ceil(max(average_identification_num)/500)*500 < 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=15.5, capthick = 2, elinewidth=2) 

                elif (len(dataset_path) == 4) & (math.ceil(max(average_identification_num)/500)*500 >= 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=14.6, capthick = 2, elinewidth=2) 

                elif (len(dataset_path) == 3) & (math.ceil(max(average_identification_num)/500)*500 < 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=20, capthick = 2, elinewidth=2) 

                elif (len(dataset_path) == 3) & (math.ceil(max(average_identification_num)/500)*500 >= 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=19.1, capthick = 2, elinewidth=2) 

                elif (len(dataset_path) == 2) & (math.ceil(max(average_identification_num)/500)*500 < 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=28.5, capthick = 2, elinewidth=2)

                elif (len(dataset_path) == 2) & (math.ceil(max(average_identification_num)/500)*500 >= 10000):
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=27.6, capthick = 2, elinewidth=2)

                else:
                    plt.errorbar(x, y, yerr=np.array([[0], [err_high]]), fmt='none', ecolor=fillin_color[count], capsize=15.5, capthick = 2, elinewidth=2)

                count += 1
 
        add_errorbars(bars_, sd)


        
        # Scatter plot
        for j in index:
            for i, d in enumerate([identification_data[j]]):
                np.random.seed(42) 
                scatter_width_range = 0.15

                x = np.random.normal(j, scatter_width_range, size=len(d))
                plt.scatter(x, d, alpha=0.7, color='#4e4e4e', linewidths=0, s = 10, zorder=3) 


        plt.tick_params(labelsize=14) 
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        plt.xlim(-0.65, len(average_identification_num)-0.35)

        y_lim = math.ceil(max(average_identification_num)/500)*500

        # Upper limit of the y-axis scale value of the bar chart
        y_lim = 4000
        if (Protein_or_Peptide == 'Protein'):
            y_lim = math.ceil(max(average_identification_num)/1000)*1000
            plt.ylim(0, y_lim*1.4)
            plt.yticks(np.linspace(0, y_lim, int(y_lim/1000 + 1)))
        else:
            y_lim = math.ceil(max(average_identification_num)/3000)*3000
            plt.ylim(0, y_lim*1.4)
            plt.yticks(np.linspace(0, y_lim, int(y_lim/3000 + 1)))

        #if len(df_list) == 3:
        #    plt.ylim(0, y_lim*1.2)
        #elif len(df_list) == 4:
        #    plt.ylim(0, y_lim*1.4)
        #else:
        #    plt.ylim(0, y_lim*1.4)

        #if (Protein_or_Peptide == 'Protein'):
        #    plt.yticks(np.linspace(0, y_lim, int(y_lim/1000 + 1)))
        #else:
        #    plt.yticks(np.linspace(0, y_lim, int(y_lim/2000 + 1)))

        plt.ylabel('# {0}s'.format(Protein_or_Peptide), fontsize=16)


        # p-value
        p_value_and_position = []  # p value, starting position of the line segment, ending position of the line segment, height of the line segment
        def T_Test(df1, df2):
            data1 = df1.count().values.tolist()
            data2 = df2.count().values.tolist()
            t_statistic, p_value = ttest_ind(data1, data2, equal_var=False)
            return p_value

        if len(df_list) == 2:
            
            p_value_and_position.append([T_Test(df_list[0], df_list[1]), 0, 1, y_lim*1.35])


        if len(df_list) == 3:
            
            p_value_and_position.append([T_Test(df_list[0], df_list[2]), 0, 2, y_lim*1.35])
            p_value_and_position.append([T_Test(df_list[0], df_list[1]), 0, 0.95, y_lim*1.25])
            p_value_and_position.append([T_Test(df_list[1], df_list[2]), 1.05, 2, y_lim*1.25])

            for data in p_value_and_position:
                p_value = data[0]
                if (p_value >= 0.05) & (p_value_and_position.index(data) == 0):
                    p_value_and_position[1][3] += y_lim*0.1
                    p_value_and_position[2][3] += y_lim*0.1



        if len(df_list) == 4:

            p_value_and_position.append([T_Test(df_list[0], df_list[3]), 0, 3, y_lim*1.35])
            p_value_and_position.append([T_Test(df_list[1], df_list[3]), 1, 3, y_lim*1.25])
            p_value_and_position.append([T_Test(df_list[0], df_list[2]), 0, 2, y_lim*1.15])
            p_value_and_position.append([T_Test(df_list[0], df_list[1]), 0, 0.95, y_lim*1.05])
            p_value_and_position.append([T_Test(df_list[1], df_list[2]), 1.05, 1.95, y_lim*1.05])
            p_value_and_position.append([T_Test(df_list[2], df_list[3]), 2.05, 3, y_lim*1.05])

            for data in p_value_and_position:
                p_value = data[0]
                if (p_value >= 0.05) & (p_value_and_position.index(data) == 0):
                    p_value_and_position[1][3] += y_lim*0.1
                    p_value_and_position[2][3] += y_lim*0.1
                    p_value_and_position[3][3] += y_lim*0.1
                    p_value_and_position[4][3] += y_lim*0.1
                    p_value_and_position[5][3] += y_lim*0.1
                if (p_value >= 0.05) & (p_value_and_position.index(data) == 1):
                    p_value_and_position[2][3] += y_lim*0.1
                    p_value_and_position[3][3] += y_lim*0.1
                    p_value_and_position[4][3] += y_lim*0.1
                    p_value_and_position[5][3] += y_lim*0.1
                if (p_value >= 0.05) & (p_value_and_position.index(data) == 2):
                    p_value_and_position[3][3] += y_lim*0.1
                    p_value_and_position[4][3] += y_lim*0.1
                    p_value_and_position[5][3] += y_lim*0.1



        for data in p_value_and_position:
            p_value = data[0]
            # Draw black line
            if p_value >= 0.05:
                pass
            else:
                plt.plot([data[1], data[2]], [data[3], data[3]], color='black')
                plt.plot([data[1], data[1]], [data[3]*0.98, data[3]], color='black')
                plt.plot([data[2], data[2]], [data[3]*0.98, data[3]], color='black')
            # Annotating p-value
            if p_value >= 0.05:
                pass
            elif p_value <0.001:
                #plt.text((data[1] + data[2])/2, data[3]*1.015, "${p}$" + '=' + format(data[0], '.1E'), horizontalalignment='center', fontsize=8, color='black')
                plt.text((data[1] + data[2])/2, data[3]*1.015, format(data[0], '.0E'), horizontalalignment='center', fontsize=13, color='black')
            else:
                #plt.text((data[1] + data[2])/2, data[3]*1.015, "${p}$" + '={0}'.format(str(round(data[0], 3))), horizontalalignment='center', fontsize=8, color='black')
                plt.text((data[1] + data[2])/2, data[3]*1.015, '{0}'.format(str(round(data[0], 3))), horizontalalignment='center', fontsize=13, color='black')





            
        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_linewidth(2) 
        axes.spines['left'].set_linewidth(2) 

        plt.xticks([])

        if y_lim < 10000:
            plt.subplots_adjust(left=0.24, right=0.99, bottom=0.02, top=0.97, wspace=0.1)
        else:
            plt.subplots_adjust(left=0.27, right=0.99, bottom=0.02, top=0.97, wspace=0.1)

        
        if savefig:
            plt.savefig(savefolder + 'Comparison_Identifications_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 

        plt.show()
        plt.close()


        # >>>>> Generate Legend <<<<<
        if savefig:

            fig_legend = plt.figure(figsize=(2.5,2.5))
 
            axes = plt.gca()
            parts = axes.violinplot([[10000]]*len(dataset_name), list(range(1, len(dataset_name)+1)), widths=0.8,
                                    showmeans=False, showmedians=False, showextrema=False)
            count = 0
            for pc in parts['bodies']:
                pc.set_facecolor(fillin_color[count])
                pc.set_edgecolor('white')
                pc.set_alpha(1)

                count += 1

            axes.legend(labels = dataset_name, title='Dataset', title_fontsize=18, fontsize=16, 
                        loc = 'center',
                        markerfirst=True, markerscale=2.0) 

            plt.ylim(-5, 5)
            plt.xlim(-5, 5)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

            plt.xticks([])
            plt.yticks([])

            plt.savefig(savefolder + 'Legend_Datasets.svg', dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()



        # >>>>>> Venn Diagram <<<<<<
        subsets = self.prepare_subsets(df_list)
        self.Plot_Veen_Diagram(venn_num = len(df_list), subsets = subsets, 
                               names = dataset_name, 
                               Protein_or_Peptide = Protein_or_Peptide,
                               savefig = True, 
                               savefolder = savefolder)



        # >>>>>> Comparison of Data Completeness <<<<<<

        def DataCompletenessResult(df_all):
            x_list = [] 
            y_list = [] 

            missing_ratio = df_all.isnull().sum(axis=1) / df_all.shape[1]
            missing_ratio = missing_ratio.values

            for i in range(101):
            
                indices = np.where(missing_ratio <= i/100)
            
                if len(x_list) >= 1:
                    if y_list[-1] == len(indices[0]):
                        pass
                    else:
                        x_list.append(i)
                        y_list.append(len(indices[0]))

                if len(x_list) == 0:
                    x_list.append(i)
                    y_list.append(len(indices[0]))

                if len(indices[0]) == df_all.shape[0]:
                    break

            return x_list, y_list


        fig, ax = plt.subplots(1, 1, figsize=(4.5,4.5))

        plt.subplot(1,1,1)

        y_max = 0
        count = 0
        for df in df_list:

            x_list, y_list = DataCompletenessResult(df)

            x = np.array(x_list)
            x = 100-x
            y = np.array(y_list)

            plt.plot(x, y , linewidth=2, color = fillin_color[:len(dataset_path)][count])

            count += 1

            if max(y_list) > y_max:
                y_max = max(y_list)

        plt.axvline( x = 66, linestyle='--', color='gray')
        plt.axvline( x = 75, linestyle='--', color='gray')
        plt.axvline( x = 90, linestyle='--', color='gray')

        
        plt.tick_params(labelsize=14) 
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        plt.xlabel('Data Completeness (%)', y=0.5, fontsize=16)
        ylabel = 'Proteins'
        if 'Peptide' in Protein_or_Peptide:
            ylabel = 'Peptides'
        plt.ylabel('# ' + ylabel, y=0.5, fontsize=16)


        plt.xlim(-2, 103 ) 
        xticks = [0, 25, 50, 66, 75, 90, 100]
        plt.xticks(xticks, ['0', '25', '50', '66', '75', '90', '100'])


        plt.ylim(0, math.ceil(y_max/500)*500) 
        plt.yticks(np.linspace(0, math.ceil(y_max/500)*500, 6)) 
        axes = plt.gca()
        axes.spines['top'].set_visible(False) 
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(True)
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2) 
        axes.invert_xaxis() 

        if math.ceil(y_max/500)*500 < 10000:
            plt.subplots_adjust(left=0.19, right=1, bottom=0.13, top=0.97, wspace=0.05)
        else:
            plt.subplots_adjust(left=0.215, right=1, bottom=0.13, top=0.97, wspace=0.05)

        
        if savefig:
            plt.savefig(savefolder + 'Comparison_DataCompleteness_{0}s.svg'.format(Protein_or_Peptide), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()



        # >>>>> Comparison_of_Quantitative_Accuracy <<<
        self.Comparison_of_Quantitative_Accuracy_From_FoldChange_CSV(
                                                        fc_path = fc_path,
                                                        Compared_groups_label = Compared_groups_label,
                                                        Compared_softwares = dataset_name,  # ['Spectronaut', 'DIA-NN', 'Peaks', 'MaxQuant'],
                                                        linecolor = linecolor,
                                                        fillin_color = fillin_color,
                                                        Protein_or_Peptide = Protein_or_Peptide,
                                                        savefolder = savefolder)


        # >>>>> Comparison of CV distributions of different software/library construction methods  <<<<<
        def CV_list_from_dataset(dataset):
            df_all = dataset
            groups = self.groups
            total_samples = self.total_samples
            sample_index_of_each_group = list(self.sample_index_of_each_group.values())
            group_name = list(self.sample_index_of_each_group.keys())
    
            PG_list = []
            cv_list = []
            for j in range(groups):
                temp_PG_list = []
                temp_list = []
                for i in range(df_all.shape[0]):

                    sample_index_begin = len(sum(sample_index_of_each_group[:j], []))
                    sample_index_end = len(sum(sample_index_of_each_group[:(j+1)], []))

                    # Process the i-th protein/peptide of group j
                    df = df_all.iloc[i:i+1, np.r_[sample_index_of_each_group[j]]]
                    # Calculating CV value requires at least 3 data points, ignoring missing values and 0
                    # Only count protein/peptide with a proportion of missing values <=50% in each group
                    if df.isnull().sum().sum() <= len(sample_index_of_each_group[j])/2:
                        df = df.dropna(axis=1)  # Delete columns containing missing values
                        df_list = df.values[0].tolist() 
                        # Delete 0 value
                        if 0 in df_list:
                            df_list.remove(0)
                        if len(df_list) >= 3:
                            std = np.std(df_list, ddof=1)  # Calculate standard deviation
                            mean = np.mean(df_list)  # Calculate the average value
                            cv = std / mean  # Calculate CV
                            if cv != np.nan:
                                temp_list.append(cv)
                                temp_PG_list.append(df_all.index.values[i])

                cv_list.append(temp_list)
                PG_list.append(temp_PG_list)

            return cv_list

        # Calculate the CV value list for each data set 
        dataset_cv_list = {}
        for name in dataset_name:
            dataset = df_list[dataset_name.index(name)]
            cv_list = CV_list_from_dataset(dataset)
            dataset_cv_list.update({name: cv_list})

        # CV distribution diagram
        for group in self.group_name:
        
            cv_list = []
            for name in dataset_name:
                cv_list.append(dataset_cv_list[name][self.group_name.index(group)])

            
            fig, ax = plt.subplots(len(dataset_name), 1, figsize=(4.5,4.5))
            count = 0
            for data in cv_list:
                count += 1
                # Build a "density" function based on the dataset
                # When you give a value from the X axis to this function, it returns the according value on the Y axis
                density = gaussian_kde(data)
                density.covariance_factor = lambda : .25
                density._compute_covariance()

                # Create a vector of 200 values going from min to max:
                xs = np.linspace(min(data), max(data), 200)

                plt.subplot(len(dataset_name), 1, count)

                # Make the chart
                # We're actually building a line chart where x values are set all along the axis and y value are
                # the corresponding values from the density function
                plt.plot(xs, density(xs), color = linecolor[count-1], lw = 1)
                # Draw a vertical dashed line representing the median
                median_ = np.median(data)
                y_max = max(density(xs))
                plt.axvline(x=median_, ymin=0, ymax=y_max, c=linecolor[count-1], ls="--", lw=1.5, label=None)
                plt.text(median_ + 0.01, y_max*0.3, '{:.1%}'.format(median_), ha='left', va='center', color = linecolor[count-1], size=14, family="Arial")
                plt.fill_between(xs, 0, density(xs), facecolor = fillin_color[count-1]) 

                x_lim = 1.1
                plt.xlim(-0.05, x_lim)
                plt.xticks(np.linspace(0, 1, 6))  
                plt.ylim(-0.1, 0.5+y_max)  

                plt.tick_params(axis='x', labelsize=14) 
                plt.tick_params(axis='y', labelsize=14)
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                axes = plt.gca()
        
                axes.yaxis.set_label_position('right')
                axes.set_ylabel(dataset_name[count-1], fontsize=18, rotation=0, horizontalalignment = 'right') 

                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                if count == len(dataset_name):
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(True)
                    axes.spines['left'].set_visible(False)
                    axes.spines['bottom'].set_position(('data', 0)) 
                    axes.spines['left'].set_position(('data', 0))
                    plt.tick_params(axis='both', which='both', bottom=True, left=False, labelbottom=True, labelleft=False) 
                    plt.xlabel(Protein_or_Peptide + ' CV', fontsize=16)
            
                else:
                    axes.spines['top'].set_visible(False) 
                    axes.spines['right'].set_visible(False)
                    axes.spines['bottom'].set_visible(False)
                    axes.spines['left'].set_visible(False)
                    axes.spines['bottom'].set_position(('data', 0)) 
                    axes.spines['left'].set_position(('data', 0))
                    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 
        
            plt.subplots_adjust(left=0, right=0.97, bottom=0.13, top=0.98, wspace=None, hspace=0.2) 

            if savefig:
                plt.savefig(savefolder + 'Comparison_CVDistribution_{0}s_{1}.svg'.format(Protein_or_Peptide, group), dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()




    # Protein/peptide expression matrix - sparsity reduction method (remove missing values)
    def Sparsity_Reduction(self, df, method = 'No SR'):

        
        df2 = df.copy(deep=True)
    
        if (method == 'No SR') | (method == 'NoSR'):
            # Do not remove missing values
            print('Sparsity reduction method: ' + method)

            # Number of protein groups/peptides after reducing sparsity
            Rows_after_SR = df2.shape[0]
            print('Number of protein groups/peptides after reducing sparsity: ' + str(Rows_after_SR))

            self.PG_num_SR = Rows_after_SR

            return df2
            
        elif (method == 'SR 66%') | (method == 'SR66'):
            # Remove rows with more than 34% missing values
            print('Sparsity reduction method: ' + method)
            na_ratio = np.sum(df2.isna(), axis=1) / df2.shape[1] * 100
            index_list = df2.index.tolist()
            count = 0
            delete_index = []
            for i in na_ratio:
                if i >= 34:
                    delete_index.append(index_list[count])
                    count+=1
                else:
                    count+=1
            df2 = df2.drop(index = delete_index)

            # Number of protein groups/peptides after reducing sparsity
            Rows_after_SR = df2.shape[0]
            print('Number of protein groups/peptides after reducing sparsity: ' + str(Rows_after_SR))

            self.PG_num_SR = Rows_after_SR

            return df2

        elif (method == 'SR 75%') | (method == 'SR75'):
            # Remove rows with more than 25% missing values
            print('Sparsity reduction method: ' + method)
            na_ratio = np.sum(df2.isna(), axis=1) / df2.shape[1] * 100
            index_list = df2.index.tolist()
            count = 0
            delete_index = []
            for i in na_ratio:
                if i >= 25:
                    delete_index.append(index_list[count])
                    count+=1
                else:
                    count+=1
            df2 = df2.drop(index = delete_index)

            # Number of protein groups/peptides after reducing sparsity
            Rows_after_SR = df2.shape[0]
            print('Number of protein groups/peptides after reducing sparsity: ' + str(Rows_after_SR))

            self.PG_num_SR = Rows_after_SR

            return df2

        elif (method == 'SR 90%') | (method == 'SR90'):
            # Remove rows with more than 10% missing values
            print('Sparsity reduction method: ' + method)
            na_ratio = np.sum(df2.isna(), axis=1) / df2.shape[1] * 100
            index_list = df2.index.tolist()
            count = 0
            delete_index = []
            for i in na_ratio.values:
                if i >= 10:
                    delete_index.append(index_list[count])
                    count+=1
                else:
                    count+=1
            df2 = df2.drop(index = delete_index)

            # Number of protein groups/peptides after reducing sparsity
            Rows_after_SR = df2.shape[0]
            print('Number of protein groups/peptides after reducing sparsity: ' + str(Rows_after_SR))

            self.PG_num_SR = Rows_after_SR

            return df2

            



    # Missing value imputation methods
    def Missing_Data_Imputation(self, df, method = 'HalfRowMinimum'):

        print('Missing value filling method: ' + method)
        data = df.values
        if (method == 'HalfRowMinimum') | (method == 'HRMin') | (method == 'HalfRowMin'):
            # Filling method 1, missing values ​​are filled with 1/2 of the minimum value in each row
            for i in range(len(data[:,0])):
                row_data = data[i,:]
                min_value = np.nanmin(row_data)
                data[i,:][np.isnan(data[i,:])] = min_value/2
        elif (method == 'RowMedian')  | (method == 'RMedian'):
            # Filling method 2, missing values ​​are filled with the median of each row
            for i in range(len(data[:,0])):
                row_data = data[i,:]
                median_value = np.nanmedian(row_data)
                data[i,:][np.isnan(data[i,:])] = median_value
        elif method == 'Zero':
            # Filling method 3, missing values ​​are filled with 0
            data = SimpleFill(fill_method='zero').fit_transform(data)
            #df2 = df.copy()
            #df2.fillna(0, inplace=True)
            #data = df2.values

        elif (method == 'RowMean') | (method == 'RMean'):
            # Filling method 4, missing values ​​are filled with the average value of each row
            data = SimpleFill(fill_method='mean').fit_transform(data.T)
            data = data.T
        elif (method == 'IterativeSVD') | (method == 'ISVD'):
            # Filling method 5, IterativeSVD
            data = IterativeSVD().fit_transform(data)
        elif (method == 'SoftImpute') | (method == 'Soft'):
            # Filling method 6, SoftImpute
            data = SoftImpute().fit_transform(data)
        elif (method == 'KNN') | (method == 'KNN'):
            # Filling method 7，KNN
            data = KNN(k=6, print_interval=3).fit_transform(data)
        elif (method == 'KeepNA') | (method == 'keepNA'):
            pass


        # Reconvert data to dataframe
        df_imputated = pd.DataFrame(data, index = df.index, columns = df.columns)
        # If there is a negative value after filling, convert it to 0
        df_imputated = df_imputated.clip(lower=0)
        # Percentage of missing values ​​in the data
        print('After missing values ​​are filled, the percentage of missing values ​​in the data is: {0:.2%}'.format(df_imputated.isnull().sum().sum()/df_imputated.size))

        return df_imputated



    # Data normalization methods
    def Data_Normalization(self, df, method = 'unnormalized'):

        print('Data normalization method: ' + method)

        

        if (method == 'sum') | (method == 'Sum'):
            
            factor = df.sum(axis=0, skipna=True)
            factor = factor.mean() / factor
            df = df.multiply(factor, axis=1)

            return df


        # R script for data normalization method
        Normalization_R = '''
            library(MBQN)
            library(limma)


            # NORMALIZATION
            # "unnormalized", "TRQN", "QN", "median"
            getNormalizedDf <- function(modus, df) {
              if ((modus == "unnormalized") | (modus == "Unnormalized")){
                df.model <- df
              } else if (modus == "TRQN") {
                mtx <- as.matrix(df)
                df.trqn <- mbqn(mtx, FUN = mean)
                row.names(df.trqn) <- row.names(df)
                df.model <- as.data.frame(df.trqn)
              } else if (modus == "QN"){
                mtx <- as.matrix(df)
                df.qn <- mbqn(mtx, FUN = NULL)
                row.names(df.qn) <- row.names(df)
                df.model <- as.data.frame(df.qn)
              }else if  ((modus == "median") | (modus == "Median")){
                mtx <- as.matrix(df)
                df.median <- limma::normalizeMedianValues(mtx)
                df.model <- as.data.frame(df.median)
              } else {
                print("Undefined modus")
              }
              return(df.model)
            }

            NormalizedDf <- getNormalizedDf(method, data)

            NormalizedDf

        '''


        # Execute R script
        with lc(robjects.default_converter + pandas2ri.converter):
            r_dataframe = robjects.conversion.py2rpy(df)
        globalenv['data'] = r_dataframe 
        globalenv['method'] = method

        Normalized_df = robjects.r(Normalization_R)
        pandas2ri.activate()
        df_normalized = pandas2ri.rpy2py(Normalized_df) 
        pandas2ri.deactivate()


        # Median normalization may produce missing values
        if (method == 'median') | (method == 'Median'):
            if (df.isnull().values.any() == True):
                # KeepNA
                pass
            else:
                df_normalized.fillna(0, inplace=True)


        if method == 'TRQN':
            # TRQN will produce negative values
            df_normalized = df_normalized.clip(lower=0)

            

        return df_normalized



    # Batch effect correction methods
    def Batch_Correction(self, df, method = 'No Correction', UseCovariates = True):

        
        sample_batch_list = self.sample_batch_list
        sample_group_list = self.sample_group_list
        batches = self.batches

        sample_index_of_each_batch = []
        for batch in self.batch_name:
            row_indices = self.df_samples[self.df_samples['Batch'] == batch].index

            sample_index = row_indices.values.tolist()
            sample_index_of_each_batch.append(sample_index)

        print('Batch effect correction method: ' + method)
        

        # Import the object into the R environment beforehand
        robjects.r('rm(list=ls())')
        with lc(robjects.default_converter + pandas2ri.converter):
            r_dataframe = robjects.conversion.py2rpy(df)
        globalenv['data'] = r_dataframe 

        robjects.r('batch=c(%s)'%( str(sample_batch_list)[1:-1] ))
        robjects.r('grouplist=c(%s)'%( str(sample_group_list)[1:-1] ))


        if (method == 'No Correction') | (method == 'NoBC'):
            # No batch effect correction is performed, 
            # but in order for subsequent analysis to proceed normally, 
            # the negative numbers in the expression matrix are changed to 0.
            df[df < 0] = 0
            #df = df.dropna(axis=0, how='all')
            return df
        elif (method == 'Limma') | (method == 'limma'):

            # R script
            r_script_use_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)

            #data
            #batch 
            #grouplist

            g=factor(grouplist)
            design1=model.matrix(~g)

            data_corr <- removeBatchEffect(data, batch=batch, design=design1)
            data_corr

            '''

            r_script_no_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)

            #data
            #batch 
            #grouplist

            #g=factor(grouplist)
            #design1=model.matrix(~g)

            data_corr <- removeBatchEffect(data, batch=batch)
            data_corr

            '''

            # Execute R script
            metadata = None
            if UseCovariates:
                metadata = robjects.r(r_script_use_covariates)
            else:
                metadata = robjects.r(r_script_no_covariates)
            #metadata = robjects.r(r_script)
            pandas2ri.activate()
            metadata_py = pandas2ri.rpy2py(metadata) 
            pandas2ri.deactivate()

            # Reconvert data to dataframe
            df_correction = pd.DataFrame(metadata_py, index = df.index, columns = df.columns)

            # Change the values ​​<0 in the corrected data to 0
            df_correction[df_correction < 0] = 0
            #df_correction = df_correction.dropna(axis=0, how='all')

            return df_correction
        elif (method == 'Combat parametric with covariates') | (method == 'Combat-P'):
            # R script
            r_script_use_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(sva)

            #data
            #batch 
            #grouplist

            #data=na.omit(data)

            g=factor(grouplist)
            design1=model.matrix(~g)

            data_corr <- ComBat(dat = data, batch=batch, mod=design1, par.prior = TRUE)
            data_corr

            '''

            r_script_no_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(sva)

            #data
            #batch 
            #grouplist

            #data=na.omit(data)

            #g=factor(grouplist)
            #design1=model.matrix(~g)

            data_corr <- ComBat(dat = data, batch=batch, par.prior = TRUE)
            data_corr

            '''

            # Execute R script
            metadata = None
            if UseCovariates:
                metadata = robjects.r(r_script_use_covariates)
            else:
                metadata = robjects.r(r_script_no_covariates)
            #metadata = robjects.r(r_script)
            pandas2ri.activate()
            metadata_py = pandas2ri.rpy2py(metadata) 
            pandas2ri.deactivate()

            # Reconvert data to dataframe
            df_correction = pd.DataFrame(metadata_py, index = df.index, columns = df.columns)

            # Change the values ​​<0 in the corrected data to 0
            df_correction[df_correction < 0] = 0
            #df_correction = df_correction.dropna(axis=0, how='all')

            return df_correction
        elif (method == 'Combat non-parametric with covariates') | (method == 'Combat-NP'):
            # R script
            r_script_use_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(sva)

            #data
            #batch 
            #grouplist

            g=factor(grouplist)
            design1=model.matrix(~g)

            data_corr <- ComBat(dat = data, batch=batch, mod=design1, par.prior = FALSE)
            data_corr

            '''

            r_script_no_covariates = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(sva)

            #data
            #batch 
            #grouplist

            #g=factor(grouplist)
            #design1=model.matrix(~g)

            data_corr <- ComBat(dat = data, batch=batch, par.prior = FALSE)
            data_corr

            '''

            # Execute R script
            metadata = None
            if UseCovariates:
                metadata = robjects.r(r_script_use_covariates)
            else:
                metadata = robjects.r(r_script_no_covariates)
            #metadata = robjects.r(r_script)
            pandas2ri.activate()
            metadata_py = pandas2ri.rpy2py(metadata) 
            pandas2ri.deactivate()

            # Reconvert data to dataframe
            df_correction = pd.DataFrame(metadata_py, index = df.index, columns = df.columns)

            # Change the values ​​<0 in the corrected data to 0
            df_correction[df_correction < 0] = 0
            #df_correction = df_correction.dropna(axis=0, how='all')

            return df_correction
        elif method == 'Scanorama':

            # Extract the expression matrix and protein/peptide of the samples
            datasets = []
            protein_or_peptide_list = []

            for i in range(batches):
                df_batch_i = df.iloc[:, sample_index_of_each_batch[i]]
                datasets.append(df_batch_i.values.T)
                protein_or_peptide_list.append(df_batch_i.index.tolist())

            # Integration and batch correction.
            corrected, protein_or_peptide = scanorama.correct(datasets, protein_or_peptide_list)
            #integrated, corrected, protein_or_peptide = scanorama.correct(datasets, protein_or_peptide_list, return_dimred=True)

            corrected_data = []
            for i in range(batches):
                corrected_data.append(corrected[i].A.T)

            columns_corrected = []
            for i in range(batches):
                columns = df.iloc[:,sample_index_of_each_batch[i]].columns.tolist()
                columns_corrected = columns_corrected + columns

            df_corrected = pd.DataFrame(np.concatenate(corrected_data, axis=1), index = protein_or_peptide, columns = columns_corrected)
            df_corrected = df_corrected.reindex(index = df.index, columns=df.columns, copy=False)
            #df_corrected = pd.DataFrame(np.concatenate(corrected_data, axis=1), index = protein_or_peptide, columns = df.columns)
            #df_corrected = df_corrected.reindex(df.index.tolist())

            # Change the values ​​<0 in the corrected data to 0
            df_corrected[df_corrected < 0] = 0
            # Amplify the corrected data so that subsequent difference analysis methods can work properly

            df_corrected = df_corrected.apply(lambda x: x*(12/df_corrected.values.max())+1)

            return df_corrected


    
    # Cluster analysis and plotting clustering results
    def Cluster_Analysis(self, df, 
                         title = 'Figure title',
                         savefig = True, savefolder = './', savename = 'Cluster_Analysis'):


        group_label = self.group_name
        true_label = []
        for i in range(self.total_samples):
            true_label.append(group_label.index(self.sample_group_list[i]))

        batch_label = self.batch_name
        batches = self.batches
        groups = self.groups

        sample_index_list = []
        for batch in range(batches):
            for group in range(groups):

                row_indices = self.df_samples[(self.df_samples['Group'] == group_label[group]) & (self.df_samples['Batch'] == batch_label[batch])].index

                sample_index = row_indices.values.tolist()
                sample_index_list.append(sample_index)



        # Execute R script
        robjects.r('rm(list=ls())')
        with lc(robjects.default_converter + pandas2ri.converter):
            r_dataframe = robjects.conversion.py2rpy(df)
        globalenv['data'] = r_dataframe 


        # R script
        r_script_using_pca = '''

        options(java.parameters = "-Xmx10000m")

        library(Seurat)
        library(tidyr)
        #library(patchwork)

        # data
        data <- drop_na(data)  # Remove rows containing NA

        # Create Seurat object
        sce <- CreateSeuratObject(counts = data, min.cells = 0, min.features = 0, project = 'louvain_analysis')  # 3   200


        #sce <- NormalizeData(sce, normalization.method = "LogNormalize", scale.factor = 10000)
        sce  <- FindVariableFeatures(sce,selection.method = "vst",nfeatures = 3000) 
        sce <- ScaleData(sce, verbose = FALSE) 
        sce <- RunPCA(object = sce, npcs = 20, verbose = F) 
        #ElbowPlot(sce)

        # clustering
        sce <- FindNeighbors(sce, k.param = 10, dims = 1:20)
        sce <- FindClusters(sce, resolution = {0})

        #sce <- RunUMAP(sce, dims = 1:20, n.neighbors = 20L, n.components = 2L, verbose = F)

        sce@meta.data
        #sce@reductions[["pca"]]@cell.embeddings
        #sce@reductions[["umap"]]@cell.embeddings
        
        '''.format(self.FindClusters_resolution)


        r_script_using_umap = '''

        options(java.parameters = "-Xmx10000m")

        library(Seurat)
        library(tidyr)
        #library(patchwork)

        # data
        data <- drop_na(data)  # Remove rows containing NA

        # Create Seurat object
        sce <- CreateSeuratObject(counts = data, min.cells = 0, min.features = 0, project = 'louvain_analysis')  # 3   200


        #sce <- NormalizeData(sce, normalization.method = "LogNormalize", scale.factor = 10000)
        sce  <- FindVariableFeatures(sce,selection.method = "vst",nfeatures = 3000) 
        sce <- ScaleData(sce, verbose = FALSE) 
        sce <- RunPCA(object = sce, npcs = 20, verbose = F) 
        #ElbowPlot(sce)

        sce <- RunUMAP(sce, dims = 1:20, n.neighbors = 20L, n.components = 20L, verbose = F)

        # clustering
        sce <- FindNeighbors(sce, reduction = "umap", k.param = 10, dims = 1:20)
        sce <- FindClusters(sce, resolution = {0})


        sce@meta.data
        #sce@reductions[["pca"]]@cell.embeddings
        #sce@reductions[["umap"]]@cell.embeddings
        
        '''.format(self.FindClusters_resolution)


        # Get clustering results
        metadata = None
        if (additional_plot_methods.Reduction == 'pca'):
            metadata = robjects.r(r_script_using_pca)
        elif (additional_plot_methods.Reduction == 'umap'):
            metadata = robjects.r(r_script_using_umap)

        pandas2ri.activate()
        metadata_py = pandas2ri.rpy2py(metadata) 
        pandas2ri.deactivate()

        seurat_clusters = metadata_py['seurat_clusters'].values.T.codes.tolist() 
        #print('seurat_clusters: {0}\n'.format(seurat_clusters))

        # get pca results
        pca_data = None
        if (additional_plot_methods.Reduction == 'pca'):
            pca_data = np.array(robjects.r('sce@reductions[["pca"]]@cell.embeddings'))
            pca_stdev = np.array(robjects.r('sce@reductions[["pca"]]@stdev'))
            additional_plot_methods.PC1_2_Ratio = [pca_stdev[0]/np.sum(pca_stdev), pca_stdev[1]/np.sum(pca_stdev)]

        # get umap results
        umap_data = None
        if (additional_plot_methods.Reduction == 'umap'):
            umap_data = np.array(robjects.r('sce@reductions[["umap"]]@cell.embeddings'))


        # Calculate clustering index
        purity = accuracy(true_label, seurat_clusters)
        ri, ari, f_beta = get_rand_index_and_f_measure(true_label, seurat_clusters, beta=1.)

        # Save data
        df_data = pd.DataFrame()
        df_data['Run Name'] = self.df_samples['Run Name'].values.tolist()
        df_data['Group'] = self.df_samples['Group'].values.tolist()
        df_data['Batch'] = self.df_samples['Batch'].values.tolist()
        if (additional_plot_methods.Reduction == 'pca'):
            df_data['PCA 1'] = pca_data[:,0]
            df_data['PCA 2'] = pca_data[:,1]
        if (additional_plot_methods.Reduction == 'umap'):
            df_data['UMAP 1'] = umap_data[:,0]
            df_data['UMAP 2'] = umap_data[:,1]

        df_data['Expected Label'] = true_label
        df_data['Cluster Label'] = seurat_clusters
        df_data.to_csv(savefolder + savename + '.csv', index=False)

        
        # Plot clustering results
        if savefig:
            Plot_ARI_Version_Cluster_Diagram(df_data = df_data, ari = ari,
                                               Groups = group_label,
                                               Batches = batch_label,
                                               savefig = savefig, savefolder = savefolder, savename = savename)


        return df, ari



    # Draw the ARI/Purity Score histogram of different sparsity reduction methods
    def Plot_BatchCorrection_ARI(self, result_csv_path, 
                                 SR_method = 'SR90', 
                                 Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                 Normalization_methods = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                 BC_methods = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                 Difference_analysis_methods = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'],
                                 ARI_or_Purity = 'ARI',
                                 savefig = True, savefolder = ''):

        # Read csv file
        df_result = pd.read_csv(result_csv_path, index_col=0)
        # Sort the 'No' column in ascending order
        df_result = df_result.sort_values('No', ascending=True)
        df_screened = df_result[df_result['Sparsity Reduction'] == SR_method]
        if (ARI_or_Purity == 'ARI'):
            ARI_list = df_screened['ARI'].values.tolist()  # ARI Results List
        if (ARI_or_Purity == 'Purity Score'):
            ARI_list = df_screened['Purity Score'].values.tolist()  # Use Purity Score instead of ARI for plotting

        BC_methods_count = len(BC_methods)
        Statistica_methods_count = len(Difference_analysis_methods)

        Temp_list = []
        for i in range(0, len(ARI_list), Statistica_methods_count): 
            Temp_list.append(ARI_list[i])

        ARI_list = []
        for j in range(0, len(Temp_list), BC_methods_count):
            ARI_list.append(Temp_list[j:j+BC_methods_count])


        # BatchCorrection_ARI_{SR}.svg
        fig, ax = plt.subplots(len(Fill_NaN_methods), len(Normalization_methods), figsize=(10,12))  

        colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890']

        for row in range(len(Fill_NaN_methods)):
            for column in range(len(Normalization_methods)):

                plot_index = row*len(Normalization_methods) + column
                plt.subplot(len(Fill_NaN_methods), len(Normalization_methods), plot_index+1)

                bar1 = plt.bar(list(range(BC_methods_count)), ARI_list[plot_index], width = 0.75, color = colors) 

                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                plt.ylim(0, 1) 
                plt.yticks(np.linspace(0, 1, 3)) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(False)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 

                if row == 0:
                    title_name = Normalization_methods[column]

                    if title_name == 'TRQN':
                        title_name = '        TRQN         '
                    if title_name == 'QN':
                        title_name = '            QN            '
                    if title_name == 'Sum':
                        title_name = '          Sum          '
                    if title_name == 'Median':
                        title_name = '        Median         '
                    if title_name == 'Unnormalized':
                        title_name = '  Unnormalized   '

                    bbox = plt.title(title_name, horizontalalignment='center', verticalalignment='center', fontsize=15, pad = 15, bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1))
                    

                if column == 0:
                    if (ARI_or_Purity == 'ARI'):
                        plt.ylabel('ARI', fontsize=16) 
                    if (ARI_or_Purity == 'Purity Score'):
                        plt.ylabel('Purity Score', fontsize=16) 

                    axes.spines['left'].set_visible(True) 
                    plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 

                if column == (len(Normalization_methods) - 1):

                    axes.yaxis.set_label_position('right') 

                    y_label = Fill_NaN_methods[row]
                    
                    if y_label == 'HalfRowMin':
                        y_label = ' HalfRowMin '
                    if y_label == 'RowMedian':
                        y_label = '  RowMedian '
                    if y_label == 'Zero':
                        y_label = '       Zero       '
                    if y_label == 'RowMean':
                        y_label = '   RowMean  '
                    if y_label == 'SoftImpute':
                        y_label = '  SoftImpute  '
                    if y_label == 'IterativeSVD':
                        y_label = ' IterativeSVD'
                    if y_label == 'KNN':
                        y_label = '       KNN       '
                    plt.ylabel(y_label, rotation = 270, horizontalalignment='center', verticalalignment='center', fontsize=15, labelpad=15, 
                               bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1)) 


        plt.subplots_adjust(left=0.07, right=0.96, bottom=0.01, top=0.965, wspace=0.05)

        if savefig:
            if (ARI_or_Purity == 'ARI'):
                plt.savefig(savefolder + 'BatchCorrection_ARI_{0}.svg'.format(SR_method), dpi=600, format="svg", transparent=True) 
            if (ARI_or_Purity == 'Purity Score'):
                plt.savefig(savefolder + 'BatchCorrection_PurityScore_{0}.svg'.format(SR_method), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()


        # BatchCorrection_ARI_{SR}_{Imputation}_{Norm}.svg

        colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890']

        for row in range(len(Fill_NaN_methods)):
            for column in range(len(Normalization_methods)):

                fig, ax = plt.subplots(1, 1, figsize=(2.5,2))

                plot_index = row*len(Normalization_methods) + column

                bar1 = plt.bar(list(range(BC_methods_count)), ARI_list[plot_index], width = 0.75, color = colors) 

                plt.tick_params(labelsize=14)
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                plt.ylim(0, 1) 
                plt.yticks(np.linspace(0, 1, 3)) 

                axes = plt.gca()
                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(True)
                axes.spines['left'].set_visible(True)
                axes.spines['bottom'].set_linewidth(2) 
                axes.spines['left'].set_linewidth(2) 

                if (ARI_or_Purity == 'ARI'):
                    plt.ylabel('ARI', fontsize=16) 
                if (ARI_or_Purity == 'Purity Score'):
                    plt.ylabel('Purity Score', fontsize=16) 

                plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 

                plt.subplots_adjust(left=0.28, right=0.99, bottom=0.05, top=0.94, wspace=0.05)

                if savefig:
                    if (ARI_or_Purity == 'ARI'):
                        plt.savefig(savefolder + 'BatchCorrection_ARI_{0}_{1}_{2}.svg'.format(SR_method, Fill_NaN_methods[row], Normalization_methods[column]), dpi=600, format="svg", transparent=True) 
                    if (ARI_or_Purity == 'Purity Score'):
                        plt.savefig(savefolder + 'BatchCorrection_PurityScore_{0}_{1}_{2}.svg'.format(SR_method, Fill_NaN_methods[row], Normalization_methods[column]), dpi=600, format="svg", transparent=True) 
                plt.show()
                plt.close()




        # Legend
        if savefig:
            fig_legend = plt.figure(figsize=(2.5,2.5))
 
            axes = plt.gca()
            parts = axes.violinplot([[10000]]*len(BC_methods), list(range(1, len(BC_methods)+1)), widths=0.8,
                                    showmeans=False, showmedians=False, showextrema=False)
            count = 0
            for pc in parts['bodies']:
                pc.set_facecolor(colors[count])
                pc.set_edgecolor('white')
                pc.set_alpha(1)

                count += 1

            axes.legend(labels = BC_methods, title='Batch Correction', title_fontsize=18, fontsize=16, 
                        loc = 'center',
                        markerfirst=True, markerscale=2.0) 

            plt.ylim(-5, 5)
            plt.xlim(-5, 5)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

            plt.xticks([])
            plt.yticks([])

            plt.savefig(savefolder + 'Legend_BatchCorrection.svg', dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()


        # Plot a dot plot for each batch effect correction method
        for BC in BC_methods:
            fig, ax = plt.subplots(1, 1, figsize=(4,5))

            for row in range(len(Fill_NaN_methods)):
                for column in range(len(Normalization_methods)):

                    plot_index = row*len(Normalization_methods) + column

                    ARI_data = ARI_list[plot_index][BC_methods.index(BC)]


                    # Init color and scatter size
                    Init_scatter_color_list = ['#e95b1b', '#b0203f', '#a14ee0', '#1883b8', '#39a139']
                    Init_scatter_color = Init_scatter_color_list[BC_methods.index(BC)]
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
                    plt.scatter([column], [len(Fill_NaN_methods) -1 - row], color = Init_scatter_color, edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)

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

            plt.xlim(-0.7, (len(Normalization_methods)-1) + 0.7)
            
            plt.xticks(list(range(0, len(Normalization_methods), 1)), Normalization_methods)

            plt.ylim(-0.7, (len(Fill_NaN_methods)-1) + 0.7)
            plt.yticks(list(range(0, len(Fill_NaN_methods), 1)), list(reversed(Fill_NaN_methods)))

            plt.subplots_adjust(left=0.31, right=1, bottom=0.28, top=1, wspace=0.05)

            if savefig:
                if (ARI_or_Purity == 'ARI'):
                    plt.savefig(savefolder + 'Imputation_Normalization_ARI_{0}_{1}.svg'.format(SR_method, BC), dpi=600, format="svg", transparent=True) 
                if (ARI_or_Purity == 'Purity Score'):
                    plt.savefig(savefolder + 'Imputation_Normalization_PurityScore_{0}_{1}.svg'.format(SR_method, BC), dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()


        # Legend
        if savefig:
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

                parts = axes.scatter([100], [100], color = '#e95b1b', edgecolor='white', marker = 'o', s = scatter_size, alpha=scatter_color_alpha)
            

            axes.legend(labels = labels, title=ARI_or_Purity, title_fontsize=18, fontsize=16, 
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

            if (ARI_or_Purity == 'ARI'):
                plt.savefig(savefolder + 'Legend_Imputation_Normalization_ARI.svg', dpi=600, format="svg", transparent=True) 
            if (ARI_or_Purity == 'Purity Score'):
                plt.savefig(savefolder + 'Legend_Imputation_Normalization_PurityScore.svg', dpi=600, format="svg", transparent=True)
            plt.show()
            plt.close()



    # Draw the 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score' histogram of different sparsity reduction methods
    def Plot_StatisticalTest_Result(self, result_csv_path, 
                                 SR_method = 'SR90', 
                                 Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                 Normalization_methods = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                 BC_methods = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                 Difference_analysis_methods = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'],
                                 Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                 Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                                 savefig = True, savefolder = ''):

        # Read csv file
        df_result = pd.read_csv(result_csv_path, index_col=0) 
        df_result = df_result.sort_values('No', ascending=True)
        df_screened = df_result[df_result['Sparsity Reduction'] == SR_method]  # Filter data that belongs to this sparsity reduction method

        # Plot each batch effect correction method
        for BC in BC_methods:
            # Filter data that are relevant to this batch effect correction method
            df_screened_BC = df_screened[df_screened['Batch Correction'] == BC]

            # Plot each indicator in turn
            for Indicator in Indicator_type_list:
                Indicator_column_names = []
                for Compared_group in Compared_groups_label:
                    Indicator_column_names.append(Compared_group + ' ' + Indicator)

                Compared_groups_indicator_data = [[]]*len(Compared_groups_label)
                for Indicator_column_name in Indicator_column_names:
                    Compared_groups_indicator_data[Indicator_column_names.index(Indicator_column_name)] = df_screened_BC[Indicator_column_name].values.tolist()

                # Draw bar charts and line charts
                fig, ax = plt.subplots(len(Fill_NaN_methods), len(Normalization_methods), figsize=(10,12))  

                colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890', '#fbe29d', '#c8d961']
                marker_list = ['o', '^', 's', 'd', '*', 'v', '<', '>']

                count = 0
                for row in range(len(Fill_NaN_methods)):
                    for column in range(len(Normalization_methods)):

                        plot_index = row*len(Normalization_methods) + column
                        plt.subplot(len(Fill_NaN_methods), len(Normalization_methods), plot_index+1)

                        # Histogram Plotting Data
                        bar_plot_data = []
                        # Data for line chart
                        scatter_plot_data = []
                        for i in range(len(Compared_groups_label)):
                            scatter_plot_data.append(Compared_groups_indicator_data[i][count*len(Difference_analysis_methods):(count*len(Difference_analysis_methods) + len(Difference_analysis_methods))])

                        count += 1

                        array = np.array(scatter_plot_data)
                        bar_plot_data = array.mean(axis = 0).tolist()

                        bar1 = plt.bar(list(range(len(Difference_analysis_methods))), bar_plot_data, width = 0.75, color = colors) 
                        for data in scatter_plot_data:
                            plt.plot(list(range(len(Difference_analysis_methods))), data, marker = marker_list[scatter_plot_data.index(data)], markersize = 4, linewidth = 0.5, color = 'black') 


                        plt.tick_params(labelsize=14) 
                        plt.tick_params(axis='x', width=2)
                        plt.tick_params(axis='y', width=2)

                        if Indicator == 'pAUC':
                            plt.ylim(-0.005, 0.105) 
                            plt.yticks(np.linspace(0, 0.1, 3)) 
                        else:
                            plt.ylim(-0.05, 1.05) 
                            plt.yticks(np.linspace(0, 1, 3)) 

                        axes = plt.gca()
                        axes.spines['top'].set_visible(False) 
                        axes.spines['right'].set_visible(False)
                        axes.spines['bottom'].set_visible(True)
                        axes.spines['left'].set_visible(False)
                        axes.spines['bottom'].set_linewidth(2) 
                        axes.spines['left'].set_linewidth(2) 

                        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 

                        if row == 0:
                            title_name = Normalization_methods[column]

                            if title_name == 'TRQN':
                                title_name = '         TRQN        '
                            if title_name == 'QN':
                                title_name = '           QN            '
                            if title_name == 'Sum':
                                title_name = '          Sum          '
                            if title_name == 'Median':
                                title_name = '        Median         '
                            if title_name == 'Unnormalized':
                                title_name = '  Unnormalized   '

                            
                            bbox = plt.title(title_name, horizontalalignment='center', verticalalignment='center', fontsize=15, pad = 15, bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1))
                    

                        if column == 0:
                            if Indicator == 'accuracy':
                                Indicator = 'Accuracy'
                            if Indicator == 'precision':
                                Indicator = 'Precision'
                            if Indicator == 'recall':
                                Indicator = 'Recall'
                            if Indicator == 'f1_score':
                                Indicator = 'F1-Score'
                            plt.ylabel(Indicator, fontsize=16) 

                            axes.spines['left'].set_visible(True) 
                            plt.tick_params(axis='both', which='both', bottom=False, left=True, labelbottom=False, labelleft=True) 

                        if column == (len(Normalization_methods) - 1):
                            axes.yaxis.set_label_position('right') 

                            y_label = Fill_NaN_methods[row]
                    
                            if y_label == 'HalfRowMin':
                                y_label = ' HalfRowMin '
                            if y_label == 'RowMedian':
                                y_label = '  RowMedian '
                            if y_label == 'Zero':
                                y_label = '       Zero       '
                            if y_label == 'RowMean':
                                y_label = '   RowMean  '
                            if y_label == 'SoftImpute':
                                y_label = '  SoftImpute  '
                            if y_label == 'IterativeSVD':
                                y_label = ' IterativeSVD'
                            if y_label == 'KNN':
                                y_label = '       KNN       '
                            plt.ylabel(y_label, rotation = 270, horizontalalignment='center', verticalalignment='center', fontsize=15, labelpad=15, 
                                       bbox=dict(facecolor='#dadada', edgecolor = 'white', alpha=1)) 


                plt.subplots_adjust(left=0.08, right=0.96, bottom=0.01, top=0.965, wspace=0.05)


                if savefig:
                    plt.savefig(savefolder + 'StatisticalTest_{0}_{1}_{2}.svg'.format(Indicator, SR_method, BC), dpi=600, format="svg", transparent=True) 
                plt.show()
                plt.close()


                # Legend_StatisticalTest_Comparison
                if savefig:
                    fig_legend = plt.figure(figsize=(2.5, 2.5))
                    axes = plt.gca()

                    for i in range(len(Compared_groups_label)):
                        parts = axes.scatter([100], [100], color = 'black', edgecolor='white', marker = marker_list[i], s = 4, alpha=1)
            
                    
                    axes.legend(labels = Compared_groups_label, title='Comparison', title_fontsize=18, fontsize=16, 
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

                    plt.savefig(savefolder + 'Legend_StatisticalTest_Comparison.svg', dpi=600, format="svg", transparent=True) 
                    plt.show()
                    plt.close()


                # Legend_StatisticalTest
                if savefig:
                    fig_legend = plt.figure(figsize=(2.5, 3.5))
                    axes = plt.gca()

                    if 'DESeq2-parametric' in Difference_analysis_methods:
                        Difference_analysis_methods[Difference_analysis_methods.index('DESeq2-parametric')] = 'DESeq2'
                    if 'Wilcoxon-test' in Difference_analysis_methods:
                        Difference_analysis_methods[Difference_analysis_methods.index('Wilcoxon-test')] = 'Wilcox'


                    parts = axes.violinplot([[10000]]*len(Difference_analysis_methods), list(range(1, len(Difference_analysis_methods)+1)), widths=0.8,
                                            showmeans=False, showmedians=False, showextrema=False)
                    count = 0
                    for pc in parts['bodies']:
                        pc.set_facecolor(colors[count])
                        pc.set_edgecolor('white')
                        pc.set_alpha(1)

                        count += 1
            

                    axes.legend(labels = Difference_analysis_methods, title='Statistical Test', title_fontsize=18, fontsize=16, 
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

                    plt.savefig(savefolder + 'Legend_StatisticalTest.svg', dpi=600, format="svg", transparent=True) 
                    plt.show()
                    plt.close()



    # Draw parallel coordinates plots of performance indicators for different comparison groups under different sparsity reduction methods
    def Plot_StatisticalMetrics(self, result_csv_path, 
                                 SR_method = 'SR90', 
                                 Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                 Normalization_methods = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                 BC_methods = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                 Difference_analysis_methods = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'],
                                 Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                 Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                                 ARI_or_Purity = 'ARI',
                                 savefig = True, savefolder = '',
                                 rank_scheme = 1):

        # Read csv file
        df_result = pd.read_csv(result_csv_path, index_col=0) 
        df_result = df_result.sort_values('No', ascending=True)
        df_screened = df_result[df_result['Sparsity Reduction'] == SR_method]  # Filter data that belongs to this sparsity reduction method

        colors = ['#f4ac8c', '#ed9dae', '#d39feb', '#67bfeb', '#90d890', '#fbe29d', '#c8d961']

        # When Rank is Top %
        if (rank_scheme == 1):
            # Plot for each comparison
            for Compared_groups in Compared_groups_label:
                # Filter the plot data 

                # If use given FC and p-value, add Precision when plotting
                columns = None
                if self.Use_Given_PValue_and_FC:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                               ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Precision', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                               Compared_groups + ' Rank']
                else:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                               ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                               Compared_groups + ' Rank']

                df_plot = df_screened[columns].copy(deep = True) 
                # Change the strings in the first 4 columns to numeric values
                value_list = []
                for  i in df_plot['Missing Value Imputation'].values.tolist():
                    value_list.append(Fill_NaN_methods.index(i))
                df_plot['Missing Value Imputation'] = value_list

                value_list = []
                for  i in df_plot['Normalization'].values.tolist():
                    value_list.append(Normalization_methods.index(i))
                df_plot['Normalization'] = value_list

                value_list = []
                for  i in df_plot['Batch Correction'].values.tolist():
                    value_list.append(BC_methods.index(i))
                df_plot['Batch Correction'] = value_list

                value_list = []
                for  i in df_plot['Statistical Test'].values.tolist():
                    value_list.append(Difference_analysis_methods.index(i))
                df_plot['Statistical Test'] = value_list


                # Set df data type to float
                df_plot.astype('float')

                # Modify Rank
                current_rank_list = df_plot[Compared_groups + ' Rank'].values.tolist()
                new_rank_list = []
                for rank in current_rank_list:
                    rank_ratio = rank/4900
                    if rank_ratio == 1/4900:
                        new_rank_list.append(1)
                    elif rank_ratio <= 0.01:
                        new_rank_list.append(int(rank*700/49))
                    elif rank_ratio <= 0.02:
                        new_rank_list.append(int(700 + (rank-49)*700/49))
                    elif rank_ratio <= 0.04:
                        new_rank_list.append(int(700*2 + (rank-49*2)*700/(49*2)))
                    elif rank_ratio <= 0.06:
                        new_rank_list.append(int(700*3 + (rank-49*4)*700/(49*2)))
                    elif rank_ratio <= 0.08:
                        new_rank_list.append(int(700*4 + (rank-49*6)*700/(49*2)))
                    elif rank_ratio <= 0.10:
                        new_rank_list.append(int(700*5 + (rank-49*8)*700/(49*2)))
                    else:
                        new_rank_list.append(int(700*6 + (rank-49*10)*700/(49*90)))

                df_plot[Compared_groups + ' Rank'] = new_rank_list

                max_rank_value = 4900 
                min_rank_value = 1 
                df_plot[Compared_groups + ' Rank'] = max_rank_value - df_plot[Compared_groups + ' Rank'] + min_rank_value


                # Draw a parallel coordinates plot
                ynames = None
                if self.Use_Given_PValue_and_FC:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                              ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
                else:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                              ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']

                

                ys = np.array(df_plot.values.tolist())
                # Add a row of data at the end to adjust the maximum and minimum values ​​of the parallel coordinate graph indicators
                #ys = np.append(ys, [[0, 0, 0, 0, 0.1, 1, 1, 1, 1]], axis=0)
                if self.Use_Given_PValue_and_FC:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 4900]], axis=0)
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)
                    ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
                else:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 4900]], axis=0)
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 0]], axis=0)

                    # Enlarge the gap between the last few columns
                    ys[:, -5:-1] = 1/(1.5 - (ys[:, -5:-1] ** 4))
                    #ys[:, -5:] = 1/(1.5 - (ys[:, -5:] ** 4))
                    #ys[:, -4:] = ys[:, -4:] ** 3

                # Add 5% noise to the first 4 columns of data
                noise_upper_bound = 1.05
                noise_lower_bound = 0.95
                #np.ones((ys.shape[0], 4))
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
                            
                            ax.set_yticks(range(len(Fill_NaN_methods)))
                            ax.set_yticklabels(Fill_NaN_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')

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
                                
                                ax.set_yticks(range(len(Normalization_methods)))
                                ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 2:
                                
                                ax.set_yticks(range(len(BC_methods)))
                                ax.set_yticklabels(BC_methods, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 3:
                                
                                ax.set_yticks(range(len(Difference_analysis_methods)))
                                ax.set_yticklabels(Difference_analysis_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')
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

                                ax.set_yticks([0, 700, 700*2, 700*3, 700*4, 700*5, 700*6, 4900])
                                str_list = ['100%', '10%', '8%', '6%', '4%', '2%', '1%', '0%']

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

                                #print(ytick_label)
                                #time.sleep(2)

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


                    for j in range(ys.shape[0] - 2):
                        verts = list(
                            zip(
                                [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                                np.repeat(zs[j, :], 3)[1:-1],
                            )
                        )
                        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                        path = Path(verts, codes)


                        sorted_Rank_list = df_plot[Compared_groups + ' Rank'].values.tolist()
                        sorted_Rank_list = sorted(sorted_Rank_list, reverse=True) 
                        current_Rank = df_plot[Compared_groups + ' Rank'].values.tolist()[j]
                        current_Rank_rank = (sorted_Rank_list.index(current_Rank))/len(sorted_Rank_list)



                        alpha = 0.85
                        linewidth = 0.95

                        # Sort by pAUC, using different line colors
                        if current_Rank_rank <= 0.01:
                            edgecolor = '#b0203f'
                        elif current_Rank_rank <= 0.02:
                            edgecolor = '#e95b1b'
                        elif current_Rank_rank <= 0.04:
                            edgecolor = '#a14ee0'
                        elif current_Rank_rank <= 0.06:
                            edgecolor = '#1883b8'
                        elif current_Rank_rank <= 0.08:
                            edgecolor = '#39a139'
                        elif current_Rank_rank <= 0.10:
                            edgecolor = '#bc8f00'
                        else:
                            edgecolor = '#d9d8d2'
                            alpha = 0.2
                            linewidth = 0.2

                
                        if plot_index == 0:

                            if current_Rank_rank > 0.1:
                                patch = patches.PathPatch(
                                    path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                )
                                #legend_handles[j] = patch
                                host.add_patch(patch)

                        else:
                            if current_Rank_rank <= 0.1:
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
                svg1.save(savefolder + 'StatisticalMetrics_{0}_{1}_vs_{2}.svg'.format(SR_method, Compared_groups.split('/')[0], Compared_groups.split('/')[1]))


            

                # Legend_Rank_Top
                fig_legend = plt.figure(figsize=(2, 4))
 
                axes = plt.gca()
                edgecolor = ['#b0203f', '#e95b1b', '#a14ee0', '#1883b8', '#39a139', '#bc8f00', '#d9d8d2']
                labels = ['≤ 1%', '1% - 2%', '2% - 4%', '4% - 6%', '6% - 8%', '8% - 10%', '> 10%']
                for i in range(7):
                    axes.plot([10000, 20000], [10000, 20000], lw=2, color = edgecolor[i])
            

                axes.legend(labels = labels, title='Top %', title_fontsize=18, fontsize=16, 
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

                plt.savefig(savefolder + 'Legend_Rank_Top.svg', dpi=600, format="svg", transparent=True) 
                plt.show()
                plt.close()



            # StatisticalMetrics_{0}_Average.svg
            columns = None
            if self.Use_Given_PValue_and_FC:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1-Score',
                            'Rank']
            else:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Recall', 'Average F1-Score',
                            'Rank']

            df_plot = df_screened[columns].copy(deep = True) 
            value_list = []
            for  i in df_plot['Missing Value Imputation'].values.tolist():
                value_list.append(Fill_NaN_methods.index(i))
            df_plot['Missing Value Imputation'] = value_list

            value_list = []
            for  i in df_plot['Normalization'].values.tolist():
                value_list.append(Normalization_methods.index(i))
            df_plot['Normalization'] = value_list

            value_list = []
            for  i in df_plot['Batch Correction'].values.tolist():
                value_list.append(BC_methods.index(i))
            df_plot['Batch Correction'] = value_list

            value_list = []
            for  i in df_plot['Statistical Test'].values.tolist():
                value_list.append(Difference_analysis_methods.index(i))
            df_plot['Statistical Test'] = value_list


            df_plot.astype('float')

            current_rank_list = df_plot['Rank'].values.tolist()
            new_rank_list = []
            for rank in current_rank_list:
                rank_ratio = rank/4900
                if rank_ratio == 1/4900:
                    new_rank_list.append(1)
                elif rank_ratio <= 0.01:
                    new_rank_list.append(int(rank*700/49))
                elif rank_ratio <= 0.02:
                    new_rank_list.append(int(700 + (rank-49)*700/49))
                elif rank_ratio <= 0.04:
                    new_rank_list.append(int(700*2 + (rank-49*2)*700/(49*2)))
                elif rank_ratio <= 0.06:
                    new_rank_list.append(int(700*3 + (rank-49*4)*700/(49*2)))
                elif rank_ratio <= 0.08:
                    new_rank_list.append(int(700*4 + (rank-49*6)*700/(49*2)))
                elif rank_ratio <= 0.10:
                    new_rank_list.append(int(700*5 + (rank-49*8)*700/(49*2)))
                else:
                    new_rank_list.append(int(700*6 + (rank-49*10)*700/(49*90)))

            df_plot['Rank'] = new_rank_list

            max_rank_value = 4900
            min_rank_value = 1
            df_plot['Rank'] = max_rank_value - df_plot['Rank'] + min_rank_value

            ynames = None
            if self.Use_Given_PValue_and_FC:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                          ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
            else:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                          ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']
            ys = np.array(df_plot.values.tolist())
            if self.Use_Given_PValue_and_FC:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 4900]], axis=0)
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)

                ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
            else:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 4900]], axis=0)
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 0]], axis=0)

                ys[:, -5:-1] = 1/(1.5 - (ys[:, -5:-1] ** 4))
                #ys[:, -5:] = 1/(1.5 - (ys[:, -5:] ** 4))
                #ys[:, -4:] = ys[:, -4:] ** 3

            noise_upper_bound = 1.05
            noise_lower_bound = 0.95
            #np.ones((ys.shape[0], 4))
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
                        ax.set_yticks(range(len(Fill_NaN_methods)))
                        ax.set_yticklabels(Fill_NaN_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')
                    

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
                            ax.set_yticks(range(len(Normalization_methods)))
                            ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 2:
                            ax.set_yticks(range(len(BC_methods)))
                            ax.set_yticklabels(BC_methods, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 3:
                            ax.set_yticks(range(len(Difference_analysis_methods)))
                            ax.set_yticklabels(Difference_analysis_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')
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

                            ax.set_yticks([0, 700, 700*2, 700*3, 700*4, 700*5, 700*6, 4900])
                            str_list = ['100%', '10%', '8%', '6%', '4%', '2%', '1%', '0%']

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


                for j in range(ys.shape[0] - 2):
                    verts = list(
                        zip(
                            [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                            np.repeat(zs[j, :], 3)[1:-1],
                        )
                    )
                    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                    path = Path(verts, codes)


                    sorted_Rank_list = df_plot['Rank'].values.tolist()
                    sorted_Rank_list = sorted(sorted_Rank_list, reverse=True) 
                    current_Rank = df_plot['Rank'].values.tolist()[j]
                    current_Rank_rank = (sorted_Rank_list.index(current_Rank))/len(sorted_Rank_list)


                    alpha = 0.85
                    linewidth = 0.95

                    if current_Rank_rank <= 0.01:
                        edgecolor = '#b0203f'
                    elif current_Rank_rank <= 0.02:
                        edgecolor = '#e95b1b'
                    elif current_Rank_rank <= 0.04:
                        edgecolor = '#a14ee0'
                    elif current_Rank_rank <= 0.06:
                        edgecolor = '#1883b8'
                    elif current_Rank_rank <= 0.08:
                        edgecolor = '#39a139'
                    elif current_Rank_rank <= 0.10:
                        edgecolor = '#bc8f00'
                    else:
                        edgecolor = '#d9d8d2'
                        alpha = 0.2
                        linewidth = 0.2

                
                    if plot_index == 0:

                        if current_Rank_rank > 0.1:
                            patch = patches.PathPatch(
                                path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                            )
                            #legend_handles[j] = patch
                            host.add_patch(patch)

                    else:
                        if current_Rank_rank <= 0.1:
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
            svg1.save(savefolder + 'StatisticalMetrics_{0}_Average.svg'.format(SR_method))



        # When Rank sorting within each comparison group
        if (rank_scheme == 2):
            for Compared_groups in Compared_groups_label:

                # If use given FC and p-value, add Precision when plotting
                columns = None
                if self.Use_Given_PValue_and_FC:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                               ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Precision', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                               Compared_groups + ' Rank']
                else:
                    columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                               ARI_or_Purity, Compared_groups + ' pAUC', Compared_groups + ' Accuracy', Compared_groups + ' Recall', Compared_groups + ' F1-Score',
                               Compared_groups + ' Rank']



                df_plot = df_screened[columns].copy(deep = True) 
                value_list = []
                for  i in df_plot['Missing Value Imputation'].values.tolist():
                    value_list.append(Fill_NaN_methods.index(i))
                df_plot['Missing Value Imputation'] = value_list

                value_list = []
                for  i in df_plot['Normalization'].values.tolist():
                    value_list.append(Normalization_methods.index(i))
                df_plot['Normalization'] = value_list

                value_list = []
                for  i in df_plot['Batch Correction'].values.tolist():
                    value_list.append(BC_methods.index(i))
                df_plot['Batch Correction'] = value_list

                value_list = []
                for  i in df_plot['Statistical Test'].values.tolist():
                    value_list.append(Difference_analysis_methods.index(i))
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
                        new_rank_list.append(int((4900/6)*rank_ratio/0.01))
                    elif rank_ratio <= 0.02:
                        new_rank_list.append(int((4900/6)*1+(4900/6)*(rank_ratio-0.01)/0.01))
                    elif rank_ratio <= 0.03:
                        new_rank_list.append(int((4900/6)*2+(4900/6)*(rank_ratio-0.02)/0.01))
                    elif rank_ratio <= 0.04:
                        new_rank_list.append(int((4900/6)*3+(4900/6)*(rank_ratio-0.03)/0.01))
                    elif rank_ratio <= 0.05:
                        new_rank_list.append(int((4900/6)*4+(4900/6)*(rank_ratio-0.04)/0.01))
                    else:
                        new_rank_list.append(int((4900/6)*5+(4900/6)*(rank_ratio-0.05)/0.95))


                df_plot[Compared_groups + ' Rank'] = new_rank_list

                max_rank_value = 4900 
                min_rank_value = 1 
                df_plot[Compared_groups + ' Rank'] = max_rank_value - df_plot[Compared_groups + ' Rank'] + min_rank_value


                ynames = None
                if self.Use_Given_PValue_and_FC:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                              ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
                else:
                    ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                              ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']

                ys = np.array(df_plot.values.tolist())

                if self.Use_Given_PValue_and_FC:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 4900]], axis=0)
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)
                    ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
                else:
                    ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 4900]], axis=0)
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
                            ax.set_yticks(range(len(Fill_NaN_methods)))
                            ax.set_yticklabels(Fill_NaN_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')

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
                                ax.set_yticks(range(len(Normalization_methods)))
                                ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 2:
                                ax.set_yticks(range(len(BC_methods)))
                                ax.set_yticklabels(BC_methods, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                                ax.tick_params(axis="y", pad = -6)

                                if plot_index == 0:
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.spines['right'].set_visible(False)
                                    ax.spines['left'].set_visible(False)

                            elif i == 3:
                                ax.set_yticks(range(len(Difference_analysis_methods)))
                                ax.set_yticklabels(Difference_analysis_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')
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

                                #ax.set_yticks([0, 700*1, 700*2, 700*3, 700*4, 700*5, 700*6, 4900])
                                #str_list = ['100%', '10%', '8%', '6%', '4%', '2%', '1%', '0%']

                                ax.set_yticks([0, (4900/6)*1, (4900/6)*2, (4900/6)*3, (4900/6)*4, (4900/6)*5, 4900])
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

                    
                    for tuceng in range(7):
                        for j in range(ys.shape[0] - 2):
                            verts = list(
                                zip(
                                    [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                                    np.repeat(zs[j, :], 3)[1:-1],
                                )
                            )
                            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                            path = Path(verts, codes)

                            sorted_Rank_list = df_plot[Compared_groups + ' Rank'].values.tolist()
                            sorted_Rank_list = sorted(sorted_Rank_list, reverse=True) 
                            current_Rank = df_plot[Compared_groups + ' Rank'].values.tolist()[j]
                            current_Rank_rank = (sorted_Rank_list.index(current_Rank))/len(sorted_Rank_list)


                            alpha = 0.85
                            linewidth = 0.95

                            if current_Rank_rank < 3/1225:
                                if (tuceng != 6):
                                    continue
                                edgecolor = '#b0203f'
                                linewidth = 2.0
                                alpha = 1.0
                            elif current_Rank_rank <= 0.01:
                                if (tuceng != 5):
                                    continue
                                edgecolor = '#e95b1b'
                                alpha = 0.85
                                linewidth = 0.95
                            elif current_Rank_rank <= 0.02:
                                if (tuceng != 4):
                                    continue
                                edgecolor = '#a14ee0'
                                alpha = 0.85
                                linewidth = 0.95
                            elif current_Rank_rank <= 0.03:
                                if (tuceng != 3):
                                    continue
                                edgecolor = '#1883b8'
                                alpha = 0.85
                                linewidth = 0.95
                            elif current_Rank_rank <= 0.04:
                                if (tuceng != 2):
                                    continue
                                edgecolor = '#39a139'
                                alpha = 0.85
                                linewidth = 0.95
                            elif current_Rank_rank <= 0.05:
                                if (tuceng != 1):
                                    continue
                                edgecolor = '#bc8f00'
                                alpha = 0.85
                                linewidth = 0.95
                            else:
                                if (tuceng != 0):
                                    continue
                                edgecolor = '#d9d8d2'
                                alpha = 0.2
                                linewidth = 0.2


                            #if current_Rank_rank <= 0.01:
                            #    edgecolor = '#b0203f'
                            #elif current_Rank_rank <= 0.02:
                            #    edgecolor = '#e95b1b'
                            #elif current_Rank_rank <= 0.04:
                            #    edgecolor = '#a14ee0'
                            #elif current_Rank_rank <= 0.06:
                            #    edgecolor = '#1883b8'
                            #elif current_Rank_rank <= 0.08:
                            #    edgecolor = '#39a139'
                            #elif current_Rank_rank <= 0.10:
                            #    edgecolor = '#bc8f00'
                            #else:
                            #    edgecolor = '#d9d8d2'
                            #    alpha = 0.2
                            #    linewidth = 0.2

                
                            if plot_index == 0:

                                if current_Rank_rank > 0.1:
                                    patch = patches.PathPatch(
                                        path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                    )
                                    #legend_handles[j] = patch
                                    host.add_patch(patch)

                            else:
                                if current_Rank_rank <= 0.1:
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
                svg1.save(savefolder + 'StatisticalMetrics_{0}_{1}_vs_{2}_Scheme2.svg'.format(SR_method, Compared_groups.split('/')[0], Compared_groups.split('/')[1]))


            
                fig_legend = plt.figure(figsize=(2, 4))
 
                axes = plt.gca()
                edgecolor = ['#b0203f', '#e95b1b', '#a14ee0', '#1883b8', '#39a139', '#bc8f00', '#d9d8d2']
                labels = ['≤ 1%', '1% - 2%', '2% - 4%', '4% - 6%', '6% - 8%', '8% - 10%', '> 10%']
                for i in range(7):
                    axes.plot([10000, 20000], [10000, 20000], lw=2, color = edgecolor[i])
            

                axes.legend(labels = labels, title='Rank', title_fontsize=18, fontsize=16, 
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

                plt.savefig(savefolder + 'Legend_Rank.svg', dpi=600, format="svg", transparent=True) 
                plt.show()
                plt.close()


            columns = None
            if self.Use_Given_PValue_and_FC:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1-Score',
                            'Rank']
            else:
                columns = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                            ARI_or_Purity, 'Average pAUC', 'Average Accuracy', 'Average Recall', 'Average F1-Score',
                            'Rank']

            df_plot = df_screened[columns].copy(deep = True) 

            value_list = []
            for  i in df_plot['Missing Value Imputation'].values.tolist():
                value_list.append(Fill_NaN_methods.index(i))
            df_plot['Missing Value Imputation'] = value_list

            value_list = []
            for  i in df_plot['Normalization'].values.tolist():
                value_list.append(Normalization_methods.index(i))
            df_plot['Normalization'] = value_list

            value_list = []
            for  i in df_plot['Batch Correction'].values.tolist():
                value_list.append(BC_methods.index(i))
            df_plot['Batch Correction'] = value_list

            value_list = []
            for  i in df_plot['Statistical Test'].values.tolist():
                value_list.append(Difference_analysis_methods.index(i))
            df_plot['Statistical Test'] = value_list


            df_plot.astype('float')

            current_rank_list = df_plot['Rank'].values.tolist()

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
                    new_rank_list.append(int((4900/6)*rank_ratio/0.01))
                elif rank_ratio <= 0.02:
                    new_rank_list.append(int((4900/6)*1+(4900/6)*(rank_ratio-0.01)/0.01))
                elif rank_ratio <= 0.03:
                    new_rank_list.append(int((4900/6)*2+(4900/6)*(rank_ratio-0.02)/0.01))
                elif rank_ratio <= 0.04:
                    new_rank_list.append(int((4900/6)*3+(4900/6)*(rank_ratio-0.03)/0.01))
                elif rank_ratio <= 0.05:
                    new_rank_list.append(int((4900/6)*4+(4900/6)*(rank_ratio-0.04)/0.01))
                else:
                    new_rank_list.append(int((4900/6)*5+(4900/6)*(rank_ratio-0.05)/0.95))

            df_plot['Rank'] = new_rank_list

            max_rank_value = 4900
            min_rank_value = 1
            df_plot['Rank'] = max_rank_value - df_plot['Rank'] + min_rank_value

            ynames = None
            if self.Use_Given_PValue_and_FC:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                          ARI_or_Purity, 'pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
            else:
                ynames = ['Missing Value Imputation', 'Normalization', 'Batch Correction', 'Statistical Test',
                          ARI_or_Purity, 'pAUC', 'Accuracy', 'Recall', 'F1-Score', 'Rank']
            ys = np.array(df_plot.values.tolist())

            if self.Use_Given_PValue_and_FC:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 4900]], axis=0)
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], ys.max(axis=0)[9], 0]], axis=0)

                ys[:, -6:-1] = 1/(1.5 - (ys[:, -6:-1] ** 4))
            else:
                ys = np.append(ys, [[0, 0, 0, 0, ys.max(axis=0)[4], ys.max(axis=0)[5], ys.max(axis=0)[6], ys.max(axis=0)[7], ys.max(axis=0)[8], 4900]], axis=0)
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
                        ax.set_yticks(range(len(Fill_NaN_methods)))
                        ax.set_yticklabels(Fill_NaN_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')


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
                            ax.set_yticks(range(len(Normalization_methods)))
                            ax.set_yticklabels(['Unnorm', 'Median', 'Sum', 'QN', 'TRQN'], fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 2:
                            ax.set_yticks(range(len(BC_methods)))
                            ax.set_yticklabels(BC_methods, fontsize=10, rotation = 270, ha = 'right', va = 'top')
                            ax.tick_params(axis="y", pad = -6)

                            if plot_index == 0:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_visible(False)

                        elif i == 3:
                            ax.set_yticks(range(len(Difference_analysis_methods)))
                            ax.set_yticklabels(Difference_analysis_methods, fontsize=10, rotation = 250, ha = 'right', va = 'top')
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
                            
                            #ax.set_yticks([0, 700*1, 700*2, 700*3, 700*4, 700*5, 700*6, 4900])
                            #str_list = ['100%', '10%', '8%', '6%', '4%', '2%', '1%', '0%']

                            ax.set_yticks([0, (4900/6)*1, (4900/6)*2, (4900/6)*3, (4900/6)*4, (4900/6)*5, 4900])
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


                for tuceng in range(7):

                    for j in range(ys.shape[0] - 2):
                        verts = list(
                            zip(
                                [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                                np.repeat(zs[j, :], 3)[1:-1],
                            )
                        )
                        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
                        path = Path(verts, codes)


                        sorted_Rank_list = df_plot['Rank'].values.tolist()
                        sorted_Rank_list = sorted(sorted_Rank_list, reverse=True) 
                        current_Rank = df_plot['Rank'].values.tolist()[j]
                        current_Rank_rank = (sorted_Rank_list.index(current_Rank))/len(sorted_Rank_list)


                        alpha = 0.85
                        linewidth = 0.95

                        if current_Rank_rank < 3/1225:
                            if (tuceng != 6):
                                continue
                            edgecolor = '#b0203f'
                            linewidth = 2.0
                            alpha = 1.0
                        elif current_Rank_rank <= 0.01:
                            if (tuceng != 5):
                                continue
                            edgecolor = '#e95b1b'
                            alpha = 0.85
                            linewidth = 0.95
                        elif current_Rank_rank <= 0.02:
                            if (tuceng != 4):
                                continue
                            edgecolor = '#a14ee0'
                            alpha = 0.85
                            linewidth = 0.95
                        elif current_Rank_rank <= 0.03:
                            if (tuceng != 3):
                                continue
                            edgecolor = '#1883b8'
                            alpha = 0.85
                            linewidth = 0.95
                        elif current_Rank_rank <= 0.04:
                            if (tuceng != 2):
                                continue
                            edgecolor = '#39a139'
                            alpha = 0.85
                            linewidth = 0.95
                        elif current_Rank_rank <= 0.05:
                            if (tuceng != 1):
                                continue
                            edgecolor = '#bc8f00'
                            alpha = 0.85
                            linewidth = 0.95
                        else:
                            if (tuceng != 0):
                                continue
                            edgecolor = '#d9d8d2'
                            alpha = 0.2
                            linewidth = 0.2


                        #if current_Rank_rank <= 0.01:
                        #    edgecolor = '#b0203f'
                        #elif current_Rank_rank <= 0.02:
                        #    edgecolor = '#e95b1b'
                        #elif current_Rank_rank <= 0.04:
                        #    edgecolor = '#a14ee0'
                        #elif current_Rank_rank <= 0.06:
                        #    edgecolor = '#1883b8'
                        #elif current_Rank_rank <= 0.08:
                        #    edgecolor = '#39a139'
                        #elif current_Rank_rank <= 0.10:
                        #    edgecolor = '#bc8f00'
                        #else:
                        #    edgecolor = '#d9d8d2'
                        #    alpha = 0.2
                        #    linewidth = 0.2

                
                        if plot_index == 0:

                            if current_Rank_rank > 0.1:
                                patch = patches.PathPatch(
                                    path, facecolor="none", lw=linewidth, alpha=alpha, edgecolor=edgecolor , rasterized = False 
                                )
                                #legend_handles[j] = patch
                                host.add_patch(patch)

                        else:
                            if current_Rank_rank <= 0.1:
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
            svg1.save(savefolder + 'StatisticalMetrics_{0}_Average_Scheme2.svg'.format(SR_method))




    # Draw the TN, TP, FP, and FN histograms of the highest Rank under different sparsity reduction methods
    def Plot_SparsityReduction_DifferentialProteins(self, result_csv_path, 
                                 SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'], 
                                 Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute'],
                                 Normalization_methods = ['Unnormalized', 'Median', 'Sum', 'QN', 'TRQN'],
                                 BC_methods = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama'],
                                 Difference_analysis_methods = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2'],
                                 Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                 Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                                 savefig = True, savefolder = ''):

        
        df_result = pd.read_csv(result_csv_path) 

        colors = ['#8bd2ca', '#fe708a', '#67b7fd', '#d29fea']

        # Draw for each comparison - by overall Rank
        for Compared_groups in Compared_groups_label:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))
            
            # Data
            TN_list = []
            TP_list = []
            FP_list = []
            FN_list = []
            No_list = []

            for SR in SR_methods:
                df_screened = df_result[df_result['Sparsity Reduction'] == SR]
                # Sort in ascending order by Rank
                df_screened2 = df_screened.sort_values('Rank', ascending=True)
                TN_list.append(df_screened2[Compared_groups + ' TN'].values.tolist()[0])
                TP_list.append(df_screened2[Compared_groups + ' TP'].values.tolist()[0])
                FP_list.append(df_screened2[Compared_groups + ' FP'].values.tolist()[0])
                FN_list.append(df_screened2[Compared_groups + ' FN'].values.tolist()[0])

                No_list.append(df_screened2['No'].values.tolist()[0])


            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if height > 450:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.5*height, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                    else:
                        if height > -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100 + 0.05*height, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.05*height, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 

            def autolabel2(rects, bottom):
                count = 0
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if bottom[count] < 300:
                            bottom[count] = 300
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    else:
                        if bottom[count] > -300:
                            bottom[count] = -300
                        if height > -200:
                            bottom[count] += -240
                        if height < -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., bottom[count], '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial")
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
            autolabel(bar2) 

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

            bar4 = plt.bar(bar_x_FN, bar_y_FN, bottom = bar_y_FP, width = 0.75, color = colors[3]) 
            autolabel2(bar4, bottom = bar_y_FP) 


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

            y_max = math.ceil(y_max/500)*500 + 500
            y_min = math.ceil(-y_min/500)*500 + 500
            plt.ylim(-y_min, y_max)
            plt.yticks(list(range(0, y_max, 1000)))

            plt.xlim(0.35, 0.5+len(SR_methods))
            plt.xticks(list(range(1, 1+len(SR_methods))))
            xticklabels = []
            for SR in SR_methods:
                xticklabels.append('[{0}] '.format(No_list[SR_methods.index(SR)]) + SR)

            axes.set_xticklabels(xticklabels, fontsize=14, rotation = 90, ha = 'center', va = 'top')
            plt.ylabel('# Proteins', y=0.5, fontsize=16) 
            plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

            plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98, wspace=0.05, hspace=0.1) 

            plt.savefig(savefolder + 'SparsityReduction_DifferentialProteins_{0}_vs_{1}.svg'.format(Compared_groups.split('/')[0], Compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)
            plt.show()
            plt.close()


        # Draw for each comparison - by Rank of each comparison
        for Compared_groups in Compared_groups_label:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))
            
            # Data
            TN_list = []
            TP_list = []
            FP_list = []
            FN_list = []
            No_list = []

            for SR in SR_methods:
                df_screened = df_result[df_result['Sparsity Reduction'] == SR]
                # Sort in ascending order by Rank
                df_screened2 = df_screened.sort_values(Compared_groups + ' Rank', ascending=True)
                TN_list.append(df_screened2[Compared_groups + ' TN'].values.tolist()[0])
                TP_list.append(df_screened2[Compared_groups + ' TP'].values.tolist()[0])
                FP_list.append(df_screened2[Compared_groups + ' FP'].values.tolist()[0])
                FN_list.append(df_screened2[Compared_groups + ' FN'].values.tolist()[0])

                No_list.append(df_screened2['No'].values.tolist()[0])


            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if height > 450:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.5*height, '%s' % int(height), ha='center', va='center', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height, '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial") 
                    else:
                        if height > -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., -100 + 0.05*height, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 0.05*height, '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 

            def autolabel2(rects, bottom):
                count = 0
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0:
                        if bottom[count] < 300:
                            bottom[count] = 300
                        plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(height), ha='center', va='bottom', rotation=90, size=11.5, family="Arial")
                    else:
                        if bottom[count] > -300:
                            bottom[count] = -300
                        if height > -200:
                            bottom[count] += -240
                        if height < -200:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., 1.05*height + bottom[count], '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial") 
                        else:
                            plt.text(rect.get_x()+1.1*rect.get_width()/2., bottom[count], '%s' % int(-height), ha='center', va='top', rotation=90, size=11.5, family="Arial")
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
            autolabel(bar2) 

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

            bar4 = plt.bar(bar_x_FN, bar_y_FN, bottom = bar_y_FP, width = 0.75, color = colors[3]) 
            autolabel2(bar4, bottom = bar_y_FP) 


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

            y_max = math.ceil(y_max/500)*500 + 500
            y_min = math.ceil(-y_min/500)*500 + 500
            plt.ylim(-y_min, y_max)
            plt.yticks(list(range(0, y_max, 1000)))

            plt.xlim(0.35, 0.5+len(SR_methods))
            plt.xticks(list(range(1, 1+len(SR_methods))))
            xticklabels = []
            for SR in SR_methods:
                xticklabels.append('[{0}] '.format(No_list[SR_methods.index(SR)]) + SR)

            axes.set_xticklabels(xticklabels, fontsize=14, rotation = 90, ha = 'center', va = 'top')
            plt.ylabel('# Proteins', y=0.5, fontsize=16) 
            plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)


            plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98, wspace=0.05, hspace=0.1) 

            plt.savefig(savefolder + 'SparsityReduction_DifferentialProteins_{0}_vs_{1}_Using_Comparison_Rank.svg'.format(Compared_groups.split('/')[0], Compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)
            plt.show()
            plt.close()


    # Draw a Spearman correlation coefficient plot
    def Plot_Spearmanr_Result(self, result_csv_path, 
                              Indicator_type_list = ['Rank', 'ARI', 'pAUC', 'F1-Score'], 
                              SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'],
                              Compared_groups_label = ['S4/S2', 'S5/S1'],
                              ShowValue = False,
                              savefig = True, savefolder = ''):


        df_result = pd.read_csv(result_csv_path, index_col=0) 

        # color
        color_minus_one = (130/255, 30/255, 35/255)
        color_zero = (1, 1, 1) 
        color_one = (5/255, 49/255, 99/255) 


        for Indicator in Indicator_type_list:

            scores_list = []

            # Data
            list_Comparision = []
            list_SR = []
            list_Comparision_column = []
            list_SR_column = []
            list_Correlation = []

            for Compared_groups in Compared_groups_label:
                for SR in SR_methods:
                    scores = []
                    # Y Data
                    Y = []
                    df_screened = df_result[df_result['Sparsity Reduction'] == SR]
                    # Sort by 'No' in ascending order
                    df_screened = df_screened.sort_values('No', ascending=True)
                    if (Indicator == 'ARI') | (Indicator == 'Purity Score'):
                        Y = df_screened[Indicator].values.tolist()
                    else:
                        Y = df_screened[Compared_groups + ' ' + Indicator].values.tolist()

                    # X Data
                    X = []
                    X_num = (Compared_groups_label.index(Compared_groups))*len(SR_methods) + SR_methods.index(SR) + 1
                
                    count = 1
                    for i in Compared_groups_label:
                        for j in SR_methods:
                            if (count <= X_num):
                                df_screened2 = df_result[df_result['Sparsity Reduction'] == j]
                                # Sort by 'No' in ascending order
                                df_screened2 = df_screened2.sort_values('No', ascending=True)
                                if (Indicator == 'ARI') | (Indicator == 'Purity Score'):
                                    x = df_screened2[Indicator].values.tolist()
                                    X.append(x)
                                else:
                                    x = df_screened2[i + ' ' + Indicator].values.tolist()
                                    X.append(x)


                                if (count < X_num):
                                    list_Comparision.append(Compared_groups)
                                    list_SR.append(SR)

                                    list_Comparision_column.append(i)
                                    list_SR_column.append(j)

                            count += 1

                    # Calculate the Spearman correlation coefficient
                    count = 1
                    for x in X:
                        pccs = spearmanr(np.array(x), np.array(Y))
                        scores.append(pccs[0])

                        if (count != len(X)):
                            list_Correlation.append(pccs[0])

                        count += 1


                    scores_list.append(scores)
                        

            dict_data = {'Comparision 1':list_Comparision,
                         'Sparsity Reduction 1':list_SR,
                         'Comparision 2':list_Comparision_column,
                         'Sparsity Reduction 2':list_SR_column,
                         'Correlation':list_Correlation}

            df_data = pd.DataFrame(dict_data)
            if (Indicator == 'Purity Score'):
                df_data.to_csv(savefolder + 'Correlation_{0}_Comparison_SR.csv'.format('PurityScore'), index=False)
            else:
                df_data.to_csv(savefolder + 'Correlation_{0}_Comparison_SR.csv'.format(Indicator), index=False)


            fig, ax = plt.subplots(len(Compared_groups_label), len(Compared_groups_label), figsize=(4.5,4.5))


            count = 0
            for i in range(len(Compared_groups_label)):
                for j in range(len(Compared_groups_label)):
                
                    if (j <= i):
                        plt.subplot(len(Compared_groups_label), len(Compared_groups_label), i*len(Compared_groups_label)+j+1)

                        for row in range(len(SR_methods)):
                            for column in range(len(SR_methods)):

                                if ((j*len(SR_methods)+column+1) < len(scores_list[i*len(SR_methods)+row])):

                                    value = scores_list[i*len(SR_methods)+row][j*len(SR_methods)+column]
                                    position_x = column
                                    position_y = len(SR_methods) - 1 - row

                                    if value >=0:
                                        color = (color_zero[0] + (value-0)*(color_one[0]-color_zero[0])/1,
                                                 color_zero[1] + (value-0)*(color_one[1]-color_zero[1])/1,
                                                 color_zero[2] + (value-0)*(color_one[2]-color_zero[2])/1)
                                    else:
                                        color = (color_minus_one[0] + (value+1)*(color_zero[0]-color_minus_one[0])/1,
                                                 color_minus_one[1] + (value+1)*(color_zero[1]-color_minus_one[1])/1,
                                                 color_minus_one[2] + (value+1)*(color_zero[2]-color_minus_one[2])/1)
                                    rect = patches.Rectangle((position_x, position_y), width=1, height=1,
                                                     color=color, fill=True)


                                    if ShowValue:
                                        plt.text(position_x+0.5, position_y+0.5, '{:.2f}'.format(value), fontsize=12, color='black', ha='center', va='center')

                                    plt.xlim(0, len(SR_methods))
                                    plt.ylim(0, len(SR_methods)) 

                                    axes = plt.gca()
                                    axes.add_patch(rect)

                                    plt.xticks([])
                                    plt.yticks([])
                                    axes.spines['top'].set_visible(False) 
                                    axes.spines['right'].set_visible(False)
                                    axes.spines['bottom'].set_visible(False) 
                                    axes.spines['left'].set_visible(False)


                        if (i==0) & (j==0):

                            plt.ylabel(Compared_groups_label[i], y=((len(SR_methods)-1)/2)/(len(SR_methods)), fontsize=16)
                            plt.ylim(0, len(SR_methods))

                            plt.yticks(list(np.array(list(range(0, len(SR_methods)-1)))+0.5), (SR_methods[1:])[::-1])
                            plt.tick_params(labelsize=14) 
                            axes = plt.gca()
                            axes.yaxis.set_tick_params(width=0) 

                        
                        if (i==(len(Compared_groups_label)-1)) & (j==(len(Compared_groups_label)-1)):

                            plt.xlabel(Compared_groups_label[j], x=((len(SR_methods)-1)/2)/(len(SR_methods)), fontsize=16)
                            plt.xlim(0, len(SR_methods))

                            plt.xticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods[:-1], rotation=90)
                            plt.tick_params(labelsize=14) 
                            axes = plt.gca()
                            axes.xaxis.set_tick_params(width=0) 

                        if (i==(len(Compared_groups_label)-1)) & (j!=(len(Compared_groups_label)-1)):

                            plt.xlabel(Compared_groups_label[j], fontsize=16)
                            plt.xlim(0, len(SR_methods))

                            plt.xticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods, rotation=90)
                            plt.tick_params(labelsize=14) 
                            axes = plt.gca()
                            axes.xaxis.set_tick_params(width=0) 

                        if (i!=0) & (j==0):

                            plt.ylabel(Compared_groups_label[i], fontsize=16)
                            plt.ylim(0, len(SR_methods))

                            plt.yticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods[::-1])
                            plt.tick_params(labelsize=14) 
                            axes = plt.gca()
                            axes.yaxis.set_tick_params(width=0) 

                    else:
                        plt.subplot(len(Compared_groups_label), len(Compared_groups_label), i*len(Compared_groups_label)+j+1)
                        
                        plt.xticks([])
                        plt.yticks([])
                        axes = plt.gca()
                        axes.spines['top'].set_visible(False) 
                        axes.spines['right'].set_visible(False)
                        axes.spines['bottom'].set_visible(False) 
                        axes.spines['left'].set_visible(False)


                plt.subplots_adjust(left=0.21, right=0.99, bottom=0.21, top=0.96, wspace=0.05, hspace=0.05)

            if savefig:
                if (Indicator == 'Purity Score'):
                    plt.savefig(savefolder + 'Correlation_{0}_Comparison_SR.svg'.format('PurityScore'), dpi=600, format="svg", transparent=True)
                else:
                    plt.savefig(savefolder + 'Correlation_{0}_Comparison_SR.svg'.format(Indicator), dpi=600, format="svg", transparent=True)

            plt.show()
            plt.close()


        # Legend_Correlation
        fig, ax = plt.subplots(figsize=(2.5, 5))
        list_of_values = np.arange(-1, 1.01, 0.01)
        axes = plt.gca()
        for value in list_of_values:

            color = (0,0,0)
            if value >=0:
                color = (color_zero[0] + (value-0)*(color_one[0]-color_zero[0])/1,
                            color_zero[1] + (value-0)*(color_one[1]-color_zero[1])/1,
                            color_zero[2] + (value-0)*(color_one[2]-color_zero[2])/1)
            else:
                color = (color_minus_one[0] + (value+1)*(color_zero[0]-color_minus_one[0])/1,
                            color_minus_one[1] + (value+1)*(color_zero[1]-color_minus_one[1])/1,
                            color_minus_one[2] + (value+1)*(color_zero[2]-color_minus_one[2])/1)

            rect = patches.Rectangle((-1, value), width=1, height=0.01,
                                     color=color, fill=True)

        
            axes.add_patch(rect)


        plt.xlim(-2, 1.5)
        plt.ylim(-1, 1) 

        plt.xticks([])
        axes.yaxis.set_ticks_position('right')
        axes.spines['right'].set_position(('data', 0)) 

        plt.yticks(np.arange(-1, 1.25, 0.25), ['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
        plt.tick_params(labelsize=14) 

        yticks = axes.get_yticklabels()
 
        for label in yticks:
            label.set_va('center') 

    
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False) 
        axes.spines['left'].set_visible(False)

        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        if savefig:
            plt.savefig(savefolder + 'Legend_Correlation.svg', dpi=600, format="svg", transparent=True)

        plt.show()
        plt.close()



    # Plot the Spearman correlation coefficient result of three softwares
    def Plot_Spearmanr_Result_of_Three_Softwares(self, DatasetPath = [], DatasetName = ['DIA-NN', 'Spectronaut', 'PEAKS'], 
                              Indicator_type_list = ['Rank', 'ARI', 'pAUC', 'F1-Score'], 
                              SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90'],
                              Compared_groups_label = ['S4/S2', 'S5/S1'],
                              ShowValue = False,
                              savefig = True, savefolder = ''):

        df_dataset1 = pd.read_csv(DatasetPath[0], index_col=0) 
        df_dataset2 = pd.read_csv(DatasetPath[1], index_col=0) 
        df_dataset3 = pd.read_csv(DatasetPath[2], index_col=0) 
        df_datasets = [df_dataset1, df_dataset2, df_dataset3]

    
        # color
        color_minus_one = (130/255, 30/255, 35/255)
        color_zero = (1, 1, 1) 
        color_one = (5/255, 49/255, 99/255) 

        for Indicator in Indicator_type_list:
            for Compared_groups in Compared_groups_label:

                scores_list = []

                list_Dataset1 = []
                list_SR1 = []
                list_Dataset2 = []
                list_SR2 = []
                list_Correlation = []

                for Dataset in DatasetName:
                    for SR in SR_methods:
                        scores = []

                        # Y data
                        Y = []
                        df_result = df_datasets[DatasetName.index(Dataset)]
                        df_screened = df_result[df_result['Sparsity Reduction'] == SR]

                        df_screened = df_screened.sort_values('No', ascending=True)
                        if (Indicator == 'ARI') | (Indicator == 'Purity Score'):
                            Y = df_screened[Indicator].values.tolist()
                        else:
                            Y = df_screened[Compared_groups + ' ' + Indicator].values.tolist()

                        # X data
                        X = []
                        X_num = (DatasetName.index(Dataset))*len(SR_methods) + SR_methods.index(SR) + 1

                        count = 1
                        for i in DatasetName:
                            for j in SR_methods:
                                if (count <= X_num):
                                    df_result = df_datasets[DatasetName.index(i)]
                                    df_screened2 = df_result[df_result['Sparsity Reduction'] == j]

                                    df_screened2 = df_screened2.sort_values('No', ascending=True)
                                    if (Indicator == 'ARI') | (Indicator == 'Purity Score'):
                                        x = df_screened2[Indicator].values.tolist()
                                        X.append(x)
                                    else:
                                        x = df_screened2[Compared_groups + ' ' + Indicator].values.tolist()
                                        X.append(x)

                                    if (count < X_num):
                                        list_Dataset1.append(Dataset)
                                        list_SR1.append(SR)

                                        list_Dataset2.append(i)
                                        list_SR2.append(j)

                                count += 1

                        # Calculate the Spearman correlation coefficient
                        count = 1
                        for x in X:
                            pccs = spearmanr(np.array(x), np.array(Y))
                            scores.append(pccs[0])

                            if (count != len(X)):
                                list_Correlation.append(pccs[0])

                            count += 1

                        scores_list.append(scores)


                dict_data = {'Dataset 1':list_Dataset1,
                             'Sparsity Reduction 1':list_SR1,
                             'Dataset 2':list_Dataset2,
                             'Sparsity Reduction 2':list_SR2,
                             'Correlation':list_Correlation}

                df_data = pd.DataFrame(dict_data)
                if (Indicator == 'Purity Score'):
                    df_data.to_csv(savefolder + 'Correlation_{0}_Dataset_SR_{1}_vs_{2}.csv'.format('PurityScore', Compared_groups.split('/')[0], Compared_groups.split('/')[1]), index=False)
                else:
                    df_data.to_csv(savefolder + 'Correlation_{0}_Dataset_SR_{1}_vs_{2}.csv'.format(Indicator, Compared_groups.split('/')[0], Compared_groups.split('/')[1]), index=False)


                fig, ax = plt.subplots(len(DatasetName), len(DatasetName), figsize=(4.5,4.5))


                count = 0

                for i in range(len(DatasetName)):
                    for j in range(len(DatasetName)):
                
                        if (j <= i):
                            plt.subplot(len(DatasetName), len(DatasetName), i*len(DatasetName)+j+1)

                            for row in range(len(SR_methods)):
                                for column in range(len(SR_methods)):

                                    if ((j*len(SR_methods)+column+1) < len(scores_list[i*len(SR_methods)+row])):

                                        value = scores_list[i*len(SR_methods)+row][j*len(SR_methods)+column]
                                        position_x = column
                                        position_y = len(SR_methods) - 1 - row

                                        if value >=0:
                                            color = (color_zero[0] + (value-0)*(color_one[0]-color_zero[0])/1,
                                                     color_zero[1] + (value-0)*(color_one[1]-color_zero[1])/1,
                                                     color_zero[2] + (value-0)*(color_one[2]-color_zero[2])/1)
                                        else:
                                            color = (color_minus_one[0] + (value+1)*(color_zero[0]-color_minus_one[0])/1,
                                                     color_minus_one[1] + (value+1)*(color_zero[1]-color_minus_one[1])/1,
                                                     color_minus_one[2] + (value+1)*(color_zero[2]-color_minus_one[2])/1)
                                        rect = patches.Rectangle((position_x, position_y), width=1, height=1,
                                                         color=color, fill=True)


                                        if ShowValue:
                                            plt.text(position_x+0.5, position_y+0.5, '{:.2f}'.format(value), fontsize=12, color='black', ha='center', va='center')

                                        plt.xlim(0, len(SR_methods))
                                        plt.ylim(0, len(SR_methods)) 

                                        axes = plt.gca()
                                        axes.add_patch(rect)

                                        plt.xticks([])
                                        plt.yticks([])
                                        axes.spines['top'].set_visible(False) 
                                        axes.spines['right'].set_visible(False)
                                        axes.spines['bottom'].set_visible(False) 
                                        axes.spines['left'].set_visible(False)


                            if (i==0) & (j==0):

                                plt.ylabel(DatasetName[i], y=((len(SR_methods)-1)/2)/(len(SR_methods)), fontsize=15.5)
                                plt.ylim(0, len(SR_methods))

                                plt.yticks(list(np.array(list(range(0, len(SR_methods)-1)))+0.5), (SR_methods[1:])[::-1])
                                plt.tick_params(labelsize=14) 
                                axes = plt.gca()
                                axes.yaxis.set_tick_params(width=0) 


                            if (i==(len(DatasetName)-1)) & (j==(len(DatasetName)-1)):

                                plt.xlabel(DatasetName[j], x=((len(SR_methods)-1)/2)/(len(SR_methods)), fontsize=15.5)
                                plt.xlim(0, len(SR_methods))

                                plt.xticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods[:-1], rotation=90)
                                plt.tick_params(labelsize=14) 
                                axes = plt.gca()
                                axes.xaxis.set_tick_params(width=0) 

                            if (i==(len(DatasetName)-1)) & (j!=(len(DatasetName)-1)):

                                plt.xlabel(DatasetName[j], fontsize=15.5)
                                plt.xlim(0, len(SR_methods))

                                plt.xticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods, rotation=90)
                                plt.tick_params(labelsize=14) 
                                axes = plt.gca()
                                axes.xaxis.set_tick_params(width=0)

                            if (i!=0) & (j==0):

                                plt.ylabel(DatasetName[i], fontsize=15.5)
                                plt.ylim(0, len(SR_methods))

                                plt.yticks(list(np.array(list(range(0, len(SR_methods))))+0.5), SR_methods[::-1])
                                plt.tick_params(labelsize=14) 
                                axes = plt.gca()
                                axes.yaxis.set_tick_params(width=0) 

                        else:
                            plt.subplot(len(DatasetName), len(DatasetName), i*len(DatasetName)+j+1)

                            plt.xticks([])
                            plt.yticks([])
                            axes = plt.gca()
                            axes.spines['top'].set_visible(False) 
                            axes.spines['right'].set_visible(False)
                            axes.spines['bottom'].set_visible(False) 
                            axes.spines['left'].set_visible(False)

                
                plt.subplots_adjust(left=0.21, right=0.99, bottom=0.21, top=0.96, wspace=0.05, hspace=0.05)
                

                if savefig:
                    if (Indicator == 'Purity Score'):
                        plt.savefig(savefolder + 'Correlation_{0}_Dataset_SR_{1}_vs_{2}.svg'.format('PurityScore', Compared_groups.split('/')[0], Compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)
                    else:
                        plt.savefig(savefolder + 'Correlation_{0}_Dataset_SR_{1}_vs_{2}.svg'.format(Indicator, Compared_groups.split('/')[0], Compared_groups.split('/')[1]), dpi=600, format="svg", transparent=True)

                plt.show()
                plt.close()


        # Legend_Correlation
        fig, ax = plt.subplots(figsize=(2.5, 5))
        list_of_values = np.arange(-1, 1.01, 0.01)
        axes = plt.gca()
        for value in list_of_values:


            color = (0,0,0)
            if value >=0:
                color = (color_zero[0] + (value-0)*(color_one[0]-color_zero[0])/1,
                            color_zero[1] + (value-0)*(color_one[1]-color_zero[1])/1,
                            color_zero[2] + (value-0)*(color_one[2]-color_zero[2])/1)
            else:
                color = (color_minus_one[0] + (value+1)*(color_zero[0]-color_minus_one[0])/1,
                            color_minus_one[1] + (value+1)*(color_zero[1]-color_minus_one[1])/1,
                            color_minus_one[2] + (value+1)*(color_zero[2]-color_minus_one[2])/1)

            rect = patches.Rectangle((-1, value), width=1, height=0.01,
                                     color=color, fill=True)

            axes.add_patch(rect)


        plt.xlim(-2, 1.5)
        plt.ylim(-1, 1) 

        plt.xticks([])
        axes.yaxis.set_ticks_position('right')
        axes.spines['right'].set_position(('data', 0)) 

        plt.yticks(np.arange(-1, 1.25, 0.25), ['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
        plt.tick_params(labelsize=14) 

        yticks = axes.get_yticklabels()
 
        for label in yticks:
            label.set_va('center') 

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False) 
        axes.spines['left'].set_visible(False)

        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        if savefig:
            plt.savefig(savefolder + 'Legend_Correlation.svg', dpi=600, format="svg", transparent=True)

        plt.show()
        plt.close()





    # Draw ROC curve, PR curve and volcano plot
    def plot_ROC_and_Volcano(self, df, label_true, df_PG, species_PG_names, plot_index, method, up_down_label,
                              compared_groups_num = 3, 
                              compared_groups_label = ['S_2E1Y/S_2Y1E', 'C_2E1Y/C_2Y1E', '2E1Y/2Y1E'],
                              methods_used = 'title_methods',
                              savefig = True,
                              savefolder = './'):

        # Draw ROC curve
        y = label_true

        # Replace the values ​​<= 0 in df['padjust'] with finfo(np.float64).eps
        df['padjust'] = df['padjust'].clip(lower = np.finfo(np.float64).eps)
        scores = -np.log10(df['padjust'].values)
        
        y_single = []
        scores_single = []

        for i in range(len(scores)):
            speices_i = df['Species'].values.tolist()[i]
            for j in self.species:
                if speices_i == j + ' ':
                    y_single.append(y[i])

                    true_change = up_down_label[self.species.index(j)] 
                    log2fc_i = df['log2fc'].values.tolist()[i]
                    if (true_change == 1) & (log2fc_i < 0):
                        scores_single.append(-1)
                    elif (true_change == 2) & (log2fc_i > 0):
                        scores_single.append(-1)
                    else:
                        scores_single.append(scores[i])

        fpr, tpr, thresholds = metrics.roc_curve(y_single, scores_single, pos_label=1)
        if thresholds[-1] < 0:
            thresholds = thresholds[:-1]
            fpr = fpr[:-1]
            tpr = tpr[:-1]

        thresholds = np.append(thresholds[1:], -1)


        # When calculating pAUC, take FPR<=0.1
        if (len(fpr[fpr<=0.1]) >= 2):
            pauc = metrics.auc(fpr[fpr<=0.1], tpr[0:len(fpr[fpr<=0.1])])
        else:
            pauc = 0


        # Draw ROC Curve
        fig_ROC = plt.figure(figsize=(5,5))

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',lw=lw) 
        plt.axvline(0.1, ymax = tpr[find_nearest(fpr, 0.1)]/1.02, linewidth = lw, color='#FF0000') # Draw a vertical dashed line, FPR = 0.1
        plt.fill_between(fpr, tpr, 0, where = (fpr<0.1) & (fpr>0), color = '#FFE4E1')  # Fill color of pAUC region


        if (len(fpr[fpr<=0.1]) >= 2):
            # Mark the optimal value under the premise of FPR<=0.1
            optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr[0:len(fpr[fpr<=0.1])], FPR=fpr[fpr<=0.1], threshold=thresholds[0:len(fpr[fpr<=0.1])])
    
            plt.plot(optimal_point[0], optimal_point[1], marker='o', markersize=8, color='black')

            if optimal_point[1] < 0.13:
                optimal_point[1] = 0.14

            plt.text(optimal_point[0] + 0.03, optimal_point[1] - 0.13, 'Thr: {:.3f} \nFPR: {:.2f} \nTPR: {:.2f}'.format(optimal_th, optimal_point[0], optimal_point[1]), fontsize=14) 
        else:
            optimal_th = thresholds[0]
            optimal_point = [0, 0]
            plt.plot(optimal_point[0], optimal_point[1], marker='o', markersize=8, color='black')
            plt.text(optimal_point[0] + 0.03, optimal_point[1] + 0.01, 'Thr: {:.3f} \nFPR: {:.2f} \nTPR: {:.2f}'.format(optimal_th, optimal_point[0], optimal_point[1]), fontsize=14) 
        
    
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate', fontsize=16) 
        plt.ylabel('True Positive Rate', fontsize=16) 
        plt.tick_params(axis='x', labelsize=14) 
        plt.tick_params(axis='y', labelsize=14)

        plt.text(0.8, 0.08, 'pAUC = {0}'.format(round(pauc, 3)), ha = 'center', va = 'center', fontsize=14)

        plt.tick_params(labelsize=14)
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        axes = plt.gca()
        axes.spines['bottom'].set_linewidth(2) 
        axes.spines['left'].set_linewidth(2) 
        axes.spines['top'].set_linewidth(2) 
        axes.spines['right'].set_linewidth(2)  

        plt.subplots_adjust(left=0.14, right=0.97, bottom=0.115, top=0.99, wspace=0.05, hspace=0.1)

        # Save Threshold, FPR, TPR to csv
        if savefig:
            data = {'Threshold': thresholds.tolist(),
                    'FPR': fpr.tolist(),
                    'TPR': tpr.tolist()}
            df_save = pd.DataFrame(data)
            # Delete the first row of invalid data
            df_save = df_save.drop(df_save.index[0])
            df_save.to_csv(savefolder + 'ROC_{0}_{1}_vs_{2}.csv'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), index=False)

        if savefig:
            plt.savefig(savefolder + 'ROC_{0}_{1}_vs_{2}.svg'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()




        # Plotting the PR curve
        labels = label_true
        scores = df['log2fc'].abs()

        p_cutoff_list = [0.1, 0.05, 0.01, 0.001]
        line_color_list = ['#b0203f', '#e95b1b', '#a14ee0', '#1883b8']

        labels_single = [[],[],[],[]]
        scores_single = [[],[],[],[]]

        precision_list = [[],[],[],[]] 
        recall_list = [[],[],[],[]] 
        threshold_list = [[],[],[],[]]

        fig_precision_recall = plt.figure(figsize=(5,5))

        plot_line_1_4 = []

        for p_cutoff in p_cutoff_list:
            for i in range(len(scores)):
                speices_i = df['Species'].values.tolist()[i]
                for j in self.species:
                    if speices_i == j + ' ':
                        labels_single[p_cutoff_list.index(p_cutoff)].append(labels[i])

                        true_change = up_down_label[self.species.index(j)]  # 1 Up 2 Down 3 No Sig.
                        log2fc_i = df['log2fc'].values.tolist()[i]
                        if (true_change == 1) & (log2fc_i < 0):
                            scores_single[p_cutoff_list.index(p_cutoff)].append(-1)
                        elif (true_change == 2) & (log2fc_i > 0):
                            scores_single[p_cutoff_list.index(p_cutoff)].append(-1)
                        elif (df['padjust'].values.tolist()[i] >= p_cutoff):
                            scores_single[p_cutoff_list.index(p_cutoff)].append(-1)
                        else:
                            scores_single[p_cutoff_list.index(p_cutoff)].append(scores[i])

            # For the case where scores_single has missing values
            y_test = labels_single[p_cutoff_list.index(p_cutoff)]
            y_score = np.array(scores_single[p_cutoff_list.index(p_cutoff)])
            new_y_test = []
            new_y_score = []
            count = 0
            for item in y_score:
                is_nan = np.isnan(item)
                if is_nan:
                    pass
                else:
                    new_y_test.append(y_test[count])
                    new_y_score.append(y_score[count])
                count += 1
            precision, recall, thresholds = precision_recall_curve(new_y_test, new_y_score)


            #precision, recall, thresholds = precision_recall_curve(labels_single[p_cutoff_list.index(p_cutoff)], scores_single[p_cutoff_list.index(p_cutoff)])
            
            if thresholds[0] < 0:
                thresholds = thresholds[1:]
                precision = precision[1:]
                recall = recall[1:]

            if (len(precision) >= 2):
                thresholds = np.insert(thresholds, 0, -1)
                precision[-1] = precision[-2]

                precision_list[p_cutoff_list.index(p_cutoff)] = precision.tolist()
                recall_list[p_cutoff_list.index(p_cutoff)] = recall.tolist()
                threshold_list[p_cutoff_list.index(p_cutoff)] = thresholds.tolist()

                plt.plot(recall, precision, color=line_color_list[p_cutoff_list.index(p_cutoff)], lw=2)

                plot_line_1_4.append(True)
            else:
                thresholds = [0]
                precision = [0]
                recall = [0]

                precision_list[p_cutoff_list.index(p_cutoff)] = precision
                recall_list[p_cutoff_list.index(p_cutoff)] = recall
                threshold_list[p_cutoff_list.index(p_cutoff)] = thresholds

                plot_line_1_4.append(False)


        plt.xlim([-0.03, 1.03])
        plt.ylim([-0.03, 1.03])
        plt.xlabel('Recall', fontsize=16) 

        plt.ylabel('Precision', fontsize=16) 
        plt.tick_params(axis='x', labelsize=14) 
        plt.tick_params(axis='y', labelsize=14)

        plt.tick_params(labelsize=14) 
        plt.tick_params(axis='x', width=2)
        plt.tick_params(axis='y', width=2)

        axes = plt.gca()
        axes.spines['bottom'].set_linewidth(2) 
        axes.spines['left'].set_linewidth(2) 
        axes.spines['top'].set_linewidth(2) 
        axes.spines['right'].set_linewidth(2) 

        # Draw a horizontal dashed line
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
        axes.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '0.95', '1.0'])
        plt.axhline(0.95, color='grey', linestyle='--')

        # Find and mark the optimal threshold point
        # Calculate the index value of the intersection of the four polylines and y=0.95
        idx_0 = np.argwhere(np.diff(np.sign(np.array(precision_list[0]) - np.array([0.95]*len(precision_list[0]))))).flatten() +1
        idx_1 = np.argwhere(np.diff(np.sign(np.array(precision_list[1]) - np.array([0.95]*len(precision_list[1]))))).flatten() +1
        idx_2 = np.argwhere(np.diff(np.sign(np.array(precision_list[2]) - np.array([0.95]*len(precision_list[2]))))).flatten() +1
        idx_3 = np.argwhere(np.diff(np.sign(np.array(precision_list[3]) - np.array([0.95]*len(precision_list[3]))))).flatten() +1


        # If the points of the 4 curves are all <2
        if (plot_line_1_4 == [False, False, False, False]):
            x_best = 0
            y_best = 0
            Thr_log2fc = 0

            optimal_pvalue = 0.1
            optimal_th = -np.log10(0.1)

        # If the accuracy of the four lines is greater than or equal to 0.95, and the curve intersects with 0.95
        elif ((len(idx_0) + len(idx_1) + len(idx_2) + len(idx_3)) > 0) & ((max(precision_list[0]) >= 0.95) | (max(precision_list[1]) >= 0.95) | (max(precision_list[2]) >= 0.95) | (max(precision_list[3]) >= 0.95)):
            max_index = [0,0,0,0]
            if len(idx_0) > 0:
                max_index[0] = recall_list[0][idx_0[0]]
            if len(idx_1) > 0:
                max_index[1] = recall_list[1][idx_1[0]]
            if len(idx_2) > 0:
                max_index[2] = recall_list[2][idx_2[0]]
            if len(idx_3) > 0:
                max_index[3] = recall_list[3][idx_3[0]]

            x_best = max(max_index)
            index = max_index.index(max(max_index)) 

            if index == 0:
                loc_index = idx_0[0]

                # Determine whether there is a point on the line segment with precision>=0.95 and recall greater than the current point
                current_recall = recall_list[0][loc_index]
                other_point_list = (np.array(recall_list[0]) > current_recall) & (np.array(precision_list[0]) >= 0.95)
                if (True in other_point_list):
                    loc_index = np.where(other_point_list == True)[0][0]
                    x_best = recall_list[0][loc_index]


            if index == 1:
                loc_index = idx_1[0]

                # Determine whether there is a point on the line segment with precision>=0.95 and recall greater than the current point
                current_recall = recall_list[1][loc_index]
                other_point_list = (np.array(recall_list[1]) > current_recall) & (np.array(precision_list[1]) >= 0.95)
                if (True in other_point_list):
                    loc_index = np.where(other_point_list == True)[0][0]
                    x_best = recall_list[1][loc_index]

            if index == 2:
                loc_index = idx_2[0]

                # Determine whether there is a point on the line segment with precision>=0.95 and recall greater than the current point
                current_recall = recall_list[2][loc_index]
                other_point_list = (np.array(recall_list[2]) > current_recall) & (np.array(precision_list[2]) >= 0.95)
                if (True in other_point_list):
                    loc_index = np.where(other_point_list == True)[0][0]
                    x_best = recall_list[2][loc_index]

            if index == 3:
                loc_index = idx_3[0]

                # Determine whether there is a point on the line segment with precision>=0.95 and recall greater than the current point
                current_recall = recall_list[3][loc_index]
                other_point_list = (np.array(recall_list[3]) > current_recall) & (np.array(precision_list[3]) >= 0.95)
                if (True in other_point_list):
                    loc_index = np.where(other_point_list == True)[0][0]
                    x_best = recall_list[3][loc_index]


            # Optimal p-value threshold
            optimal_pvalue = p_cutoff_list[index]
            optimal_th = -np.log10(p_cutoff_list[index])

            y_best = precision_list[index][loc_index]
            if (threshold_list[index][loc_index]) == -1:
                Thr_log2fc = threshold_list[index][loc_index+1]
            else:
                Thr_log2fc = threshold_list[index][loc_index]

        # If the accuracy of all four lines is < 0.95, take the 0,1 point of the first line
        else:
            x_best = 0
            y_best = precision_list[0][-1]
            Thr_log2fc = threshold_list[0][-1]

            optimal_pvalue = p_cutoff_list[3]
            optimal_th = -np.log10(p_cutoff_list[3])


        if ((x_best <0.1) & (y_best >0.2)):
            plt.text(x_best+0.02, y_best - 0.03, 'Thr: {:.3f} \nTPR: {:.2f} \nPPV: {:.2f}'.format(Thr_log2fc, x_best, y_best), ha='left', va='top', fontsize=14)
        elif ((x_best >0.9) & (y_best >0.2)):
            plt.text(x_best-0.02, y_best - 0.03, 'Thr: {:.3f} \nTPR: {:.2f} \nPPV: {:.2f}'.format(Thr_log2fc, x_best, y_best), ha='right', va='top', fontsize=14)
        elif (y_best <=0.2):
            plt.text(x_best+0.02, y_best +0.15, 'Thr: {:.3f} \nTPR: {:.2f} \nPPV: {:.2f}'.format(Thr_log2fc, x_best, y_best), ha='left', va='top', fontsize=14)
        else:
            plt.text(x_best, y_best - 0.035, 'Thr: {:.3f} \nTPR: {:.2f} \nPPV: {:.2f}'.format(Thr_log2fc, x_best, y_best), ha='center', va='top', fontsize=14)


        plt.plot(x_best, y_best, marker='o', markersize=8, color='black')


        # Save Threshold, FPR, TPR to csv
        if savefig:
            if plot_line_1_4[0]:
                data1 = pd.DataFrame({'Threshold (p-value=0.1)': threshold_list[0],
                        'Recall (p-value=0.1)': recall_list[0],
                        'Precision (p-value=0.1)': precision_list[0]})
            else:
                data1 = pd.DataFrame({'Threshold (p-value=0.1)': [],
                        'Recall (p-value=0.1)': [],
                        'Precision (p-value=0.1)': []})
        
            if plot_line_1_4[1]:
                data2 = pd.DataFrame({
                        'Threshold (p-value=0.05)': threshold_list[1],
                        'Recall (p-value=0.05)': recall_list[1],
                        'Precision (p-value=0.05)': precision_list[1]})
            else:
                data2 = pd.DataFrame({
                        'Threshold (p-value=0.05)': [],
                        'Recall (p-value=0.05)': [],
                        'Precision (p-value=0.05)': []})

            if plot_line_1_4[2]:
                data3 = pd.DataFrame({
                        'Threshold (p-value=0.01)': threshold_list[2],
                        'Recall (p-value=0.01)': recall_list[2],
                        'Precision (p-value=0.01)': precision_list[2]})
            else:
                data3 = pd.DataFrame({
                        'Threshold (p-value=0.01)': [],
                        'Recall (p-value=0.01)': [],
                        'Precision (p-value=0.01)': []})

            if plot_line_1_4[3]:
                data4 = pd.DataFrame({
                        'Threshold (p-value=0.001)': threshold_list[3],
                        'Recall (p-value=0.001)': recall_list[3],
                        'Precision (p-value=0.001)': precision_list[3]})
            else:
                data4 = pd.DataFrame({
                        'Threshold (p-value=0.001)': [],
                        'Recall (p-value=0.001)': [],
                        'Precision (p-value=0.001)': []})

            df_save = pd.concat([data1, data2, data3, data4], axis = 1)
            df_save.columns = ['Threshold (p-value=0.1)', 'Recall (p-value=0.1)', 'Precision (p-value=0.1)',
                               'Threshold (p-value=0.05)', 'Recall (p-value=0.05)', 'Precision (p-value=0.05)',
                               'Threshold (p-value=0.01)', 'Recall (p-value=0.01)', 'Precision (p-value=0.01)',
                               'Threshold (p-value=0.001)', 'Recall (p-value=0.001)', 'Precision (p-value=0.001)']
            df_save.to_csv(savefolder + 'PR_{0}_{1}_vs_{2}.csv'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), index=False)

        plt.subplots_adjust(left=0.155, right=0.98, bottom=0.115, top=0.99, wspace=0.05, hspace=0.1)
        
        if savefig:
            plt.savefig(savefolder + 'PR_{0}_{1}_vs_{2}.svg'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), dpi=600, format="svg", transparent=True) 
        plt.show()
        plt.close()

        if savefig:
            # Generate PR curve legend
            fig_legend = plt.figure(figsize=(2.5, 2.5))
 
            axes = plt.gca()
            edgecolor = ['#b0203f', '#e95b1b', '#a14ee0', '#1883b8', '#39a139', '#bc8f00', '#d9d8d2']
            labels = ['0.10', '0.05', '0.01', '0.001']
            for i in range(4):
                axes.plot([10000, 20000], [10000, 20000], lw=2, color = edgecolor[i])
            

            axes.legend(labels = labels, title='p-value', title_fontsize=18, fontsize=16, 
                        loc = 'center',
                        markerfirst=True, markerscale=2.0) 

            plt.ylim(-5, 5)
            plt.xlim(-5, 5)

            axes.spines['top'].set_visible(False) 
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

            plt.xticks([])
            plt.yticks([])

            plt.savefig(savefolder + 'Legend_PR.svg', dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()



        # Plot volcano diagram
        #fig_Vol = plt.figure(figsize=(5,5))

        df2 = df.copy(deep=True)
        df2['padjust'] = df2['padjust'].clip(lower = np.finfo(np.float64).eps)
        df2['padjust'] = -np.log10(df2['padjust'])

        # Delete protein data from multiple species in df2
        drop_index_list = []
        count = 0
        for species in df2['Species'].values.tolist():
            if species[:-1] in self.species:
                pass
            else:
                drop_index_list.append(df2.index.values[count])
            count += 1

        df2 = df2.drop(drop_index_list) 


        # If using the given list of p-values
        if self.Use_Given_PValue_and_FC & self.Use_PValue_List:
            pauc_list = []
            optimal_pvalue_list = []
            Thr_log2fc_list = []
            TP_list = []
            TN_list = []
            FP_list = [] 
            FN_list = []
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_score_list = []
            label_true_list = []
            label_predict_list = []

            for p in self.PValue_List:
                optimal_pvalue = p
                optimal_th = -np.log10(optimal_pvalue)
                #Thr_log2fc = np.log2(self.Given_FC) 
                Thr_log2fc = np.log2(self.FC_For_Groups[plot_index]) 

                c2part1=df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)]  # Up-regulated points
                c2part2=df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))]  # Down-regulated points
                c2part3=df2[(df2['padjust']<=optimal_th) | ((df2['log2fc']>=(-Thr_log2fc)) & (df2['log2fc']<=(Thr_log2fc)))]  # Points with no significant difference

                species_list = self.species
                sepcies_df_list = []

                # Saved drawing data
                Protein_data_list = []
                Organism_data_list = []
                log2FC_data_list = []
                minus_log10pvalue_data_list = []
                for species in species_list:
                    df_species = df2[df2['Species'] == (species + ' ')]
                    sepcies_df_list.append(df_species)

                    Protein_data_list.append(df_species.index.tolist())
                    Organism_data_list.append([species]*df_species.shape[0])
                    log2FC_data_list.append(df_species['log2fc'].values.tolist())
                    minus_log10pvalue_data_list.append(df_species['padjust'].values.tolist())

                Protein_data_list = sum(Protein_data_list, [])
                Organism_data_list = sum(Organism_data_list, [])
                log2FC_data_list = sum(log2FC_data_list, [])
                minus_log10pvalue_data_list = sum(minus_log10pvalue_data_list, [])

                scatter_color = ['#339dff', '#65c3ba', '#ff6680', '#ff9900', '#996633', '#660066', '#006600', '#ff3300']
    
    
                # Statistics of the true and predicted up- and down-regulation of all proteins in the two compared groups
                # 1 up-regulated 2 down-regulated 3 unchanged
                label_true = [0] * df2.shape[0]
                label_predict = [0] * df2.shape[0]


                count = 0

                for i in df2.index.values.tolist():
                    for species in species_list:
                        if i in species_PG_names[species]:
                            label_true[count] = up_down_label[species_list.index(species)] 

                    if i in c2part1.index.tolist():
                        label_predict[count] = 1  # Up
                    if i in c2part2.index.tolist():
                        label_predict[count] = 2  # Down
                    if i in c2part3.index.tolist():
                        label_predict[count] = 3  # No Sig.
                    count += 1

                # Statistically predicted number of differentially expressed proteins
                DEPs_up = label_predict.count(1)
                DEPs_down = label_predict.count(2)
                DEPs = DEPs_up + DEPs_down

        
                # Calculate the number of TP, TN, FP, FN proteins
                TP = 0 
                TN = 0 
                FP = 0 
                FN = 0 

                for species in species_list:
                    # Theoretical downregulation of proteins in this species
                    up_down_true = up_down_label[species_list.index(species)]

                    if up_down_true == 1:
                        # Up
                        # TP
                        df_temp = df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        TP += df_temp.shape[0]
                        # FN
                        df_temp = df2[(df2['padjust']<=optimal_th) | (df2['log2fc']<=(Thr_log2fc))]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        FN += df_temp.shape[0]

                    if up_down_true == 2:
                        # Down
                        # TP
                        df_temp = df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        TP += df_temp.shape[0]
                        # FN
                        df_temp = df2[(df2['padjust']<=optimal_th) | (df2['log2fc']>=(-Thr_log2fc)) ]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        FN += df_temp.shape[0]

                    if up_down_true == 3:
                        # No Sig.
                        # FP
                        df_temp = df2[((df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)) | ((df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))) ]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        FP += df_temp.shape[0]
                        # TN
                        df_temp = df2[(df2['padjust']<=optimal_th) | ((df2['log2fc']>=(-Thr_log2fc)) & (df2['log2fc']<=(Thr_log2fc)))]
                        df_temp = df_temp[df_temp['Species'].str.contains(species)]
                        TN += df_temp.shape[0]


                # Calculate 4 indicators
                if (TP + FP) != 0:
                    accuracy = (TP + TN)/(TP + TN + FP + FN)
                    precision = TP/(TP + FP)
                    recall = TP/(TP + FN)
                    f1_score = 2*TP/(2*TP + FP + FN)
                else:
                    accuracy = (TP + TN)/(TP + TN + FP + FN)
                    precision = y_best
                    recall = 0
                    f1_score = 0

                pauc_list.append(pauc)
                optimal_pvalue_list.append(optimal_pvalue)
                Thr_log2fc_list.append(Thr_log2fc)
                TP_list.append(TP)
                TN_list.append(TN)
                FP_list.append(FP)
                FN_list.append(FN)
                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1_score)
                label_true_list.append(label_true)
                label_predict_list.append(label_predict)

            return pauc_list, optimal_pvalue_list, Thr_log2fc_list, TP_list, TN_list, FP_list, FN_list, accuracy_list, precision_list, recall_list, f1_score_list, label_true_list, label_predict_list

        else:
            # If fixed p-value and FC are used 
            if self.Use_Given_PValue_and_FC:
                optimal_pvalue = self.Given_PValue
                optimal_th = -np.log10(optimal_pvalue)
                Thr_log2fc = np.log2(self.Given_FC)

            # If only a fixed p-value is used, FC is automatically determined
            if self.Only_Use_Given_PValue:
                optimal_pvalue = self.Given_PValue
                optimal_th = -np.log10(optimal_pvalue)

            c2part1=df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)]  # Up-regulated points
            c2part2=df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))]  # Down-regulated points
            c2part3=df2[(df2['padjust']<=optimal_th) | ((df2['log2fc']>=(-Thr_log2fc)) & (df2['log2fc']<=(Thr_log2fc)))]  # Points with no significant difference

            species_list = self.species
            sepcies_df_list = []

            # Saved drawing data
            Protein_data_list = []
            Organism_data_list = []
            log2FC_data_list = []
            minus_log10pvalue_data_list = []
            for species in species_list:
                df_species = df2[df2['Species'] == (species + ' ')]
                sepcies_df_list.append(df_species)

                Protein_data_list.append(df_species.index.tolist())
                Organism_data_list.append([species]*df_species.shape[0])
                log2FC_data_list.append(df_species['log2fc'].values.tolist())
                minus_log10pvalue_data_list.append(df_species['padjust'].values.tolist())

            Protein_data_list = sum(Protein_data_list, [])
            Organism_data_list = sum(Organism_data_list, [])
            log2FC_data_list = sum(log2FC_data_list, [])
            minus_log10pvalue_data_list = sum(minus_log10pvalue_data_list, [])

            scatter_color = ['#339dff', '#65c3ba', '#ff6680', '#ff9900', '#996633', '#660066', '#006600', '#ff3300']
    
    
            # Statistics of the true and predicted up- and down-regulation of all proteins in the two compared groups
            # 1 up-regulated 2 down-regulated 3 unchanged
            label_true = [0] * df2.shape[0]
            label_predict = [0] * df2.shape[0]


            count = 0

            for i in df2.index.values.tolist():
                for species in species_list:
                    if i in species_PG_names[species]:
                        label_true[count] = up_down_label[species_list.index(species)] 

                if i in c2part1.index.tolist():
                    label_predict[count] = 1  # Up
                if i in c2part2.index.tolist():
                    label_predict[count] = 2  # Down
                if i in c2part3.index.tolist():
                    label_predict[count] = 3  # No Sig.
                count += 1

            # Statistically predicted number of differentially expressed proteins
            DEPs_up = label_predict.count(1)
            DEPs_down = label_predict.count(2)
            DEPs = DEPs_up + DEPs_down

        
            # Calculate the number of TP, TN, FP, FN proteins
            TP = 0 
            TN = 0 
            FP = 0 
            FN = 0 

            for species in species_list:
                # Theoretical downregulation of proteins in this species
                up_down_true = up_down_label[species_list.index(species)]

                if up_down_true == 1:
                    # Up
                    # TP
                    df_temp = df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    TP += df_temp.shape[0]
                    # FN
                    df_temp = df2[(df2['padjust']<=optimal_th) | (df2['log2fc']<=(Thr_log2fc))]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    FN += df_temp.shape[0]

                if up_down_true == 2:
                    # Down
                    # TP
                    df_temp = df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    TP += df_temp.shape[0]
                    # FN
                    df_temp = df2[(df2['padjust']<=optimal_th) | (df2['log2fc']>=(-Thr_log2fc)) ]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    FN += df_temp.shape[0]

                if up_down_true == 3:
                    # No Sig.
                    # FP
                    df_temp = df2[((df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)) | ((df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))) ]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    FP += df_temp.shape[0]
                    # TN
                    df_temp = df2[(df2['padjust']<=optimal_th) | ((df2['log2fc']>=(-Thr_log2fc)) & (df2['log2fc']<=(Thr_log2fc)))]
                    df_temp = df_temp[df_temp['Species'].str.contains(species)]
                    TN += df_temp.shape[0]


            # Calculate 4 indicators
            if (TP + FP) != 0:
                accuracy = (TP + TN)/(TP + TN + FP + FN)
                precision = TP/(TP + FP)
                recall = TP/(TP + FN)
                f1_score = 2*TP/(2*TP + FP + FN)
            else:
                accuracy = (TP + TN)/(TP + TN + FP + FN)
                precision = y_best
                recall = 0
                f1_score = 0


            # Plot volcano diagram
            if savefig:
                fig_Vol = plt.figure(figsize=(5,5))

                plt.axhline(optimal_th,color='grey',linestyle='--') # Draw a horizontal dashed line to mark the position where pvalue=optimal_th
                plt.axvline(Thr_log2fc,color='grey',linestyle='--') 
                plt.axvline(-Thr_log2fc,color='grey',linestyle='--') 

    
                ax = plt.gca()

                legend_list = []
                outdata_list = []

        
                for i in range(len(species_list)):
                    # Draw scatter plot and generate a legend

                    up_down_true = up_down_label[i]

                    df_temp = sepcies_df_list[i]

                    if up_down_true == 1:
                        df_plot_color = df_temp[(df_temp['padjust']>optimal_th)&(df_temp['log2fc']>Thr_log2fc)]
                        df_plot_grey = df_temp[(df_temp['padjust']<=optimal_th) | (df_temp['log2fc']<=Thr_log2fc)]
                        ax.scatter(df_plot_grey['log2fc'], df_plot_grey['padjust'], s = 3, marker = 'o', color='grey', alpha=0.5, rasterized=True)

                    if up_down_true == 2:
                        df_plot_color = df_temp[(df_temp['padjust']>optimal_th)&(df_temp['log2fc']<(-Thr_log2fc))]
                        df_plot_grey = df_temp[(df_temp['padjust']<=optimal_th) | (df_temp['log2fc']>=(-Thr_log2fc))]
                        ax.scatter(df_plot_grey['log2fc'], df_plot_grey['padjust'], s = 3, marker = 'o', color='grey', alpha=0.5, rasterized=True)

                    if up_down_true == 3:
                        df_plot_color = df_temp[(df_temp['padjust']<=optimal_th) | ((df_temp['log2fc']>=(-Thr_log2fc)) & (df_temp['log2fc']<=(Thr_log2fc)))]
                        df_plot_grey = df_temp[((df_temp['padjust']>optimal_th)&(df_temp['log2fc']>Thr_log2fc)) | ((df_temp['padjust']>optimal_th)&(df_temp['log2fc']<(-Thr_log2fc))) ]
                        ax.scatter(df_plot_grey['log2fc'], df_plot_grey['padjust'], s = 3, marker = 'o', color='grey', alpha=0.5, rasterized=True)

                for i in range(len(species_list)):
                    # Draw scatter plot and generate a legend

                    up_down_true = up_down_label[i]

                    df_temp = sepcies_df_list[i]

                    if up_down_true == 1:
                        df_plot_color = df_temp[(df_temp['padjust']>optimal_th)&(df_temp['log2fc']>Thr_log2fc)]
                        df_plot_grey = df_temp[(df_temp['padjust']<=optimal_th) | (df_temp['log2fc']<=Thr_log2fc)]
                        legend_list.append(ax.scatter(df_plot_color['log2fc'], df_plot_color['padjust'], s = 3, marker = 'o', color=scatter_color[i], alpha=0.5, rasterized=True))

                    if up_down_true == 2:
                        df_plot_color = df_temp[(df_temp['padjust']>optimal_th)&(df_temp['log2fc']<(-Thr_log2fc))]
                        df_plot_grey = df_temp[(df_temp['padjust']<=optimal_th) | (df_temp['log2fc']>=(-Thr_log2fc))]
                        legend_list.append(ax.scatter(df_plot_color['log2fc'], df_plot_color['padjust'], s = 3, marker = 'o', color=scatter_color[i], alpha=0.5, rasterized=True))

                    if up_down_true == 3:
                        df_plot_color = df_temp[(df_temp['padjust']<=optimal_th) | ((df_temp['log2fc']>=(-Thr_log2fc)) & (df_temp['log2fc']<=(Thr_log2fc)))]
                        df_plot_grey = df_temp[((df_temp['padjust']>optimal_th)&(df_temp['log2fc']>Thr_log2fc)) | ((df_temp['padjust']>optimal_th)&(df_temp['log2fc']<(-Thr_log2fc))) ]
                        legend_list.append(ax.scatter(df_plot_color['log2fc'], df_plot_color['padjust'], s = 3, marker = 'o', color=scatter_color[i], alpha=0.5, rasterized=True))


                    outdata_list.append(pd.DataFrame(list(sepcies_df_list[i]['log2fc']), columns=[compared_groups_label[plot_index] + ' Volcano {0} log2fc'.format(species_list[i])]))
                    outdata_list.append(pd.DataFrame(list(-np.log10(sepcies_df_list[i]['padjust'])), columns=[compared_groups_label[plot_index] + ' Volcano {0} -Log10PValue'.format(species_list[i])]))
    
        
        
                plt.tick_params(labelsize=14) 
                plt.tick_params(axis='x', width=2)
                plt.tick_params(axis='y', width=2)

                ax.spines['bottom'].set_linewidth(2) 
                ax.spines['left'].set_linewidth(2) 
                ax.spines['top'].set_linewidth(2) 
                ax.spines['right'].set_linewidth(2)  

                ax.set_xlabel('log$_{2}$FC', fontsize=16)
                ax.set_ylabel('-log$_{10}$P-value', fontsize=16)

                text_str = 'TP: {0}\nTN: {1}\nFP: {2}\nFN: {3}'.format(TP, TN, FP, FN)
                x_min = ax.get_xlim()[0]
                x_width = (ax.get_xlim()[1] - ax.get_xlim()[0])
                y_half = (ax.get_ylim()[0] + ax.get_ylim()[1])/2
                plt.text(x_min + 0.02*x_width, y_half + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.38, text_str, ha='left', va='center', size=14, color = 'black', family="Arial", rotation=0) 

                plt.rcParams['savefig.dpi'] = 600
                plt.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98, wspace=0.05, hspace=0.1)


                data = {'Protein': Protein_data_list,
                        'Organism': Organism_data_list,
                        'log2FC': log2FC_data_list,
                        '-log10pvalue': minus_log10pvalue_data_list}
                df_save = pd.DataFrame(data)
                df_save.to_csv(savefolder + 'Volcano_{0}_{1}_vs_{2}.csv'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), index=False)

    
                plt.savefig(savefolder + 'Volcano_{0}_{1}_vs_{2}.svg'.format(methods_used, compared_groups_label[plot_index].split('/')[0], compared_groups_label[plot_index].split('/')[1]), dpi=600, format="svg", transparent=True)  # , bbox_inches='tight'
                plt.show()
                plt.close()


                # Generate Legend
                fig_legend = plt.figure(figsize=(2.5,2.5))
 
                axes = plt.gca()
                for i in range(len(self.species)):
                    axes.scatter([10000], [10000], s = 3, marker = 'o', color=scatter_color[i], alpha=0.9)

                axes.legend(labels = self.species, title='Organism', title_fontsize=18, fontsize=16, 
                            loc = 'center',
                            markerfirst=True, markerscale=4.0) 

                plt.ylim(-5, 5)
                plt.xlim(-5, 5)

                axes.spines['top'].set_visible(False) 
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(False)
                axes.spines['left'].set_visible(False)

                plt.xticks([])
                plt.yticks([])

                plt.savefig(savefolder + 'Legend_Volcano.svg', dpi=600, format="svg", transparent=True) 
                plt.show()
                plt.close()
        

            return pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict




    # Difference analysis, used in the fourth part of the article
    def Difference_Analysis_Part_4(self, df_all, 
                                   method = 't-test',
                                   Compared_groups_label = ['T/C'],
                                   FC = 1.5,
                                   pValue = 0.05,
                                   up_down_scatter_color = [[157/255, 48/255, 238/255, 1.0],
                                                            [53/255, 131/255, 99/255, 1.0]],
                                   MethodSelection = '',
                                   savefolder = './'):

        print('Difference analysis method: ' + method)
        df = df_all.copy(deep = True)

        Compared_groups_index = []
        for item in Compared_groups_label:
            item_A = item.split('/')[0]
            item_B = item.split('/')[1]
            Compared_groups_index.append([self.sample_index_of_each_group[item_A], self.sample_index_of_each_group[item_B]])


        df_list = []
        

        if method == 't-test':
            
            all_data = df.values.astype(float)
            all_data = np.where(all_data <= 0, 1, all_data)

            for i in range(len(Compared_groups_index)):
                df_list.append(df.iloc[:, sum(Compared_groups_index[i], []) ])

            # Perform differential analysis on the selected 2 groups of samples
            def DE_analysis(df, df_treat, df_control):
                Treat_mean = scipy.mean(df_treat, axis = 1) 
                Control_mean = scipy.mean(df_control, axis = 1) 
                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean 

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]

                    t, p = ttest_ind(Treat, Control, equal_var=False)  # Welch's t-test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 
                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue)  # P adjust

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean

                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()
            
                return df

            for i in range(len(Compared_groups_index)):
                df_treat = all_data[:, Compared_groups_index[i][0]].astype(float)
                df_control = all_data[:, Compared_groups_index[i][1]].astype(float)

                df_list[i] = DE_analysis(df_list[i], df_treat, df_control)

        elif (method == 'Wilcoxon-test') | (method == 'Wilcox'):

            all_data = df.values.astype(float)
            all_data = np.where(all_data <= 0, 1, all_data)

            for i in range(len(Compared_groups_index)):
                df_list.append(df.iloc[:, sum(Compared_groups_index[i], []) ])

            # Perform differential analysis on the selected 2 groups of samples
            def DE_analysis(df, df_treat, df_control):
                Treat_mean = scipy.mean(df_treat, axis = 1) 
                Control_mean = scipy.mean(df_control, axis = 1) 
                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean 

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]

                    statistic, p = ranksums(Treat, Control)  # Wilcoxon test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 
                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue)  # P adjust

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean

                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()
            
                return df

            for i in range(len(Compared_groups_index)):
                df_treat = all_data[:, Compared_groups_index[i][0]].astype(float)
                df_control = all_data[:, Compared_groups_index[i][1]].astype(float)

                df_list[i] = DE_analysis(df_list[i], df_treat, df_control)

        elif (method == 'Limma-trend') | (method == 'limma-trend'):
            # R script
            Limma_trend = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

        
            # data
            # cols

            targets <- data


            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            dgelist <- DGEList(counts = targets, group = group)

            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'TMM') 
            lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            dge <- estimateDisp(targets, design, robust = TRUE)  # dgelist_norm

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-trend
            #de <- voom(dge,design,plot=TRUE, normalize="quantile")
            fit1 <- lmFit(targets, design)   # lcpmyf
            fit2 <- contrasts.fit(fit1,cont.matrix) 
            efit <- eBayes(fit2, trend=TRUE)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_trend  <- na.omit(tempDEG)

            DEG_limma_trend

            '''
            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after limma-trend difference analysis
                DEG_limma_trend = robjects.r(Limma_trend)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_trend)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B']  #adj.P.Val


                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif (method == 'Limma-voom') | (method == 'limma-voom'):

            #  R script
            Limma_voom = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols

            targets <- data

            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'TMM')
            #lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-voom
            de <- voom(targets, design, plot=FALSE) # dge, normalize="quantile"
            fit1 <- lmFit(de, design) 
            fit2 <- contrasts.fit(fit1,cont.matrix)
            efit <- eBayes(fit2, trend=F)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_voom  <- na.omit(tempDEG)

            DEG_limma_voom 

            '''

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after limma-voom difference analysis
                DEG_limma_voom = robjects.r(Limma_voom)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_voom)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B']

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif method == 'edgeR-LRT':

            # R script
            edgeR_LRT = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'

            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)   # dgelist_norm

            fit <- glmFit(dge, design, robust = TRUE) 
            lrt <- glmLRT(fit) 
            topTags(lrt, adjust.method="BH")

            lrt[["table"]] 

            '''

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_groups_index[index][1], Compared_groups_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after edgeR-LRT difference analysis
                lrt_table = robjects.r(edgeR_LRT)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(lrt_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif method == 'edgeR-QLF':

            # R script
            edgeR_QLF = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'


            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)  #dgelist_norm

            
            fit <- glmQLFit(dge, design, robust = TRUE) 
            qlf <- glmQLFTest(fit)
            topTags(qlf, adjust.method="BH") 

            qlf[["table"]]

            '''

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_groups_index[index][1], Compared_groups_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after edgeR-QLF difference analysis
                qlf_table = robjects.r(edgeR_QLF)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(qlf_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif (method == 'DESeq2-parametric') | (method == 'DESeq2'):
            # The input of DESeq2 must use the original count. Log normalization and minmax normalization cannot be performed because it has already been normalized.
            # Otherwise, an error will be reported because the differences between groups are too small, 
            # and the estimated dispersion values ​​of proteins are very similar and standard fitting cannot be performed.
        

            # R script
            DESeq2_parametric = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)
            library(DESeq2)

            # data
            # cols
        
            targets <- data
            targets <- targets[,cols]

            countData <- as.matrix(targets)
            colnames(countData) <- rep(c('A', 'B'), c(group_A_num, group_B_num))
            condition <- factor(rep(c('A', 'B'), c(group_A_num, group_B_num)))


            dds <- DESeqDataSetFromMatrix(round(countData), DataFrame(condition), ~condition)


            #keep <- rowSums(counts(dds)) >= 1.5*ncol(counts) 
            #dds <- dds[keep,] 
            dds <- DESeq(dds, fitType = 'parametric', quiet = F) 

            #dds <- estimateSizeFactors(dds)
            #dds <- estimateDispersionsGeneEst(dds)
            #dispersions(dds) <- mcols(dds)$dispGeneEst
            #dds <- nbinomLRT(dds, reduced = ~ 1)


            res <- results(dds,contrast=c("condition", 'A', 'B'), pAdjustMethod = "BH") 
            resOrdered <- res[order(res$padj),] 
            tempDEG <- as.data.frame(resOrdered)
            DEG_DEseq2 <- na.omit(tempDEG)

            DEG_DEseq2 

            '''

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # R integer maximum value
                threshold = 2147483648-1
                df2 = df2.clip(upper=threshold)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after DESeq2 differential analysis
                DEG_DEseq2 = robjects.r(DESeq2_parametric)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_DEseq2)
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'baseMean', 'log2fc', 'lfcSE', 'stat', 'pvalue', 'padjust']

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        # List used to store the number of up-regulated, down-regulated, and non-significantly different proteins
        list_Up_Num = []
        list_Down_Num = []
        list_NoSig_Num = []

        # Used to store the number of enrichment analysis terms
        list_GO_Terms = [] 
        list_KEGG_Terms = [] 
        list_Reactome_Terms = []

        # Plotting a volcano
        for df2 in df_list:

            fig_Vol = plt.figure(figsize=(5,5))
            ax = plt.gca()

            df2['padjust'] = df2['padjust'].clip(lower = np.finfo(np.float64).eps)
            df2['padjust'] = -np.log10(df2['padjust'])

            optimal_th = -np.log10(pValue)
            Thr_log2fc = np.log2(FC)
            c2part1=df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)] # Up-regulated points
            c2part2=df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))] # Down-regulated points
            c2part3=df2[(df2['padjust']<=optimal_th) | ((df2['log2fc']>=(-Thr_log2fc)) & (df2['log2fc']<=(Thr_log2fc)))] # Points with no significant difference

            ax.scatter(c2part1['log2fc'], c2part1['padjust'], s = 3, marker = 'o', color = up_down_scatter_color[0], alpha=0.9, rasterized=True) 
            ax.scatter(c2part2['log2fc'], c2part2['padjust'], s = 3, marker = 'o', color = up_down_scatter_color[1], alpha=0.9, rasterized=True)
            ax.scatter(c2part3['log2fc'], c2part3['padjust'], s = 3, marker = 'o', color='grey', alpha=0.9, rasterized=True) 

            list_Up_Num.append(c2part1.shape[0])
            list_Down_Num.append(c2part2.shape[0])
            list_NoSig_Num.append(c2part3.shape[0])

            # Horizontal and vertical dashed lines
            plt.axhline(optimal_th,color='grey',linestyle='--') 
            plt.axvline(Thr_log2fc,color='grey',linestyle='--') 
            plt.axvline(-Thr_log2fc,color='grey',linestyle='--') 

            # Mark the amount
            text_up = 'Up: {0}'.format(str(c2part1.shape[0]))
            text_down = 'Down: {0}'.format(str(c2part2.shape[0]))
            x_min = ax.get_xlim()[0]
            x_width = (ax.get_xlim()[1] - ax.get_xlim()[0])
            y_half = (ax.get_ylim()[0] + ax.get_ylim()[1])/2
            plt.text(x_min + 0.04*x_width, y_half + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.44, text_down, ha='left', va='center', size=16, color = 'black', family="Arial", rotation=0) 
            plt.text(x_min + 0.77*x_width, y_half + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.44, text_up, ha='left', va='center', size=16, color = 'black', family="Arial", rotation=0) 

            plt.tick_params(labelsize=14) 
            plt.tick_params(axis='x', width=2)
            plt.tick_params(axis='y', width=2)

            ax.spines['bottom'].set_linewidth(2) 
            ax.spines['left'].set_linewidth(2) 
            ax.spines['top'].set_linewidth(2) 
            ax.spines['right'].set_linewidth(2)  

            ax.set_xlabel('log$_{2}$FC', fontsize=16)
            ax.set_ylabel('-log$_{10}$P-value', fontsize=16)

            plt.rcParams['savefig.dpi'] = 600
            plt.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98, wspace=0.05, hspace=0.1)


            plot_index = df_list.index(df2)
            Protein_data_list = df2.index.values.tolist()
            log2FC_data_list = df2['log2fc'].values.tolist()
            minus_log10pvalue_data_list = df2['padjust'].values.tolist()

            data = {'Protein': Protein_data_list,
                    'log2FC': log2FC_data_list,
                    '-log10pvalue': minus_log10pvalue_data_list}
            df_save = pd.DataFrame(data)
            # Sort by log2FC descending
            df_sorted = df_save.sort_values(by='log2FC', ascending=False)
            volcano_csv_path = savefolder + 'Volcano_{0}_{1}_vs_{2}.csv'.format(MethodSelection, Compared_groups_label[plot_index].split('/')[0], Compared_groups_label[plot_index].split('/')[1])
            df_sorted.to_csv(volcano_csv_path, index=False)

    
            plt.savefig(savefolder + 'Volcano_{0}_{1}_vs_{2}.svg'.format(MethodSelection, Compared_groups_label[plot_index].split('/')[0], Compared_groups_label[plot_index].split('/')[1]), dpi=600, format="svg", transparent=True) 
            plt.show()
            plt.close()

            # Only if there is data on up- and down-regulation of proteins, do subsequent analysis
            if (c2part1.shape[0] > 0) & (c2part2.shape[0] > 0):
                # R script
                R_GO = '''

                options(java.parameters = "-Xmx10000m")

                library(readr)

                # csv_path

                # GO_csv_path

                differential_proteins = read_csv(csv_path)

                differential_proteins = subset(differential_proteins, abs(log2FC) > log2(1.5) & `-log10pvalue` > -log10(0.05))

                # Packages
                # https://www.bioconductor.org/packages/release/bioc/html/clusterProfiler.html
                # https://bioconductor.org/packages/release/data/annotation/html/org.Hs.eg.db.html
                # https://bioconductor.org/packages/release/bioc/html/ReactomePA.html
                # https://mp.weixin.qq.com/s/PwrdQAkG3pTlwMB6Mj8wXQ

                library(clusterProfiler)

                get_entrez_id = function(differential_proteins, OrgDb = 'org.Hs.eg.db') {
                  protein_id = sapply(strsplit(differential_proteins$Protein, ';'), head, 1)
                  protein_mapping = bitr(unique(protein_id), fromType = 'UNIPROT', toType = 'ENTREZID', OrgDb = OrgDb)
                  differential_proteins$EntrezID = protein_mapping$ENTREZID[match(protein_id, protein_mapping$UNIPROT)]
                  differential_proteins
                }

                add_protein_info = function(enrich_result, differential_proteins) {
                  if (nrow(enrich_result) == 0) {
                    return(
                      cbind(
                        enrich_result,
                        proteinID = character(),
                        CountUp = integer(),
                        CountDown = integer()
                      )
                    )
                  }
                  enrich_result = cbind(
                    enrich_result, 
                    t(sapply(strsplit(enrich_result$geneID, '/'), function(x){
                      index = match(x, differential_proteins$EntrezID)
                      proteinID = paste(differential_proteins$Protein[index], collapse = '/')
                      CountUp = sum(differential_proteins$log2FC[index] > 0, na.rm = TRUE)
                      CountDown = sum(differential_proteins$log2FC[index] < 0, na.rm = TRUE)
                      data.frame(
                        proteinID = proteinID,
                        CountUp = CountUp,
                        CountDown = CountDown,
                        stringsAsFactors = FALSE
                      )
                    }))
                  )
                  enrich_result
                }

                differential_proteins = get_entrez_id(differential_proteins)

                enrich_GO = function(differential_proteins, OrgDb = 'org.Hs.eg.db', pvalueCutoff = 0.05) {
                  differential_proteins = subset(differential_proteins, !is.na(EntrezID))
                  GO = enrichGO(
                    differential_proteins$EntrezID,
                    OrgDb = OrgDb,
                    ont = "All",
                    pvalueCutoff = pvalueCutoff,
                    qvalueCutoff = Inf,
                  )
                  GO = simplify(GO)
                  GO = subset(GO@result, p.adjust < pvalueCutoff)
                  GO = add_protein_info(GO, differential_proteins)
                  GO
                }

                GO = enrich_GO(differential_proteins)
                GO2<-data.frame(c(unlist(GO$proteinID)),c(unlist(GO$CountUp)),c(unlist(GO$CountDown)))
                colnames(GO2) <- c("proteinID", "CountUp", "CountDown")
                GO_combined <- cbind(GO[,1:10], GO2)
                GO_combined

                '''

                # GO enrichment analysis
                robjects.r('rm(list=ls())')
                robjects.r('csv_path = "{0}"'.format(volcano_csv_path))

                GO = robjects.r(R_GO)

                pandas2ri.activate()
                df_GO = pandas2ri.rpy2py(GO)
                pandas2ri.deactivate()
                # Save GO results
                df_GO_path = savefolder + 'GO_{0}_{1}_vs_{2}.csv'.format(MethodSelection, Compared_groups_label[plot_index].split('/')[0], Compared_groups_label[plot_index].split('/')[1])
                df_GO.to_csv(df_GO_path, index=False)
                # Number of enriched terms
                GO_Terms = df_GO.shape[0]
                list_GO_Terms.append(GO_Terms)

                # KEGG
                R_KEGG = '''
                enrich_KEGG = function(differential_proteins, organism = "hsa", pvalueCutoff = 0.05) {
                  differential_proteins = subset(differential_proteins, !is.na(EntrezID))
                  KEGG = enrichKEGG(
                    differential_proteins$EntrezID,
                    organism = organism,
                    pvalueCutoff = pvalueCutoff,
                    qvalueCutoff = Inf,
                    use_internal_data = TRUE
                  )
                  KEGG = subset(KEGG@result, p.adjust < pvalueCutoff)
                  KEGG = add_protein_info(KEGG, differential_proteins)
                  KEGG
                }

                KEGG = enrich_KEGG(differential_proteins)
                KEGG2<-data.frame(c(unlist(KEGG$proteinID)),c(unlist(KEGG$CountUp)),c(unlist(KEGG$CountDown)))
                colnames(KEGG2) <- c("proteinID", "CountUp", "CountDown")
                KEGG_combined <- cbind(KEGG[,1:9], KEGG2)
                KEGG_combined

                '''

                KEGG = robjects.r(R_KEGG)

                pandas2ri.activate()
                df_KEGG = pandas2ri.rpy2py(KEGG)
                pandas2ri.deactivate()
                # Save KEGG results
                df_KEGG_path = savefolder + 'KEGG_{0}_{1}_vs_{2}.csv'.format(MethodSelection, Compared_groups_label[plot_index].split('/')[0], Compared_groups_label[plot_index].split('/')[1])
                df_KEGG.to_csv(df_KEGG_path, index=False)
                # Number of enriched terms
                KEGG_Terms = df_KEGG.shape[0]
                list_KEGG_Terms.append(KEGG_Terms)

                # Reactome
                R_Reactome = '''
                library(ReactomePA)

                enrich_Reactome = function(differential_proteins, organism = "human", pvalueCutoff = 0.05) {
                  differential_proteins = subset(differential_proteins, !is.na(EntrezID))
                  Reactome = enrichPathway(
                    differential_proteins$EntrezID,
                    organism = organism,
                    pvalueCutoff = pvalueCutoff,
                    qvalueCutoff = Inf
                  )
                  Reactome = subset(Reactome@result, p.adjust < pvalueCutoff)
                  Reactome = add_protein_info(Reactome, differential_proteins)
                  Reactome
                }

                Reactome = enrich_Reactome(differential_proteins)
                Reactome2<-data.frame(c(unlist(Reactome$proteinID)),c(unlist(Reactome$CountUp)),c(unlist(Reactome$CountDown)))
                colnames(Reactome2) <- c("proteinID", "CountUp", "CountDown")
                Reactome_combined <- cbind(Reactome[,1:9], Reactome2)
                Reactome_combined

                '''

                Reactome = robjects.r(R_Reactome)

                pandas2ri.activate()
                df_Reactome = pandas2ri.rpy2py(Reactome)
                pandas2ri.deactivate()
                # Saving Reactome Results
                df_Reactome_path = savefolder + 'Reactome_{0}_{1}_vs_{2}.csv'.format(MethodSelection, Compared_groups_label[plot_index].split('/')[0], Compared_groups_label[plot_index].split('/')[1])
                df_Reactome.to_csv(df_Reactome_path, index=False)
                # Number of enriched terms
                Reactome_Terms = df_Reactome.shape[0]
                list_Reactome_Terms.append(Reactome_Terms)

            else:
                list_GO_Terms.append(0)
                list_KEGG_Terms.append(0)
                list_Reactome_Terms.append(0)


            
        return list_Up_Num, list_Down_Num, list_NoSig_Num, list_GO_Terms, list_KEGG_Terms, list_Reactome_Terms


    # For differential analysis between different batches
    def Difference_Analysis_For_Batches(self, df_all, dict_species,
                                        method = 't-test',
                                        Compared_batches_label = ['Batch1/Batch2', 'Batch2/Batch3', 'Batch1/Batch3'],
                                        FC = 1.5,
                                        pValue_List = [0.001, 0.01, 0.05, 0.1]):

        df = df_all.copy(deep = True)

        Compared_batches_index = []

        for item in Compared_batches_label:
            item_A = item.split('/')[0]
            item_B = item.split('/')[1]
            Compared_batches_index.append([self.sample_index_of_each_batch[item_A], self.sample_index_of_each_batch[item_B]])


        # Used to store differential analysis results of different batches
        df_list = []
        

        if method == 't-test':
            
            all_data = df.values.astype(float)
            all_data = np.where(all_data <= 0, 1, all_data)

            for i in range(len(Compared_batches_index)):
                df_list.append(df.iloc[:, sum(Compared_batches_index[i], []) ])

            # Perform differential analysis on the selected 2 groups of samples
            def DE_analysis(df, df_treat, df_control):
                #Treat_mean = scipy.mean(df_treat, axis = 1) 
                #Control_mean = scipy.mean(df_control, axis = 1) 
                Treat_mean = np.nanmean(df_treat, axis=1)
                Control_mean = np.nanmean(df_control, axis=1)

                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean 

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]
                    Treat_No_NaN = Treat[~np.isnan(Treat)]
                    Control_No_NaN = Control[~np.isnan(Control)]

                    t, p = ttest_ind(Treat_No_NaN, Control_No_NaN, equal_var=False)  # Welch's t-test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 
                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue)  # P adjust

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean

                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()
            
                return df

            for i in range(len(Compared_batches_index)):
                df_treat = all_data[:, Compared_batches_index[i][0]].astype(float)
                df_control = all_data[:, Compared_batches_index[i][1]].astype(float)

                df_list[i] = DE_analysis(df_list[i], df_treat, df_control)

        elif (method == 'Wilcoxon-test') | (method == 'Wilcox'):

            all_data = df.values.astype(float)
            all_data = np.where(all_data <= 0, 1, all_data)

            for i in range(len(Compared_batches_index)):
                df_list.append(df.iloc[:, sum(Compared_batches_index[i], []) ])

            # Perform differential analysis on the selected 2 groups of samples
            def DE_analysis(df, df_treat, df_control):
                #Treat_mean = scipy.mean(df_treat, axis = 1) 
                #Control_mean = scipy.mean(df_control, axis = 1) 
                Treat_mean = np.nanmean(df_treat, axis=1)
                Control_mean = np.nanmean(df_control, axis=1)

                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean 

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]

                    statistic, p = ranksums(Treat, Control)  # Wilcoxon test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 
                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue)  # P adjust

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean

                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()
            
                return df

            for i in range(len(Compared_batches_index)):
                df_treat = all_data[:, Compared_batches_index[i][0]].astype(float)
                df_control = all_data[:, Compared_batches_index[i][1]].astype(float)

                df_list[i] = DE_analysis(df_list[i], df_treat, df_control)

        elif (method == 'Limma-trend') | (method == 'limma-trend'):
            # R script
            Limma_trend = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

        
            # data
            # cols

            targets <- data


            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            #dgelist <- DGEList(counts = targets, group = group)

            ##keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            ##dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            #dgelist_norm <- calcNormFactors(dgelist, method = 'TMM') 
            #lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            #dge <- estimateDisp(targets, design, robust = TRUE)  # dgelist_norm

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-trend
            #de <- voom(dge,design,plot=TRUE, normalize="quantile")
            fit1 <- lmFit(targets, design)   # lcpmyf
            fit2 <- contrasts.fit(fit1,cont.matrix) 
            efit <- eBayes(fit2, trend=TRUE)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_trend  <- na.omit(tempDEG)

            DEG_limma_trend

            '''
            for index in range(len(Compared_batches_index)):

                df2 = df.copy(deep=True)
                df3 = None

                # In the case where missing value filling is not used, the protein expression matrix needs to be screened 
                # so that the number of valid samples of proteins in the two comparison groups is greater than or equal to 2.
                if self.FilterProteinMatrix:
                    Batch_A_Index = Compared_batches_index[index][0]
                    Batch_B_Index = Compared_batches_index[index][1]

                    df2 = df2[df2.apply(lambda row: row[Batch_A_Index].notna().sum() >= 2, axis=1)]
                    df2 = df2[df2.apply(lambda row: row[Batch_B_Index].notna().sum() >= 2, axis=1)]

                    df3 = df2.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_batches_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_batches_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_batches_index[index][1])))

                # Obtain the dataframe after limma-trend difference analysis
                DEG_limma_trend = robjects.r(Limma_trend)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_trend)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B']  #adj.P.Val


                # Add experimental and control group data
                if self.FilterProteinMatrix:
                    columns_of_df = df3.columns.tolist()
                    columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                    df_to_add = df3[columns_to_add]
                    df2 = pd.concat([df2, df_to_add], axis=1)
                else:
                    columns_of_df = df.columns.tolist()
                    columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                    df_to_add = df[columns_to_add]
                    df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif (method == 'Limma-voom') | (method == 'limma-voom'):

            #  R script
            Limma_voom = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols

            targets <- data

            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            #dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            #dgelist_norm <- calcNormFactors(dgelist, method = 'TMM')
            #lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            #dge <- estimateDisp(dgelist_norm, design, robust = TRUE)

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-voom
            de <- voom(targets, design, plot=FALSE) # dge, normalize="quantile"
            fit1 <- lmFit(de, design) 
            fit2 <- contrasts.fit(fit1,cont.matrix)
            efit <- eBayes(fit2, trend=F)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_voom  <- na.omit(tempDEG)

            DEG_limma_voom  

            '''

            for index in range(len(Compared_batches_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_batches_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_batches_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_batches_index[index][1])))

                # Obtain the dataframe after limma-voom difference analysis
                DEG_limma_voom = robjects.r(Limma_voom)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_voom)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B']

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif method == 'edgeR-LRT':

            # R script
            edgeR_LRT = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'

            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)   # dgelist_norm

            fit <- glmFit(dge, design, robust = TRUE) 
            lrt <- glmLRT(fit) 
            topTags(lrt, adjust.method="BH")

            lrt[["table"]] 

            '''

            for index in range(len(Compared_batches_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_batches_index[index][1], Compared_batches_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_batches_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_batches_index[index][1])))

                # Obtain the dataframe after edgeR-LRT difference analysis
                lrt_table = robjects.r(edgeR_LRT)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(lrt_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif method == 'edgeR-QLF':

            # R script
            edgeR_QLF = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'


            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)  #dgelist_norm

            
            fit <- glmQLFit(dge, design, robust = TRUE) 
            qlf <- glmQLFTest(fit)
            topTags(qlf, adjust.method="BH") 

            qlf[["table"]]

            '''

            for index in range(len(Compared_batches_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_batches_index[index][1], Compared_batches_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_batches_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_batches_index[index][1])))

                # Obtain the dataframe after edgeR-QLF difference analysis
                qlf_table = robjects.r(edgeR_QLF)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(qlf_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

        elif (method == 'DESeq2-parametric') | (method == 'DESeq2'):
            # The input of DESeq2 must use the original count. Log normalization and minmax normalization cannot be performed because it has already been normalized.
            # Otherwise, an error will be reported because the differences between groups are too small, 
            # and the estimated dispersion values ​​of proteins are very similar and standard fitting cannot be performed.
        

            # R script
            DESeq2_parametric = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)
            library(DESeq2)

            # data
            # cols
        
            targets <- data
            targets <- targets[,cols]

            countData <- as.matrix(targets)
            colnames(countData) <- rep(c('A', 'B'), c(group_A_num, group_B_num))
            condition <- factor(rep(c('A', 'B'), c(group_A_num, group_B_num)))


            dds <- DESeqDataSetFromMatrix(round(countData), DataFrame(condition), ~condition)


            #keep <- rowSums(counts(dds)) >= 1.5*ncol(counts) 
            #dds <- dds[keep,] 
            dds <- DESeq(dds, fitType = 'parametric', quiet = F) 

            #dds <- estimateSizeFactors(dds)
            #dds <- estimateDispersionsGeneEst(dds)
            #dispersions(dds) <- mcols(dds)$dispGeneEst
            #dds <- nbinomLRT(dds, reduced = ~ 1)


            res <- results(dds,contrast=c("condition", 'A', 'B'), pAdjustMethod = "BH") 
            resOrdered <- res[order(res$padj),] 
            tempDEG <- as.data.frame(resOrdered)
            DEG_DEseq2 <- na.omit(tempDEG)

            DEG_DEseq2 

            '''

            for index in range(len(Compared_batches_index)):

                df2 = df.copy(deep=True)

                # R integer maximum value
                threshold = 2147483648-1
                df2 = df2.clip(upper=threshold)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                cols = sum(Compared_batches_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_batches_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_batches_index[index][1])))

                # Obtain the dataframe after DESeq2 differential analysis
                DEG_DEseq2 = robjects.r(DESeq2_parametric)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_DEseq2)
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names
                df2.columns = ['Names', 'baseMean', 'log2fc', 'lfcSE', 'stat', 'pvalue', 'padjust']

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_batches_index[index][0]] + [columns_of_df[i] for i in Compared_batches_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)


        # Start counting the number of up- and down-regulated proteins in different comparison batches and different P values
        dict_Up_Num = {}
        dict_Down_Num = {}
        for p in pValue_List:
            #up_num_list = [[]]*len(Compared_batches_label)
            #down_num_list = [[]]*len(Compared_batches_label)
            dict_Up_Num[str(p)] = []
            dict_Down_Num[str(p)] = []

        for p in pValue_List:
            compared_batches_count = 0
            for df3 in df_list:
                df2 = df3.copy(deep = True)
                df2['padjust'] = df2['padjust'].clip(lower = np.finfo(np.float64).eps)
                df2['padjust'] = -np.log10(df2['padjust'])
                optimal_th = -np.log10(p)
                Thr_log2fc = np.log2(FC)
                c2part1=df2[(df2['padjust']>optimal_th)&(df2['log2fc']>Thr_log2fc)] # Up-regulated points
                c2part2=df2[(df2['padjust']>optimal_th)&(df2['log2fc']<(-Thr_log2fc))] # Down-regulated points

                dict_Up_Num[str(p)].append(c2part1.shape[0])
                dict_Down_Num[str(p)].append(c2part2.shape[0])
                compared_batches_count += 1


        return dict_Up_Num, dict_Down_Num







    
    # Difference analysis methods
    def Difference_Analysis(self, df_all, dict_species,
                            method = 't-test',
                            Compared_groups_label = ['S1/S3', 'S2/S3', 'S4/S3', 'S5/S3'],
                            DoROC = True,
                            title_methods = 'methods used',
                            savefig = True, savefolder = './', savename = 'Difference_Analysis'):


        print('Difference analysis method: ' + method)
        df = df_all

        Compared_groups_index = []
        Compared_groups_up_down_label = [] 
        for item in Compared_groups_label:
            item_A = item.split('/')[0]
            item_B = item.split('/')[1]
            Compared_groups_index.append([self.sample_index_of_each_group[item_A], self.sample_index_of_each_group[item_B]])

            compared_ratio = self.df_composition[item_A].values/self.df_composition[item_B].values
            up_down_label = []
            for i in range(len(compared_ratio)):
                if compared_ratio[i] > self.FC_threshold:
                    up_down_label.append(1)  # up
                if compared_ratio[i] < 1/self.FC_threshold:
                    up_down_label.append(2)  # down
                if (compared_ratio[i] <= self.FC_threshold) & (compared_ratio[i] >= 1/self.FC_threshold):
                    up_down_label.append(3)  # no sig.
            Compared_groups_up_down_label.append(up_down_label)
        

        species_PG_names = {}  # The name of the protein group contained in each species. Dictionary structure
        for species in self.species:
            species_PG_names[species] = dict_species[species].index.tolist()


        if method == 't-test':
            all_data = df.values.astype(float) 
            # In order to perform subsequent calculations normally, change the value <=0 in the data to 1
            all_data = np.where(all_data <= 0, 1, all_data)
            # Perform log2 processing on the data
            #all_data = np.log2(all_data)

            df_list = []
            for i in range(len(Compared_groups_index)):
                df_list.append(df.iloc[:, sum(Compared_groups_index[i], []) ])


            # Perform differential analysis on the selected 2 groups of samples
            # group_label: The true differences of all species' protein groups in df. 1-has a difference, 0-no difference
            def DE_analysis(df, df_treat, df_control, group_label = [1, 1, 0]):
                #Treat_mean = scipy.mean(df_treat, axis = 1) 
                #Control_mean = scipy.mean(df_control, axis = 1) 
                Treat_mean = np.nanmean(df_treat, axis=1)
                Control_mean = np.nanmean(df_control, axis=1)

                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean 

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]
                    Treat_No_NaN = Treat[~np.isnan(Treat)]
                    Control_No_NaN = Control[~np.isnan(Control)]

                    t, p = ttest_ind(Treat_No_NaN, Control_No_NaN, equal_var=False)  # Welch's t-test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 

                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue)  # P adjust

                # Statistics of species origin of protein groups in df
                species_list = []
                PG = df.index.tolist()
                for i in PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                    df.insert(loc=5,column='Species',value=species_list)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean
                    df['Species'] = species_list


                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                label_true = [0] * df.shape[0]
                count = 0
                df_PG = df.index.tolist()

                for i in df_PG:
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]

                    count += 1

                try:
                    df.insert(loc=6,column='label_true',value=label_true)
                except:
                    df[label_true] = label_true


                return df, label_true, df_PG, species_PG_names


            # Lists used to store differential analysis results
            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []


            for i in range(len(Compared_groups_index)):

                df_treat = all_data[:, Compared_groups_index[i][0]].astype(float)
                df_control = all_data[:, Compared_groups_index[i][1]].astype(float)

                up_down_label = Compared_groups_up_down_label[i]

                group_label = [0 if element == 3 else 1 for element in up_down_label]

                df_list[i], label_true, df_PG, species_PG_names = DE_analysis(df_list[i], df_treat, df_control, group_label)
                
                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df_list[i], label_true, df_PG, species_PG_names, i, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)
                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)

        
            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list

        elif (method == 'Wilcoxon-test') | (method == 'Wilcox'):
            all_data = df.values.astype(float) 
            # In order to perform subsequent calculations normally, change the value <=0 in the data to 1
            all_data = np.where(all_data <= 0, 1, all_data)
            # Perform log2 processing on the data
            #all_data = np.log2(all_data)

            df_list = []
            for i in range(len(Compared_groups_index)):
                df_list.append(df.iloc[:, sum(Compared_groups_index[i], []) ])


            # Perform differential analysis on the selected 2 groups of samples
            # group_label: The true differences of all species' protein groups in df. 1-has a difference, 0-no difference
            def DE_analysis(df, df_treat, df_control, group_label = [1, 1, 0]):
                #Treat_mean = scipy.mean(df_treat, axis = 1) 
                #Control_mean = scipy.mean(df_control, axis = 1) 
                Treat_mean = np.nanmean(df_treat, axis=1)
                Control_mean = np.nanmean(df_control, axis=1)


                divide = np.divide(Treat_mean,Control_mean)
                Treat_VS_Control_log2fc = Treat_mean - Control_mean

                Treat_VS_Control_pvalue = [] 
                # P value
                for i in range(len(Treat_mean)):
                    Treat = df_treat[i,:]
                    Control = df_control[i,:]

                    statistic, p = ranksums(Treat, Control)  # Wilcoxon test

                    Treat_VS_Control_pvalue.append(p)
                Treat_VS_Control_pvalue = np.array(Treat_VS_Control_pvalue) 

                Treat_VS_Control_padjust = correct_pvalues_for_multiple_testing(Treat_VS_Control_pvalue) 

                # Statistics of species origin of protein groups in df
                species_list = []
                PG = df.index.tolist()
                for i in PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            temp += '{0} '.format(j)


                    species_list.append(temp)

                # Update data
                try:
                    df.insert(loc=0,column='log2fc',value=Treat_VS_Control_log2fc)
                    df.insert(loc=1,column='pvalue',value=Treat_VS_Control_pvalue)
                    df.insert(loc=2,column='padjust',value=Treat_VS_Control_padjust)
                    df.insert(loc=3,column='SA_mean',value=Treat_mean)
                    df.insert(loc=4,column='SB_mean',value=Control_mean)
                    df.insert(loc=5,column='Species',value=species_list)
                except:
                    df['log2fc'] = Treat_VS_Control_log2fc
                    df['pvalue'] = Treat_VS_Control_pvalue
                    df['padjust'] = Treat_VS_Control_padjust
                    df['SA_mean'] = Treat_mean
                    df['SB_mean'] = Control_mean
                    df['Species'] = species_list


                # Delete rows in df with missing pvalues
                df = df.dropna(how='all', subset=['pvalue']).copy()

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                label_true = [0] * df.shape[0]
                count = 0
                df_PG = df.index.tolist()

                for i in df_PG:
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]


                    count += 1

                try:
                    df.insert(loc=6,column='label_true',value=label_true)
                except:
                    df[label_true] = label_true


                return df, label_true, df_PG, species_PG_names


            # Lists used to store differential analysis results
            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []


            for i in range(len(Compared_groups_index)):
                
                df_treat = all_data[:, Compared_groups_index[i][0]].astype(float)
                df_control = all_data[:, Compared_groups_index[i][1]].astype(float)

                up_down_label = Compared_groups_up_down_label[i]

                group_label = [0 if element == 3 else 1 for element in up_down_label]


                df_list[i], label_true, df_PG, species_PG_names = DE_analysis(df_list[i], df_treat, df_control, group_label)
                
                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df_list[i], label_true, df_PG, species_PG_names, i, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)
                    
                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list

        elif (method == 'Limma-trend') | (method == 'limma-trend'):

            # R script
            Limma_trend = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

        
            # data
            # cols

            targets <- data


            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            #dgelist <- DGEList(counts = targets, group = group)

            ##keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            ##dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            #dgelist_norm <- calcNormFactors(dgelist, method = 'TMM') 
            #lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            #dge <- estimateDisp(targets, design, robust = TRUE)  # dgelist_norm

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-trend
            #de <- voom(dge,design,plot=TRUE, normalize="quantile")
            fit1 <- lmFit(targets, design)   # lcpmyf
            fit2 <- contrasts.fit(fit1,cont.matrix) 
            efit <- eBayes(fit2, trend=TRUE)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_trend  <- na.omit(tempDEG)

            DEG_limma_trend

            '''

            

            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []

            df_list = []

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)
                df3 = None

                # In the case where missing value filling is not used, the protein expression matrix needs to be screened 
                # so that the number of valid samples of proteins in the two comparison groups is greater than or equal to 2.
                if self.FilterProteinMatrix:
                    Group_A_Index = Compared_groups_index[index][0]
                    Group_B_Index = Compared_groups_index[index][1]

                    df2 = df2[df2.apply(lambda row: row[Group_A_Index].notna().sum() >= 2, axis=1)]
                    df2 = df2[df2.apply(lambda row: row[Group_B_Index].notna().sum() >= 2, axis=1)]

                    df3 = df2.copy(deep=True)


                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after limma-trend difference analysis
                DEG_limma_trend = robjects.r(Limma_trend)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_trend)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                up_down_label = Compared_groups_up_down_label[index]
                group_label = [0 if element == 3 else 1 for element in up_down_label]

                label_true = [0] * df2.shape[0]
                count = 0
                df_PG = df2.index.tolist()

                species_list = []  # Statistics of species origin of protein groups in df2

                for i in df_PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                    count += 1

                df2['label_true'] = label_true
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B', 'label_true']  #adj.P.Val

                df2['Species'] = species_list

                # Add experimental and control group data
                if self.FilterProteinMatrix:
                    columns_of_df = df3.columns.tolist()
                    columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                    df_to_add = df3[columns_to_add]
                    df2 = pd.concat([df2, df_to_add], axis=1)
                else:
                    columns_of_df = df.columns.tolist()
                    columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                    df_to_add = df[columns_to_add]
                    df2 = pd.concat([df2, df_to_add], axis=1)


                df_list.append(df2)

    
                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df2, label_true, df_PG, species_PG_names, index, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list


        elif (method == 'Limma-voom') | (method == 'limma-voom'):

            #  R script
            Limma_voom = '''
            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols

            targets <- data

            targets <- targets[,cols]
            group <- rep(c('C1', 'C2'), c(group_A_num, group_B_num)) 
            #dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            #dgelist_norm <- calcNormFactors(dgelist, method = 'TMM')
            #lcpmyf <- cpm(dgelist_norm,log=T)

            design <- model.matrix(~0+factor(group)) 
            colnames(design)=levels(factor(group))
            #dge <- estimateDisp(dgelist_norm, design, robust = TRUE)

            cont.matrix <- makeContrasts(contrasts=paste0('C1','-','C2'), levels = design)

            # limma-voom
            de <- voom(targets, design, plot=FALSE) # dge, normalize="quantile"
            fit1 <- lmFit(de, design) 
            fit2 <- contrasts.fit(fit1,cont.matrix)
            efit <- eBayes(fit2, trend=F)  #Apply empirical Bayes smoothing to the standard errors

            tempDEG <- topTable(efit, coef=paste0('C1','-','C2'), n=Inf, adjust.method='BH') 
            DEG_limma_voom  <- na.omit(tempDEG)

            DEG_limma_voom 

            '''


            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []

            df_list = []

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 

                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after limma-voom difference analysis
                DEG_limma_voom = robjects.r(Limma_voom)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_limma_voom)  # R dataframe to Python dataframe
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                up_down_label = Compared_groups_up_down_label[index]
                group_label = [0 if element == 3 else 1 for element in up_down_label]

                label_true = [0] * df2.shape[0]
                count = 0
                df_PG = df2.index.tolist()

                species_list = []  # Statistics of species origin of protein groups in df2

                for i in df_PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                    count += 1

                df2['label_true'] = label_true
                df2.columns = ['Names', 'log2fc', 'AveExpr', 't', 'pvalue', 'padjust', 'B', 'label_true']

                df2['Species'] = species_list

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)
    
                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df2, label_true, df_PG, species_PG_names, index, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list


        elif method == 'edgeR-LRT':

            # R script
            edgeR_LRT = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'

            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)   # dgelist_norm

            fit <- glmFit(dge, design, robust = TRUE) 
            lrt <- glmLRT(fit) 
            topTags(lrt, adjust.method="BH")

            lrt[["table"]] 

            '''


            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []

            df_list = []

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_groups_index[index][1], Compared_groups_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after edgeR-LRT difference analysis
                lrt_table = robjects.r(edgeR_LRT)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(lrt_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                up_down_label = Compared_groups_up_down_label[index]
                group_label = [0 if element == 3 else 1 for element in up_down_label]

                label_true = [0] * df2.shape[0]
                count = 0
                df_PG = df2.index.tolist()

                species_list = []  # Statistics of species origin of protein groups in df2

                for i in df_PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                    count += 1

                df2['label_true'] = label_true
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust', 'label_true']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                df2['Species'] = species_list

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df2, label_true, df_PG, species_PG_names, index, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list


        elif method == 'edgeR-QLF':

            # R script
            edgeR_QLF = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)

            # data
            # cols
        
            targets <- data
            
            targets <- targets[,cols]
            group <- rep(c('Control', 'Case'), c(group_B_num, group_A_num)) 
            dgelist <- DGEList(counts = targets, group = group)


            #keep <- rowSums(cpm(dgelist) > 1 ) >= 2 
            #dgelist <- dgelist[keep, ,keep.lib.sizes = FALSE]
            dgelist_norm <- calcNormFactors(dgelist, method = 'none')  # method = 'TMM'


            design <- model.matrix(~group) 
            dge <- estimateDisp(dgelist_norm, design, robust = TRUE)  #dgelist_norm

            
            fit <- glmQLFit(dge, design, robust = TRUE) 
            qlf <- glmQLFTest(fit)
            topTags(qlf, adjust.method="BH") 

            qlf[["table"]]

            '''


            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []

            df_list = []

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                # Here, the control group is in front and the treatment group is in the back
                cols = sum([Compared_groups_index[index][1], Compared_groups_index[index][0]], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))

                # Obtain the dataframe after edgeR-QLF difference analysis
                qlf_table = robjects.r(edgeR_QLF)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(qlf_table) 
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                up_down_label = Compared_groups_up_down_label[index]
                group_label = [0 if element == 3 else 1 for element in up_down_label]

                label_true = [0] * df2.shape[0]
                count = 0
                df_PG = df2.index.tolist()

                species_list = []  # Statistics of species origin of protein groups in df2

                for i in df_PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                    count += 1

                df2['label_true'] = label_true
                df2.columns = ['Names', 'log2fc', 'logCPM', 'LR', 'padjust', 'label_true']

                # Since the log2fc output by edgeR is the result of the control group compared with the experimental group, 
                # we want the result of the experimental group compared with the control group, so we need to take a negative number.
                df2[['log2fc']] = df2[['log2fc']] * -1

                df2['Species'] = species_list

                # Add experimental and control group data
                columns_of_df = df.columns.tolist()
                columns_to_add = [columns_of_df[i] for i in Compared_groups_index[index][0]] + [columns_of_df[i] for i in Compared_groups_index[index][1]]
                df_to_add = df[columns_to_add]
                df2 = pd.concat([df2, df_to_add], axis=1)

                df_list.append(df2)

                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df2, label_true, df_PG, species_PG_names, index, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                   
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list


        elif (method == 'DESeq2-parametric') | (method == 'DESeq2'):
            # The input of DESeq2 must use the original count. Log normalization and minmax normalization cannot be performed because it has already been normalized.
            # Otherwise, an error will be reported because the differences between groups are too small, 
            # and the estimated dispersion values ​​of proteins are very similar and standard fitting cannot be performed.
        

            # R script
            DESeq2_parametric = '''

            options(java.parameters = "-Xmx10000m")

            library(Seurat)
            #library(ggsci)
            #library(patchwork)
            library(limma)
            #library(cowplot)
            library(edgeR)
            library(statmod)
            library(DESeq2)

            # data
            # cols
        
            targets <- data
            targets <- targets[,cols]

            countData <- as.matrix(targets)
            colnames(countData) <- rep(c('A', 'B'), c(group_A_num, group_B_num))
            condition <- factor(rep(c('A', 'B'), c(group_A_num, group_B_num)))


            dds <- DESeqDataSetFromMatrix(round(countData), DataFrame(condition), ~condition)


            #keep <- rowSums(counts(dds)) >= 1.5*ncol(counts) 
            #dds <- dds[keep,] 
            dds <- DESeq(dds, fitType = 'parametric', quiet = F) 

            #dds <- estimateSizeFactors(dds)
            #dds <- estimateDispersionsGeneEst(dds)
            #dispersions(dds) <- mcols(dds)$dispGeneEst
            #dds <- nbinomLRT(dds, reduced = ~ 1)


            res <- results(dds,contrast=c("condition", 'A', 'B'), pAdjustMethod = "BH") 
            resOrdered <- res[order(res$padj),] 
            tempDEG <- as.data.frame(resOrdered)
            DEG_DEseq2 <- na.omit(tempDEG)

            DEG_DEseq2 

            '''



            list_pauc = [] 
            list_pvalue = []
            list_log2fc = []

            list_TP = []
            list_TN = []
            list_FP = []
            list_FN = []
            list_accuracy = []
            list_precision = []
            list_recall = []
            list_f1_score = []

            list_overall_label_true_data = []
            list_overall_label_predict_data = []

            df_list = []

            for index in range(len(Compared_groups_index)):

                df2 = df.copy(deep=True)
                
                # R integer maximum value
                threshold = 2147483648-1
 
                df2 = df2.clip(upper=threshold)

                # Run the R script
                robjects.r('rm(list=ls())')
            
                with lc(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df2)
                globalenv['data'] = r_dataframe 


                cols = sum(Compared_groups_index[index], [])
                cols = [x + 1 for x in cols] 
                robjects.r('cols=c(%s)'%( str(cols)[1:-1] ))
                robjects.r('group_A_num = '+str(len(Compared_groups_index[index][0])))
                robjects.r('group_B_num = '+str(len(Compared_groups_index[index][1])))


                

                # Obtain the dataframe after DESeq2 differential analysis
                DEG_DEseq2 = robjects.r(DESeq2_parametric)
                pandas2ri.activate()
                df2 = pandas2ri.rpy2py(DEG_DEseq2)
                pandas2ri.deactivate()

                df2.insert(loc=0, column='Names', value=df2.index)  # Insert a column of protein group names

                # The true differences of the protein groups in df
                # 1-has a difference, 0-no difference
                up_down_label = Compared_groups_up_down_label[index]
                group_label = [0 if element == 3 else 1 for element in up_down_label]

                label_true = [0] * df2.shape[0]
                count = 0
                df_PG = df2.index.tolist()

                species_list = []  # Statistics of species origin of protein groups in df2

                for i in df_PG:
                    temp = ''
                    for j in self.species:
                        if i in species_PG_names[j]:
                            label_true[count] = group_label[self.species.index(j)]
                            temp += '{0} '.format(j)

                    species_list.append(temp)

                    count += 1

                df2['label_true'] = label_true
                df2.columns = ['Names', 'baseMean', 'log2fc', 'lfcSE', 'stat', 'pvalue', 'padjust', 'label_true']
                df2['Species'] = species_list

                df_list.append(df2)
    
                if DoROC:

                    pauc, optimal_pvalue, Thr_log2fc, TP, TN, FP, FN, accuracy, precision, recall, f1_score, label_true, label_predict = self.plot_ROC_and_Volcano(df2, label_true, df_PG, species_PG_names, index, method, up_down_label,
                                                                                                                        compared_groups_num = len(Compared_groups_index), 
                                                                                                                        compared_groups_label = Compared_groups_label,
                                                                                                                        methods_used = title_methods,
                                                                                                                        savefig = savefig,
                                                                                                                        savefolder = savefolder)

                    
                    list_pauc.append(pauc)
                    list_pvalue.append(optimal_pvalue)
                    list_log2fc.append(Thr_log2fc)

                    list_TP.append(TP)
                    list_TN.append(TN)
                    list_FP.append(FP)
                    list_FN.append(FN)

                    list_accuracy.append(accuracy)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1_score.append(f1_score)

                    list_overall_label_true_data.append(label_true)
                    list_overall_label_predict_data.append(label_predict)


            if DoROC:
                return list_pauc, list_pvalue, list_log2fc, list_TP, list_TN, list_FP, list_FN, list_accuracy, list_precision, list_recall, list_f1_score, df_list, list_overall_label_true_data, list_overall_label_predict_data
            else:
                return df_list








if __name__ == '__main__': 

    
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument("--Task", type=str, default='MethodSelection')

    parser.add_argument("--SoftwareQuantity", type=int, default=1)

    parser.add_argument("--Software", type=str, default='')
    parser.add_argument("--FolderPath", type=str, default='./')
    parser.add_argument("--FileName", type=str, default='DIANN.tsv')
    parser.add_argument("--ReportPath", type=str, default='./DIANN.tsv')

    parser.add_argument("--Software2", type=str, default='')
    parser.add_argument("--FolderPath2", type=str, default='./')
    parser.add_argument("--FileName2", type=str, default='Spectronaut.tsv')
    parser.add_argument("--ReportPath2", type=str, default='./Spectronaut.tsv')

    parser.add_argument("--Software3", type=str, default='')
    parser.add_argument("--FolderPath3", type=str, default='./')
    parser.add_argument("--FileName3", type=str, default='Peaks.csv')
    parser.add_argument("--ReportPath3", type=str, default='./Peaks.csv')

    parser.add_argument("--Software4", type=str, default='')
    parser.add_argument("--FolderPath4", type=str, default='./')
    parser.add_argument("--FileName4", type=str, default='MaxQuant.txt')
    parser.add_argument("--ReportPath4", type=str, default='./MaxQuant.txt')

    # The user's sample information csv file path
    parser.add_argument("--SamplesPath", type=str, default='')
    # The user's species composition information csv file path
    parser.add_argument("--CompositionPath", type=str, default='')
    # Protein/Peptide
    parser.add_argument("--Level", type=str, default='Protein')

    # Dimensionality reduction method: pca or umap
    parser.add_argument("--Reduction", type=str, default='pca')

    # Whether to provide covariates when correcting for batch effects
    parser.add_argument("--UseCovariates", type=str, default='True')

    # Is the clustering indicator ARI or PurityScore?
    parser.add_argument("--ClusteringEvaluation", type=str, default='ARI')

    # Whether to output figures and tables
    parser.add_argument("--OutputMethodSelectionFigures", type=str, default='False')

    # Groups that need differential analysis
    parser.add_argument("--Comparison", nargs='+', type=str, default="S1/S3")
    # Parameters for differential analysis
    parser.add_argument("--FC", type=float, default=1.5)
    parser.add_argument("--p-value", type=float, default=0.05)

    # MethodSelectionTask Type: UseCov, NoCov, KeepNA, AutoFC
    parser.add_argument("--Type", type=str, default='UseCov')

    # FC for each comparison group
    parser.add_argument("--ComparisonFC", nargs='+', type=str, default="1.5")

    ## Using fixed FC and P-Value
    #parser.add_argument("--Use_Given_PValue_and_FC", type=str, default='False')
    #parser.add_argument("--Use_PValue_List", type=str, default='True')

    # SHAPFolder
    parser.add_argument("--SHAPFolder", type=str, default='./')

    parser.add_argument("--ProteinMatrixPath", type=str, default='')
    parser.add_argument("--MethodSelectionPath", type=str, default='')

    # Hierarchical clustering parameter
    parser.add_argument("--PathwayResultPath", type=str, default='')


    # The folder path of the dataset to be compared
    # The name of the software/library construction method to be compared
    parser.add_argument("--DatasetName1", type=str, default='')
    parser.add_argument("--DatasetFolder1", type=str, default='./')
    parser.add_argument("--DatasetName2", type=str, default='')
    parser.add_argument("--DatasetFolder2", type=str, default='./')
    parser.add_argument("--DatasetName3", type=str, default='')
    parser.add_argument("--DatasetFolder3", type=str, default='./')
    parser.add_argument("--DatasetName4", type=str, default='')
    parser.add_argument("--DatasetFolder4", type=str, default='./')
    parser.add_argument("--DatasetName5", type=str, default='')
    parser.add_argument("--DatasetFolder5", type=str, default='./')

    # Path to the result file of the combined analysis
    parser.add_argument("--ResultCSVPath", type=str, default='./')
    parser.add_argument("--ResultCSVPath2", type=str, default='./')
    parser.add_argument("--ResultCSVPath3", type=str, default='./')

    # Save the results
    parser.add_argument("--SaveResult", type=bool, default=True)
    parser.add_argument("--SavePath", type=str, default='./')
    parser.add_argument("--ResultFileName", type=str, default='Combined_Analysis_Result')

    args = parser.parse_args()

    args.FolderPath, args.FileName = os.path.split(args.ReportPath)
    args.FolderPath += '/'
    args.FolderPath2, args.FileName2 = os.path.split(args.ReportPath2)
    args.FolderPath2 += '/'
    args.FolderPath3, args.FileName3 = os.path.split(args.ReportPath3)
    args.FolderPath3 += '/'
    args.FolderPath4, args.FileName4 = os.path.split(args.ReportPath4)
    args.FolderPath4 += '/'

    print('\n--- User input information ---')
    print('Task: ' + args.Task)
    

    # Generate sample grouping information template
    if args.Task == 'Generate_Samples_Template':

        # python SCPDA.py --Task Generate_Samples_Template --Software DIANN --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230904_0904hela-yeast-ecoli_200pg_test01_DIAnn.pg_matrix.tsv" --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Samples_Template --Software Spectronaut --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230901_161854_20230904_0904Hela-Yeast-Ecoli_200pg_tesr01_18_Report.tsv" --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Samples_Template --Software Peaks --ReportPath "G:/MS_data/Single_cell_benchmarking_data/PeaksStudio_report/HYE0904_DIA_DDALib/lfq.dia.proteins.csv" --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Samples_Template --Software MaxQuant --ReportPath "G:/MS_data/Single_cell_benchmarking_data/MaxQuant_report/20230904/proteinGroups.txt" --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 


        a = SCPDA(samples_csv_path = 'Generate_Template',
                  composition_csv_path = 'Generate_Template')

        a.Generate_Samples_Template(FolderPath = args.FolderPath, 
                                    FileName = args.FileName, 
                                    Protein_or_Peptide = args.Level, 
                                    Software = args.Software, 
                                    SavePath = args.SavePath)

        exit()

    # Generate species composition information template
    if args.Task == 'Generate_Composition_Template':

        # python SCPDA.py --Task Generate_Composition_Template --Software DIANN --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230904_0904hela-yeast-ecoli_200pg_test01_DIAnn.pg_matrix.tsv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/Samples_Template.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Composition_Template --Software Spectronaut --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230901_161854_20230904_0904Hela-Yeast-Ecoli_200pg_tesr01_18_Report.tsv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/Samples_Template.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Composition_Template --Software Peaks --ReportPath "G:/MS_data/Single_cell_benchmarking_data/PeaksStudio_report/HYE0904_DIA_DDALib/lfq.dia.proteins.csv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/Samples_Template.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 
        # python SCPDA.py --Task Generate_Composition_Template --Software MaxQuant --ReportPath "G:/MS_data/Single_cell_benchmarking_data/MaxQuant_report/20230904/proteinGroups.txt" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/Samples_Template.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 


        a = SCPDA(samples_csv_path = 'Generate_Template',
                  composition_csv_path = 'Generate_Template')

        a.Generate_Composition_Template(FolderPath = args.FolderPath, 
                                        FileName = args.FileName, 
                                        SamplesCSVPath = args.SamplesPath,
                                        Protein_or_Peptide = args.Level, 
                                        Software = args.Software, 
                                        SavePath = args.SavePath)

        exit()


    # Plot for UseCov
    if args.Task == 'PlotForUseCov':

        # python SCPDA.py --Task PlotForUseCov --ResultCSVPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis.csv" --Software DIANN --ReportPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\three_mix_report.pg_matrix.tsv" --SamplesPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Samples_Template.csv" --CompositionPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Composition_Template.csv" --Comparison S4/S2 S5/S1 --ComparisonFC 1.2 1.3 --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/PlotForUseCov/"

        from Example_Part2_02_UseCov_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts import *
        from Example_Part2_02_UseCov_Plot_2_ARI_and_pAUC_DotPlot import *
        from Example_Part2_02_UseCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators import *
        from Example_Part2_02_UseCov_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot import *
        from Example_Part2_02_UseCov_Plot_5_FeatureImportance_and_SHAP import *
        from Example_Part2_02_UseCov_Plot_6_Cluster_ROC_Volcano import *

        # Parameters that require user input
        ResultCSVPath = args.ResultCSVPath  # MethodSelection_DifferentialExpressionAnalysis.csv table with 4 p-values
        SavePath = args.SavePath  # Path for saving drawing results  'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Results/'

        SavePath_DotPlot = SavePath + 'Dot_Plot/'  # Point chart save path
        SavePath_Cluster_ROC_Volcano = SavePath + 'Cluster_ROC_Volcano/'  # Cluster ROC volcano save path
        SavePath_Others = SavePath + 'Other_Figures/'  # Other images save path

        Comparison = args.Comparison  # Comparison Group
        ComparisonFC = args.ComparisonFC  # FC of the comparison group
        FC_For_Groups = []
        for FC in ComparisonFC:
            FC_For_Groups.append(float(FC))


        if not os.path.exists(SavePath_DotPlot):
            os.makedirs(SavePath_DotPlot)

        if not os.path.exists(SavePath_Cluster_ROC_Volcano):
            os.makedirs(SavePath_Cluster_ROC_Volcano)

        if not os.path.exists(SavePath_Others):
            os.makedirs(SavePath_Others)

        Software = args.Software  # Software: DIANN, Spectronaut, PEAKS
        ReportPath = args.ReportPath  # Report path
        FolderPath = args.FolderPath  
        FileName = args.FileName  
        samples_csv_path = args.SamplesPath 
        composition_csv_path = args.CompositionPath 


        UseCov_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        UseCov_Plot_2_ARI_and_pAUC_DotPlot(result_csv_path = ResultCSVPath,savefolder = SavePath_DotPlot)

        UseCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        UseCov_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        UseCov_Plot_5_FeatureImportance_and_SHAP(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        UseCov_Plot_6_Cluster_ROC_Volcano(
            result_csv_path = ResultCSVPath,
            savefolder = SavePath_Cluster_ROC_Volcano,
            Software = Software,
            Report_Folder = FolderPath,
            Report_Name = FileName,

            samples_csv_path = samples_csv_path,
            composition_csv_path = composition_csv_path,

            FC_For_Groups = FC_For_Groups,
            Compared_groups_label = Comparison)

        exit()



    # Plot for NoCov
    if args.Task == 'PlotForNoCov':

        # python SCPDA.py --Task PlotForNoCov --ResultCSVPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Results\MethodSelection_DifferentialExpressionAnalysis.csv" --ResultCSVPath2 "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Results\与使用协变量表格合并重新排序后的表格\MethodSelection_DifferentialExpressionAnalysis-NC.csv" --ResultCSVPath3 "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Results\与使用协变量表格合并重新排序后的表格\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv" --Software DIANN --ReportPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\three_mix_report.pg_matrix.tsv" --SamplesPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Samples_Template.csv" --CompositionPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_NoCovariates\Composition_Template.csv" --Comparison S4/S2 S5/S1 --ComparisonFC 1.2 1.3 --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/PlotForNoCov/"

        from Example_Part2_04_NoCov_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts import *
        from Example_Part2_04_NoCov_Plot_2_ARI_and_pAUC_DotPlot import *
        from Example_Part2_04_NoCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators import *
        from Example_Part2_04_NoCov_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot import *
        from Example_Part2_04_NoCov_Plot_6_Cluster_ROC_Volcano import *

        # Parameters that require user input
        ResultCSVPath = args.ResultCSVPath  # NoCov's original table
        ResultCSVPath2 = args.ResultCSVPath2  # Renamed NoCovariate tables of 'limma', 'Combat-P', 'Combat-NP' to 'limma-NC', 'Combat-P-NC', 'Combat-NP-NC'
        ResultCSVPath3 = args.ResultCSVPath3  # The NoCov table selects the data of 3 batch effect correction methods that do not use covariates, and the result file after merging with the UseCov table

        SavePath = args.SavePath  # Path for saving drawing results  'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Results/'

        SavePath_DotPlot = SavePath + 'Dot_Plot/' 
        SavePath_Cluster_ROC_Volcano = SavePath + 'Cluster_ROC_Volcano/' 
        SavePath_Others = SavePath + 'Other_Figures/' 

        Comparison = args.Comparison 
        ComparisonFC = args.ComparisonFC 
        FC_For_Groups = []
        for FC in ComparisonFC:
            FC_For_Groups.append(float(FC))

        if not os.path.exists(SavePath_DotPlot):
            os.makedirs(SavePath_DotPlot)

        if not os.path.exists(SavePath_Cluster_ROC_Volcano):
            os.makedirs(SavePath_Cluster_ROC_Volcano)

        if not os.path.exists(SavePath_Others):
            os.makedirs(SavePath_Others)

        Software = args.Software  # Software: DIANN, Spectronaut, PEAKS
        ReportPath = args.ReportPath 
        FolderPath = args.FolderPath 
        FileName = args.FileName 
        samples_csv_path = args.SamplesPath 
        composition_csv_path = args.CompositionPath 



        NoCov_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts(result_csv_path = ResultCSVPath3,savefolder = SavePath_Others)

        NoCov_Plot_2_ARI_and_pAUC_DotPlot(result_csv_path = ResultCSVPath2,savefolder = SavePath_DotPlot)

        NoCov_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators(result_csv_path = ResultCSVPath2,savefolder = SavePath_Others)

        NoCov_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot(result_csv_path = ResultCSVPath2,savefolder = SavePath_Others)


        NoCov_Plot_6_Cluster_ROC_Volcano(
            result_csv_path = ResultCSVPath,
            savefolder = SavePath_Cluster_ROC_Volcano,
            Software = Software,
            Report_Folder = FolderPath,
            Report_Name = FileName,

            samples_csv_path = samples_csv_path,
            composition_csv_path = composition_csv_path,

            FC_For_Groups = FC_For_Groups,
            Compared_groups_label = Comparison)

        exit()


    # Plot for KeepNA
    if args.Task == 'PlotForKeepNA':

        # python SCPDA.py --Task PlotForKeepNA --ResultCSVPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Results\MethodSelection_DifferentialExpressionAnalysis-AddUseCov.csv" --Software DIANN --ReportPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\three_mix_report.pg_matrix.tsv" --SamplesPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Samples_Template.csv" --CompositionPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_KeepNA_UsePCA_UseCovariates\Composition_Template.csv" --Comparison S4/S2 S5/S1 --ComparisonFC 1.2 1.3 --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/Cluster_ROC_Volcano/"


        from Example_Part2_06_KeepNA_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts import *
        from Example_Part2_06_KeepNA_Plot_2_ARI_and_pAUC_DotPlot import *
        from Example_Part2_06_KeepNA_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators import *
        from Example_Part2_06_KeepNA_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot import *

        from Example_Part2_06_KeepNA_Plot_6_Cluster_ROC_Volcano import *

        # Parameters that require user input
        ResultCSVPath = args.ResultCSVPath  # Table after the merger of KeepNA and UseCov
        SavePath = args.SavePath  # Result save path  'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/Results/'

        SavePath_DotPlot = SavePath + 'Dot_Plot/'  
        SavePath_Cluster_ROC_Volcano = SavePath + 'Cluster_ROC_Volcano/' 
        SavePath_Others = SavePath + 'Other_Figures/' 

        Comparison = args.Comparison 
        ComparisonFC = args.ComparisonFC 
        FC_For_Groups = []
        for FC in ComparisonFC:
            FC_For_Groups.append(float(FC))

        if not os.path.exists(SavePath_DotPlot):
            os.makedirs(SavePath_DotPlot)

        if not os.path.exists(SavePath_Cluster_ROC_Volcano):
            os.makedirs(SavePath_Cluster_ROC_Volcano)

        if not os.path.exists(SavePath_Others):
            os.makedirs(SavePath_Others)

        Software = args.Software 
        ReportPath = args.ReportPath 
        FolderPath = args.FolderPath 
        FileName = args.FileName  
        samples_csv_path = args.SamplesPath  
        composition_csv_path = args.CompositionPath  



        KeepNA_Plot_1_Parallel_Coordinates_Plots_and_Frequency_Bar_Charts(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        KeepNA_Plot_2_ARI_and_pAUC_DotPlot(result_csv_path = ResultCSVPath,savefolder = SavePath_DotPlot)

        KeepNA_Plot_3_Bar_Charts_and_Scatter_of_Statistical_Indicators(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)

        KeepNA_Plot_4_Differential_Protein_Bar_Charts_and_Correlation_Coefficient_Box_Plot(result_csv_path = ResultCSVPath,savefolder = SavePath_Others)


        KeepNA_Plot_6_Cluster_ROC_Volcano(
            result_csv_path = ResultCSVPath,
            savefolder = SavePath_Cluster_ROC_Volcano,
            Software = Software,
            Report_Folder = FolderPath,
            Report_Name = FileName,

            samples_csv_path = samples_csv_path,
            composition_csv_path = composition_csv_path,

            FC_For_Groups = FC_For_Groups,
            Compared_groups_label = Comparison)

        exit()


    # Article Part 3 - Selection of the high-performing method combinations
    if args.Task == 'BeamSearch':

        SHAPFolder = args.SHAPFolder
        SavePath = args.SavePath

        from Example_Part3_01_Beam_Search import *

        Analysis_From_SHAP_Folder(shap_dir = SHAPFolder,
                                  save_dir = SavePath)



    # Article Part 4 - Hierarchical Clustering Diagram
    if args.Task == 'HierarchicalClustering':

        # python SCPDA.py --Task HierarchicalClustering --MethodSelectionPath "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\第四部分\DIANN\MethodSelection_DIANN_MCF7.csv" --PathwayResultPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/第四部分/DIANN/MCF7_3_Batches_MethodSelection_Result_2Y1E_不区分聚类簇/" --Comparison T/C --SavePath "C:/Users/Administrator/Downloads/Test/"

        
        TopMethodsFilePath = args.MethodSelectionPath
        ComparisonList = args.Comparison
        PathwayResultPath = args.PathwayResultPath
        SavePath = args.SavePath

        matplotlib.use('Agg')
        for PathWay in ['GO', 'Reactome']:
            for SR in ['NoSR', 'SR66', 'SR75', 'SR90']: 
            #for SR in ['SR75']:
                for C in ComparisonList:

                    X_label_after_clustering = Hierarchical_Clustering(TopMethodsFilePath = TopMethodsFilePath, 
                                            PathWay = PathWay, SR = SR, Comparison = C,
                                            Result_Folder = PathwayResultPath,
                                            savefolder = SavePath)

                    #Plot_PurityScore_Histogram(GO = PathWay, SR = SR, 
                    #                           X_label_after_clustering = X_label_after_clustering, 
                    #                           DifferentialExpressionAnalysis_CSV_Path = PathwayResultPath + "MethodSelection_DifferentialExpressionAnalysis.csv", 
                    #                           savefolder = SavePath)

                    Plot_ARI_Histogram(GO = PathWay, SR = SR, 
                                       X_label_after_clustering = X_label_after_clustering, 
                                       DifferentialExpressionAnalysis_CSV_Path = PathwayResultPath + "MethodSelection_DifferentialExpressionAnalysis.csv", 
                                       savefolder = SavePath)

        exit()



    # Article Part 4 - Finding Differential Proteins
    if args.Task == 'DifferentialProtein':

        # python SCPDA.py --Task DifferentialProtein --ProteinMatrixPath  "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\第四部分\MCF7_3_Batches_Protein_Matrix_Only_Human_1Y2E.csv" --MethodSelectionPath "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\第四部分\DIANN\MethodSelection_DIANN_MCF7.csv" --SamplesPath "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\第四部分\DIANN\Samples_DIANN_MCF7_3_Batches_T1Y2E_C1Y2E.csv" --Comparison T/C --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/第四部分/DIANN/MCF7_3_Batches_MethodSelection_Result_1Y2E_不区分聚类簇/"

        matplotlib.use('Agg')

        # Reading protein matrix
        df_protein_matrix = pd.read_csv(args.ProteinMatrixPath, index_col = 0) 
        if 'Organism' in df_protein_matrix.columns:
            # If it exists, delete the column
            df_protein_matrix.drop('Organism', axis=1, inplace=True)
        df_protein_matrix = df_protein_matrix.dropna(axis=0, how='all')
        # Reading method combination
        df_method_selection = pd.read_csv(args.MethodSelectionPath)
        # Reading sample information
        df_samples = pd.read_csv(args.SamplesPath)
        Groups = df_samples['Group'].unique().tolist()
        Batches = df_samples['Batch'].unique().tolist()

        a = SCPDA(samples_csv_path = 'DifferentialProtein',
                  composition_csv_path = 'DifferentialProtein')

        # Initialize class parameters
        a.df_samples = df_samples
        a.total_samples = df_samples['Run Name'].unique().size
        a.batches = df_samples['Batch'].unique().size
        a.batch_name = Batches
        a.sample_batch_list = df_samples['Batch'].values.tolist()
        a.sample_group_list = df_samples['Group'].values.tolist()
        a.group_name = Groups
        a.groups = len(Groups)  

        a.sample_index_of_each_group = {}
        a.run_name_of_each_group = {}
        for group in a.group_name:
            row_indices = a.df_samples[a.df_samples['Group'] == group].index
            sample_index = row_indices.values.tolist()
            a.sample_index_of_each_group.update({group:sample_index})
            a.run_name_of_each_group.update({group: a.df_samples.iloc[row_indices, 0].values.tolist()})



        # Clustering parameters
        a.FindClusters_resolution = '1.0'


        sample_index_of_each_group = {}
        Compared_groups_label = []


        Distinguish_Clusters = False

        # >>>> User input parameters
        if (Distinguish_Clusters == False):
            Compared_groups_label = args.Comparison
        SavePath = args.SavePath

        Reduction = args.Reduction
        if (Reduction == 'pca') | (Reduction == 'PCA'):
            additional_plot_methods.Reduction = 'pca'
        if (Reduction == 'umap') | (Reduction == 'UMAP'):
            additional_plot_methods.Reduction = 'umap'

        
        Using_ARI_or_PurityScore = 'ARI'  # Clustering evaluation metrics   PurityScore, ARI

        # lists of analysis method combinations and indicators
        list_SR_method = [] 
        list_Fill_NaN_method = [] 
        list_Normalization = [] 
        list_Batch_correction = [] 
        list_Difference_analysis = [] 

        list_ARI = []  # ARI

        list_Purity_Score = []  # Purity Score
        list_Sample_Purity = []  # Sample Purity
        list_Batch_Purity = []  # Batch Purity

        list_Comparison = [] 

        list_Up_Num = []
        list_Down_Num = []
        list_NoSig_Num = []

        # Enrichment results - Number of terms
        list_GO_Terms = []
        list_KEGG_Terms = []
        list_Reactome_Terms = []


        # Traverse each method combination
        for row in range(df_method_selection.shape[0]):
            if (row >= 0):
                i = df_method_selection['Sparsity Reduction'].values.tolist()[row]
                j = df_method_selection['Missing Value Imputation'].values.tolist()[row]
                k = df_method_selection['Normalization'].values.tolist()[row]
                l = df_method_selection['Batch Correction'].values.tolist()[row]
                m = df_method_selection['Statistical Test'].values.tolist()[row]

                df = a.Sparsity_Reduction(df_protein_matrix, method = i)
                df = a.Missing_Data_Imputation(df, method = j)
                df = a.Data_Normalization(df, method = k)
                # Apply log2(x+1) to each element in the DataFrame
                df = df.apply(lambda x: np.log2(x+1))

                df = a.Batch_Correction(df, method = l)
                df, ari = a.Cluster_Analysis(df, savefig = True, savefolder = SavePath, 
                                                savename= 'Clustering_{0}_{1}_{2}_{3}'.format(i, j, k, l))

                
                if (Using_ARI_or_PurityScore == 'PurityScore'):
                    if (Distinguish_Clusters):
                        purity_score, sample_purity, batch_purity, sample_index_of_each_group, Compared_groups_label = Replot_Cluster_Result_From_File_Version2(filepath = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l), 
                                                                                                    Groups = Groups,
                                                                                                    Batches = Batches,
                                                                                                    min_samples = 6,
                                                                                                    group_fillin_color = [[234/255, 184/255, 250/255, 1.0],
                                                                                                                          [144/255, 208/255, 184/255, 1.0]],
                                                                                                    group_edge_color = [[157/255, 48/255, 238/255, 1.0],
                                                                                                                        [53/255, 131/255, 99/255, 1.0]])
                    else:
                        purity_score, sample_purity, batch_purity = Replot_Cluster_Result_From_File(filepath = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l), 
                                                                                                    Groups = Groups,
                                                                                                    Batches = Batches,
                                                                                                    group_fillin_color = [[234/255, 184/255, 250/255, 1.0],
                                                                                                                          [144/255, 208/255, 184/255, 1.0]],
                                                                                                    group_edge_color = [[157/255, 48/255, 238/255, 1.0],
                                                                                                                        [53/255, 131/255, 99/255, 1.0]])


                # If the difference analysis method is 'edgeR-QLF', 'edgeR-LRT', 'DESeq2', reverse the df data to the data before log2 processing
                if (m == 'edgeR-QLF') | (m == 'edgeR-LRT') | (m == 'Limma-voom') | (m == 'limma-voom'):
                    df = df.apply(lambda x: np.power(2, x)-1)
                if (m == 'DESeq2') | (m == 'DESeq2-parametric'):
                    df = df.apply(lambda x: np.power(2, x)-1)
                    if df.values.max() > 10000:
                        pass
                    else:
                        df = df.apply(lambda x: x*10000)

                # Difference Analysis - Part 4
                for Comparison in Compared_groups_label:

                    if (Distinguish_Clusters):
                        a.sample_index_of_each_group = sample_index_of_each_group

                    Up_Num, Down_Num, NoSig_Num, GO_Terms, KEGG_Terms, Reactome_Terms = a.Difference_Analysis_Part_4(df, 
                                                                                method = m,
                                                                                Compared_groups_label = [Comparison],
                                                                                FC = 1.5,
                                                                                pValue = 0.05,
                                                                                up_down_scatter_color = [[157/255, 48/255, 238/255, 1.0],
                                                                                                         [53/255, 131/255, 99/255, 1.0]], 
                                                                                MethodSelection = '{0}_{1}_{2}_{3}_{4}'.format(i, j, k, l, m),
                                                                                savefolder = SavePath)
            

                    list_SR_method.append(i)
                    list_Fill_NaN_method.append(j)
                    list_Normalization.append(k)
                    list_Batch_correction.append(l)
                    list_Difference_analysis.append(m)

                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                        list_Purity_Score.append(purity_score)
                        list_Sample_Purity.append(sample_purity)
                        list_Batch_Purity.append(batch_purity)
                    else:
                        list_ARI.append(ari)

                    list_Comparison.append(Comparison)
                    list_Up_Num.append(Up_Num[0])
                    list_Down_Num.append(Down_Num[0])
                    list_NoSig_Num.append(NoSig_Num[0])

                    list_GO_Terms.append(GO_Terms[0])
                    list_KEGG_Terms.append(KEGG_Terms[0])
                    list_Reactome_Terms.append(Reactome_Terms[0])

                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                        dict_result = {'Sparsity Reduction': list_SR_method,
                                                  'Missing Value Imputation': list_Fill_NaN_method,
                                                  'Normalization': list_Normalization,
                                                  'Batch Correction': list_Batch_correction,
                                                  'Statistical Test': list_Difference_analysis,
                                                  'Purity Score': list_Purity_Score,
                                                  'Sample Purity': list_Sample_Purity,
                                                  'Batch Purity': list_Batch_Purity,
                                                  'Comparison': list_Comparison,
                                                  'Up': list_Up_Num,
                                                  'Down': list_Down_Num,
                                                  'No Sig.': list_NoSig_Num,
                                                  '# GO Terms': list_GO_Terms,
                                                  '# KEGG Terms': list_KEGG_Terms,
                                                  '# Reactome Terms': list_Reactome_Terms}
                    else:
                        dict_result = {'Sparsity Reduction': list_SR_method,
                                                  'Missing Value Imputation': list_Fill_NaN_method,
                                                  'Normalization': list_Normalization,
                                                  'Batch Correction': list_Batch_correction,
                                                  'Statistical Test': list_Difference_analysis,
                                                  'ARI': list_ARI,
                                                  'Comparison': list_Comparison,
                                                  'Up': list_Up_Num,
                                                  'Down': list_Down_Num,
                                                  'No Sig.': list_NoSig_Num,
                                                  '# GO Terms': list_GO_Terms,
                                                  '# KEGG Terms': list_KEGG_Terms,
                                                  '# Reactome Terms': list_Reactome_Terms}


                    df_result = pd.DataFrame(dict_result)
                    df_result.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis.csv', index=False)


        exit()




    a = SCPDA(samples_csv_path = args.SamplesPath,
              composition_csv_path = args.CompositionPath)
    
    if args.Level == 'Protein':
        if (args.Software == 'DIANN') | (args.Software == 'DIA-NN'):
            df_all, dict_species = a.Get_Protein_Groups_Data_From_DIANN_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if args.Software == 'Spectronaut':
            df_all, dict_species = a.Get_Protein_Groups_Data_From_Spectronaut_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if (args.Software == 'Peaks') | (args.Software == 'PEAKS'):
            df_all, dict_species = a.Get_Protein_Groups_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if args.Software == 'MaxQuant':
            df_all, dict_species = a.Get_Protein_Groups_Data_From_MaxQuant_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

    elif args.Level == 'Peptide':
        if (args.Software == 'DIANN') | (args.Software == 'DIA-NN'):
            #df_all, dict_species = a.Get_Peptide_Data_From_DIANN_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)
            df_all, dict_species = a.Get_Peptide_Data_From_DIANN_Main_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if args.Software == 'Spectronaut':
            df_all, dict_species = a.Get_Peptide_Data_From_Spectronaut_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if (args.Software == 'Peaks') | (args.Software == 'PEAKS'):
            df_all, dict_species = a.Get_Peptide_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

        if args.Software == 'MaxQuant':
            df_all, dict_species = a.Get_Peptide_Data_From_MaxQuant_Report(FolderPath = args.FolderPath, FileName = args.FileName, SaveResult = True, SavePath = args.SavePath)

    if args.SoftwareQuantity >= 2:
        print('Software2: ' + args.Software2)
        print('FileName2: ' + args.FileName2)
        if args.Level == 'Protein':
            if (args.Software2 == 'DIANN') | (args.Software2 == 'DIA-NN'):
                df_all2, dict_species2 = a.Get_Protein_Groups_Data_From_DIANN_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if args.Software2 == 'Spectronaut':
                df_all2, dict_species2 = a.Get_Protein_Groups_Data_From_Spectronaut_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if (args.Software2 == 'Peaks') | (args.Software2 == 'PEAKS'):
                df_all2, dict_species2 = a.Get_Protein_Groups_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if args.Software2 == 'MaxQuant':
                df_all2, dict_species2 = a.Get_Protein_Groups_Data_From_MaxQuant_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

        elif args.Level == 'Peptide':
            if (args.Software2 == 'DIANN') | (args.Software2 == 'DIA-NN'):
                #df_all2, dict_species2 = a.Get_Peptide_Data_From_DIANN_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)
                df_all2, dict_species2 = a.Get_Peptide_Data_From_DIANN_Main_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if args.Software2 == 'Spectronaut':
                df_all2, dict_species2 = a.Get_Peptide_Data_From_Spectronaut_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if (args.Software2 == 'Peaks') | (args.Software2 == 'PEAKS'):
                df_all2, dict_species2 = a.Get_Peptide_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

            if args.Software2 == 'MaxQuant':
                df_all2, dict_species2 = a.Get_Peptide_Data_From_MaxQuant_Report(FolderPath = args.FolderPath2, FileName = args.FileName2, SaveResult = False)

    if args.SoftwareQuantity >= 3:
        print('Software3: ' + args.Software3)
        print('FileName3: ' + args.FileName3)
        if args.Level == 'Protein':
            if (args.Software3 == 'DIANN') | (args.Software3 == 'DIA-NN'):
                df_all3, dict_species3 = a.Get_Protein_Groups_Data_From_DIANN_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if args.Software3 == 'Spectronaut':
                df_all3, dict_species3 = a.Get_Protein_Groups_Data_From_Spectronaut_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if (args.Software3 == 'Peaks') | (args.Software3 == 'PEAKS'):
                df_all3, dict_species3 = a.Get_Protein_Groups_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if args.Software3 == 'MaxQuant':
                df_all3, dict_species3 = a.Get_Protein_Groups_Data_From_MaxQuant_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

        elif args.Level == 'Peptide':
            if (args.Software3 == 'DIANN') | (args.Software3 == 'DIA-NN'):
                #df_all3, dict_species3 = a.Get_Peptide_Data_From_DIANN_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)
                df_all3, dict_species3 = a.Get_Peptide_Data_From_DIANN_Main_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if args.Software3 == 'Spectronaut':
                df_all3, dict_species3 = a.Get_Peptide_Data_From_Spectronaut_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if (args.Software3 == 'Peaks') | (args.Software3 == 'PEAKS'):
                df_all3, dict_species3 = a.Get_Peptide_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

            if args.Software3 == 'MaxQuant':
                df_all3, dict_species3 = a.Get_Peptide_Data_From_MaxQuant_Report(FolderPath = args.FolderPath3, FileName = args.FileName3, SaveResult = False)

    if args.SoftwareQuantity >= 4:
        print('Software4: ' + args.Software4)
        print('FileName4: ' + args.FileName4)
        if args.Level == 'Protein':
            if (args.Software4 == 'DIANN') | (args.Software4 == 'DIA-NN'):
                df_all4, dict_species4 = a.Get_Protein_Groups_Data_From_DIANN_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if args.Software4 == 'Spectronaut':
                df_all4, dict_species4 = a.Get_Protein_Groups_Data_From_Spectronaut_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if (args.Software4 == 'Peaks') | (args.Software4 == 'PEAKS'):
                df_all4, dict_species4 = a.Get_Protein_Groups_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if args.Software4 == 'MaxQuant':
                df_all4, dict_species4 = a.Get_Protein_Groups_Data_From_MaxQuant_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

        elif args.Level == 'Peptide':
            if (args.Software4 == 'DIANN') | (args.Software4 == 'DIA-NN'):
                #df_all4, dict_species4 = a.Get_Peptide_Data_From_DIANN_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)
                df_all4, dict_species4 = a.Get_Peptide_Data_From_DIANN_Main_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if args.Software4 == 'Spectronaut':
                df_all4, dict_species4 = a.Get_Peptide_Data_From_Spectronaut_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if (args.Software4 == 'Peaks') | (args.Software4 == 'PEAKS'):
                df_all4, dict_species4 = a.Get_Peptide_Data_From_PeaksStudio_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)

            if args.Software4 == 'MaxQuant':
                df_all4, dict_species4 = a.Get_Peptide_Data_From_MaxQuant_Report(FolderPath = args.FolderPath4, FileName = args.FileName4, SaveResult = False)


    


    # Plotting software's identification results
    if args.Task == 'IdentificationResult':

        # python SCPDA.py --Task IdentificationResult --Software DIANN --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230904_0904hela-yeast-ecoli_200pg_test01_DIAnn.pg_matrix.tsv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Samples_Template_DIANN_20230904.csv" --CompositionPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Composition_20230904.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/"


        # Identification Number and Data Completeness
        a.Plot_Identification_Result_of_Each_Sample(
            df_all, 
            software = args.Software, Protein_or_Peptide = args.Level,
            c = ['#8bd2cb', '#68b6fc', '#ff6681', '#7d7d73'],
            savefig = True, 
            savefolder = args.SavePath)

        # Cumulative Identifications
        a.Plot_Cumulative_Identification_Quantity(
            df_all, 
            software = args.Software, 
            Protein_or_Peptide = args.Level,
            label_size = 14,
            savefig = True, 
            savefolder = args.SavePath) 

        # CV Distribution
        a.plot_CV(df_all = df_all, 
                  software = args.Software, Protein_or_Peptide = args.Level, x_lim = 1.1, 
                  linecolor = '#3f6d96', fillin_color = '#cde7fe',
                  savefig = True, 
                  savefolder = args.SavePath)




    # Plotting software's quantitative accuracy results
    if args.Task == 'FoldChange':

        # python SCPDA.py --Task FoldChange --Software DIANN --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230904_0904hela-yeast-ecoli_200pg_test01_DIAnn.pg_matrix.tsv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Samples_Template_DIANN_20230904.csv" --CompositionPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Composition_20230904.csv" --Level Protein --Comparison S1/S3 S2/S3 S4/S3 S5/S3 --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 


        a.Comparison_of_Quantitative_Accuracy_1_Software(dict_species, 
                                                          Compared_groups_label = args.Comparison,
                                          
                                                          Compared_softwares = [args.Software], 
                                                          linecolor = ['#3f6d96', '#689893', '#8b2c3c', '#ff9900', '#996633', '#660066', '#006600', '#ff3300'],
                                                          fillin_color = ['#cde7fe', '#d9f0ee', '#ffcdd5', '#ffff66', '#ffcc99', '#cc00cc', '#00cc00', '#ff9933'],

                                                          Protein_or_Peptide = args.Level,

                                                          scatter_plot_box_position_list = [19, 19, 19, 19],
                                                          scatter_plot__xlim = [[-2, 20], [-2, 20], [-2, 20], [-2, 20]],
                                                          scatter_plot__xticks = [np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7), np.linspace(-2, 16, 7)],
                                                          scatter_plot__xticklabels = [['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16'], ['-2','1','4','7','10', '13', '16']],
                                                          scatter_plot__ylim = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]],
                                                          scatter_plot__yticks = [np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5), np.linspace(-5, 5, 5)],

                                                          savefig = True, 
                                                          savefolder = args.SavePath)



    # Identification error rate
    if args.Task == 'Entrapment':

        # python SCPDA.py --Task Entrapment --Software DIANN --ReportPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example data/20230904_with-single_HYE_libraryfree_DIAnn.pg_matrix.tsv" --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Samples_Template_DIANN_20230904_with_single.csv" --CompositionPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Composition_20230904_with_single.csv" --Level Protein --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/" 

        a.MBR_Error_Rate(dict_species, 
                          software = args.Software, Protein_or_Peptide = args.Level, 
                          savefig = True, 
                          savefolder = args.SavePath)



    # Comparison of results from different software/library construction methods
    if args.Task == 'ResultComparison':

        # python SCPDA.py --Task ResultComparison --SamplesPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Samples_Template_DIANN_20230904.csv" --CompositionPath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Composition_20230904.csv" --Level "Protein" --DatasetName1 "DIAN-NN" --DatasetName2 "Spectronaut" --DatasetName3 "Peaks" --DatasetName4 "MaxQuant" --DatasetFolder1 "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/20230904_DIANN/" --DatasetFolder2 "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/20230904_Spectronaut/" --DatasetFolder3 "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/20230904_Peaks/" --DatasetFolder4 "E:/WJW_Code_Hub/MS_Report_Treat/Paper/Example result/20230904_MaxQuant/" --Comparison S1/S3 S2/S3 S4/S3 S5/S3 --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/"

        dataset_path = []
        fc_path = []
        dataset_name = []

        if args.DatasetFolder1 != './':
            dataset_path.append(args.DatasetFolder1 + args.Level + '_Matrix.csv')
            fc_path.append(args.DatasetFolder1 + 'FoldChange_{0}s.csv'.format(args.Level))
            dataset_name.append(args.DatasetName1)

        if args.DatasetFolder2 != './':
            dataset_path.append(args.DatasetFolder2 + args.Level + '_Matrix.csv')
            fc_path.append(args.DatasetFolder2 + 'FoldChange_{0}s.csv'.format(args.Level))
            dataset_name.append(args.DatasetName2)

        if args.DatasetFolder3 != './':
            dataset_path.append(args.DatasetFolder3 + args.Level + '_Matrix.csv')
            fc_path.append(args.DatasetFolder3 + 'FoldChange_{0}s.csv'.format(args.Level))
            dataset_name.append(args.DatasetName3)

        if args.DatasetFolder4 != './':
            dataset_path.append(args.DatasetFolder4 + args.Level + '_Matrix.csv')
            fc_path.append(args.DatasetFolder4 + 'FoldChange_{0}s.csv'.format(args.Level))
            dataset_name.append(args.DatasetName4)

        if args.DatasetFolder5 != './':
            dataset_path.append(args.DatasetFolder5 + args.Level + '_Matrix.csv')
            fc_path.append(args.DatasetFolder5 + 'FoldChange_{0}s.csv'.format(args.Level))
            dataset_name.append(args.DatasetName5)



        a.ResultComparison(dataset_path = dataset_path, 
                           fc_path = fc_path,
                           dataset_name = dataset_name,
                           Protein_or_Peptide = args.Level,
                           Compared_groups_label = args.Comparison,
                           savefolder = args.SavePath)




    # Comparison of performance metrics for different analytical method combinations
    if args.Task == 'MethodSelectionAutoFC':

        matplotlib.use('Agg')

        
        # python SCPDA.py --Task MethodSelectionAutoFC --Reduction PCA --ClusteringEvaluation ARI --Software DIANN --ReportPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\three_mix_report.pg_matrix.tsv" --SamplesPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Samples_Template.csv" --CompositionPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Composition_Template.csv" --Comparison S4/S2 S5/S1 --OutputMethodSelectionFigures False --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/"


        SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
        Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
        Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
        Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
        Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']

        

        # >>> User input parameters
        # Dimensionality reduction method
        Reduction = args.Reduction
        if (Reduction == 'pca') | (Reduction == 'PCA'):
            additional_plot_methods.Reduction = 'pca'
        if (Reduction == 'umap') | (Reduction == 'UMAP'):
            additional_plot_methods.Reduction = 'umap'

        print('Reduction: {0}'.format(str(Reduction)))

        # Whether to use covariates when correcting for batch effects
        UseCovariates = None
        User_Selection = args.UseCovariates
        if (User_Selection == 'True') | (User_Selection == 'TRUE'):
            UseCovariates = True
        else:
            UseCovariates = False
        print('Use Covariates: {0}'.format(UseCovariates))


        # Use ARI or Purity Score for clustering
        Using_ARI_or_PurityScore = args.ClusteringEvaluation 
        print('Use ARI or PurityScore: {0}'.format(Using_ARI_or_PurityScore))


        # Whether to use the given FC and P value
        a.Use_Given_PValue_and_FC = False
        a.Use_PValue_List = False  


        # Choose whether to export plots and CSV
        Output_Figure_And_CSV_For_MethodSelection = None
        User_Selection = args.OutputMethodSelectionFigures
        if (User_Selection == 'True') | (User_Selection == 'TRUE'):
            Output_Figure_And_CSV_For_MethodSelection = True
        else:
            Output_Figure_And_CSV_For_MethodSelection = False
        print('Output_Figure_And_CSV_For_MethodSelection: {0}'.format(str(Output_Figure_And_CSV_For_MethodSelection)))
        time.sleep(3)

        # User input parameters
        Compared_groups_label = args.Comparison
        SavePath = args.SavePath

        list_No = [] 
        list_SR_method = [] 
        list_PG_num_after_SR = [] 
        list_Fill_NaN_method = [] 
        list_Normalization = [] 
        list_Batch_correction = [] 
        list_Difference_analysis = [] 

        # Adjusted Rand coefficient
        list_ari = [] 

        list_PurityScore = []
        list_SamplePurity = []
        list_BatchPurity = []

        # Average value of indicators
        list_Average_pAUC = []
        list_Average_Accuracy = []
        list_Average_Precision = []
        list_Average_Recall = []
        list_Average_F1Score = []


        # Difference analysis result indicators
        list_pauc = []
        list_pvalue = [] 
        list_log2fc = []


        list_TP = []
        list_TN = []
        list_FP = []
        list_FN = []
        list_accuracy = [] 
        list_precision = [] 
        list_recall = [] 
        list_f1_score = [] 

        list_compared_group_rank_value = [] 
        list_total_rank_value = [] 


        for i in range(len(Compared_groups_label)):
            list_pauc.append([])
            list_pvalue.append([])
            list_log2fc.append([])

            list_TP.append([])
            list_TN.append([])
            list_FP.append([])
            list_FN.append([])
            list_accuracy.append([])
            list_precision.append([])
            list_recall.append([])
            list_f1_score.append([])

            list_compared_group_rank_value.append([])


        count = 0
        for i in SR_methods:
            for j in Fill_NaN_methods:
                for k in Normalization:
                    for l in Batch_correction:

                        for m in Difference_analysis:

                            #if count >= 0:  # To continue running after interruption, enter No at the last end position

                                print('\n--- Differential Expression Analysis: No ' + str(count+1) + ' ---\n')

                                print("User Notice:\n(1) Do not open the csv result file directly before the program runs to the end, otherwise the file will be occupied and the program will not be able to write the results.\n(2) If you need to view the csv result file, please copy the file and view the copy.\n")


                                df = a.Sparsity_Reduction(df_all, method = i)
                                df = a.Missing_Data_Imputation(df, method = j)
                                df = a.Data_Normalization(df, method = k)
                                # Apply log2(x+1) to each element in the DataFrame
                                df = df.apply(lambda x: np.log2(x+1))


                                df = a.Batch_Correction(df, method = l, UseCovariates = UseCovariates)
                                df, ari = a.Cluster_Analysis(df, savefig = Output_Figure_And_CSV_For_MethodSelection, savefolder = SavePath, 
                                                             savename= 'Clustering_{0}_{1}_{2}_{3}'.format(i, j, k, l))

                                # Whether to calculate the Purity Score - Part 3 of this article
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    purity_score, sample_purity, batch_purity = Replot_Cluster_Result_From_File(filepath = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l), 
                                                                                        Groups = a.group_name,
                                                                                        Batches = a.batch_name,
                                                                                        group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
                                                                                                              [51/255, 157/255, 255/255, 1.0]],
                                                                                        group_edge_color = [[139/255, 44/255, 60/255, 1.0],
                                                                                                            [63/255, 109/255, 150/255, 1.0]],
                                                                                        savefig = Output_Figure_And_CSV_For_MethodSelection)
                                
                                    list_PurityScore.append(purity_score)
                                    list_SamplePurity.append(sample_purity)
                                    list_BatchPurity.append(batch_purity)
                                
                                if (Output_Figure_And_CSV_For_MethodSelection == False):
                                    filepath_to_delete = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l)
                                    if os.path.exists(filepath_to_delete):
                                        os.remove(filepath_to_delete)


                                # If the difference analysis method is 'edgeR-QLF', 'edgeR-LRT', 'DESeq2', reverse the df data to the data before log2 processing
                                if (m == 'edgeR-QLF') | (m == 'edgeR-LRT') | (m == 'Limma-voom') | (m == 'limma-voom'):
                                    df = df.apply(lambda x: np.power(2, x)-1)
                                if (m == 'DESeq2') | (m == 'DESeq2-parametric'):
                                    df = df.apply(lambda x: np.power(2, x)-1)
                                    if df.values.max() > 10000:
                                        pass
                                    else:
                                        df = df.apply(lambda x: x*10000)


                                list_pauc_, list_pvalue_, list_log2fc_, list_TP_, list_TN_, list_FP_, list_FN_, list_accuracy_, list_precision_, list_recall_, list_f1_score_, df_list, list_overall_label_true_data_, list_overall_label_predict_data_ = a.Difference_Analysis(
                                                       df, dict_species, method = m,
                                                       Compared_groups_label = Compared_groups_label,
                                                       title_methods = '{0}_{1}_{2}_{3}_{4}'.format(i, j, k, l, m),
                                                       savefig = Output_Figure_And_CSV_For_MethodSelection,
                                                       savefolder = SavePath)

                                
                                list_No.append(count + 1)
                                list_SR_method.append(i)
                                list_Fill_NaN_method.append(j)
                                list_Normalization.append(k)
                                list_Batch_correction.append(l)
                                list_Difference_analysis.append(m)

                                list_PG_num_after_SR.append(a.PG_num_SR)
                                list_ari.append(ari)

                                for index in range(len(Compared_groups_label)):
                                    list_pauc[index].append(list_pauc_[index])
                                    list_pvalue[index].append(list_pvalue_[index])
                                    list_log2fc[index].append(list_log2fc_[index])

                                    list_TP[index].append(list_TP_[index])
                                    list_TN[index].append(list_TN_[index])
                                    list_FP[index].append(list_FP_[index])
                                    list_FN[index].append(list_FN_[index])

                                    list_accuracy[index].append(list_accuracy_[index])
                                    list_precision[index].append(list_precision_[index])
                                    list_recall[index].append(list_recall_[index])
                                    list_f1_score[index].append(list_f1_score_[index])


                                    # For each comparision:  Rank = Rank((Rank(ARI) + Rank(pAUC) + Rank(F1Score)) / 3)

                                    sorted_list_ari = []
                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        sorted_list_ari = sorted(list_ari)
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        sorted_list_ari = sorted(list_PurityScore)
                                    sorted_list_pauc = sorted(list_pauc[index])
                                    sorted_list_f1_score = sorted(list_f1_score[index])

                                    list_compared_group_rank_value[index] = []
                                    for row in range(len(list_ari)):
                                        ARI_rank = 0
                                        if (Using_ARI_or_PurityScore == 'ARI'):
                                            ARI_rank = sorted_list_ari.index(list_ari[row])
                                        if (Using_ARI_or_PurityScore == 'PurityScore'):
                                            ARI_rank = sorted_list_ari.index(list_PurityScore[row])
                                        pAUC_rank = sorted_list_pauc.index(list_pauc[index][row])
                                        F1Score_rank = sorted_list_f1_score.index(list_f1_score[index][row])

                                        list_compared_group_rank_value[index].append((ARI_rank + pAUC_rank + F1Score_rank)/3)


                                df_info = pd.DataFrame()
                                df_info['No'] = list_No
                                df_info['Sparsity Reduction'] = list_SR_method
                                df_info['Missing Value Imputation'] = list_Fill_NaN_method
                                df_info['Normalization'] = list_Normalization
                                df_info['Batch Correction'] = list_Batch_correction
                                df_info['Statistical Test'] = list_Difference_analysis
                        
                                if (Using_ARI_or_PurityScore == 'ARI'):
                                    df_info['ARI'] = list_ari 
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    df_info['Purity Score'] = list_PurityScore
                                    df_info['Sample Purity'] = list_SamplePurity
                                    df_info['Batch Purity'] = list_BatchPurity
                        

                                # Average pAUC, Accuracy, Precision, Recall, F1-Score
                                list_Average_pAUC.append(sum(list_pauc_)/len(list_pauc_))
                                df_info['Average pAUC'] = list_Average_pAUC 

                                list_Average_Accuracy.append(sum(list_accuracy_)/len(list_accuracy_))
                                df_info['Average Accuracy'] = list_Average_Accuracy 

                                list_Average_Precision.append(sum(list_precision_)/len(list_precision_))
                                df_info['Average Precision'] = list_Average_Precision 

                                list_Average_Recall.append(sum(list_recall_)/len(list_recall_))
                                df_info['Average Recall'] = list_Average_Recall 

                                list_Average_F1Score.append(sum(list_f1_score_)/len(list_f1_score_))
                                df_info['Average F1-Score'] = list_Average_F1Score 

                                # Overall Rank
                                sorted_list_ari = []
                                if (Using_ARI_or_PurityScore == 'ARI'):
                                    sorted_list_ari = sorted(list_ari)
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    sorted_list_ari = sorted(list_PurityScore)

                                sorted_list_pauc = sorted(list_Average_pAUC)
                                sorted_list_f1_score = sorted(list_Average_F1Score)

                                list_total_rank_value = []
                                for row in range(df_info.shape[0]):
                                    ARI_rank = 0
                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        ARI_rank = sorted_list_ari.index(list_ari[row])
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        ARI_rank = sorted_list_ari.index(list_PurityScore[row])

                                    pAUC_rank = sorted_list_pauc.index(list_Average_pAUC[row])
                                    F1Score_rank = sorted_list_f1_score.index(list_Average_F1Score[row])

                                    list_total_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)

                                

                                for compared_group in Compared_groups_label:

                                    # Each comparision rank
                                    Group_rank_list = []
                                    for row in range(df_info.shape[0]):
                                        Group_rank_value = list_compared_group_rank_value[Compared_groups_label.index(compared_group)]
                                        Sorted_group_rank_value = sorted(Group_rank_value, reverse=True)
                                        Group_rank_list.append(Sorted_group_rank_value.index(list_compared_group_rank_value[Compared_groups_label.index(compared_group)][row]) + 1)


                                    df_info['{0} Rank'.format(compared_group)] = Group_rank_list

                                    df_info['{0} pAUC'.format(compared_group)] = list_pauc[Compared_groups_label.index(compared_group)]
                                    df_info['{0} p-value'.format(compared_group)] = list_pvalue[Compared_groups_label.index(compared_group)]
                                    df_info['{0} log2FC'.format(compared_group)] = list_log2fc[Compared_groups_label.index(compared_group)]

                                    df_info['{0} TP'.format(compared_group)] = list_TP[Compared_groups_label.index(compared_group)]
                                    df_info['{0} TN'.format(compared_group)] = list_TN[Compared_groups_label.index(compared_group)]
                                    df_info['{0} FP'.format(compared_group)] = list_FP[Compared_groups_label.index(compared_group)]
                                    df_info['{0} FN'.format(compared_group)] = list_FN[Compared_groups_label.index(compared_group)]

                                    df_info['{0} Accuracy'.format(compared_group)] = list_accuracy[Compared_groups_label.index(compared_group)]
                                    df_info['{0} Precision'.format(compared_group)] = list_precision[Compared_groups_label.index(compared_group)]
                                    df_info['{0} Recall'.format(compared_group)] = list_recall[Compared_groups_label.index(compared_group)]
                                    df_info['{0} F1-Score'.format(compared_group)] = list_f1_score[Compared_groups_label.index(compared_group)]


                                # Overall Rank
                                Rank_list = []
                                Sorted_rank_value = sorted(list_total_rank_value, reverse=True)  # Sort from largest to smallest

                                for row in range(df_info.shape[0]):
                                    Rank_list.append(Sorted_rank_value.index(list_total_rank_value[row]) + 1)

                                if (Using_ARI_or_PurityScore == 'ARI'):
                                    df_info.insert(loc=df_info.columns.get_loc('ARI'), column='Rank', value=Rank_list) 
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    df_info.insert(loc=df_info.columns.get_loc('Purity Score'), column='Rank', value=Rank_list) 

                                # Sort by Rank from small to large
                                df_sorted = df_info.sort_values(by='Rank', ascending=True)

                                if (Using_ARI_or_PurityScore == 'ARI'):
                                    df_sorted.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis.csv', index=False)
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    df_sorted.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis - PurityScore.csv', index=False)

                                count+=1



    # Comparison of performance metrics for different analytical method combinations
    if args.Task == 'MethodSelection':

        matplotlib.use('Agg')

        # UseCov
        # python SCPDA.py --Task MethodSelection --Type UseCov --Reduction PCA --ClusteringEvaluation ARI --Software DIANN --ReportPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\three_mix_report.pg_matrix.tsv" --SamplesPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Samples_Template.csv" --CompositionPath "E:\WJW_Code_Hub\SCPDA\Report_Results_Part2\DIANN_QC_3Mix_UsePCA_UseCovariates\Composition_Template.csv" --Comparison S4/S2 S5/S1 --ComparisonFC 1.2 1.3 --OutputMethodSelectionFigures False --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/" 


        SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
        Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
        Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
        Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
        Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']


        ## Analysis of differences between batches
        #Compared_batches_label = ['Batch1/Batch2', 'Batch2/Batch3', 'Batch1/Batch3']



        # >>> User input parameters
        # Dimensionality reduction method
        Reduction = args.Reduction
        if (Reduction == 'pca') | (Reduction == 'PCA'):
            additional_plot_methods.Reduction = 'pca'
        if (Reduction == 'umap') | (Reduction == 'UMAP'):
            additional_plot_methods.Reduction = 'umap'
        print('Reduction: {0}'.format(str(Reduction)))



        # use ARI or Purity Score for clustering
        Using_ARI_or_PurityScore = args.ClusteringEvaluation 
        print('Use ARI or PurityScore: {0}'.format(Using_ARI_or_PurityScore))

        ComparisonFC = args.ComparisonFC
        FC_For_Groups = []
        for FC in ComparisonFC:
            FC_For_Groups.append(float(FC))

        UseCovariates = None
        # Task Type: UseCov, NoCov, KeepNA, AutoFC
        MethodSelectionTask = args.Type
        if (MethodSelectionTask == 'UseCov'):
            UseCovariates = True
            a.Use_Given_PValue_and_FC = True
            a.Use_PValue_List = True  

            a.PValue_List = [0.001, 0.01, 0.05, 0.1]
            a.FC_For_Groups = FC_For_Groups 

        if (MethodSelectionTask == 'NoCov'):
            UseCovariates = False
            a.Use_Given_PValue_and_FC = True
            a.Use_PValue_List = True 

            a.PValue_List = [0.001, 0.01, 0.05, 0.1]
            a.FC_For_Groups = FC_For_Groups 

        if (MethodSelectionTask == 'KeepNA'):
            UseCovariates = True
            a.Use_Given_PValue_and_FC = True
            a.Use_PValue_List = True 

            a.PValue_List = [0.001, 0.01, 0.05, 0.1]
            a.FC_For_Groups = FC_For_Groups 

            # Only use method combinations that support missing values
            SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
            Fill_NaN_methods = ['KeepNA']
            Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
            Batch_correction = ['NoBC', 'limma']
            Difference_analysis = ['t-test', 'Wilcox', 'limma-trend']




        # Choose whether to export plots and CSV
        Output_Figure_And_CSV_For_MethodSelection = False
        User_Selection = args.OutputMethodSelectionFigures
        if (User_Selection == 'True') | (User_Selection == 'TRUE'):
            Output_Figure_And_CSV_For_MethodSelection = True
        else:
            Output_Figure_And_CSV_For_MethodSelection = False

        # User input parameters
        Compared_groups_label = args.Comparison
        SavePath = args.SavePath

        list_No = [] 
        list_SR_method = [] 
        list_PG_num_after_SR = [] 
        list_Fill_NaN_method = [] 
        list_Normalization = [] 
        list_Batch_correction = [] 
        list_Difference_analysis = [] 

        # Adjusted Rand coefficient
        list_ari = [] 

        list_PurityScore = []
        list_SamplePurity = []
        list_BatchPurity = []

        # Average value of indicators
        list_Average_pAUC = []
        list_Average_Accuracy = []
        list_Average_Precision = []
        list_Average_Recall = []
        list_Average_F1Score = []


        # Difference analysis result indicators
        list_pauc = []
        list_pvalue = [] 
        list_log2fc = []


        list_TP = []
        list_TN = []
        list_FP = []
        list_FN = []
        list_accuracy = [] 
        list_precision = [] 
        list_recall = [] 
        list_f1_score = [] 

        list_compared_group_rank_value = [] 
        list_total_rank_value = [] 


        for i in range(len(Compared_groups_label)):
            list_pauc.append([])
            list_pvalue.append([])
            list_log2fc.append([])

            list_TP.append([])
            list_TN.append([])
            list_FP.append([])
            list_FN.append([])
            list_accuracy.append([])
            list_precision.append([])
            list_recall.append([])
            list_f1_score.append([])

            list_compared_group_rank_value.append([])

        ## Used to save the number of up- and down-regulated proteins for differential analysis between batches
        #list_compared_batches_up_num = []
        #list_compared_batches_down_num = []

        #for i in range(len(Compared_batches_label)):
        #    list_compared_batches_up_num.append([])
        #    list_compared_batches_down_num.append([])


        count = 0
        for i in SR_methods:
            for j in Fill_NaN_methods:
                for k in Normalization:
                    for l in Batch_correction:

                        for m in Difference_analysis:


                            #if count >= 0:  # To continue running after interruption, enter No at the last end position

                                print('\n--- Differential Expression Analysis: No ' + str(count+1) + ' ---\n')

                                print("User Notice:\n(1) Do not open the csv result file directly before the program runs to the end, otherwise the file will be occupied and the program will not be able to write the results.\n(2) If you need to view the csv result file, please copy the file and view the copy.\n")


                                df = a.Sparsity_Reduction(df_all, method = i)
                                df = a.Missing_Data_Imputation(df, method = j)
                                df = a.Data_Normalization(df, method = k)
                                # Apply log2(x+1) to each element in the DataFrame
                                df = df.apply(lambda x: np.log2(x+1))


                                df = a.Batch_Correction(df, method = l, UseCovariates = UseCovariates)
                                df, ari = a.Cluster_Analysis(df, savefig = Output_Figure_And_CSV_For_MethodSelection, savefolder = SavePath, 
                                                                savename= 'Clustering_{0}_{1}_{2}_{3}'.format(i, j, k, l))

                                # Whether to calculate the Purity Score - Part 3 of this article
                                if (Using_ARI_or_PurityScore == 'PurityScore'):
                                    purity_score, sample_purity, batch_purity = Replot_Cluster_Result_From_File(filepath = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l), 
                                                                                        Groups = a.group_name,
                                                                                        Batches = a.batch_name,
                                                                                        group_fillin_color = [[255/255, 102/255, 128/255, 1.0],
                                                                                                                [51/255, 157/255, 255/255, 1.0]],
                                                                                        group_edge_color = [[139/255, 44/255, 60/255, 1.0],
                                                                                                            [63/255, 109/255, 150/255, 1.0]],
                                                                                        savefig = Output_Figure_And_CSV_For_MethodSelection)
                                
                                    list_PurityScore.append(purity_score)
                                    list_SamplePurity.append(sample_purity)
                                    list_BatchPurity.append(batch_purity)
                                
                                if (Output_Figure_And_CSV_For_MethodSelection == False):
                                    filepath_to_delete = SavePath + 'Clustering_{0}_{1}_{2}_{3}.csv'.format(i, j, k, l)
                                    if os.path.exists(filepath_to_delete):
                                        os.remove(filepath_to_delete)


                                # If the difference analysis method is 'edgeR-QLF', 'edgeR-LRT', 'DESeq2', reverse the df data to the data before log2 processing
                                if (m == 'edgeR-QLF') | (m == 'edgeR-LRT') | (m == 'Limma-voom') | (m == 'limma-voom'):
                                    df = df.apply(lambda x: np.power(2, x)-1)
                                if (m == 'DESeq2') | (m == 'DESeq2-parametric'):
                                    df = df.apply(lambda x: np.power(2, x)-1)
                                    if df.values.max() > 10000:
                                        pass
                                    else:
                                        df = df.apply(lambda x: x*10000)


                                a.FilterProteinMatrix = True  #Screen the protein expression matrix so that the comparison group contains at least 2 valid data

                                # Using a list of p-values
                                # The return value format is: [[Comparison group 1 indicators: p1, p2, p3, p4], [Comparison group 2 indicators: p1, p2, p3, p4]]
                                list_pauc_, list_pvalue_, list_log2fc_, list_TP_, list_TN_, list_FP_, list_FN_, list_accuracy_, list_precision_, list_recall_, list_f1_score_, df_list, list_overall_label_true_data_, list_overall_label_predict_data_ = a.Difference_Analysis(
                                                        df, dict_species, method = m,
                                                        Compared_groups_label = Compared_groups_label,
                                                        title_methods = '{0}_{1}_{2}_{3}_{4}'.format(i, j, k, l, m),
                                                        savefig = Output_Figure_And_CSV_For_MethodSelection,
                                                        savefolder = SavePath)

                                ## Analysis of differences between batches
                                ## The return value format is: {p1:[Comparison batch 1,Comparison batch 2, Comparison batch 2], p2: ...}
                                #dict_Up_Num, dict_Down_Num = a.Difference_Analysis_For_Batches(df, dict_species,
                                #                                                                method = m,
                                #                                                                Compared_batches_label = Compared_batches_label,
                                #                                                                FC = 1.5,
                                #                                                                pValue_List = a.PValue_List)

                                
                                for p_count in range(len(a.PValue_List)):

                                    p = a.PValue_List[p_count]

                                    list_No.append(count + 1)
                                    list_SR_method.append(i)
                                    list_Fill_NaN_method.append(j)
                                    list_Normalization.append(k)
                                    list_Batch_correction.append(l)
                                    list_Difference_analysis.append(m)

                                    list_PG_num_after_SR.append(a.PG_num_SR)
                                    list_ari.append(ari)

                                    for index in range(len(Compared_groups_label)):
                                        list_pauc[index].append(list_pauc_[index][p_count])
                                        list_pvalue[index].append(list_pvalue_[index][p_count])
                                        list_log2fc[index].append(list_log2fc_[index][p_count])

                                        list_TP[index].append(list_TP_[index][p_count])
                                        list_TN[index].append(list_TN_[index][p_count])
                                        list_FP[index].append(list_FP_[index][p_count])
                                        list_FN[index].append(list_FN_[index][p_count])

                                        list_accuracy[index].append(list_accuracy_[index][p_count])
                                        list_precision[index].append(list_precision_[index][p_count])
                                        list_recall[index].append(list_recall_[index][p_count])
                                        list_f1_score[index].append(list_f1_score_[index][p_count])


                                        # For each comparision:  Rank = Rank((Rank(ARI) + Rank(pAUC) + Rank(F1Score)) / 3)

                                        sorted_list_ari = []
                                        if (Using_ARI_or_PurityScore == 'ARI'):
                                            sorted_list_ari = sorted(list_ari)
                                        if (Using_ARI_or_PurityScore == 'PurityScore'):
                                            sorted_list_ari = sorted(list_PurityScore)
                                        sorted_list_pauc = sorted(list_pauc[index])
                                        sorted_list_f1_score = sorted(list_f1_score[index])

                                        list_compared_group_rank_value[index] = []
                                        for row in range(len(list_ari)):
                                            ARI_rank = 0
                                            if (Using_ARI_or_PurityScore == 'ARI'):
                                                ARI_rank = sorted_list_ari.index(list_ari[row])
                                            if (Using_ARI_or_PurityScore == 'PurityScore'):
                                                ARI_rank = sorted_list_ari.index(list_PurityScore[row])
                                            pAUC_rank = sorted_list_pauc.index(list_pauc[index][row])
                                            F1Score_rank = sorted_list_f1_score.index(list_f1_score[index][row])

                                            list_compared_group_rank_value[index].append((ARI_rank + pAUC_rank + F1Score_rank)/3)


                                    ## Differentially expressed proteins between batches
                                    #for index in range(len(Compared_batches_label)):
                                    #    list_compared_batches_up_num[index].append(dict_Up_Num[str(p)][index])
                                    #    list_compared_batches_down_num[index].append(dict_Down_Num[str(p)][index])



                                    df_info = pd.DataFrame()
                                    df_info['No'] = list_No
                                    df_info['Sparsity Reduction'] = list_SR_method
                                    df_info['Missing Value Imputation'] = list_Fill_NaN_method
                                    df_info['Normalization'] = list_Normalization
                                    df_info['Batch Correction'] = list_Batch_correction
                                    df_info['Statistical Test'] = list_Difference_analysis
                        
                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        df_info['ARI'] = list_ari 
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        df_info['Purity Score'] = list_PurityScore
                                        df_info['Sample Purity'] = list_SamplePurity
                                        df_info['Batch Purity'] = list_BatchPurity
                        

                                    # Average pAUC, Accuracy, Precision, Recall, F1-Score
                                    temp_list = []
                                    for data in list_pauc_:
                                        temp_list.append(data[p_count])

                                    list_Average_pAUC.append(sum(temp_list)/len(temp_list))
                                    df_info['Average pAUC'] = list_Average_pAUC 


                                    temp_list = []
                                    for data in list_accuracy_:
                                        temp_list.append(data[p_count])

                                    list_Average_Accuracy.append(sum(temp_list)/len(temp_list))
                                    df_info['Average Accuracy'] = list_Average_Accuracy 


                                    temp_list = []
                                    for data in list_precision_:
                                        temp_list.append(data[p_count])

                                    list_Average_Precision.append(sum(temp_list)/len(temp_list))
                                    df_info['Average Precision'] = list_Average_Precision 


                                    temp_list = []
                                    for data in list_recall_:
                                        temp_list.append(data[p_count])

                                    list_Average_Recall.append(sum(temp_list)/len(temp_list))
                                    df_info['Average Recall'] = list_Average_Recall 


                                    temp_list = []
                                    for data in list_f1_score_:
                                        temp_list.append(data[p_count])

                                    list_Average_F1Score.append(sum(temp_list)/len(temp_list))
                                    df_info['Average F1-Score'] = list_Average_F1Score 

                                    # Overall Rank
                                    sorted_list_ari = []
                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        sorted_list_ari = sorted(list_ari)
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        sorted_list_ari = sorted(list_PurityScore)

                                    sorted_list_pauc = sorted(list_Average_pAUC)
                                    sorted_list_f1_score = sorted(list_Average_F1Score)

                                    list_total_rank_value = []
                                    for row in range(df_info.shape[0]):
                                        ARI_rank = 0
                                        if (Using_ARI_or_PurityScore == 'ARI'):
                                            ARI_rank = sorted_list_ari.index(list_ari[row])
                                        if (Using_ARI_or_PurityScore == 'PurityScore'):
                                            ARI_rank = sorted_list_ari.index(list_PurityScore[row])

                                        pAUC_rank = sorted_list_pauc.index(list_Average_pAUC[row])
                                        F1Score_rank = sorted_list_f1_score.index(list_Average_F1Score[row])

                                        list_total_rank_value.append((ARI_rank + pAUC_rank + F1Score_rank)/3)

                                

                                    for compared_group in Compared_groups_label:

                                        # Each comparision rank
                                        Group_rank_list = []
                                        for row in range(df_info.shape[0]):
                                            Group_rank_value = list_compared_group_rank_value[Compared_groups_label.index(compared_group)]
                                            Sorted_group_rank_value = sorted(Group_rank_value, reverse=True)
                                            Group_rank_list.append(Sorted_group_rank_value.index(list_compared_group_rank_value[Compared_groups_label.index(compared_group)][row]) + 1)


                                        df_info['{0} Rank'.format(compared_group)] = Group_rank_list

                                        df_info['{0} pAUC'.format(compared_group)] = list_pauc[Compared_groups_label.index(compared_group)]
                                        df_info['{0} p-value'.format(compared_group)] = list_pvalue[Compared_groups_label.index(compared_group)]
                                        df_info['{0} log2FC'.format(compared_group)] = list_log2fc[Compared_groups_label.index(compared_group)]

                                        df_info['{0} TP'.format(compared_group)] = list_TP[Compared_groups_label.index(compared_group)]
                                        df_info['{0} TN'.format(compared_group)] = list_TN[Compared_groups_label.index(compared_group)]
                                        df_info['{0} FP'.format(compared_group)] = list_FP[Compared_groups_label.index(compared_group)]
                                        df_info['{0} FN'.format(compared_group)] = list_FN[Compared_groups_label.index(compared_group)]

                                        df_info['{0} Accuracy'.format(compared_group)] = list_accuracy[Compared_groups_label.index(compared_group)]
                                        df_info['{0} Precision'.format(compared_group)] = list_precision[Compared_groups_label.index(compared_group)]
                                        df_info['{0} Recall'.format(compared_group)] = list_recall[Compared_groups_label.index(compared_group)]
                                        df_info['{0} F1-Score'.format(compared_group)] = list_f1_score[Compared_groups_label.index(compared_group)]


                                    ## Differentially expressed proteins between batches
                                    #for index in range(len(Compared_batches_label)):
                                    #    df_info[Compared_batches_label[index] + ' Up'] = list_compared_batches_up_num[index][p_count]
                                    #    df_info[Compared_batches_label[index] + ' Down'] = list_compared_batches_down_num[index][p_count]

                                        


                                    # Overall Rank
                                    Rank_list = []
                                    Sorted_rank_value = sorted(list_total_rank_value, reverse=True)  # Sort from largest to smallest

                                    for row in range(df_info.shape[0]):
                                        Rank_list.append(Sorted_rank_value.index(list_total_rank_value[row]) + 1)

                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        df_info.insert(loc=df_info.columns.get_loc('ARI'), column='Rank', value=Rank_list) 
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        df_info.insert(loc=df_info.columns.get_loc('Purity Score'), column='Rank', value=Rank_list) 

                                    # Sort by Rank from small to large
                                    df_sorted = df_info.sort_values(by='Rank', ascending=True)

                                    if (Using_ARI_or_PurityScore == 'ARI'):
                                        df_sorted.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis.csv', index=False)
                                    if (Using_ARI_or_PurityScore == 'PurityScore'):
                                        df_sorted.to_csv(SavePath + 'MethodSelection_DifferentialExpressionAnalysis - PurityScore.csv', index=False)

                                    count+=1


    # Table conversion: The sorting method of the original table with 4 p-values ​​is converted to TotalRank AverageRank
    if args.Task == 'MethodSelectionTableConversion':
        
        # python SCPDA.py --Task MethodSelectionTableConversion --ResultCSVPath "E:\WJW_Code_Hub\SCPDA\Test\TestCMD\MethodSelection_DifferentialExpressionAnalysis.csv" --Comparison S4/S2 S5/S1 --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/"

        Compared_groups_label = args.Comparison
        SavePath = args.SavePath
        result_csv_path = args.ResultCSVPath

        MethodSelectionTableConversion(csv_path = result_csv_path,
                                       SavePath = SavePath,
                                       Compared_groups_label = Compared_groups_label)


    # When the user manually merges two tables (such as merging the KeepNA table with the UseCov table) and wants to re-rank it 
    if args.Task == 'MethodSelectionTableRerank':
        
        # python SCPDA.py --Task MethodSelectionTableRerank --ResultCSVPath "E:\WJW_Code_Hub\SCPDA\Test\TestCMD\MethodSelection_DifferentialExpressionAnalysis.csv" --Comparison S4/S2 S5/S1 --SavePath "E:/WJW_Code_Hub/SCPDA/Test/TestCMD/"

        Compared_groups_label = args.Comparison
        SavePath = args.SavePath
        result_csv_path = args.ResultCSVPath

        RerankMethodSelectionTable(csv_path = result_csv_path,
                                   SavePath = SavePath,
                                   Compared_groups_label = Compared_groups_label)
    




    ## Plot Metrics Results - Part 2 of the Article
    #if args.Task == 'PlotMetricsResult':

    #    # python SCPDA.py --Task PlotMetricsResult --ResultCSVPath "E:\WJW_Code_Hub\MS_Report_Treat\Paper\Example result\DIANN_3_Batches_Result\MethodSelection_DifferentialExpressionAnalysis_DIA-NN.csv" --Comparison S1/S3 S2/S3 S4/S3 S5/S3 --SavePath "E:/WJW_Code_Hub/MS_Report_Treat/Paper/test_cmd/Metrics_Result/"

    #    matplotlib.use('Agg')

    #    SR_methods = ['NoSR', 'SR66', 'SR75', 'SR90']
    #    Fill_NaN_methods = ['Zero', 'HalfRowMin', 'RowMean', 'RowMedian', 'KNN', 'IterativeSVD', 'SoftImpute']
    #    Normalization = ["Unnormalized", "Median", "Sum", "QN", "TRQN"]
    #    Batch_correction = ['NoBC', 'limma', 'Combat-P', 'Combat-NP', 'Scanorama']
    #    Difference_analysis = ['t-test', 'Wilcox', 'limma-trend', 'limma-voom', 'edgeR-QLF', 'edgeR-LRT', 'DESeq2']


    #    # User input parameters
    #    Compared_groups_label = args.Comparison
    #    SavePath = args.SavePath
    #    result_csv_path = args.ResultCSVPath

    #    a.Use_Given_PValue_and_FC = False
    #    Use_Given_PValue_and_FC = args.Use_Given_PValue_and_FC
    #    if (Use_Given_PValue_and_FC == 'True') | (Use_Given_PValue_and_FC == 'TRUE'):
    #        a.Use_Given_PValue_and_FC = True
    #    else:
    #        a.Use_Given_PValue_and_FC = False

    #    ARI_or_Purity = 'ARI'  # Purity Score  ARI
    #    Using_ARI_or_PurityScore = args.ClusteringEvaluation
    #    if (Using_ARI_or_PurityScore == 'PurityScore'):
    #        ARI_or_Purity = 'Purity Score'

    #    # Plot BatchCorrection_ARI and Imputation_Normalization_ARI graphs
    #    for SR in SR_methods:
    #        a.Plot_BatchCorrection_ARI(result_csv_path = result_csv_path, 
    #                                   SR_method = SR, 
    #                                   Fill_NaN_methods = Fill_NaN_methods,
    #                                   Normalization_methods = Normalization, 
    #                                   BC_methods = Batch_correction,
    #                                   Difference_analysis_methods = Difference_analysis,
    #                                   ARI_or_Purity = ARI_or_Purity,
    #                                   savefig = True, savefolder = SavePath)


    #    # Plot StatisticalTest_pAUC Accuracy and other graphs
    #    for SR in SR_methods:
    #        a.Plot_StatisticalTest_Result(result_csv_path = result_csv_path, 
    #                                       SR_method = SR, 
    #                                       Fill_NaN_methods = Fill_NaN_methods,
    #                                       Normalization_methods = Normalization, 
    #                                       BC_methods = Batch_correction,
    #                                       Difference_analysis_methods = Difference_analysis,
    #                                       Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    #                                       Compared_groups_label = Compared_groups_label,
    #                                       savefig = True, savefolder = SavePath)


    #    # Draw parallel coordinates plot
    #    for SR in SR_methods:
    #        #a.Plot_StatisticalMetrics(result_csv_path = result_csv_path, 
    #        #                               SR_method = SR, 
    #        #                               Fill_NaN_methods = Fill_NaN_methods,
    #        #                               Normalization_methods = Normalization, 
    #        #                               BC_methods = Batch_correction,
    #        #                               Difference_analysis_methods = Difference_analysis,
    #        #                               Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], 
    #        #                               Compared_groups_label = Compared_groups_label,
    #        #                               ARI_or_Purity = ARI_or_Purity,
    #        #                               savefig = True, savefolder = SavePath,
    #        #                               rank_scheme = 1)

    #        a.Plot_StatisticalMetrics(result_csv_path = result_csv_path, 
    #                                       SR_method = SR, 
    #                                       Fill_NaN_methods = Fill_NaN_methods,
    #                                       Normalization_methods = Normalization, 
    #                                       BC_methods = Batch_correction,
    #                                       Difference_analysis_methods = Difference_analysis,
    #                                       Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], 
    #                                       Compared_groups_label = Compared_groups_label,
    #                                       ARI_or_Purity = ARI_or_Purity,
    #                                       savefig = True, savefolder = SavePath,
    #                                       rank_scheme = 2)


    #    ## SparsityReduction_DifferentialProteins
    #    #a.Plot_SparsityReduction_DifferentialProteins(result_csv_path = result_csv_path, 
    #    #                                   SR_methods = SR_methods, 
    #    #                                   Fill_NaN_methods = Fill_NaN_methods,
    #    #                                   Normalization_methods = Normalization, 
    #    #                                   BC_methods = Batch_correction,
    #    #                                   Difference_analysis_methods = Difference_analysis,
    #    #                                   Indicator_type_list = ['pAUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    #    #                                   Compared_groups_label = Compared_groups_label,
    #    #                                   savefig = True, savefolder = SavePath)

    #    ## Spearman correlation coefficient plot
    #    #a.Plot_Spearmanr_Result(result_csv_path = result_csv_path, 
    #    #                      Indicator_type_list = ['Rank', ARI_or_Purity, 'pAUC', 'F1-Score'],
    #    #                      SR_methods = SR_methods,
    #    #                      Compared_groups_label = Compared_groups_label,
    #    #                      ShowValue = False,
    #    #                      savefig = True, savefolder = SavePath)


