import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats

class sampleSplit():

    def __init__(self, df, col: str, target: str):
        # initialize the splitters
        self.median = df[col].median()
        self.iq_range_mean_point = (((df[col].quantile([0.25,0.75])).iloc[0] + (df[col].quantile([0.25,0.75])).iloc[1]))/2
        self.m = df[col].mean()
        self.minmax_mean_point = (df[col].max() + df[col].min())/2
        # initialize the arguments
        self.df = df
        self.col = col
        self.target = target
        
    def split(self):
        # create the splitters
        splitters = [self.median,self.iq_range_mean_point,self.m,self.minmax_mean_point]

        # for loop that populates two lists that will be the column of the resulting df
        # the lists will be composed of the low and high samples divided by the splitters
        l_samples = []
        h_samples = []
        for split in splitters:
            l_sample = self.df[self.target][self.df[self.col]<split]
            h_sample = self.df[self.target][self.df[self.col]>=split]
            l_samples.append(l_sample)
            h_samples.append(h_sample)
        # create the final dataframe and associate each column to the low and high list of dataframes
        samples = pd.DataFrame()
        samples[f"low_{self.col}"]  = l_samples # inserting target where low col values
        samples[f"high_{self.col}"] = h_samples # inserting target where high col values
        # we create the two df of target rate for low and high variable
        samples[f"low_{self.col}_mean"] = [l_samples[i].mean() for i in range(len(splitters))] #inserting target mean on low col
        samples[f"high_{self.col}_mean"] = [h_samples[i].mean() for i in range(len(splitters))]#inserting target mean on high col
        samples[f"low_{self.col}_var"] = [l_samples[i].var() for i in range(len(splitters))]   #inserting target mean on low col
        samples[f"high_{self.col}_var"] = [h_samples[i].var() for i in range(len(splitters))]  #inserting target mean on high col
        samples.index = ["median","iq_range_mean_point","m","minmax_mean_point"]
        return samples