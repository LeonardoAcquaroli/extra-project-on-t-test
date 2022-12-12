import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats 
import streamlit as st
# from tqdm import tqdm      
# from sampleSplit import *
# from sampleSplit_copy import *
# import seaborn as sns
# import plotly.express as px

#sampleSplit_copy function
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

states = pd.read_excel(r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Unimi\Subjects\Statistics\Extra project\Project 5 - States data\State data.xlsx")
states.columns = ['Acronym', 'Population', 'Income', 'Illiteracy', 'Life.Exp',
       'Murder', 'HS.Grad', 'Frost', 'Area']
states["Density"] = states.Population/states.Area
states = states.sort_values("Population",ascending=False)

def f_test(a,b):
    F = np.var(a,ddof=1)/np.var(b,ddof=1)
    df1 = np.array(a).size - 1
    df2 = np.array(b).size - 1
    if np.var(a,ddof=1) >= np.var(b,ddof=1):
        p_value = 1 - scipy.stats.f.cdf(F,df1,df2)
    else: 
        p_value = scipy.stats.f.cdf(F,df1,df2)
    if p_value >= 0.05:
        return F, p_value, "f_p_value >= 0.05 Variances are not different"
    else:   
        return F, p_value, "f_p_value < 0.05 Variances are different"

variables = ["Income","Illiteracy","Density"]
splitters = ['median','iq_range_mean_point','m','minmax_mean_point']
mainlist = [] # list for the main dataframe of bigdfs

Z = np.zeros((3,4)) # create the df cells to fill them with the tuples of p_values (f,t) by the .loc[variable][splitter]
p_values_df = pd.DataFrame(Z,index=variables, columns=splitters, dtype=object) # zero dfs with variables as index and splitters as columns that will host the tuples of p_values (f,t)
                                                                               # WARNING use dtype = object to avoid the rejection of a tuple as element of the df

# after the for loop we set variables as the index of the pvalues df

for col in variables:
    ss = sampleSplit(df = states, col = col, target="Murder")
    big_df = ss.split()
    mainlist.append(big_df)
    for splitter in splitters:
        a = big_df.loc[splitter][0]
        b = big_df.loc[splitter][1]
        f_p_value = f_test(a,b)[1]
        f_str_outcome = f_test(a,b)[2]
        # fig, ax = plt.subplots(1, ncols=2)
        # a.hist(ax=ax[0],bins=15)
        # b.hist(ax=ax[1],bins=15)
        if f_p_value >= 0.05: # F-TEST --> H0: variances are equal
            print('\n'+f"{col}, {splitter}, var_low: {round(np.var(a),2)}, var_high: {round(np.var(b),2)}, {f_str_outcome}")
            print(scipy.stats.ttest_ind(a,b,equal_var=True))
            t_p_value = scipy.stats.ttest_ind(a,b,equal_var=True)[1]
            p_values_df.loc[col][splitter] = (f_p_value,t_p_value) # add the tuple of p-values (f,t) to the corresponding cell of the p_values_df
            if t_p_value >= 0.05: # t-TEST --> H0: means are equal
                print(f"p_value of the t-test is: {t_p_value} >= 0.05 hence is ACCEPTED the null hypotesis of no statistical difference in the mean murder rate between the low and the high {col} samples divided by the {splitter}")
            else: # t-TEST --> H1: means are not equal
                print(f"p_value of the t-test is: {t_p_value} < 0.05 hence is REJECTED the null hypotesis of no statistical difference in the mean murder rate between the low and the high {col} samples divided by the {splitter}")                
        else: # F-TEST --> H1: variances are not equal
            print('\n'+f"{col}, {splitter}, var_low: {round(np.var(a),2)}, var_high: {round(np.var(b),2)}, {f_str_outcome}")
            print(scipy.stats.ttest_ind(a,b,equal_var=False))
            t_p_value = scipy.stats.ttest_ind(a,b,equal_var=False)[1]
            p_values_df.loc[col][splitter] = (f_p_value,t_p_value) # add the tuple of p-values (f,t) to the corresponding cell of the p_values_df
            if t_p_value >= 0.05: # t-TEST --> H0: means are equal
                print(f"p_value of the t-test is: {t_p_value} >= 0.05 hence is ACCEPTED the null hypotesis of no statistical difference in the mean murder rate between the low and the high {col} samples divided by the {splitter}")
            else: # t-TEST --> H1: means are not equal
                print(f"p_value of the t-test is: {t_p_value} < 0.05 hence is REJECTED the null hypotesis of no statistical difference in the mean murder rate between the low and the high {col} samples divided by the {splitter}")

# If p-value slightly less then 0.05 the level of significance alpha can impact the result

#create main dataframe to plot on streamlit
mainframe = pd.DataFrame()
mainframe["dfs"] = mainlist
mainframe.index = variables

variable = st.sidebar.selectbox('Variable',variables)
splitter = st.sidebar.selectbox('Splitter',splitters)
st.title("Results of the t-test")
st.write(f"{variable} divided by {splitter}",)

corresponding_big_df = mainframe.loc[variable][0]
low_mean = round(corresponding_big_df.loc[splitter][2],2)  # mean of the low variable sample divided by splitter
high_mean = round(corresponding_big_df.loc[splitter][3],2) # mean of the high variable sample divided by splitter
offset = 0.1

fig,ax = plt.subplots(ncols=2)
ax[0].boxplot(corresponding_big_df.loc[splitter][0], showmeans=True, meanline=True)
ax[0].set_ylim(0,16) #15.1 is the max of Murder
ax[0].set_title(f"Murder rate in\nlow {variable}\nsample (n = {len(corresponding_big_df.loc[splitter][0])})")
ax[0].annotate(str(low_mean),xy=(1+offset,low_mean+offset), c='green') # annotate the mean with a bit of offset (to right for the low plot and to left for the high)
ax[1].boxplot(corresponding_big_df.loc[splitter][1], showmeans=True, meanline=True)
ax[1].set_ylim(0,16) #15.1 is the max of Murder
ax[1].set_title(f"Murder rate in\nhigh {variable}\nsample (m = {len(corresponding_big_df.loc[splitter][1])})")
ax[1].annotate(str(high_mean), xy=(1-offset*2.3,high_mean+offset),c='green') # annotate the mean with a bit of offset (to right for the low plot and to left for the high)
fig.tight_layout(pad=3)
st.pyplot(fig)

# Now we print the final results based on the F-test and t-test
correspondent_f_p = p_values_df.loc[variable][splitter][0] # first element of the tuple of p_values (f,t)
correspondent_t_p = p_values_df.loc[variable][splitter][1] # second element of the tuple of p_values (f,t)

if correspondent_f_p >= 0.05: # Student's t-test (Equal var)
    st.write(f"Student's t-test for the equality of the mean Murder rate: p-value = {round(correspondent_t_p,5)}")
    if correspondent_t_p >= 0.05: # equal means
        st.write(f"**ACCEPTED the null hypotesis of non-statistical difference.**")
    else: # different means
        st.write(f"**REJECTED the null hypotesis of non-statistical difference.**")

else: # Welch's t-test (Unequal var)
    st.write(f"Welch's t-test for the equality of the mean Murder rate: p-value = {round(correspondent_t_p,5)}")
    if correspondent_t_p >= 0.05: # equal means
        st.write(f"**ACCEPTED the null hypotesis of non-statistical difference.**")
    else: # different means
        st.write(f"**REJECTED the null hypotesis of non-statistical difference.**")

# Describe all the three variables:
# Income: 
# Illiteracy: We always reject
# Density: Start from minmax (REJECTED) and then proceed to more "central" splitters in which we ACCEPT