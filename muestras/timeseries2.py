# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from matplotlib import dates
import mplfinance as mpf
import seaborn as sns
import statsmodels.api as sm
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import cufflinks as cf


# %%
start = '1-1-2019'
end = '24-4-2021'
df_1 = pdr.DataReader('AMZN', 'yahoo', start=start, end=end)
df_1.head()


# %%
fig_1 = plt.figure(figsize=(12,6), dpi=100)
axes_1 = fig_1.add_axes([0.0, 0.0, 0.9, 0.9])
axes_1.set_xlabel('Date')
axes_1.set_ylabel('Closing')
axes_1.set_title('Matplotlib Plot')
axes_1.plot(df_1.index, df_1['Close'], label='Closing Price')
axes_1.legend(loc=0)
axes_1.grid(True, color='0.6', dashes=(5,2,1,2))

fig_2 = plt.figure(figsize=(12, 6), dpi=100)
axes_2 = sns.lineplot(data=df_1, x=df_1.index, y='Close')
axes_2.set(xlabel='Date', ylabel='Closing Price')
axes_2.set_title('Seaborn Plot')
axes_2.legend(loc=0)
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})
# %%
