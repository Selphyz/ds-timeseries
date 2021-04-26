# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio as py
import plotly.express as px
import pandas_datareader as pdr


# %%
pd.date_range('Oct 01, 2020', periods=15, freq='D')
pd.to_datetime(['10/01/2020', '10/02/2020'], format='%d/%m/%Y')
arr_1 = np.random.randint(10, 50, size=(3,3))
date_arr = pd.date_range('2020-01-01', periods=3, freq='D')
df_1 = pd.DataFrame(arr_1, columns=['A', 'B', 'C'], index=date_arr)
df_1.index.argmin()


# %%
aapl = pdr.DataReader('AAPL', 'yahoo', start='01/01/2020', end='20/04/2021')
px.line(aapl, y='Close')


# %%
print(aapl['Close'].resample(rule='A').std())


# %%
aapl['Close'].plot(figsize=(12,6))
aapl.rolling(window=30).mean()['Close'].plot()


# %%



