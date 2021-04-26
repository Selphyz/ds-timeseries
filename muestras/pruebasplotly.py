# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import cufflinks as cf
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot
init_notebook_mode(connected=True)
cf.go_offline()


# %%
arr_1 = np.random.randn(50, 4)
df_1 = pd.DataFrame(arr_1, columns=['A', 'B', 'C', 'D'])
df_1.head()


# %%
df_1.plot()


# %%
df_1.iplot()


# %%
df_stocks = px.data.stocks()
px.line(df_stocks, x='date', y='GOOG', labels={'x': 'Date', 'y': 'Date'})


# %%
px.line(df_stocks, x='date', y=['GOOG', 'AMZN', 'MSFT', 'FB', 'AAPL'], labels={'x': 'Date', 'y': 'Date'}, title='BigTech Comparison')


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AAPL, mode='lines', name='Apple'))
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AMZN, mode='lines+markers', name='Amazon'))
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.GOOG, mode='lines+markers', name='Google', line=dict(color='firebrick', width=2, dash='dashdot')))
# fig.update_layout(title='Stock 2018-2020', xaxis_title='Price', yaxis_title='Date')
fig.update_layout(xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')))
fig.show()


# %%
df_us = px.data.gapminder().query("country == 'United States'")
px.bar(df_us, x='year', y='pop')


# %%
df_tips = px.data.tips()
px.bar(df_tips, x='sex', y='total_bill', color='smoker', barmode='group')


# %%
df_europe = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df_europe, y='pop', x='country', text='pop', color='country')
fig.show()


# %%
df_iris = px.data.iris()
px.scatter(df_iris, x='sepal_width', y='sepal_length', color='species', size='petal_length', hover_data=['petal_width'])


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_iris.sepal_width, y=df_iris.sepal_length, mode='markers',
marker_color=df_iris.petal_width, text=df_iris.species, marker=dict(showscale=True)))
fig.update_traces(marker_line_width=2, marker_size=10)


# %%
fig = go.Figure(data=go.Scattergl(x=np.random.randn(10000), y=np.random.randn(10000), mode='markers', marker=dict(color=np.random.randn(10000), colorscale='Viridis', line_width=1)))
fig.show()


# %%
df_asia = px.data.gapminder().query("year == 2007").query("continent == 'Asia'").query('pop > 1.e8')
px.pie(df_asia, values='pop', names='country', title='Populatiion of Asian Continent',
color_discrete_sequence=px.colors.sequential.Rainbow)


# %%
flights = sns.load_dataset('flights')
fig = px.density_heatmap(flights, x='year', y='month', z='passengers', color_continuous_scale='Viridis')
fig.show()


# %%
fig = px.line_3d(flights, x='year', y='month', z='passengers', color='year')
fig.show()


# %%
fig = px.scatter_matrix(flights, color='month')
fig.show()


# %%
df=px.data.gapminder().query('year == 2007')
fig = px.scatter_geo(df, locations='iso_alpha', color='continent', hover_name='country', size='pop', projection='orthographic')
fig.show()


# %%
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv',
dtype={"fips": str})
fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp', color_continuous_scale='Viridis', range_color=(0, 12), scope='usa', labels={'unemp': 'unemployement rate'})
fig.show()


# %%
df_wind = px.data.wind()
df_wind.head()


# %%

px.line_polar(df_wind, r='frequency', theta='direction', color='strength', line_close=True, template='plotly_dark')


# %%
df_exp = px.data.experiment()
df_exp.head()


# %%
px.scatter_ternary(df_exp, a='experiment_1', b='experiment_2', c='experiment_3', hover_name='group', color='gender')


# %%
df_tips.head()


# %%
px.scatter(df_tips, x='total_bill', y='tip', color='smoker', facet_col='sex')


# %%
px.histogram(df_tips, x='total_bill', y='tip', color='sex', facet_col='day', facet_row='time', category_orders={'day': ['Thur','Fri','Sat','Sun'],
'time': ['Lunch', 'Dinner']})


# %%
df_att = sns.load_dataset("attention")
fig = px.line(df_att, x='solutions', y='score', facet_col='subject', facet_col_wrap=5, title='Attention Scores')
fig.show()


# %%
df_cnt = px.data.gapminder()
px.scatter(df_cnt, x="gdpPercap", y="lifeExp", animation_frame="year", 
           animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])


# %%
px.bar(df_cnt, x="continent", y="pop", color="continent",
  animation_frame="year", animation_group="country", range_y=[0,4000000000])


# %%



