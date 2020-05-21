---
title: "Analysis of Covid-19"
date: 2020-05-1
tags: [data visualisation, modelling, web scraping]
header:
  image: "/images/cdc-w9KEokhajKw-unsplash.jpg"
excerpt: "Web scraping, data visualisation and basic modelling of Covid-19 Cases"
mathjax: "true"
---

```python
import pandas as pd

#98 days of data for each county
df = pd.read_csv("confirmed-covid-19-cases-in-us-by-state-and-county.csv") 
df = df[df.county_name != "Statewide Unallocated"]
df.dropna(inplace = True)
df = df.reset_index()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>county_fips</th>
      <th>county_name</th>
      <th>state_name</th>
      <th>state_fips</th>
      <th>date</th>
      <th>confirmed</th>
      <th>lat</th>
      <th>long</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98</td>
      <td>1001</td>
      <td>Autauga County</td>
      <td>AL</td>
      <td>1</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>32.539527</td>
      <td>-86.644082</td>
      <td>POINT (-86.64408227 32.53952745)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>1001</td>
      <td>Autauga County</td>
      <td>AL</td>
      <td>1</td>
      <td>2020-01-23</td>
      <td>0</td>
      <td>32.539527</td>
      <td>-86.644082</td>
      <td>POINT (-86.64408227 32.53952745)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>1001</td>
      <td>Autauga County</td>
      <td>AL</td>
      <td>1</td>
      <td>2020-01-24</td>
      <td>0</td>
      <td>32.539527</td>
      <td>-86.644082</td>
      <td>POINT (-86.64408227 32.53952745)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101</td>
      <td>1001</td>
      <td>Autauga County</td>
      <td>AL</td>
      <td>1</td>
      <td>2020-01-25</td>
      <td>0</td>
      <td>32.539527</td>
      <td>-86.644082</td>
      <td>POINT (-86.64408227 32.53952745)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>102</td>
      <td>1001</td>
      <td>Autauga County</td>
      <td>AL</td>
      <td>1</td>
      <td>2020-01-26</td>
      <td>0</td>
      <td>32.539527</td>
      <td>-86.644082</td>
      <td>POINT (-86.64408227 32.53952745)</td>
    </tr>
  </tbody>
</table>
</div>



### Plot Total Case Distribution Across Counties by Date


```python
import folium
from folium import plugins

data = df
data = data[["date", "lat", "long", "county_name", "county_fips", "confirmed"]]

####DATE####
date = "2020-04-15"
data = data[data.date == date]
```

Marker Clustering


```python
usa_map = folium.Map(
    location=[data["lat"].mean(), 
    data["long"].mean()], 
    zoom_start=4)

cases = plugins.MarkerCluster().add_to(usa_map)

for lat, long, confirmed in zip(data.lat, data.long, data.confirmed):
    for case in range(0, confirmed):
        folium.Marker(
            location=[lat, long],
            icon=None,
        ).add_to(cases)
    
usa_map
```







Choropleth Map


```python
import branca

colorscale = branca.colormap.linear.YlOrRd_09.scale(0, 100) #Scale 

fips_cases = data.set_index("county_fips")["confirmed"]

def style_function(feature):
    cases = fips_cases.get(int(feature['id'][-5:]), None)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if cases is None else colorscale(cases)
    }

c_map = folium.Map(
    location = [data["lat"].mean(), data["long"].mean()],
    tiles = "cartodbpositron",
    zoom_start = 4
)

folium.TopoJson(
    open('us_counties_20m_topo.geojson'),
    'objects.us_counties_20m',
    style_function=style_function
).add_to(c_map)

c_map
```





