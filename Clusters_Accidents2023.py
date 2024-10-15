#streamlit run C:\Github\RTA_App-1\Clusters_Accidents2023.py
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import folium
import datetime as datetime
import matplotlib.pyplot as plt

st.header("UK Fatal Road Accident clustering tool")
st.markdown("Data Source: [www.data.gov.uk/road-safety-data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data)")

if 'fullresponse' not in st.session_state:
    st.session_state['fulldataset'] = None

@st.cache_data
def plot_accidents_by_year(fulldf):
    accidents_by_year = fulldf['accident_year'].value_counts().sort_index()
    plt.figure(figsize=(6, 3))
    fig, ax = plt.subplots()
    accidents_by_year.plot(kind='bar', ax=ax)
    ax.set_title('Number of Fatal Accidents by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Accidents')
    st.pyplot(fig)

def greatcircle(x,y):
    lat1, long1 = x[0], x[1]
    lat2, long2 = y[0], y[1]
    dist = great_circle((lat1,long1),(lat2,long2)).meters
    return dist

@st.cache_data
def load_dataset(mystart_year,myend_year):
    #READ RAW DATA
    fulldf = pd.DataFrame()
    for myyear in range(int(mystart_year),int(myend_year)+1):
        myfilepath = f"dft-road-casualty-statistics-collision-{myyear}.csv"
        importdataframe=pd.read_csv(myfilepath, low_memory=False)
        fulldf = pd.concat([fulldf,importdataframe])
    fulldf = fulldf.dropna(subset=['latitude', 'longitude'])
    return fulldf.loc[fulldf['accident_severity']==1].copy()

@st.cache_data
def perform_clustering(df_numeric, myepsilon, no_samples, myfulldf):
    dbc_circle = DBSCAN(eps = myepsilon, min_samples = no_samples, metric=greatcircle).fit(df_numeric)
    df_numeric['Cluster']=dbc_circle.labels_
    df_numeric['date'] = myfulldf['date'].copy()
    df_numeric['number_of_casualties'] = myfulldf['number_of_casualties'].copy()
    return df_numeric

@st.cache_data
def load_grey(mydf):
    UKMap = folium.Map(location=[53,-1.99], zoom_start=6)
    for index,entry in mydf.iterrows():
        folium.CircleMarker((entry['latitude'], entry['longitude']),popup=f'Date:{entry['date']}  Number of casualties:{entry['number_of_casualties']}', radius=5, weight=2, color='grey', fill_color='grey', fill_opacity=.2).add_to(UKMap)
    folium_static(UKMap)

@st.cache_data
def load_red(mydf):
    UKMap = folium.Map(location=[53,-1.99], zoom_start=6)
    for index,entry in mydf.iterrows():
        folium.CircleMarker((entry['latitude'], entry['longitude']),popup=f'Date:{entry['date']}  Number of casualties:{entry['number_of_casualties']}', radius=5, weight=2, color='red', fill_color='red', fill_opacity=.5).add_to(UKMap)
    folium_static(UKMap)

if 'UKMap' not in st.session_state:
    st.session_state['UKMap'] = folium.Map(location=[53,-1.99], zoom_start=6)

#load and filter for fatal accidents only
with st.spinner('Loading data...'):
    st.session_state['fulldataset'] = load_dataset(2019,2023)

#st.session_state['fulldataset'].to_csv('data.csv')
full_accident_count = len(st.session_state['fulldataset'])
st.write(f"Loaded {full_accident_count} fatal accidents from 2019-2023")
plot_accidents_by_year(st.session_state['fulldataset'])

start_year, end_year = st.select_slider(
    'Select date range to examine:',
    options=['2019','2020', '2021', '2022','2023'],
    value=('2021', '2023'))

selecteddf = st.session_state['fulldataset'].loc[(st.session_state['fulldataset']['accident_year']>=int(start_year)) & (st.session_state['fulldataset']['accident_year']<=int(end_year))].copy()
accident_count = len(selecteddf)
st.write(f"Loaded {accident_count} fatal accidents from {start_year} to {end_year}")
no_samples = st.number_input("Specify Number of Fatal accidents",min_value=2, max_value=10,step=1)
myepsilon = st.number_input("Specify density distance (radius in metres) ",min_value=500, max_value=2000,step=500,value=500)
ind_displayDBSCAN = st.button('Calculate and display DBSCAN clusters')
st.markdown("Link to explanatory article: [www.theactuary.com/top-tool-data-clustering](https://www.theactuary.com/2023/04/08/top-tool-data-clustering)")
st.markdown("Link to explanatory podcast (AI-generated): [Soundcloud.com/ai-generated-podcast-notebookdb-dbscan](https://soundcloud.com/richard-steele-752231039/ai-generated-podcast-notebookdb-dbscan?si=548a5166090746ffaefbf8932f7dd5d7&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)")

#Initialize map
st.divider()
UKMap = st.session_state['UKMap']
if ind_displayDBSCAN:
    df_numeric = selecteddf[['longitude','latitude']].copy()
    myclusterdf = perform_clustering(df_numeric, myepsilon, no_samples, selecteddf)
    mysubsetdf = myclusterdf.loc[myclusterdf['Cluster']!=-1].copy()
    accident_count = len(mysubsetdf)
    st.header(f"Displaying {accident_count} clustered accidents: {start_year}-{end_year}")
    st.write(f"{len(set(myclusterdf['Cluster']))-1} cluster(s) highlighted in red")
    load_red(mysubsetdf)
else:
    st.header(f"Displaying {accident_count} accidents: {start_year}-{end_year}")
    load_grey(selecteddf)