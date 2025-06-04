import streamlit as st
import pandas as pd
import requests
import xarray as xr
import math
import time
import netCDF4
import numpy as np
import cftime
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors  
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.feature as cfeature
import folium
from folium import Rectangle
import matplotlib.cm as cm
from streamlit_folium import folium_static

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Levelised Cost of Hydrogen Maps',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_resource
def get_input_pem_data():
    """ Import the original LCOH datafile in a NetCDF format
    
    
    Read the data in from Google Cloud and load onto the webpage"""
    storage_link = "https://www.dropbox.com/scl/fi/w6gffbv2dbli75kis1hsf/ALK_COLLATED_RESULTS.nc?rlkey=xpn4eddjfkc1y22mdoamtv6px&st=hvr7l4hy&dl=1"

    # Path to save the downloaded file
    file_path = 'PEM_COLLATED_RESULTS.nc'

    # Download the file
    response = requests.get(storage_link)
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    data_file = xr.open_dataset(Path(__file__).parent/file_path)

    return data_file

@st.cache_resource
def get_input_alk_data():
    """ Import the original LCOH ALK datafile in a NetCDF format
    
    
    Read the data in from Google Cloud and load onto the webpage"""
    storage_link = "https://www.dropbox.com/scl/fi/w6gffbv2dbli75kis1hsf/ALK_COLLATED_RESULTS.nc?rlkey=xpn4eddjfkc1y22mdoamtv6px&st=kjgjwa7q&dl=1"

    # Path to save the downloaded file
    file_path = 'ALK_COLLATED_RESULTS.nc'

    # Download the file
    response = requests.get(storage_link)
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    data_file = xr.open_dataset(Path(__file__).parent/file_path)

    return data_file


def plot_data_shading(data, tick_values=None, cmap=None):      
    
    # Apply limits
    values = data.sel(latitude=slice(-60, 90)).values
    latitudes = data.sel(latitude=slice(-60, 90)).latitude.values
    longitudes = data.longitude.values
    
    # create the heatmap using pcolormesh
    fig = plt.figure(figsize=(50, 30), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
=1, linthresh=0.5), transform=ccrs.PlateCarree(), cmap=cmap)
        
    axins = inset_axes(
    ax,
    width="1.5%",  
    height="82%",  
    loc="lower left",
    bbox_to_anchor=(1.03, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
    values_min = np.nanmin(values)
    values_max = np.nanmax(values)
    if values_min < tick_values[0]:
        extend = "min"
    if values_max > tick_values[-1]:
        extend = "max"
    if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
        extend="both"

    cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend)



    cb.ax.tick_params(labelsize=30)
    cb.ax.set_title("Levelised\n Cost of \nHydrogen\n(US$/kg)\n", fontsize=40)

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.coastlines()

    st.pyplot(fig)
    
    return 

@st.cache_data
def change_capex(_data, solar_change, wind_change, elec_change, solar_fraction):

    """ Function to examine how the LCOH changes based on the CAPEX cost of renewables"""
    # Drop existing cost 
    _data = _data.drop_vars("Calculated_LCOH")
    
    # Get the proportion of cost associated with renewables
    ren_lcoh = _data['levelised_cost_ren']
    elec_lcoh = _data['levelised_cost_elec']
    total_lcoh = _data['levelised_cost']

    # Get the proportion of renewable lcoh associated with solar
    solar_costs_frac= _data['solar_costs'] / _data['renewables_costs']
    
    # Calculate new LCOH associated with renewables costs
    new_ren_lcoh = (1 - solar_costs_frac) * ren_lcoh * (1 + wind_change / 100) + solar_costs_frac * ren_lcoh * (1 + solar_change / 100)
    new_elec_lcoh = elec_lcoh * ( 1 + elec_change / 100)
    
    # Apply the percentage increase
    calculated_lcoh = total_lcoh - ren_lcoh + new_ren_lcoh - elec_lcoh + new_elec_lcoh

    _data['Calculated_LCOH'] = calculated_lcoh
    
    return _data


@st.cache_data
def change_capex_absolute(_data, solar_capex, wind_capex, elec_capex, initial_capex):

    """ Function to examine how the LCOH changes based on the CAPEX cost of renewables"""
    # Drop existing cost 
    _data = _data.drop_vars("Calculated_LCOH")
    
    # Get the proportion of cost associated with renewables
    ren_lcoh = _data['levelised_cost_ren']
    elec_lcoh = _data['levelised_cost_elec']
    total_lcoh = _data['levelised_cost']

    # Get the proportion of renewable lcoh associated with solar
    solar_costs_frac= _data['solar_costs'] / _data['renewables_costs']
    
    # Calculate new LCOH associated with renewables costs
    new_ren_lcoh = (1 - solar_costs_frac) * ren_lcoh * (1 + (wind_capex - 1500)/1500) + solar_costs_frac * ren_lcoh * (1 + (solar_capex - 990) / 990)
    new_elec_lcoh = elec_lcoh * ( 1 + (elec_capex - initial_capex) / initial_capex)
    
    # Apply the percentage increase
    calculated_lcoh = total_lcoh - ren_lcoh + new_ren_lcoh - elec_lcoh + new_elec_lcoh

    _data['Calculated_LCOH'] = calculated_lcoh

    # Return flag
    flag = True
    
    return _data

@st.cache_data
def get_selected_tech(_PEM_data, _ALK_data, selected_tech=None):

    selected_data = _PEM_data

    if selected_tech == "Alkaline":
        selected_data = _ALK_data

    return selected_data


PEM_data = get_input_pem_data()
ALK_data = get_input_alk_data()

def display_netcdf_map(ds, variable_name):

    # Extract variable at desired time index (if 3D)
    stride = 1
    var = ds[variable_name]
    lat = var.latitude.values
    lon = var.longitude.values
    data = var.values

    # Normalize and colormap
    vmin=0.01
    vmax=10

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("YlOrRd")

    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=2, tiles='CartoDB positron', control_scale=True)

    for i in range(0, len(lat)-1, stride):
        for j in range(0, len(lon)-1, stride):
            val = data[i, j]
            if val == 0 or np.isnan(val):
                continue

            color = colors.to_hex(cmap(norm(val)))
            bounds = [[lat[i], lon[j]], [lat[i+1], lon[j+1]]]

            rect = Rectangle(
                bounds=bounds,
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                tooltip=f"{variable_name}: {val:.2f}"
            )
            rect.add_to(m)

    # Optional: Add color scale as legend
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colormap._A = []
    import branca.colormap as bcm
    linear_colormap = bcm.LinearColormap(
        [colors.to_hex(cmap(i)) for i in np.linspace(0, 1, 256)],
        vmin=vmin, vmax=vmax
    ).to_step(5)
    linear_colormap.caption = f'Levelised Cost of Hydrogen (USD/kg)'
    m.add_child(linear_colormap)

    return m

def show_map(selected_data_plotting):

    # Get map
    map = display_netcdf_map(selected_data_plotting, 'Calculated_LCOH')

    # Save in the session state
    st.session_state.map = map
    folium_static(st.session_state.map)


# -----------------------------------------------------------------------------

# Set the title that appears at the top of the page.
'''
# :earth_americas: GreenHydrogen.Ninja: A Geospatial Levelised Cost of Hydrogen Explorer

Interactive maps of the levelised cost of hydrogen (LCOH) from solar PV and
onshore wind. '''

# Add some spacing
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 About", "ℹ️ Methods",  "📈 Inputs", ":earth_americas: Global LCOH Snapshot", "🌐 Interactive LCOH Map"])

with tab1:
    about = open('about.md').read()
    st.write(about) 
    
with tab2:
    st.header("Methods")
    methods = open('methods.md').read()
    st.write(methods)

with tab3:
    st.header("Inputs")
    selected_tech = st.selectbox("Electrolyser Technology", options={"PEM", "Alkaline"})
    selected_sf = st.number_input('Specify the Solar Fraction (percentage of renewable capacity met by solar)', min_value=0, max_value=100, step=10, value=50)

    # Specify cost inputs
    solar_capex = st.number_input('Specify the global solar CAPEX (USD/kW)', min_value=100, max_value=2000, step=100, value=990)
    wind_capex = st.number_input('Specify the global wind CAPEX (USD/kW)', min_value=100, max_value=2000, step=100, value=1500)
    if selected_tech == "PEM":
        initial_capex = 2000
    else:
        initial_capex = 1700
    elec_capex = st.number_input("Specify the global electrolyser CAPEX (USD/kW)", min_value=100, max_value=2000, step=100, value=initial_capex)
    

    # Select the given technology
    selected_data = get_selected_tech(PEM_data, ALK_data, selected_tech=selected_tech)
    selected_data['Calculated_LCOH'] = selected_data['levelised_cost']

    # Apply changes in CAPEX
    selected_data_plotting = change_capex_absolute(selected_data.sel(solar_fraction=selected_sf), solar_capex, wind_capex, elec_capex, initial_capex)


with tab4: 
    st.header("Global LCOH Distribution")
    progress_text = "Generating static LCOH map. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.25)
        my_bar.progress(percent_complete + 1, text=progress_text)


    plot_data_shading(selected_data_plotting['Calculated_LCOH'], tick_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="YlOrRd")
with tab5:
    progress_text = "Generating interactive LCOH map. Please wait."
    my_interactive_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.33)
        my_interactive_bar.progress(percent_complete + 1, text=progress_text)
    
    map_obj = show_map(selected_data_plotting)

















