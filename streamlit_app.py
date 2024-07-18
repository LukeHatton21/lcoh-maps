import streamlit as st
import pandas as pd
import requests
import xarray as xr
import math
import netCDF4
import numpy as np
import cftime
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors  
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.feature as cfeature

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
    storage_link = "https://storage.googleapis.com/example-bucket-lcoh-map/PEM_COLLATED_RESULTS.nc"

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
    storage_link = "https://storage.googleapis.com/example-bucket-lcoh-map/ALK_COLLATED_RESULTS.nc"

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
def get_selected_tech(_PEM_data, _ALK_data, selected_tech=None):

    selected_data = _PEM_data

    if selected_tech == "Alkaline":
        selected_data = _ALK_data

    return selected_data


PEM_data = get_input_pem_data()
ALK_data = get_input_alk_data()
# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Levelised Cost of Hydrogen Maps Generator

Interactive maps of the levelised cost of hydrogen (LCOH) from solar PV and
onshore wind. 

Solar and wind power capacity factors have been generated from NASA's MERRA-2
dataset, which have been used in a technoeconomic model developed at Imperial 
College London with assumptions over cost to determine the LCOH at a gridpoint
resolution.


'''

# Add some spacing
''
''



selected_tech = st.select_slider(
    "Specify the Electrolyser Technology", 
    options={"PEM", "Alkaline"},
    value="PEM")

selected_sf = st.slider(
    'Specify the Solar Fraction (percentage of renewable capacity met by solar)',
    min_value=0,
    max_value=100,
    step = 10,
    value=[50])

solar_increase = st.slider(
    'Specify the percentage change in solar CAPEX',
    min_value=-100,
    max_value=100,
    step = 5,
    value=[0])

wind_increase = st.slider(
    'Specify the percentage change in wind CAPEX',
    min_value=-100,
    max_value=100,
    step = 5,
    value=[0])

elec_increase = st.slider(
    'Specify the percentage change in electrolyser CAPEX',
    min_value=-100,
    max_value=100,
    step = 5,
    value=[0])




# Select the given technology
selected_data = get_selected_tech(PEM_data, ALK_data, selected_tech=selected_tech)
selected_data['Calculated_LCOH'] = selected_data['levelised_cost']


# Apply changes in solar and wind CAPEX
selected_data_plotting = change_capex(selected_data.sel(solar_fraction=selected_sf[0]), solar_increase[0], wind_increase[0], elec_increase[0], selected_sf[0])

# Plot the data
plot_data_shading(selected_data_plotting['Calculated_LCOH'], tick_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="YlOrRd")


#
