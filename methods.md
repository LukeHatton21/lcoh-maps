## Renewable Profiles

Wind and solar weather data was sourced from NASA's MERRA-2 dataset, a global reanalysis dataset that provides a record of historical weather parameters at a hourly resolution. Weather data was converted into solar and wind power profiles using the methods underpinning Renewables.Ninja (wind) and the PV Lib package developed by Sandia National Labs (solar). Renewable profiles were then smoothed, by modelling the deployment of a lithium-ion battery on site, and for each location a combination of wind and solar at 10% capacity intervals was examined.

## Electrolyser Model

The smoothed renewable power profiles were used to calculate hydrogen production at an hourly resolution using dynamic electrolyser efficiency models, that account for how efficiency varies under different partial power loads. Models for the electrolyser efficiency were developed for both PEM and Alkaline electrolysers, with operational ranges of 10-100% and 40-100%, respectively, with the electrolyser assumed to shut off outside of this range.

## Economic Model

The established measure for comparing the cost of hydrogen production is the levelised cost of hydrogen, which accounts for deployment costs, annual hydrogen production and the time value of money (modelled here as an equivalent cost of capital for financing green hydrogen production projects). Costs associated with the electrolyser and renewables were discounted using country and technology-specific discount rates, whilst the hydrogen production was discounted with a weighted average discount rate that accounted for the proportions of total costs made up by the electrolyser and renewable components of the plant. The lifetime of the project was taken as 20 years, reflecting the typical operation of a renewables project.

## Optimisation

An important parameter to the economics and operation of a green hydrogen project without a grid connection is the relative sizing of the electrolyser, compared to the on-site renewables. In our model, we optimise the electrolyser:renewables ratio as so to minimise the overall levelised cost of hydrogen at each grid cell.

## Deployment and Financing Cost Assumptions
Costs of capital for solar and onshore wind were taken in real terms after tax from a benchmarking tool developed by IRENA (and validated by an expert survey), which covered 100 countries at a national and technology-specific granularity. For missing countries, the cost of capital was assumed either through comparison to comparator countries, or a uniform discount rate of c.10% was used. The cost of capital for the electrolyser component was taken as a 5% premium above the cost of capital for renewables at each location, reflecting the limited deployment and high risks associated with low-carbon hydrogen production.

Capital costs of US$990/kW and US$1500/kW were initially taken for solar and onshore wind, with the lithium ion battery costed at US$2526/kW and assumed to have an 8 hour max duration. An additional cost of US$115/kW was applied to offshore wind to reflect onshore foundation costs, whilst for offshore wind the costs of the foundations, substation, offshore platform(s) and transport medium (either a transmission cable or a hydrogen pipeline) were modelled. For the electrolyser, capital costs of US$1700 and US$2000/kW for Alkaline and PEM electrolysers were assumed, based on the average values reported by the International Energy Agency.

There is substantial uncertainty over the future cost of electrolysers, and renewable costs have been shown over recent decades to change very rapidly between years. Here, to account for this, we allow for modification of the input cost assumptions directly, though this does not re-run the optimisation model at each location due to the high computational weight this would incur at a global scale. 


