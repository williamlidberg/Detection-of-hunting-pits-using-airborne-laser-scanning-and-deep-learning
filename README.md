[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter Follow](https://img.shields.io/twitter/follow/William_Lidberg?style=social)](https://twitter.com/william_lidberg)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/)



# U-net-tutorial
This is an example of how to implement transfer learning for semantic segmentation. Lunar impact craters are used to pre-train a UNet model, which is then trained on high-resolution LiDAR data and manually digitised hunting pits.

<img src="images/Hunting_pit_3D_1.png" alt="hunting pit" width="75%"/>\
Hunting pit in a high resolution digital elevation model (DEM).

## Click this button to get started

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamlidberg/Unet-tutorial/blob/main/U_net_tutorial_on_impact_craters.ipynb)


## Data description

## Impact craters
Impact craters from the moon were used to pre-train the model. These creaters were digitised by NASA and are avalible from the Moon Crater Database v1 Robbins:https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018 The database contains approximately 1.3 million lunar impact craters and is approximately complete for all craters larger than about 1–2 km in diameter. Craters were manually identified and measured on Lunar Reconnaissance Orbiter Camera Wide-Angle Camera (WAC) images, in LRO Lunar Orbiter Laser Altimeter topography, SELENE Kaguya Terrain Camera images, and a merged LOLA+TC DTM.


The Moon LRO LOLA DEM 118m v1 was used as digital elevation model. This digital elevation model  is based on data from the Lunar Orbiter Laser Altimeter, an instrument on the National Aeronautics and Space Agency Lunar Reconnaissance Orbiter spacecraft. The created DEM represents more than 6.5 billion measurements gathered between July 2009 and July 2013, adjusted for consistency in the coordinate system described below, and then converted to lunar radii.
Source: https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub


<img src="images/Crater.png" alt="impact crater" width="75%"/>\
Impact crater in the lunar DEM

## Hunting pits
Coordinates of known hunting pits were downloaded from the Swedish national heritage board. The coordinates were manually adjusted based on visual observations in a hill-shaded elevation model. The area surrounding known hunting pits was inspected and additional hunting pits were manually digitized if discovered. The adjusted coordinates were also converted to polygon circles outlining the size of each hunting pit. In total, 2519 hunting pits were digitized in this way. These polygons were converted into segmentation masks where pixels inside the polygons were given a value of 1 and pixels outside the polygons were given a value of 0.

<img src="images/Lidberg_figure_1.png" alt="hunting pits" width="75%"/>\
In total, 2519 hunting pits were manually mapped in northern Sweden. Some 80% of the hunting pits were used for training the models while 20% were used for testing. 

## Links
PyPI: https://pypi.org/project/whitebox/

Contact: William.lidberg@slu.se
