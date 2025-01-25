![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white) \
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges) \
<img src="https://github.com/blaylockbk/Herbie/blob/main/images/logo_new/Herbie-logo.png?raw=True" alt="drawing" width="90" height="40">






--------------------------------------------------------------------------------
# An Exploration of HHHR Data for Short Term Electric Load Forecast

This is a research project to explore how [HRRR](https://rapidrefresh.noaa.gov/hrrr/) weather data can improve the short term electric load forecast. We use [NYISO](https://www.nyiso.com/real-time-dashboard) load data at Zone F to compare our model based on HRRR data, and NYISO model performance.

The HRRR data provide 3 km spatial resolution, and are supposed to provide more spatial information to improve forecast accuracies. Here is an example of HRRR temperature data at 2m. 
<img src='images/temp_full_2m_f000.png' alt="drawing" width="250">

The findings include:
- 1 hour ahead forecast, for NYISO Zone F, in 2021, was tested. Below are the error comparison:
<img src="images/error.png" alt="drawing" width="320">


- The improvement seems to be from the time series contribution from the 1-D convolution. 

- The weather data does not appear to have abundant spatial informaiton. A PCA analysis for the 2-m temperature shows the first component explains over 99.7% of the total variance. The HRRR model seems to capture large scale spatial changes but tend to smooth out local details. It may be better to combine the HRRR data and local observational data for further improvement.
- In addition to the temperature, humidity and wind speed, other parameters, like boundary layer height, radiation can improve the accuracy slightly.

- The prior hour weather reduces error, and the coefficients are estimated within the model.


The details on how the data are retrieved and how the model are trained are explained in the [readme_techincal.md](https://github.com/limingzhou2004/pytools/blob/rolling-forecast/readme_technical.md) file. 

*** 
The pytools code base provides the tools, 
* Extract a subset of weather parameters and spatial scope, through [Herbie](https://github.com/blaylockbk/Herbie).
* Different model components to test.
 The model architecture I have tried is <img src="images/model.png" alt="drawing" width="450">