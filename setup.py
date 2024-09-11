from setuptools import setup, find_packages
from pytools import __version__


setup(
    name='pytools',
    version=__version__,
    
    description='py package for grib2 data',
    author='Liming Zhou',
    author_email='liming.zhou2004@gmail.com',
    url='',
    packages=find_packages(include=['pytools', 'pytools.*']),
    install_requires=[
        'dask==2024.5.1',
        'dill==0.3.1.1',
        'envtoml==0.1.2',
      #  'jsonargparse',
        'mlflow==2.14.1',
        'geopandas==0.12.2',
        'pandas==1.3.3',
        'pendulum==3.0.0',
        'psycopg2-binary==2.9.9',
        'pydantic==2.6.1',
        'pytorch-lightning==2.2.4',
#        'pynio', use conda forge install 1.5.5
        'read_env==1.1.0',
        'requests==2.28.2',
        'rioxarray==0.13.3',
        'scikit-learn==1.4.2',
        'shapely==2.0.0',
        'sqlalchemy==1.4.46',
        'toml==0.10.2',
        'torch==1.13', 
        'tqdm==4.66.4',
        'xarray==2022.12.0',
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    #setup_requires=['pytest-runner', 'flake8'],
    #tests_require=['pytest'],
    #entry_points={
    #    'console_scripts': ['my-command=exampleproject.example:main']
    #},
    package_data={'pytools': ['calendar.pkl','data_prep/*.txt']}
)
