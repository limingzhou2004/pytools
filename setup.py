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
        'dask',
        'dill',
        'envtoml',
        'jsonargparse',
        'mlflow',
        'geopandas',
        'pandas',
        'pendulum',
        'psycopg2-binary',
        'pytorch-lightning',
        'pynio',
        'read_env',
        'requests',
        'rioxarray',
        'scikit-learn',
        'shapely',
        'sqlalchemy',
        'toml',
        'torch', 
        'tqdm',
        'xarray',
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    #setup_requires=['pytest-runner', 'flake8'],
    #tests_require=['pytest'],
    #entry_points={
    #    'console_scripts': ['my-command=exampleproject.example:main']
    #},
    package_data={'pytools': ['calendar.pkl','data_prep/*.txt', 'data/*', 'data/*']}
)
