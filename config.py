import os

# Get the absolute path to the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw_data')
ANALYSIS_DATA_DIR = os.path.join(BASE_DIR, 'data', 'analysis_data')
RISKMACRO_DATA_DIR = os.path.join(BASE_DIR, 'data', 'riskmacro_data')
INDEX_DATA_DIR = os.path.join(BASE_DIR, 'data', 'index_components')

base_url = "https://optioncharts.io/options"
cpus = 10
tries = 3
timeout = 50000
headless = True
account = "yuweiyan@uchicago.edu"
password = "Yyw@903957503"
