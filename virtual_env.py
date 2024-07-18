import os
import subprocess

# Get the location of the rasterio package
rasterio_location = subprocess.check_output(['pip', 'show', 'rasterio']).decode()

# Find the line containing 'Location' in the output
for line in rasterio_location.split('\n'):
    if line.startswith('Location:'):
        location = line.replace('Location:', '').strip()

# Set GDAL_DATA environment variable
os.environ['GDAL_DATA'] = os.path.join(location, 'rasterio', 'gdal_data')
