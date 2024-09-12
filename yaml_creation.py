import yaml
import os
base_path = os.path.join('.', 'croppedimages')
# Your data to be saved to YAML format
data = {
    'path':  os.path.join('.', 'croppedimages'),
    'train': os.path.join(base_path, 'train'),
    'val':  os.path.join(base_path, 'val'),
    'test': '',
    'names': {
        0: 'hatchback',
        1: 'sedan',
        2: 'SUV',
        3: 'Pickup',
        4: 'Van',
        5: 'Bus',
        6: 'Semi-Truck',
        7: 'Non-Semi-Truck',
        8: 'Taxi',
    }
}

# Specify your output file name
output_file = 'cars.yaml'

# Write data to the YAML file
with open(output_file, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
