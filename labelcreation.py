import os
import re

# Define paths
base_path = os.path.join('.', 'croppedimages')
images_path = os.path.join(base_path, 'images')
label_path = os.path.join(base_path, 'labels')

# Class mappings
class_dict = {
    "Hatchback": 0,
    "Sedan": 1,
    "SUV": 2,
    "Pickup": 3,
    "Van": 4,
    "Bus": 5,
    "Semi-Truck": 6,
    "Non-Semi-Truck": 7,
    "Taxi": 8,
    "Motorbike": 9
}

# Check if label directory exists, else create it
if not os.path.exists(label_path):
    os.makedirs(label_path)

# Regex pattern to extract width and height
pattern = r'_wi([\d\.]+)_he([\d\.]+)'

# Go through train and val folders
for subfolder in ['train', 'val']:
    current_image_path = os.path.join(images_path, subfolder)
    current_label_path = os.path.join(label_path, subfolder)

    # Create label files
    for folder in os.listdir(current_image_path):
        if folder in class_dict.keys():
            class_label_path = os.path.join(current_label_path, folder)
            # Check and create class subdirectory under labels/train or labels/val
            if not os.path.exists(class_label_path):
                os.makedirs(class_label_path)

            for file in os.listdir(os.path.join(current_image_path, folder)):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Extract width and height from filename
                    match = re.search(pattern, file)
                    if match:
                        width, height = match.groups()
                        
                        # Construct label string
                        label_str = f"{class_dict[folder]} 0.5 0.5 {width} {height}"
                        
                        # Define paths
                        image_file_path = os.path.join(current_image_path, folder, file)
                        label_file_path = os.path.join(class_label_path, file.rsplit('.', 1)[0] + '.txt')

                        # Write label to file
                        with open(label_file_path, 'w') as label_file:
                            label_file.write(label_str[:-1])

print("Label generation complete!")
