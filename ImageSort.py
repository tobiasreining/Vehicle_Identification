import cv2
import os
import shutil
import numpy as np

def extract_number_from_filename(filename):
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except:
        return 0

def are_images_equal(img1, img2):
    return np.array_equal(img1, img2)

# Define your source directory and target directory structure
SOURCE_DIR = 'cropped_images/'
TARGET_DIRS = {
    '0': 'Discard',
    '1': 'Hatchback',
    '2': 'Sedan',
    '3': 'SUV',
    '4': 'Pickup',
    '5': 'Van',
    '6': 'Bus',
    '7': 'Semi-Truck',
    '8': 'Non-Semi-Truck',
    '9': 'Taxi',
    'a': 'Motorbike'
}

# Create target directories if they don't exist
for target_dir in TARGET_DIRS.values():
    os.makedirs(os.path.join(SOURCE_DIR, target_dir), exist_ok=True)

# Get a sorted list of images
all_images = sorted([img for img in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, img))],
                    key=extract_number_from_filename)

# Process each image
for image_name in all_images:
    image_path = os.path.join(SOURCE_DIR, image_name)

    # Check for duplicates in all folders
    image = cv2.imread(image_path)
    is_duplicate = False
    for target in TARGET_DIRS.values():
        target_folder = os.path.join(SOURCE_DIR, target)
        for existing_image_name in os.listdir(target_folder):
            existing_image_path = os.path.join(target_folder, existing_image_name)
            existing_image = cv2.imread(existing_image_path)
            if are_images_equal(image, existing_image):
                is_duplicate = True
                break
        if is_duplicate:
            break

    if is_duplicate:
        print(f"Duplicate found for {image_name}. Skipping.")
        continue

    # Display image
    cv2.imshow('Image Categorization', image)
    cv2.moveWindow('Image Categorization', 1200, 200)
    # Display instructions
    instructions = "Choose a category:\n"
    for key, val in TARGET_DIRS.items():
        instructions += f"{key}: {val}  "
    print(instructions)

    # Wait for key press and move image to respective folder
    key = cv2.waitKey(0)
    category_key = chr(key)

    # If key pressed matches one of our categories
    if category_key in TARGET_DIRS:
        target_path = os.path.join(SOURCE_DIR, TARGET_DIRS[category_key], image_name)
        shutil.move(image_path, target_path)
        print(f"Moved {image_name} to {TARGET_DIRS[category_key]}")
    else:
        print(f"Invalid key. Skipping {image_name}")

    # Close the image display window
    cv2.destroyAllWindows()
