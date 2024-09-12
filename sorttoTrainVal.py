import os
import shutil
import random

# Define the source and destination directories
src_dir = 'croppedimages'
train_dir = os.path.join(src_dir,"images", 'train')
val_dir = os.path.join(src_dir,"images", 'val')

# Create train and val directories if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# List all subdirectories inside VTID2
subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

# Process each subfolder
for subdir in subdirs:
    # Ignore the train and val directories if they are inside VTID2
    if subdir not in ['train', 'val']:
        subdir_path = os.path.join(src_dir, subdir)

        # Create train and val subdirectories for each type of car
        train_subdir = os.path.join(train_dir, subdir)
        val_subdir = os.path.join(val_dir, subdir)
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
        if not os.path.exists(val_subdir):
            os.makedirs(val_subdir)

        # List all image files in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle the list of image files
        random.shuffle(image_files)

        # Split into 80% train, 20% validation
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        # Copy files to train and val folders
        for file in train_files:
            shutil.copy(os.path.join(subdir_path, file), train_subdir)
        for file in val_files:
            shutil.copy(os.path.join(subdir_path, file), val_subdir)

print("Files have been split and copied to train and val folders.")
