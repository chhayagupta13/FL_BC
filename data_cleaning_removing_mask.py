import os
import shutil

# Path to the main dataset directory
main_dataset_path = 'maindataset'

# Path to the new directory
new_dataset_path = 'masterdatacleaned'

# Create the new directory if it doesn't exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Traverse the main dataset directory
for root, _, files in os.walk(main_dataset_path):
    # Get the subdirectory path relative to the main dataset path
    relative_path = os.path.relpath(root, main_dataset_path)
    
    # Create the corresponding subdirectory in the new directory
    new_subdirectory_path = os.path.join(new_dataset_path, relative_path)
    if not os.path.exists(new_subdirectory_path):
        os.makedirs(new_subdirectory_path)
    
    # Copy files without '_mask.png' suffix
    for file in files:
        if not file.endswith('_mask.png'):
            file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_subdirectory_path, file)
            shutil.copy(file_path, new_file_path)
