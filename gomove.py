import os
import shutil
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='move_go_files.log', level=logging.INFO, format='%(asctime)s %(message)s')


def move_go_files_to_single_directory(src_base_path, dest_base_path):
    # Ensure the source path exists
    if not os.path.exists(src_base_path):
        logging.error(f"The source path {src_base_path} does not exist.")
        return

    # Ensure the destination path exists, create if it does not
    os.makedirs(dest_base_path, exist_ok=True)

    # Count of moved files
    moved_files_count = 0

    # List to store paths of .go files
    go_files = []
    for root, dirs, files in os.walk(src_base_path):
        for file in files:
            if file.endswith('.go'):
                go_files.append(os.path.join(root, file))

    # Create a progress bar using tqdm
    with tqdm(total=len(go_files), desc='Moving .go files', unit='file') as pbar:
        # Move .go files to the destination folder
        for src_file_path in go_files:
            try:
                # Extract the first-level directory name from the source file path
                relative_path = os.path.relpath(src_file_path, src_base_path)
                first_level_dir = relative_path.split(os.sep)[0]

                # Construct the destination path, including the first-level directory name
                dest_file_path = os.path.join(dest_base_path, first_level_dir, os.path.basename(src_file_path))

                # Ensure the destination folder exists, create if it does not
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

                # Move the file
                shutil.move(src_file_path, dest_file_path)
                moved_files_count += 1
                logging.info(f"Moved: {src_file_path} to {dest_file_path}")
                pbar.update(1)  # Update the progress bar
            except Exception as e:
                logging.error(f"Failed to move {src_file_path}: {e}")

    logging.info(f"Total number of .go files moved: {moved_files_count}")


# Replace these paths with your actual paths
go_projects_path = r'E:\graduation project\code\Go'
go_files_target_path = r'E:\graduation project\gosrc'

# Call the function
move_go_files_to_single_directory(go_projects_path, go_files_target_path)
