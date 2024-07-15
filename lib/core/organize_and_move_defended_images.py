import json
import os
import shutil
from tqdm import tqdm

def organize_and_move_images(defended_image_dir, target_dir):
    """
    Organize and move defended images into structured directories based on attack and defense parameters.

    Args:
        defended_image_dir (str): Path to the directory containing defended images.
        target_dir (str): Target directory where the organized images will be moved.
    """
    # Get all directories to process
    all_dirs = []
    for root, dirs, files in os.walk(defended_image_dir):
        for dir_name in dirs:
            all_dirs.append(os.path.join(root, dir_name))

    with tqdm(total=len(all_dirs), desc="Processing directories", unit="dir") as pbar_dirs:
        for defended_dir in all_dirs:
            parts = os.path.basename(defended_dir).split('_')
            attack_type = parts[0]
            defense_type = parts[-1]

            files = [f for f in os.listdir(defended_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            with tqdm(total=len(files), desc=f"Processing files in {os.path.basename(defended_dir)}", unit="file") as pbar_files:
                for file_name in files:
                    src_file = os.path.join(defended_dir, file_name)
                    
                    # Load the metadata file
                    metadata_file = os.path.join(defended_dir, f"{os.path.splitext(file_name)[0]}_metadata.json")
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        attack_params = metadata.get('attack_params', {})
                        defense_params = metadata.get('defense_params', {})

                        # Handle the case where defense_params is a string
                        if isinstance(defense_params, str):
                            defense_param_str = defense_params
                        else:
                            defense_param_str = '_'.join([f"{k}_{v}" for k, v in defense_params.items()])

                        attack_param_str = '_'.join([f"{k}_{v}" for k, v in attack_params.items()])
                        new_image_dir = os.path.join(target_dir, attack_type, attack_param_str, defense_param_str)
                        
                        if not os.path.exists(new_image_dir):
                            os.makedirs(new_image_dir)
                            print(f"Created directory {new_image_dir}")

                        dest_file = os.path.join(new_image_dir, file_name)
                        shutil.copyfile(src_file, dest_file)
                        # print(f"Copied {src_file} to {dest_file}")
                    
                    pbar_files.update(1)
            pbar_dirs.update(1)
            
def main():
    """
    Main function to organize and move defended images.
    """
    
    defended_image_dir = "DefendedImages"
    target_dir = "lib/dataset/images/bdd100k"
    
    organize_and_move_images(defended_image_dir, target_dir)

if __name__ == "__main__":
    main()
