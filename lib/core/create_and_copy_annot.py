import os
import json
import itertools
from tqdm import tqdm

def create_and_link_annotations(defended_image_dir, annotation_dirs):
    """
    Create and link annotation files for defended images.

    Args:
        defended_image_dir (str): Path to the directory containing defended images.
        annotation_dirs (list): List of directories containing annotation files.
    """
    
    counter = itertools.count(1)
    
    # Get all directories to process
    all_dirs = []
    for root, dirs, files in os.walk(defended_image_dir):
        for dir_name in dirs:
            all_dirs.append(os.path.join(root, dir_name))

    with tqdm(total=len(all_dirs), desc="Processing directories", unit="dir") as pbar_dirs:
        for defended_dir in all_dirs:
            files = [f for f in os.listdir(defended_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            with tqdm(total=len(files), desc=f"Processing files in {os.path.basename(defended_dir)}", unit="file") as pbar_files:
                for file_name in files:
                    # Generate a unique identifier for the image
                    unique_id = f"{os.path.splitext(file_name)[0]}_{next(counter)}"

                    # Load the metadata file
                    metadata_file = os.path.join(defended_dir, f"{os.path.splitext(file_name)[0]}_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        attack_type = metadata.get('attack_type', 'unknown_attack')
                        attack_params = metadata.get('attack_params', {})
                        defense_params = metadata.get('defense_params', {})

                        # Create strings for attack parameters and defense parameters
                        if isinstance(attack_params, dict):
                            attack_param_str = '_'.join([f"{k}_{v}" for k, v in attack_params.items()])
                        else:
                            attack_param_str = attack_params

                        if isinstance(defense_params, dict):
                            defense_param_str = '_'.join([f"{k}_{v}" for k, v in defense_params.items()])
                        else:
                            defense_param_str = defense_params

                        # Create a directory structure that includes attack type, attack parameters, defense type, and defense parameters
                        for annotation_dir in annotation_dirs:
                            new_annotation_dir = os.path.join(annotation_dir, attack_type, attack_param_str, defense_param_str)
                            
                            if not os.path.exists(new_annotation_dir):
                                os.makedirs(new_annotation_dir)
                                print(f"Created directory {new_annotation_dir}")

                            base_name = os.path.splitext(file_name)[0]
                            if "det_annotations" in annotation_dir:
                                annotation_file_name = base_name + ".json"
                            else:
                                annotation_file_name = base_name + ".png"
                                
                            src_file = os.path.join(annotation_dir, 'val', annotation_file_name)
                            dest_file = os.path.join(new_annotation_dir, f"{base_name}_{unique_id}{os.path.splitext(annotation_file_name)[1]}")
                            
                            if os.path.exists(src_file):
                                os.symlink(src_file, dest_file)
                            else:
                                print(f"Annotation file {src_file} does not exist.")
                    pbar_files.update(1)
            pbar_dirs.update(1)
            
def main():
    """
    Main function to create and link annotation files.
    """
    
    defended_image_dir = "DefendedImages"
    annotation_dirs = [
        "/Volumes/hobbywobbies/REU-2024-YOLOP/lib/dataset/da_seg_annotations",
        "/Volumes/hobbywobbies/REU-2024-YOLOP/lib/dataset/det_annotations",
        "/Volumes/hobbywobbies/REU-2024-YOLOP/lib/dataset/ll_seg_annotations"
    ]

    create_and_link_annotations(defended_image_dir, annotation_dirs)

if __name__ == "__main__":
    main()
