import os
import json
import itertools

def create_and_link_annotations(defended_image_dir, annotation_dirs):
    counter = itertools.count(1)
    
    for root, dirs, files in os.walk(defended_image_dir):
        for dir_name in dirs:
            defended_dir = os.path.join(root, dir_name)

            for file_name in os.listdir(defended_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
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
                                print(f"Source file: {src_file}")
                                print(f"Destination file: {dest_file}")
                                
                                os.symlink(src_file, dest_file)
                                print(f"Created symlink {dest_file} -> {src_file}")
                            else:
                                print(f"Annotation file {src_file} does not exist.")

def main():
    defended_image_dir = "DefendedImages"
    annotation_dirs = [
        "/Users/annubaka/Library/Mobile Documents/com~apple~CloudDocs/Projects/YOLOP-main/lib/dataset/da_seg_annotations",
        "/Users/annubaka/Library/Mobile Documents/com~apple~CloudDocs/Projects/YOLOP-main/lib/dataset/det_annotations",
        "/Users/annubaka/Library/Mobile Documents/com~apple~CloudDocs/Projects/YOLOP-main/lib/dataset/ll_seg_annotations"
    ]

    create_and_link_annotations(defended_image_dir, annotation_dirs)

if __name__ == "__main__":
    main()
