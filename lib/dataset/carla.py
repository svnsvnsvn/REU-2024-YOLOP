import numpy as np
import json
import os
from tqdm import tqdm
from .AutoDriveDataset import AutoDriveDataset


class CarlaDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmentation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes

        annotation_files = os.listdir(self.label_root)


        for annot_file in tqdm(annotation_files):
            print(f"Annotation file: {annot_file}\n")
            label_path = os.path.join(str(self.label_root), str(annot_file)).replace(".png", ".json")
            image_path = label_path.replace(str(self.label_root), str(self.img_root)).replace(".png", ".jpg")
            
            print(f"Label path: {label_path}")
            
            with open(label_path, 'r') as f:
                print(f"path: {label_path}")
                label = json.load(f)
            
            data = label['objects']
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['label']
                x1 = float(obj['bbox']['x_min'])
                y1 = float(obj['bbox']['y_min'])
                x2 = float(obj['bbox']['x_max'])
                y2 = float(obj['bbox']['y_max'])
                cls_id = self.get_class_id(category)
                gt[idx][0] = cls_id
                box = convert((width, height), (x1, x2, y1, y2))
                gt[idx][1:] = list(box)
                
            rec = [{
                'image': image_path,
                'label': gt
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def get_class_id(self, category):
        """
        Convert category name to class ID
        """
        class_ids = {
            "Car": 0,
            "Pedestrian": 1,
            "Truck": 2,
            "TrafficLight": 3,
            "Bicycle": 4,
            "Motorcycle": 5
        }
        return class_ids.get(category, -1)  # Return -1 if category is not found

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'bbox' in obj.keys():
                remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass

# Helper function to convert bounding box
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
