from PIL import Image
from torch.utils.data import Dataset

from model_utils import *


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.annotation_paths = self.collect_paths(root_dir, max_samples)

    def collect_paths(self, root_dir, max_samples=None):
        image_paths = []
        annotation_paths = []

        annotation_root = os.path.join(root_dir, "Annotations")
        image_root = os.path.join(root_dir, "Images")

        for subdir in os.listdir(annotation_root):
            annotation_subfolder = os.path.join(annotation_root, subdir)
            image_subfolder = os.path.join(image_root, subdir)

            if os.path.isdir(annotation_subfolder) and os.path.isdir(image_subfolder):
                for filename in os.listdir(annotation_subfolder):
                    if filename.endswith(".txt"):
                        annotation_path = os.path.join(annotation_subfolder, filename)
                        image_filename = os.path.splitext(filename)[0] + ".jpg"
                        image_path = os.path.join(image_subfolder, image_filename)
                        if os.path.exists(image_path):
                            annotation_paths.append(annotation_path)
                            image_paths.append(image_path)

        if max_samples is not None:
            image_paths = image_paths[:max_samples]
            annotation_paths = annotation_paths[:max_samples]
        return image_paths, annotation_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Load and parse annotation (you'll need to implement this part based on the content of your annotation file)
        boxes = parse_annotation(annotation_path)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)

        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target
