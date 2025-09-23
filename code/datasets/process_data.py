import sys
import os
import shutil
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.animals10.translate import translate
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
size = 80

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def save_data(dataset, path):
    torch.save(dataset, path)

def load_data(path, transform=None):
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return dataset

def split_data(dataset, threshold):
    labels = [dataset[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=1-threshold, 
        stratify=labels, 
        random_state=42
    )
    
    # Create subsets using the stratified indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

def transform_and_save_data(data_path="data/animals10/raw-img"):
    import os
    from PIL import Image

    image_to_image_transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])

    for class_folder_name in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder_name)

        if os.path.isdir(class_folder_path):
            print(f"Processing images in: {class_folder_name}/")
            for image_filename in os.listdir(class_folder_path):
                image_file_path = os.path.join(class_folder_path, image_filename)

                if image_filename.lower().endswith(('.jpeg')):
                    try:
                        img = Image.open(image_file_path).convert('RGB')
                        transformed_img = image_to_image_transform(img)
                        transformed_img.save(image_file_path, quality=100)
                        print(f"  Transformed and saved: {image_filename}")
                    except Exception as e:
                        print(f"  Error processing {image_filename}: {e}")


def _remove_dir_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def export_stratified_split_to_folders(
    raw_data_root="data/animals10/raw-img",
    processed_root="data/processed",
    train_ratio=0.8,
):
    """
    Create stratified train/test folders with images copied from raw_data_root.
    """
    # Load dataset with translated classes (English) and no transform
    dataset = load_data(raw_data_root, transform=None)

    # Prepare indices and labels for stratified split
    all_indices = list(range(len(dataset.samples)))
    all_labels = [dataset.samples[i][1] for i in all_indices]

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=1 - train_ratio,
        stratify=all_labels,
        random_state=42,
    )

    # Prepare destination directories (idempotent: clear and recreate)
    train_root = os.path.join(processed_root, "train")
    test_root = os.path.join(processed_root, "test")
    _remove_dir_if_exists(train_root)
    _remove_dir_if_exists(test_root)

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    def _copy_indices(indices, dest_root):
        from PIL import Image

        for idx in indices:
            src_path, label_idx = dataset.samples[idx]
            class_name = dataset.classes[label_idx]
            class_dest_dir = os.path.join(dest_root, class_name)
            os.makedirs(class_dest_dir, exist_ok=True)

            # Validate image readability; skip if unreadable
            try:
                with Image.open(src_path) as img:
                    img.verify()
            except Exception:
                # Skip unreadable image
                continue

            filename = os.path.basename(src_path)
            dest_path = os.path.join(class_dest_dir, filename)
            shutil.copy2(src_path, dest_path)

    _copy_indices(train_indices, train_root)
    _copy_indices(test_indices, test_root)

    return {
        "num_total": len(all_indices),
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "classes": list(dataset.classes),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export stratified train/test image folders.")
    parser.add_argument("--raw", dest="raw_root", default="data/animals10/raw-img")
    parser.add_argument("--out", dest="processed_root", default="data/processed")
    parser.add_argument("--train", dest="train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    stats = export_stratified_split_to_folders(
        raw_data_root=args.raw_root,
        processed_root=args.processed_root,
        train_ratio=args.train_ratio,
    )
    print(
        f"Exported processed split -> total: {stats['num_total']}, "
        f"train: {stats['num_train']}, test: {stats['num_test']}"
    )
