import sys
import os
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

    translated_classes = [translate[class_name] for class_name in dataset.classes]
    dataset.classes = translated_classes
    dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(translated_classes)}

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
