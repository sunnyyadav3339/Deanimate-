import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

dataset_paths = [
    r"D:\Deanimate\Dataset\Anime\Test_anime",
    r"D:\Deanimate\Dataset\Anime\Train_anime",
    r"D:\Deanimate\Dataset\Human\Test_human",
    r"D:\Deanimate\Dataset\Human\Train_human"
]

transform = transforms.Compose([
    transforms.Resize((256, 256)),                
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizing between [-1, 1]
])

def transform_imgs(folder_path, transform):

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for file in tqdm(files, desc=f"Processing {folder_path}"):
        file_path = os.path.join(folder_path, file)

        image = Image.open(file_path).convert("RGB")
        processed_image = transform(image)
        processed_image_pil = transforms.ToPILImage()((processed_image * 0.5 + 0.5))  # De-normalize to [0, 1]
        processed_image_pil.save(file_path)


for folder_path in dataset_paths:
    transform_imgs(folder_path, transform)

print("Preprocessing complete. Images have been saved back to their original folders.")
