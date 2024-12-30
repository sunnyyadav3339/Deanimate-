import os
import shutil
from sklearn.model_selection import train_test_split


##Anime dataset split

dataset_path = r"D:\Deanimate\Dataset\Anime"

train_path = r"D:\Deanimate\Dataset\Anime\Train_anime"
test_path = r"D:\Deanimate\Dataset\Anime\Test_anime"


os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

for file in train_files:
    shutil.move(os.path.join(dataset_path, file), os.path.join(train_path, file))

for file in test_files:
    shutil.move(os.path.join(dataset_path, file), os.path.join(test_path, file))


##Human dataset split

dataset_path = r"D:\Deanimate\Dataset\Human"

train_path = r"D:\Deanimate\Dataset\Human\Train_human"
test_path = r"D:\Deanimate\Dataset\Human\Test_human"

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

for file in train_files:
    shutil.move(os.path.join(dataset_path, file), os.path.join(train_path, file))

for file in test_files:
    shutil.move(os.path.join(dataset_path, file), os.path.join(test_path, file))


