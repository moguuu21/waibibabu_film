import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ------------------- 1. 读取属性标注文件并构建映射 -------------------
def load_attribute_annotations(attr_file_path):
    """读取属性标注文件并构建图像名-属性映射字典"""
    with open(attr_file_path, 'r') as f:
        f.readline()  # 跳过第一行数量标识行
        attr_names = f.readline().strip().split()

    attr_df = pd.read_csv(
        attr_file_path,
        sep='\s+',
        skiprows=2,
        names=['image_id'] + attr_names
    )

    attr_dict = {}
    for idx, row in attr_df.iterrows():
        img_name = row['image_id']
        attrs = row[1:].tolist()

        # 将[-1, 1]的属性值转换为[0, 1]
        attrs = [(x + 1) / 2 for x in attrs]  # 关键修改

        attr_dict[img_name] = torch.tensor(attrs, dtype=torch.float32)

    return attr_dict, attr_names


# ------------------- 2. 自定义数据集类 -------------------
class CelebAAttributeDataset(Dataset):
    def __init__(self, img_folder, attr_dict, transform=None):
        self.img_folder = img_folder
        self.attr_dict = attr_dict
        self.transform = transform
        self.img_names = [
            f for f in os.listdir(img_folder)
            if f.endswith(('.jpg', '.png', '.jpeg')) and f in attr_dict
        ]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        attrs = self.attr_dict[img_name]

        return {
            'image': image,
            'attributes': attrs,
            'filename': img_name
        }


# ------------------- 3. 数据加载与划分主函数 -------------------
def create_data_loaders(
        img_folder,
        attr_file,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        num_workers=2,
        pin_memory=True
):
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例和需为1.0"

    attr_dict, attr_names = load_attribute_annotations(attr_file)
    print(f"成功加载 {len(attr_dict)} 条属性标注")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CelebAAttributeDataset(
        img_folder=img_folder,
        attr_dict=attr_dict,
        transform=transform
    )
    print(f"成功创建数据集，总样本数: {len(dataset)}")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"数据集划分完成: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, attr_names