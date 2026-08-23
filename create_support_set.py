# create_support_set.py
import os
import torch
import random
from PIL import Image
from torchvision import transforms

# Configuration
DATA_ROOT = r"D:\My Folder\PUC\Thesis\Archive\Acne04\acne_1024"  # Update this path
SUPPORT_DIR = "support_set"
NUM_SAMPLES_PER_CLASS = 10  # 10 per class for flexibility
INPUT_SIZE = 224

os.makedirs(SUPPORT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['acne0_1024', 'acne1_1024', 'acne2_1024', 'acne3_1024']

support_x = []
support_y = []

print("="*60)
print("  CREATING SUPPORT SET")
print("="*60)

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_ROOT, class_name)
    if not os.path.exists(class_dir):
        print(f"⚠️ Warning: {class_dir} not found!")
        continue
    
    images = [f for f in os.listdir(class_dir) 
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(images) == 0:
        print(f"⚠️ No images found in {class_dir}")
        continue
    
    selected = random.sample(images, min(NUM_SAMPLES_PER_CLASS, len(images)))
    
    for img_name in selected:
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            support_x.append(img_tensor)
            support_y.append(class_idx)
            print(f"✅ Class {class_idx} ({class_name}): {img_name}")
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")

support_x = torch.stack(support_x)
support_y = torch.tensor(support_y)

save_path = os.path.join(SUPPORT_DIR, 'support_set.pt')
torch.save({
    'x': support_x,
    'y': support_y,
    'class_names': class_names,
    'num_samples_per_class': NUM_SAMPLES_PER_CLASS
}, save_path)

print("\n" + "="*60)
print(f"✅ SUPPORT SET SAVED:")
print(f"   Total images: {support_x.shape[0]}")
print(f"   Image shape: {support_x.shape}")
print(f"   Class distribution: {torch.bincount(support_y)}")
print(f"   Saved to: {save_path}")
print("="*60)