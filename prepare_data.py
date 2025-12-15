import os
import glob
import shutil
import xml.etree.ElementTree as ET
import random
import yaml

# Set random seed for reproducibility
random.seed(42)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets')

# Sub-datasets (XML folder, Image folder)
DATA_SOURCES = [
    (
        os.path.join(DATASET_DIR, 'Annotations', 'Annotations'),
        os.path.join(DATASET_DIR, 'Indian_Number_Plates', 'Sample_Images')
    ),
    (
        os.path.join(DATASET_DIR, 'number_plate_annos_ocr', 'number_plate_annos_ocr'),
        os.path.join(DATASET_DIR, 'number_plate_images_ocr', 'number_plate_images_ocr')
    )
]

# Create output directories
for split in ['train', 'val']:
    for kind in ['images', 'labels']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, kind), exist_ok=True)

def convert_box(size, box):
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

all_samples = []

print("Processing annotations...")

for xml_dir, img_dir in DATA_SOURCES:
    print(f"Scanning {xml_dir}...")
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image filename from XML or derive from XML filename
        xml_filename = os.path.basename(xml_file)
        
        # Try finding image with same basename as XML in img_dir
        basename = os.path.splitext(xml_filename)[0]
        image_path = None
        
        # Check common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            possible_path = os.path.join(img_dir, basename + ext)
            if os.path.exists(possible_path):
                image_path = possible_path
                break
        
        if image_path is None:
            # Fallback: check filename inside XML
            try:
                fname_in_xml = root.find('filename').text
                possible_path = os.path.join(img_dir, fname_in_xml)
                if os.path.exists(possible_path):
                    image_path = possible_path
            except:
                pass

        if image_path is None:
            print(f"Warning: Image for {xml_file} not found. Skipping.")
            continue
            
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        # If size is 0 (can happen), try to read from image? 
        # For now assume XML is correct.
        
        boxes = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name != 'number_plate':
                continue
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            boxes.append(bb)
            
        if boxes:
            all_samples.append({'image': image_path, 'boxes': boxes})

# Shuffle and split
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.8)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

print(f"Total samples: {len(all_samples)}")
print(f"Training: {len(train_samples)}")
print(f"Validation: {len(val_samples)}")

def save_split(samples, split_name):
    for sample in samples:
        src_image = sample['image']
        filename = os.path.basename(src_image)
        basename = os.path.splitext(filename)[0]
        
        # Copy image
        dst_image = os.path.join(OUTPUT_DIR, split_name, 'images', filename)
        shutil.copy(src_image, dst_image)
        
        # Write label file
        label_file = os.path.join(OUTPUT_DIR, split_name, 'labels', basename + '.txt')
        with open(label_file, 'w') as f:
            for box in sample['boxes']:
                # Class 0 for number_plate
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

save_split(train_samples, 'train')
save_split(val_samples, 'val')

# Create dataset.yaml
yaml_content = {
    'path': OUTPUT_DIR,
    'train': 'train/images',
    'val': 'val/images',
    'names': {
        0: 'number_plate'
    }
}

with open(os.path.join(DATASET_DIR, 'dataset.yaml'), 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print("Data preparation complete.")
