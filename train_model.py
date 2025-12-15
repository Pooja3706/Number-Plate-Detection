from ultralytics import YOLO
import os

# Define path to dataset.yaml
# Ensure consistency with prepare_data.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, 'dataset', 'dataset.yaml')

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # Epochs set to 50 for decent convergence on small data
    model.train(data=DATA_YAML, epochs=50, imgsz=640, project=BASE_DIR, name='yolov8_number_plate')
