import os
import sys
import cv2
from ultralytics import YOLO
from roboflow import Roboflow

# Your Roboflow API key
ROBOFLOW_API_KEY = "osGFWngQm9xbVRyZgbbP"

def train_model():
    print("Starting training with Roboflow dataset...")
    os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    # Adjust project/version if needed
    project = rf.workspace("Test1").project("weapons")
    dataset = project.version(1).download("yolov8")

    data_yaml_path = dataset.data
    print(f"Dataset downloaded and prepared in: {dataset.location}")
    print(f"Data config YAML path: {data_yaml_path}")

    model = YOLO("yolov8n.pt")  # start from pretrained YOLOv8n

    # Training parameters
    model.train(data=data_yaml_path, epochs=20, batch=16, device='0')  # Adjust device to 'cpu' if no GPU

    print("Training completed.")
    print("Best weights are saved in runs/train/exp/weights/best.pt")

def is_person_in_window(box, window_region):
    """
    Check if the center of the bounding box lies inside the window region.
    box: bounding box coordinates (x1, y1, x2, y2)
    window_region: tuple (x_start, y_start, x_end, y_end)
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    x_start, y_start, x_end, y_end = window_region
    if x_start <= center_x <= x_end and y_start <= center_y <= y_end:
        return True
    return False

def detect(run_trained_model=True):
    # Load model weights: trained best weights if requested, else pretrained yolov8n
    model_path = "runs/train/exp/weights/best.pt" if run_trained_model and os.path.exists("runs/train/exp/weights/best.pt") else "yolov8n.pt"
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
        return

    height, width = frame.shape[:2]
    window_region = (int(width*0.7), int(height*0.1), int(width*0.95), int(height*0.4))

    print("Starting smart camera detection...")
    print("Press 'q' to quit.")

    COCO_CLASSES = model.names

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)[0]

        weapon_detected = False
        robbery_detected = False

        # Draw the window region where robbery detection trigger is checked
        window_color = (0, 255, 255)  # Yellow
        cv2.rectangle(frame, (window_region[0], window_region[1]), (window_region[2], window_region[3]), window_color, 2)
        cv2.putText(frame, "Window Region", (window_region[0], window_region[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, window_color, 2)

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls_id)
            confidence = float(conf)
            class_name = COCO_CLASSES[class_id]

            label = f"{class_name} {confidence:.2f}"
            color = (255, 0, 0)  # Blue bounding box by default
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Weapon detection proxy classes
            weapon_classes = ['knife', 'scissors', 'sports ball', 'baseball bat']
            if class_name in weapon_classes:
                weapon_detected = True
                cv2.putText(frame, "WEAPON ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Robbery detection: person in window region
            if class_name == "person":
                if is_person_in_window((x1, y1, x2, y2), window_region):
                    robbery_detected = True
                    cv2.putText(frame, "ROBBERY ALERT! Person in window", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Vandalism alert (basic logic)
        if weapon_detected or robbery_detected:
            cv2.putText(frame, "VANDALISM ALERT!", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Smart Camera System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting detection...")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        # By default run detection using trained model if available
        detect(run_trained_model=True)

if __name__ == "__main__":
    main()
