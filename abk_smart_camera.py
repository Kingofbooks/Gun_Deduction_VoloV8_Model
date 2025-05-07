import cv2
import numpy as np
import os
import time
import threading
from collections import deque
from roboflow import Roboflow
from PIL import Image
import io

# Configuration parameters - adjust these for performance
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
DETECTION_INTERVAL = 3  # Process every N frames
MAX_FPS = 30
CONFIDENCE_THRESHOLD = 30
OVERLAP_THRESHOLD = 30

# Gun types and their characteristics for classification
GUN_TYPES = {
    "pistol": {
        "aspect_ratio_range": (1.5, 3.0),
        "size_range": (0.01, 0.15),
        "color": (0, 0, 255)
    },
    "rifle": {
        "aspect_ratio_range": (3.0, 8.0),
        "size_range": (0.05, 0.5),
        "color": (255, 0, 0)
    },
    "shotgun": {
        "aspect_ratio_range": (5.0, 10.0),
        "size_range": (0.05, 0.5),
        "color": (255, 0, 255)
    },
    "ak47": {
        "aspect_ratio_range": (3.5, 6.0),
        "size_range": (0.05, 0.4),
        "color": (0, 165, 255)
    },
    "unknown": {
        "color": (128, 128, 128)
    }
}

# Colors for visualization
COLORS = {
    'person': (0, 255, 0),
    'gun': (0, 0, 255),
    'suspicious': (0, 165, 255)
}

class FrameBuffer:
    """Thread-safe frame buffer to decouple capture from processing"""
    def __init__(self, maxlen=5):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        
    def add_frame(self, frame):
        with self.lock:
            self.buffer.append(frame)
            
    def get_latest_frame(self):
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1].copy()

class PersonTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks = {}
        self.suspicious = set()

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.tracks[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.tracks[objectID]
        if objectID in self.suspicious:
            self.suspicious.remove(objectID)

    def update(self, detections):
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for bbox in detections:
                x, y, w, h = bbox
                centroid = (x + w // 2, y + h // 2)
                self.register(centroid, bbox)
        else:
            current_centroids = []
            for bbox in detections:
                x, y, w, h = bbox
                centroid = (x + w // 2, y + h // 2)
                current_centroids.append((centroid, bbox))
            
            objectIDs = list(self.objects.keys())
            object_centroids = []
            for objectID in objectIDs:
                x, y, w, h = self.objects[objectID]
                centroid = (x + w // 2, y + h // 2)
                object_centroids.append(centroid)

            D = np.zeros((len(objectIDs), len(current_centroids)))
            for i, obj_c in enumerate(object_centroids):
                for j, (cur_c, _) in enumerate(current_centroids):
                    D[i, j] = np.linalg.norm(np.array(obj_c) - np.array(cur_c))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                
                objectID = objectIDs[row]
                centroid, bbox = current_centroids[col]
                self.objects[objectID] = bbox
                self.disappeared[objectID] = 0
                self.tracks[objectID].append(centroid)
                if len(self.tracks[objectID]) > 20:
                    self.tracks[objectID] = self.tracks[objectID][-20:]
                self.check_suspicious(objectID)
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unused_cols:
                    centroid, bbox = current_centroids[col]
                    self.register(centroid, bbox)

        return self.objects

    def check_suspicious(self, objectID):
        positions = self.tracks[objectID]
        if len(positions) < 5:
            return False
        
        positions_array = np.array(positions[-5:])
        diffs = positions_array[1:] - positions_array[:-1]
        speeds = np.linalg.norm(diffs, axis=1)
        avg_speed = np.mean(speeds)
        if avg_speed > 30 or (len(speeds) > 3 and np.std(speeds) > 15):
            self.suspicious.add(objectID)
            return True
        return False

class CameraCapture(threading.Thread):
    def __init__(self, camera_source=0, buffer=None):
        threading.Thread.__init__(self, daemon=True)
        self.camera_source = camera_source
        self.buffer = buffer
        self.running = False
        self.camera = None
        self.frame_time = 1.0 / MAX_FPS
    
    def run(self):
        self.camera = cv2.VideoCapture(self.camera_source)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.running = True
        last_time = time.time()

        while self.running:
            current_time = time.time()
            delta = current_time - last_time
            if delta < self.frame_time:
                time.sleep(self.frame_time - delta)

            # Read to clear buffer
            ret = False
            frame = None
            for _ in range(3):
                ret, frame = self.camera.read()
                if not ret:
                    break
            
            if ret and frame is not None:
                self.buffer.add_frame(frame)
                last_time = time.time()
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()

def classify_gun_type(w, h, frame_width, frame_height):
    aspect_ratio = w / h if h > 0 else 0
    relative_size = (w * h) / (frame_width * frame_height)
    for gun_type, props in GUN_TYPES.items():
        if gun_type == 'unknown':
            continue
        min_ratio, max_ratio = props['aspect_ratio_range']
        min_size, max_size = props['size_range']
        if min_ratio <= aspect_ratio <= max_ratio and min_size <= relative_size <= max_size:
            return gun_type
    return "unknown"

def prepare_image_for_roboflow(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    return buffer

def main():
    frame_buffer = FrameBuffer(maxlen=2)
    tracker = PersonTracker()

    # Initialize Roboflow model
    print("Initializing Roboflow model...")
    rf = Roboflow(api_key="YAlfcaTyGy8t7N4W4vQe")
    project = rf.workspace("maya-mokhtar-dwf7u").project("weapons-swoty")
    model = project.version(1).model
    print("Model initialized")

    camera_thread = CameraCapture(camera_source=0, buffer=frame_buffer)
    camera_thread.start()
    print("Camera started")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.namedWindow('Weapon Detection', cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    fps = 0
    frame_idx = 0

    last_predictions = []

    gun_detected = False
    gun_detection_time = 0
    detected_gun_type = "unknown"

    processing_active = False
    processing_result = None
    processing_lock = threading.Lock()
    processing_frame = None

    debug_mode = False

    def process_frame(frame, model):
        nonlocal processing_result, processing_active
        temp_file = "temp_frame.jpg"
        try:
            image_buffer = prepare_image_for_roboflow(frame)
            with open(temp_file, 'wb') as f:
                f.write(image_buffer.getvalue())
            result = model.predict(temp_file, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD)
            predictions = result.json()
            with processing_lock:
                processing_result = predictions
                processing_active = False
        except Exception as e:
            print(f"Prediction failed: {e}")
            with processing_lock:
                processing_result = {'predictions': []}
                processing_active = False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    try:
        while True:
            frame = frame_buffer.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            current_time = time.time()

            # Start processing thread if interval reached and not processing
            if frame_idx % DETECTION_INTERVAL == 0 and not processing_active:
                with processing_lock:
                    processing_active = True
                    processing_frame = frame.copy()

                threading.Thread(target=process_frame, args=(processing_frame, model), daemon=True).start()

            with processing_lock:
                if not processing_active and processing_result is not None:
                    last_predictions = processing_result.get("predictions", [])

            person_detections = []
            display_frame = frame.copy()

            for pred in last_predictions:
                label = pred["class"]
                confidence = pred["confidence"]
                x = int(pred["x"])
                y = int(pred["y"])
                w = int(pred["width"])
                h = int(pred["height"])

                x1 = max(0, x - w // 2)
                y1 = max(0, y - h // 2)

                color = COLORS.get(label, (255, 255, 255))

                if label == 'gun':
                    gun_detected = True
                    gun_detection_time = current_time

                    gun_type = classify_gun_type(w, h, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    detected_gun_type = gun_type
                    gun_color = GUN_TYPES[gun_type]['color']

                    cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), gun_color, 3)
                    cv2.putText(display_frame, f"{gun_type.upper()} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gun_color, 2)

                    overlay = np.zeros_like(display_frame)
                    overlay[:, :, 2] = 50
                    cv2.addWeighted(overlay, 0.2, display_frame, 1.0, 0, display_frame)

                    cv2.putText(display_frame, f"WARNING: {gun_type.upper()} DETECTED",
                                (display_frame.shape[1] // 2 - 180, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
                    cv2.putText(display_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if label == 'person':
                    person_detections.append((x1, y1, w, h))

            if gun_detected and current_time - gun_detection_time > 3:
                gun_detected = False
                detected_gun_type = "unknown"

            if frame_idx % DETECTION_INTERVAL == 0:
                objects = tracker.update(person_detections)
                for object_id, (x, y, w, h) in objects.items():
                    cv2.putText(display_frame, f"ID: {object_id}", (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    track = tracker.tracks[object_id]
                    if len(track) > 1:
                        points_to_draw = min(5, len(track))
                        for i in range(1, points_to_draw):
                            cv2.line(display_frame, track[-i - 1], track[-i], (0, 255, 255), 2)

                    if object_id in tracker.suspicious:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLORS['suspicious'], 3)
                        cv2.putText(display_frame, "SUSPICIOUS", (x, y - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['suspicious'], 2)

            frame_count += 1
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_frame, "Press 'q' to quit | 'd' for debug", (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if debug_mode:
                legend_y = 60
                cv2.putText(display_frame, "Gun Types:", (10, legend_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 20
                for gun_type, props in GUN_TYPES.items():
                    if gun_type != "unknown":
                        cv2.putText(display_frame, f"- {gun_type.upper()}", (20, legend_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, props["color"], 2)
                        legend_y += 20

                cv2.putText(display_frame, f"Detection Active: {processing_active}", (10, legend_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Predictions: {len(last_predictions)}", (10, legend_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(display_frame)
            cv2.imshow('Weapon Detection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f'Debug mode {"ON" if debug_mode else "OFF"}')

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera_thread.stop()
        out.release()
        cv2.destroyAllWindows()
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
        print("Resources released and program terminated")

if __name__ == "__main__":
    main()
