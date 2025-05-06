# to run - #python distance.py webcam
import cv2
import os
import supervision as sv
from ultralytics import YOLO
import typer
import numpy as np
import math

# Load the model
model = YOLO("yolov8x.pt")  # Using YOLOv8x instead of YOLOv10x
app = typer.Typer()

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


KNOWN_DISTANCE = 100  # cm - distance used during calibration
KNOWN_WIDTH = {
    'person': 50,  # typical width of a person in cm
    'car': 180,    # typical width of a car in cm
    'bottle': 8,   # typical width of a bottle in cm
    'cell phone': 7,  # typical width of a cell phone in cm
    'laptop': 35,  # typical width of a laptop in cm
    'chair': 45   # typical width of a chair in cm
    # Add more objects as needed
}
FOCAL_LENGTH = 567  # Will be calculated during calibration

def calculate_focal_length(known_distance, real_width, pixel_width):
    """Calculate the focal length using a known object's distance and width"""
    return (pixel_width * known_distance) / real_width

def calculate_distance(focal_length, real_width, pixel_width):
    """Calculate the distance to an object using the focal length formula"""
    return (real_width * focal_length) / pixel_width

def calibrate_camera(calibration_object='person'):
    """Calibrate the camera using a known object at a known distance"""
    global FOCAL_LENGTH
    
    print(f"CALIBRATION MODE: Please place a {calibration_object} at exactly {KNOWN_DISTANCE}cm from the camera")
    print("Press 'c' to capture calibration image or 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection to find the calibration object
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Display frame with detections
        annotated_frame = frame.copy()
        for i, (box, class_id, confidence) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
            class_name = category_dict[class_id]
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.putText(annotated_frame, "CALIBRATION MODE - Press 'c' to calibrate, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Calibration", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            # Find the calibration object in the detections
            calibrated = False
            for i, (box, class_id, confidence) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
                class_name = category_dict[class_id]
                if class_name == calibration_object and confidence > 0.6:
                    # Calculate width in pixels
                    pixel_width = box[2] - box[0]
                    real_width = KNOWN_WIDTH[calibration_object]
                    FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, real_width, pixel_width)
                    print(f"Calibration successful! Focal length: {FOCAL_LENGTH}")
                    calibrated = True
                    break
            
            if calibrated:
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"Could not find a {calibration_object} in the frame. Please try again.")
        
        elif key == ord("q"):
            print("Calibration cancelled.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def process_webcam():
    global FOCAL_LENGTH
    
    # Check if we need to calibrate
    if FOCAL_LENGTH is None:
        if not calibrate_camera():
            print("Calibration failed. Using default focal length.")
            FOCAL_LENGTH = 800  # Default value, will be less accurate
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame dimensions for distance warning
    ret, test_frame = cap.read()
    if ret:
        frame_height, frame_width = test_frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
    else:
        print("Could not get frame dimensions")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Display the center of frame reference
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]
            
            # Calculate distance if we have calibration data for this object
            distance_text = ""
            if class_name in KNOWN_WIDTH:
                pixel_width = box[2] - box[0]
                real_width = KNOWN_WIDTH[class_name]
                distance = calculate_distance(FOCAL_LENGTH, real_width, pixel_width)
                distance_text = f" - {distance:.2f}cm"
                
                # Box center point
                box_center_x = (box[0] + box[2]) // 2
                box_center_y = (box[1] + box[3]) // 2
                
                # Draw line from center to object
                cv2.line(frame, (center_x, center_y), (int(box_center_x), int(box_center_y)), (0, 255, 0), 1)
                
                # Color based on distance (red if close)
                box_color = (0, 255, 0)  # Default green
                if distance < 100:  # Less than 1 meter
                    box_color = (0, 0, 255)  # Red
                elif distance < 200:  # Less than 2 meters
                    box_color = (0, 165, 255)  # Orange
            else:
                box_color = (255, 0, 0)  # Blue for objects without distance
            
            # Draw bounding box with color based on distance
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
            
            # Add label with confidence and distance
            label = f"{class_name}: {confidence:.2f}{distance_text}"
            cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        # Add status info
        cv2.putText(frame, f"Focal Length: {FOCAL_LENGTH:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Press 'q' to quit, 'r' to recalibrate", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Object Detection with Distance", frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            cap.release()
            cv2.destroyAllWindows()
            calibrate_camera()
            process_webcam()  # Restart with new calibration
            return
    
    cap.release()
    cv2.destroyAllWindows()

@app.command()
def webcam():
    """Start object detection with distance estimation on webcam feed"""
    typer.echo("Starting webcam processing with distance estimation...")
    process_webcam()

@app.command()
def calibrate(object_type: str = typer.Option("person", help="Object type to use for calibration")):
    """Calibrate camera for distance measurement using specified object"""
    if object_type not in KNOWN_WIDTH:
        available_objects = ", ".join(KNOWN_WIDTH.keys())
        typer.echo(f"Error: '{object_type}' not in known objects. Available options: {available_objects}")
        return
    
    typer.echo(f"Starting camera calibration using {object_type}...")
    calibrate_camera(object_type)

@app.command()
def add_reference(
    object_type: str = typer.Argument(..., help="Type of object to add"),
    width_cm: float = typer.Argument(..., help="Width of object in centimeters")
):
    """Add a new reference object for distance calculation"""
    global KNOWN_WIDTH
    KNOWN_WIDTH[object_type] = width_cm
    typer.echo(f"Added {object_type} with width {width_cm}cm to reference objects")
    typer.echo(f"Current reference objects: {KNOWN_WIDTH}")

if __name__ == "__main__":
    app()