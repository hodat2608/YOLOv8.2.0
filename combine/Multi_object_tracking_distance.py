import sys
from pathlib import Path
ultralytics_main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import matplotlib.pyplot as plt
current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(10, 1100), (3800, 1100), (3800, 2100), (10, 2100)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255,255,0),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for region manipulation.

    Parameters:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Notes:
        - This function is intended to be used as a callback for OpenCV mouse events.
        - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights='yolov8n.pt',
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
    track_distance= False,
    track_coordinates=False,
    track_history = defaultdict(list)
):
    vid_frame_count = 0
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")
    names = model.model.names
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    save_dir = increment_path(Path("ultralytics_source_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        results = model.track(frame, persist=True, classes=classes)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            print(results[0].orig_shape)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            xywhs = results[0].boxes.xywh.cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            dictionary= {}
            for box, track_id, cls, xywh in zip(boxes, track_ids, clss,xywhs):
                dictionary[int(track_id)] = (xywh[0], xywh[1])
                label=names[cls]
                if track_coordinates:
                    label = str(f'id:{track_id} {names[cls]} x:{xywh[0]:.1f} y:{xywh[1]:.1f}')
                annotator.box_label(box, label, color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 
                print('bbox_center',bbox_center)
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1
            print('=============================')
        if track_distance: 
            for i in range(len(list(dictionary.keys()))):
                for j in range(i+1, len(list(dictionary.keys()))):
                    x_distance = abs(dictionary[list(dictionary.keys())[i]][0]-dictionary[list(dictionary.keys())[j]][0])
                    y_distance = abs(dictionary[list(dictionary.keys())[i]][1]-dictionary[list(dictionary.keys())[j]][1])
                    if x_distance<300 or y_distance<300:
                        cv2.line(frame, tuple(map(int, dictionary[list(dictionary.keys())[i]])), tuple(map(int, dictionary[list(dictionary.keys())[j]])), (0, 0, 255), 2) 
                    else:
                        cv2.line(frame, tuple(map(int, dictionary[list(dictionary.keys())[i]])), tuple(map(int, dictionary[list(dictionary.keys())[j]])), (124,252,0), 2)

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                10,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            window_width = 2000
            window_height = 1000
            frame_resized = cv2.resize(frame, (window_width, window_height))
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame_resized)
       
        if save_img:
            video_writer.write(frame)

        for region in counting_regions:
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
        # if view_img:
        #     cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)
        # if save_img:
        #     video_writer.write(frame)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    source = r"C:\Users\CCSX009\Videos\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
    run(source=source,view_img=True,save_img=False,track_coordinates=True,track_distance=True)
    
