import sys
from pathlib import Path
ultralytics_main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO,solutions
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import supervision as sv
from ultralytics import YOLO
from collections import deque
import sys
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
# from shapely.geometry import Polygon
# from shapely.geometry.point import Point
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
'''
                    A(1252,787)       B(2298,803)
                  O _________________> x
                   |  ---------------|  <== image                
                   | /              /|         
                   |/              / |
        object==>  |              /  |      Origin Frame Size (pixel) 
                  /|             /   |
                 / |            /    |
                ---|------------     |
                   |_________________|
                   ^
                   y
              D(-550,2159)      C(5039,2159)

              
                    A'(0,0)           B'(24,0)
                      ----------------
                     /              /         
                    /              / 
                   /    object    /        Origin Actual Size (meter)
                  /              /
                 /              /
                ----------------
              D'(0,249)        C'(24,249)
'''



SOURCE = np.array([
    [1252, 787],  #A
    [2298, 803],  #B
    [5039, 2159], #C
    [-550, 2159]  #D
])

TARGET = np.array([
    [0, 0],       #A'
    [24, 0],      #B'
    [24, 249],    #C'
    [0, 249],     #D'
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
   
track_history = defaultdict(list)

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
    actual_localtion_estimation = False,
    speed_estimation = False,

):
    vid_frame_count = 0
    video_info = sv.VideoInfo.from_video_path(video_path=source)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
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
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        results = model.track(frame, persist=True, classes=classes)
        if results[0].boxes.id is not None:
            print(results[0].orig_shape)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            clss = results[0].boxes.cls.cpu().numpy().tolist()
            xywhs = results[0].boxes.xywh.cpu().numpy().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            dictionary= {}
            cfr = []
            count = []
            for box, track_id, cls, xywh in zip(boxes, track_ids, clss,xywhs):
                cfr.append([(box[0] + box[2]) / 2,box[3]])
                trans_box = np.array(cfr)
                points = view_transformer.transform_points(points=trans_box).astype(int)
                label = str(str(track_id) +' '+names[cls])
                for [x, y] in points: 
                    coordinates[track_id].append(y)
                    if actual_localtion_estimation:
                        label = str(f'id:{track_id} {names[cls]} x:{int(x*10)}mm y:{int(y*10)}mm')
                dictionary[int(track_id)] = (xywh[0], xywh[1])
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track)>30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                start_point = (950, 700)
                end_point = (1920, 700)
                color = (255, 255, 255)
                thickness = 2
                cv2.line(frame, start_point, end_point, color, thickness)
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 2)
                text = "Square"
                text_position = (110, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                text_color = (255, 255, 255)
                text_thickness = 2
                # if xywh[0] > 950 and xywh[1] < 700:
                #     if track_id in count:
                #         continue
                #     count.append(track_id)
                #     cv2.putText(frame, str(len(count)), text_position, font, font_scale, text_color, text_thickness)
                
            for box,track_id in zip(boxes,track_ids):
                if len(coordinates[track_id]) > video_info.fps / 2:
                    coordinate_start = coordinates[track_id][-1]    
                    coordinate_end = coordinates[track_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[track_id]) / video_info.fps
                    speed = distance / time * 3.6
                    if speed_estimation:
                        label = label+str(f' speed :{int(speed)}km/h')
                    annotator.box_label(box, label, color=colors(cls, True))
        if view_img:
            window_width = 2000
            window_height = 1000
            frame_resized = cv2.resize(frame, (window_width, window_height))
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame_resized)
        if save_img:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    source = r"C:\Users\CCSX009\Videos\vecteezy_cars-drive-along-the-embankment-of-the-highway-in-kiev_18733751.mp4"
    run(source=source,view_img=False,save_img=True,actual_localtion_estimation=True,speed_estimation=True)

