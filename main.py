import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

CURRENT_DIRECTORY = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(
    CURRENT_DIRECTORY,
    'sample/input/video.mp4',
)
TARGET_VIDEO_PATH = os.path.join(
    CURRENT_DIRECTORY,
    'sample/output/video.mp4',
)
MODEL = os.path.join(
    CURRENT_DIRECTORY,
    'weight/yolov10n.pt',
)

POLYGON = np.array([
    [100, 400],
    [800, 400],
    [800, 800],
    [100, 800]
])


video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=0.25,
    minimum_matching_threshold=0.6,
    lost_track_buffer=video_info.fps
)
polygon_zone = sv.PolygonZone(
    polygon=POLYGON,
    triggering_anchors=[sv.Position.CENTER]
)
thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh
)
box_annotator = sv.BoxAnnotator(
    thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK
)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=thickness,
    text_thickness=thickness,
    text_scale=text_scale,
)

model = YOLO(MODEL)
model.fuse()

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vid_writer = cv2.VideoWriter(
    TARGET_VIDEO_PATH,
    fourcc, video_info.fps,
    (video_info.width, video_info.height))
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

person_in_zone_frames = {}
for _ in tqdm(range(video_info.total_frames), desc="Rendering videos with Bounding Box: "):
    ret, frame = cap.read()
    if not ret:
        break

    result = model(
        frame,
        classes=0,
        imgsz=1280,
        verbose=False,
        device='mps'
    )[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[polygon_zone.trigger(detections)]
    detections = byte_track.update_with_detections(detections)

    labels = []
    people_counter = 0
    for person_id in detections.tracker_id:
        if person_id is not None:
            if person_id in person_in_zone_frames:
                person_in_zone_frames[person_id] += 1
            else:
                person_in_zone_frames[person_id] = 1

            time_in_zone = person_in_zone_frames[person_id] / video_info.fps
            labels.append(f"Time: {time_in_zone}")
            if time_in_zone >= 3:
                people_counter += 1

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame,
        polygon=POLYGON,
        color=sv.Color.RED,
        thickness=4
    )
    cv2.putText(
        annotated_frame,
        f'Total people > 3s: {people_counter} people',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    vid_writer.write(annotated_frame)

cap.release()
vid_writer.release()
