# NOTE

In this simple project I just want to show you my technical ability to fullfil the pre-requisite objective to do video inference that can track, count, time people in the zone. Without doing deep research of use case because of my limited of time. But I hope we can discuss it more on the technical interview I guess hehe

# People Detection and Tracking with YOLOv10n

This project processes a video file to detect and track people in a defined polygonal zone using YOLOv10, Supervision and ByteTrack. The program also counts how many people spend more than 3 seconds inside the zone and annotates the video with bounding boxes and time spent.

## Project Structure

```bash
├── sample/
│   ├── input/
│   │   └── video.mp4        # Source video
│   ├── output/
│   │   └── video.mp4        # Output video with annotations
├── weight/
│   └── yolov10n.pt          # Pretrained YOLOv10 model weights
├── main.py
└── requirements.txt
```

## How It Works

1. **Object Detection:** The script uses YOLOv10n to detect people in the input video.
2. **Tracking:** ByteTrack tracks detected individuals across video frames.
3. **Zone Filtering:** Only people within a predefined polygonal zone are tracked.
4. **Time Calculation:** The script calculates how long each detected person stays in the zone. If a person stays for more than 3 seconds, the counter increments.
5. **Video Annotation:** Bounding boxes, labels, and the total count of people who stayed for more than 3 seconds are drawn on the output video.

## Getting Started

### Clone this Repositories

```bash
git clone https://github.com/adityaazizi/people-counting-using-yolov10.git
cd people-counting-using-yolov10
```

### Prerequisites

1. Install Python 3.x
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Place your input video in the `sample/input/` directory.
2. Ensure the YOLO model weights (`yolov10n.pt`) are in the `weight/` directory.
3. Run the script:
   ```bash
   python main.py
   ```
4. The annotated video will be saved in `sample/output/video.mp4`.

## Configuration

- **Polygon Zone:** You can adjust the polygon zone in the script by modifying the `POLYGON` variable. The current polygon is set as:

  ```python
  POLYGON = np.array([
      [100, 400],
      [800, 400],
      [800, 800],
      [100, 800]
  ])
  ```

- **Model Weights:** Make sure the correct YOLO model weights are used. By default, the path is set to:
  ```python
  MODEL = os.path.join(CURRENT_DIRECTORY, 'weight/yolov10n.pt')
  ```
  you alseo can use another model from ultralytics.

## Future Improvemenet

In future iterations of this project, several enhancements can be made to improve the performance, accuracy, and flexibility of the video inference system:

- **Model Size Upgrade**: For more accurate detection, consider using a larger YOLO model, such as YOLOv10x. Larger models generally provide better performance and accuracy but may require more computational resources.
- **Enhanced Accuracy and Inference Speed**: YOLOv10b, which balances accuracy and speed, could be employed for improved results. It is designed to offer a good trade-off between detection accuracy and inference speed.
- **Model Optimization with OpenVINO: Implementing OpenVINO** for model optimization and quantization can significantly enhance inference speed and efficiency. OpenVINO helps in deploying models on various Intel hardware platforms and can reduce latency through optimized operations.
- **Real-time Processing**: To improve real-time processing capabilities, we can implement multi-threading or parallel processing. This can help in handling video frames more efficiently and speed up the processing pipeline.
- **Parameter Tuning**: We also can tune some of paramaters to get more balanced for our spesific use case.

  - **Image Size**: Experiment with different image sizes for YOLOv10. Larger images can improve detection accuracy but may increase processing time. Conversely, smaller images can speed up inference but might reduce detection performance.
  - **IOU Threshold**: Adjust the Intersection over Union (IOU) threshold to refine the detection results. A higher IOU threshold can reduce false positives but might miss some detections, while a lower threshold might include more false positives.
  - **Confidence Threshold**: Tune the confidence threshold to control the sensitivity of the detection. A higher confidence threshold can reduce false positives but might also miss some detections. A lower threshold increases detection sensitivity but may result in more false positives.
  - **Polygon Zone**: Allow dynamic adjustments to the polygonal zone used for tracking. Providing options to modify the zone’s shape and size at runtime can make the system more adaptable to different scenarios.

  These improvements aim to enhance the overall functionality and performance of the video inference system, making it more effective and versatile in different applications.

## Acknowledgement

- [Ultralytics](https://github.com/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Supervision](https://github.com/roboflow/supervision)
