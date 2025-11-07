#!/usr/bin/env python3
import uuid
import cv2
import rclpy
from rclpy.node import Node
from geographic_msgs.msg import GeoPoint
from ultralytics import YOLO

from auspex_msgs.msg import ObjectObservation


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("sens_detection")

        self.rtsp_url = "rtsp://127.0.0.1:8554/humvid"
        self.model_path = "models/yolo11s_visdrone.pt"
        self.platform_id = "hummingbird1"
        self.sensor_id = "cam1"
        self.topic = "/detections"

        self.conf_threshold = 0.50

        self.detection_counter = 0

        self.model = YOLO(self.model_path)

        self.publisher = self.create_publisher(ObjectObservation, self.topic, 10)

        self.get_logger().info(f"opening RTSP stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.get_logger().fatal("failed to open RTSP stream.")
            raise SystemExit(1)

        self.timer = self.create_timer(0.2, self.run_detection)

    def run_detection(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn("RTSP read failed; skipping frame.")
            return

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results:
            return

        results = results[0]
        if results.boxes is None or len(results.boxes) == 0:
            return

        names = results.names if hasattr(results, "names") else {}

        for box in results.boxes:
            self.detection_counter += 1

            cls_idx = int(box.cls.item()) if box.cls is not None else -1
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            det_class = names.get(cls_idx, str(cls_idx))

            msg = ObjectObservation()

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(self.detection_counter)

            msg.object_id = str(uuid.uuid4())
            msg.platform_id = self.platform_id
            msg.sensor_id = self.sensor_id
            msg.detection_class = det_class
            msg.confidence = conf

            est_pos = GeoPoint()
            est_pos.latitude = 0.0
            est_pos.longitude = 0.0
            est_pos.altitude = 0.0
            msg.estimated_position = est_pos

            self.get_logger().warn("RTSP read failed; skipping frame.")
            self.publisher.publish(msg)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #cv2.imshow('YOLO Detection', frame)
        cv2.imwrite(f'output/frame_{self.detection_counter}.jpg', frame)

    def destroy_node(self):
        try:
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
