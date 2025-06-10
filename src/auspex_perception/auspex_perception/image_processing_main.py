import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from datetime import datetime
from rclpy.node import Node
from PIL import Image
import numpy as np
import torch
import cv2
import io

from msg_context.loader import ObjectKnowledge
from msg_context.loader import FrameData
from auspex_msgs.srv import QueryKnowledge


from .ip_models import ObjectDetection, ImageClassification, ColorDetection

class ImageProcessing(Node):
    def __init__(self):
        super().__init__('image_processing')

        """
        Client for KB
        """
        self._query_client = self.create_client(QueryKnowledge, '/query_knowledge')

        """
        Define GPU usage
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """
        Flag for saving images as video stream.
        """
        self.store_video = False
        self.video_writer = None

        self.object_id = 0

        """
        Define models here
        """
        self._model_od = ObjectDetection(device=self._device)
        self._model_ic = ImageClassification(device=self._device)
        self._model_color = ColorDetection(device=self._device)

        self._current_model = self._model_od

        """
        ROS 2 Related
        """
        self._publisher = self.create_publisher(ObjectKnowledge, '/detections', 10)

        self.subscription_dict= {}
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.update_subscriptions)

        self.get_logger().info("[INFO]: Object Detection Ready...")

    def update_subscriptions(self):
        print("[INFO]: Updating Subscriptions...")
        self.get_platform_list()

    def update_platform_subscriptions(self, future):
        platforms_list = future.result().answer
        for platform_id in platforms_list:
            if platform_id not in self.subscription_dict:
                sensor_qos = qos_profile_sensor_data
                self.subscription_dict[platform_id] = self.create_subscription(FrameData,"/"+platform_id+'/raw_camera_stream',self.image_stream_cb, sensor_qos)
                self.get_logger().info(f"[INFO]: Starting Object Detection for platform: {platform_id}.")

        for key in self.subscription_dict.keys():
            if key not in platforms_list:
                del self.subscription_dict[platform_id]

    def image_stream_cb(self, msg):
        if msg.image_compressed.format == "empty":
            self.get_logger().info("[INFO]: Recevei...")
            return

        jpeg_image = msg.image_compressed.data

        if len(jpeg_image) > 0:
            if self.video_writer == None and self.store_video:
                self.get_logger().info("[INFO]: Creating Videowriter...")
                now = datetime.now()
                formatted_date_time = now.strftime('%Y_%m_%d_%H_%M')
                path = 'output_video_'+formatted_date_time+'.avi'
                self.video_writer = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'XVID'),msg.fps, (msg.res_width, msg.res_height))

            """
            Convert to PIL
            """
            image = Image.open(io.BytesIO(jpeg_image)).convert('RGB')

            """
            Write if selected
            """
            if self.store_video:
                self.video_writer.write(np.array(image)[:, :, ::-1])

            """
            Convert to RGB and extract
            """
            results = self._current_model.process_image(image, visualize=False)

            """
            Return Results
            """
            for result in results:
                detection_msg = ObjectKnowledge()
                detection_msg.id = "object_"+str(self.object_id)
                #self.object_id += 1 # id not incremented for now -> database overflow since not tracked
                detection_msg.detection_class = result
                detection_msg.priority = 0
                detection_msg.time_stamp = self.get_clock().now().to_msg()
                detection_msg.position.x = msg.gps_position.latitude
                detection_msg.position.y = msg.gps_position.longitude
                detection_msg.position.z = msg.gps_position.altitude
                detection_msg.velocity.x = 0.0
                detection_msg.velocity.y = 0.0
                detection_msg.velocity.z = 0.0
                detection_msg.confidence = 0.0
                detection_msg.state = "unknown"
                self._publisher.publish(detection_msg)

                self.get_logger().info("[INFO]: Detected Object: " + detection_msg.detection_class)


    def get_platform_list(self):
        while not self._query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query Knowledge Service, waiting again...')

        query_request = QueryKnowledge.Request()
        query_request.collection = 'platform'
        query_request.path = '$[*].platform_id'
        self.future = self._query_client.call_async(query_request)
        self.future.add_done_callback(self.update_platform_subscriptions)



def main(args=None):
    rclpy.init(args=args)

    image_processing = ImageProcessing()

    rclpy.spin(image_processing)

    image_processing.video_writer.release()
    image_processing.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()