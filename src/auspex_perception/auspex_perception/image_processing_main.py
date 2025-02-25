import rclpy
from rclpy.executors import SingleThreadedExecutor

from datetime import datetime
from rclpy.node import Node
from PIL import Image
import numpy as np
import torch
import cv2
import io

from msg_context.loader import ObjectKnowledge
from msg_context.loader import DroneImage
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
        
        """
        Get List of platforms and generate a subscription to each
        """
        self.platform_list = self.get_platform_list()
        
        # TODO make dynamic
        self.subscription_list = []
        for platform in self.platform_list:
            self.subscription_list.append(self.create_subscription(DroneImage,"/"+platform+'/raw_camera_stream',self.image_stream_cb,10))
        
        self.get_logger().info("[INFO]: Object Detection Ready...")
        
    def image_stream_cb(self, msg):
        if msg.image_compressed.format == "empty":
            return 
        
        png_image = msg.image_compressed.data
        
        if len(png_image) > 0:
            if self.video_writer == None and self.store_video:
                self.get_logger().info("[INFO]: Creating Videowriter...")
                now = datetime.now()
                formatted_date_time = now.strftime('%Y_%m_%d_%H_%M')
                path = 'output_video_'+formatted_date_time+'.avi'
                self.video_writer = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'XVID'),msg.fps, (msg.res_width, msg.res_height))     

            """
            Convert to PIL
            """
            image = Image.open(io.BytesIO(png_image)).convert('RGB')

            """
            Write if selected
            """
            if self.store_video:
                self.video_writer.write(np.array(image)[:, :, ::-1])

            """
            Convert to RGB and extract
            """
            image = image.convert('RGB')
            results = self._current_model.process_image(image)
            
            """
            Return Results
            """
            detection_msg = ObjectKnowledge()
            detection_msg.detection_class
            detection_msg.state = ",".join(results)
            detection_msg.position.x = msg.gps_position.latitude
            detection_msg.position.y = msg.gps_position.longitude
            detection_msg.position.z = msg.gps_position.altitude
            self._publisher.publish(detection_msg)
            
            self.get_logger().info("[INFO]: Detected Object: " + detection_msg.state)
                

    def get_platform_list(self):
        while not self._query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query Knowledge Service, waiting again...')
        ste = SingleThreadedExecutor()
        
        query_request = QueryKnowledge.Request()
        query_request.collection = 'platform'
        query_request.path = '$[*].platform_id'
        self.future = self._query_client.call_async(query_request)
        rclpy.spin_until_future_complete(self, self.future, ste)
        ste.shutdown()
        return self.future.result().answer
    
    

def main(args=None):
    rclpy.init(args=args)

    image_processing = ImageProcessing()

    rclpy.spin(image_processing)

    image_processing.video_writer.release()
    image_processing.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()