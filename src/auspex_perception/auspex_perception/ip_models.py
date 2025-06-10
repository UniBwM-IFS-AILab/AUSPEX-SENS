
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from ultralytics import YOLO
from PIL import Image
import matplotlib.colors as mc
import numpy as np
import webcolors
import torch
import cv2
from .image_processing_base import ImageProcessingBase
import os

class ObjectDetection(ImageProcessingBase):

    def __init__(self, device):
        self._device = device
        self._is_initalized = False

        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
        model_path = os.path.join(package_dir, "src", "auspex_perception", "auspex_perception", "models", "tank_coco_altered.pt")
        self.model = None
        if not os.path.exists(model_path):
            return
        self._is_initalized = True

        try:
            self.model = YOLO(model_path).to(device=device)
            self.model_names = self.model.names
        except Exception as e:
            print(f"error: Failed to load model: {e}")
            self.model = None

    def process_image(self, image, visualize=False):
        if self.model and self._is_initalized:
            with torch.no_grad():
                results = self.model(image, verbose=False, conf=0.70)

            if visualize:
                annotated_image = results[0].plot()
                cv2.imshow("YOLO Detection", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return [self.model_names[i] for i in results[0].boxes.cls.cpu().tolist()]
        else:
            print('error: model not initialized')
        return []

class ImageClassification(ImageProcessingBase):
    def __init__(self, device):
        self._device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152", device=self._device)
        self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-152").to(device=self._device)

    def process_image(self, image):
        if self.use_model == "resnet":
            with torch.no_grad():
                inputs = self.feature_extractor(image, return_tensors="pt").to(device=self._device)
                probs = self.model(**inputs,output_hidden_states=True).logits
            probs_real = torch.softmax(probs,dim=1)
            predicted_label = probs_real.argmax(-1).item()
            pred_label = self.model.config.id2label[predicted_label]
            return [pred_label]

class ColorDetection(ImageProcessingBase):
    def __init__(self, device="cuda:0"):
        self._device = device

        kernel_size=(10,10)
        stride=5
        padding=0
        self.pooling_layer =  torch.nn.AvgPool2d(kernel_size, stride, padding, ceil_mode=True, count_include_pad=False).to(device=device)

    def process_image(self, image):
        color_detected_rgb = self.get_main_color_rgb(image)
        color_detected_name = self.get_color_name(color_detected_rgb)
        return color_detected_name[1]

    """
    Color methods
    """
    """
    Get Closest color
    """
    def closest_color(self, requested_colour):
        min_colors = {}
        css4list = mc.CSS4_COLORS
        for name, key in css4list.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colors[np.sqrt(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

    """
    Get Color name from rgb values
    """
    def get_color_name(self, requested_color):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_color)
        except ValueError:
            closest_name = self.closest_color(requested_color)
            actual_name = None
        return actual_name, closest_name

    """
    Get most present color
    """
    def get_main_color_rgb(self, image_loaded):
        r,g,b = cv2.split(image_loaded)

        r = torch.from_numpy(r).to(dtype=torch.float32).unsqueeze(0).cuda()
        g = torch.from_numpy(g).to(dtype=torch.float32).unsqueeze(0).cuda()
        b = torch.from_numpy(b).to(dtype=torch.float32).unsqueeze(0).cuda()

        with torch.no_grad():
            r = self.pooling_layer(r)
            g = self.pooling_layer(g)
            b = self.pooling_layer(b)

        r = r.squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        g = g.squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        b = b.squeeze(0).to(dtype=torch.uint8).cpu().numpy()

        img = cv2.merge([b,g,r])

        img = Image.fromarray((img).astype(np.uint8))
        #img.show()
        colors = img.getcolors(1024*1024)
        max_occurence, most_present = 0, 0
        try:
            for c in colors:
                if c[0] > max_occurence:
                    (max_occurence, most_present) = c
            return most_present
        except TypeError:
            raise Exception("Too many colors in the image")