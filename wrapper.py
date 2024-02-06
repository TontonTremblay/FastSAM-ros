import os, sys
import random
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import cv2
import torch

from ultralytics import YOLO

import clip 
from PIL import Image

# add FastSAM submodule to path
FILE_ABS_DIR = pathlib.Path(__file__).absolute().parent
# FASTSAM_ROOT = (FILE_ABS_DIR / 'FastSAM').as_posix()
FASTSAM_ROOT = (FILE_ABS_DIR / 'fastsam').as_posix()
# print(FASTSAM_ROOT)
# raise()
if FASTSAM_ROOT not in sys.path:
    sys.path.append(FASTSAM_ROOT)

from fastsam import FastSAM, FastSAMPrompt 

from utils.tools import fast_show_mask, fast_show_mask_gpu
from utils import *

class FastSAM:
    def __init__(self, weights, conf_thresh: float = 0.4, iou_thresh: float = 0.9,
                 img_size: int = 1024, retina: bool = True, device: str = "cuda"):
        self.__img_size = img_size
        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__retina = retina
        self.__weights = weights
        self.__device = device
        self.device = device

        self.model = YOLO(self.__weights)

        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=device)


    # clip
    @torch.no_grad()
    def retrieve(self, model, preprocess, elements, search_text: str, device) -> int:
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100.0 * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)

    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < filter:
                continue
            annotation['id'] = i
            annotation['segmentation'] = mask.cpu().numpy()
            annotation['bbox'] = result.boxes.data[i]
            annotation['score'] = result.boxes.conf[i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations

    def _segment_image(self, image, bbox):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image

    def _get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                # Merge multiple bounding boxes into one.
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]


    def _crop_image(self, format_results):

        image = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ori_w, ori_h = image.size
        annotations = format_results
        mask_h, mask_w = annotations[0]['segmentation'].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = image.resize((mask_w, mask_h))
        cropped_boxes = []
        cropped_images = []
        not_crop = []
        filter_id = []
        # annotations, _ = filter_masks(annotations)
        # filter_id = list(_)
        for _, mask in enumerate(annotations):
            if np.sum(mask['segmentation']) <= 100:
                filter_id.append(_)
                continue
            bbox = self._get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
            cropped_boxes.append(self._segment_image(image, bbox))  
            # cropped_boxes.append(segment_image(image,mask["segmentation"]))
            cropped_images.append(bbox)  # Save the bounding box of the cropped image.

        return cropped_boxes, cropped_images, not_crop, filter_id, annotations

    @torch.no_grad()
    def _inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        self.img = img 

        pred_results = self.model(img)[0]
        detections = non_max_suppression(pred_results, conf_thres=self.__conf_thresh, iou_thres=self.__iou_thresh)

        if detections:
            detections = detections[0]

        return detections



    def detect(self, img0, max_det=100):
        """
        Perform inference on an image to detect classes.
        
        Parameters
        ----------
        img0 : (h, w, c) np.array -- the input image

        Returns
        -------
        dets : (n, 6) np.array -- n detections
                Each detection is 2d bbox xyxy, confidence, class
        """
        self.img = img0
        img = img0
        results = self.model(
            img0,
            imgsz=self.__img_size,
            device=self.__device,
            retina_masks=self.__retina,
            iou=self.__iou_thresh,
            conf=self.__conf_thresh,
            max_det=max_det,
        )


        ###########################################
        # def text_prompt(self, text):
        #     if self.results == None:
        #         return []
        #     format_results = self._format_results(self.results[0], 0)
        #     cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
        #     clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
        #     scores = self.retrieve(clip_model, preprocess, cropped_boxes, text, device=self.device)
        #     max_idx = scores.argsort()
        #     max_idx = max_idx[-1]
        #     max_idx += sum(np.array(filter_id) <= int(max_idx))
        #     return np.array([annotations[max_idx]['segmentation']])
        ###########################################



        text = 'mug'

        format_results = self._format_results(results[0], 0)
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
        
        scores = self.retrieve(self.clip_model, self.preprocess, cropped_boxes, text, device=self.device)
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        max_idx += sum(np.array(filter_id) <= int(max_idx))
        
        hey = np.array([annotations[max_idx]['segmentation']])

        ### make a segmentation mask with this 
        mask = hey[0]
        object_color = (0, 255, 0)  # Green
        background_color = (0, 0, 255)  # Red

        # Apply the overlay by assigning colors to the mask
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[mask] = object_color  # Assign object color to True pixels
        overlay[~mask] = background_color  # Assign background color to False pixels

        # Combine the original image and the overlay using bitwise operations
        result_img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)


        img = self.fast_process(img0, annotations=results[0].masks.data, mask_random_color=True)

        return {
            'img':img,
            'img':result_img,

            'dets':results,
        }

    # def text_prompt(self, text):
    #     if self.results == None:
    #         return []
    #     format_results = self._format_results(self.results[0], 0)
    #     cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
    #     clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
    #     scores = self.retrieve(clip_model, preprocess, cropped_boxes, text, device=self.device)
    #     max_idx = scores.argsort()
    #     max_idx = max_idx[-1]
    #     max_idx += sum(np.array(filter_id) <= int(max_idx))
    #     return np.array([annotations[max_idx]['segmentation']])

    def fast_process(self, img0, annotations, save_path='output', mask_random_color=False, retina=True, bbox=None, points=None, edges=False, better_quality=False, with_contours=False):
        if isinstance(annotations[0], dict):
            annotations = [annotation["segmentation"] for annotation in annotations]
        image = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        plt.figure(figsize=(original_w/100, original_h/100))
        plt.imshow(image)
        
        if better_quality == True:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv.morphologyEx(
                    mask.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                annotations[i] = cv.morphologyEx(
                    mask.astype(np.uint8), cv.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )

        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
        fast_show_mask_gpu(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            points=points,
            # pointlabel=None,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if with_contours == True:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask["segmentation"]
                annotation = mask.astype(np.uint8)
                if retina == False:
                    annotation = cv.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv.INTER_NEAREST,
                    )
                contours, hierarchy = cv.findContours(
                    annotation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    contour_all.append(contour)
            cv.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        save_path = os.path.join(FILE_ABS_DIR, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis("off")
        fig = plt.gcf()
        # plt.draw()
        # plt.show()
        
        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        
        cols, rows = fig.canvas.get_width_height()
        img_array = np.fromstring(buf, dtype=np.uint8).reshape(rows, cols, 3)
        # cv.imwrite(os.path.join(save_path, 'out.jpg'), cv.cvtColor(img_array, cv.COLOR_RGB2BGR))
        img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
        return img


        # return dets.cpu().detach().numpy()
if __name__ == "__main__":
    segmenter = FastSAM(
        "FastSAM-x.pt",
        conf_thresh=0.4, iou_thresh=0.9,
        img_size=1024, device="cuda")