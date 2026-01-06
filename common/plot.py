#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum
from typing import List
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


# from yolov5.utils.plots
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_bboxes(
    img: np.ndarray,
    bboxes: List[np.ndarray],
    ids: List[int],
    labels: List[int],
    names: List[str],
    confidences: List[float],
):
    """
    Draw bounding boxes in the given image. Will change the original image, so make `copy()` if necessary.

    Args:
        img (np.ndarray): Image in shape (H, W, C), BGR.
        bboxes (List[np.ndarray]): position of bounding boxes (x1, y1, x2, y2).
        ids (List[int]): object id.
        labels (List[int]): object label, e.g. (1, 19).
        names (List[str]): object name, e.g. (person, horse).
        confidences (List[float]): confidence scores.

    Returns:
        np.ndarray: image with bounding boxes and annotations, (H, W, C), BGR
    """
    for bbox, obj_id, label, name, confidence in zip(bboxes, ids, labels, names, confidences):
        img = draw_bbox(img, bbox, obj_id, label, name, confidence)
    return img


def draw_bbox(
    img: np.ndarray,
    bbox: np.ndarray,
    obj_id: int = None,
    label: int = None,
    name: str = None,
    confidence: float = None,
    line_width: int = 2,
    color=None,
    txt_color=(255, 255, 255),
):
    """
    Draw one bounding box. Will change the original image, so make `copy()` if necessary.
    Based on yolov5.utils.plots.Annotator.

    Args:
        img (np.ndarray): Image in shape (H, W, C), BGR.
        bbox (np.ndarray): position of bounding box (x1, y1, x2, y2).
        obj_id (int, optional): object id. Defaults to None.
        label (int, optional): object label, e.g. `1`. Defaults to None.
        name (str, optional): object name, e.g. `person`. Defaults to None.
        confidence (float, optional): confidence score. Defaults to None.
        line_width (int, optional): line width of bounding box. Defaults to 2.
        color ([type], optional): color of bounding box. Defaults to None.
        txt_color (tuple, optional): color of label text. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: image with bounding box and annotation, (H, W, C), BGR
    """
    # not a valid bbox, return original image
    if len(bbox) < 4:
        warnings.warn(f"bbox {bbox} not valid")
        return img
    if color is None:
        if label is not None:
            color = colors(label, True)
        else:
            color = colors(np.random.randint(20), True)
    # draw object box
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    # generate label text
    label_text = ""
    if obj_id is not None:
        label_text += f"{obj_id} "
    if name is not None:
        label_text += f"{name} "
    if confidence is not None:
        label_text += f"{confidence:.2f}"
    # draw text box
    if len(label_text) > 0:
        # font thickness
        tf = max(line_width - 1, 1)
        # text width, height
        w, h = cv2.getTextSize(label_text, 0, fontScale=line_width / 3, thickness=tf)[0]
        # label fits outside box
        outside = p1[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # filled label text area
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        # label text
        cv2.putText(
            img,
            label_text,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            line_width / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def draw_gaze_arrow(
    img,
    head_bbox,
    heatmap,
    output_resolution=64,
    color=(0, 200, 0),
    linewidth=2,
):
    """
    Draw human gaze direction as arrow

    Args:
        img (np.ndarray): original image
        head_bbox (np.ndarray): bbox location of head
        heatmap (np.ndarray): gaze attention heatmap
        output_resolution (int, optional): heatmap resolution. Defaults to 64.
        color (tuple, optional): line color. Defaults to (0, 200, 0).
        linewidth (int, optional): line width. Defaults to 2.

    Returns:
        np.ndarray: image with gaze arrows
    """
    width, height = img.shape[1], img.shape[0]
    # find maximum in heatmap
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = idx[0], idx[1]
    # scale maximum point to original image size
    norm_x, norm_y = pred_x / output_resolution, pred_y / output_resolution
    center = (int(norm_x * width), int(norm_y * height))
    radius = int(height / 50.0)
    # draw circle on gaze target
    cv2.circle(img, center=center, radius=radius, color=color, thickness=linewidth)
    # draw line from bbox to gaze target
    pt1 = (int((head_bbox[0] + head_bbox[2]) / 2), int((head_bbox[1] + head_bbox[3]) / 2))
    cv2.line(img, pt1, center, color=color, thickness=linewidth)
    return img


def draw_heatmap_overlay(
    img,
    heatmap,
    heatmap_resized=None,
):
    """
    Draw heatmap overlay

    Args:
        img (np.ndarray): original image
        heatmap (np.ndarray): original heatmap of any shape, not normalized to [0, 255]
        heatmap_resized (np.ndarray, optional): modulated and resized heatmap. Defaults to None.

    Returns:
        np.ndarray: image with heatmap overlay
    """
    if heatmap_resized is None:
        width, height = img.shape[1], img.shape[0]
        heatmap_modulated = heatmap * 255
        heatmap_resized = cv2.resize(heatmap_modulated, (width, height))
        heatmap_resized = np.clip(heatmap_resized, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    img = img * 0.6 + heatmap_resized * 0.4
    img = img.astype(np.uint8)
    return img


def draw_hoi_annotations(img, hoi_triplets, bboxes, object_names, ids, line_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draw HOI (Human-Object Interaction) annotations on image
    
    Args:
        img (np.ndarray): input image
        hoi_triplets (list): list of HOI triplets [(subject_idx, interaction_name, object_idx, score), ...]
        bboxes (list): list of bounding boxes [[x1, y1, x2, y2], ...]
        object_names (list): list of object class names
        ids (list): list of object tracking IDs
        line_color (tuple): color for connection lines
        text_color (tuple): color for text labels
    
    Returns:
        np.ndarray: annotated image
    """
    if len(hoi_triplets) == 0 or len(bboxes) == 0:
        return img
    
    # Create a copy to avoid modifying original image
    img_annotated = img.copy()
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Draw HOI connections and labels
    for triplet in hoi_triplets:
        if len(triplet) >= 4:
            subj_idx, interaction_name, obj_idx, score = triplet[:4]
            
            # Check valid indices
            if subj_idx >= len(bboxes) or obj_idx >= len(bboxes):
                continue
                
            # Get bounding boxes
            subj_bbox = bboxes[subj_idx]
            obj_bbox = bboxes[obj_idx]
            
            if len(subj_bbox) < 4 or len(obj_bbox) < 4:
                continue
            
            # Calculate center points
            subj_center = (int((subj_bbox[0] + subj_bbox[2]) / 2), 
                          int((subj_bbox[1] + subj_bbox[3]) / 2))
            obj_center = (int((obj_bbox[0] + obj_bbox[2]) / 2), 
                         int((obj_bbox[1] + obj_bbox[3]) / 2))
            
            # Draw connection line
            cv2.line(img_annotated, subj_center, obj_center, line_color, 2)
            
            # Draw arrow head
            angle = np.arctan2(obj_center[1] - subj_center[1], obj_center[0] - subj_center[0])
            arrow_length = 15
            arrow_angle = 0.5
            
            # Arrow points
            arrow_x1 = int(obj_center[0] - arrow_length * np.cos(angle - arrow_angle))
            arrow_y1 = int(obj_center[1] - arrow_length * np.sin(angle - arrow_angle))
            arrow_x2 = int(obj_center[0] - arrow_length * np.cos(angle + arrow_angle))
            arrow_y2 = int(obj_center[1] - arrow_length * np.sin(angle + arrow_angle))
            
            cv2.line(img_annotated, obj_center, (arrow_x1, arrow_y1), line_color, 2)
            cv2.line(img_annotated, obj_center, (arrow_x2, arrow_y2), line_color, 2)
            
            # Prepare text label
            subj_name = object_names[subj_idx] if subj_idx < len(object_names) else f"obj{subj_idx}"
            obj_name = object_names[obj_idx] if obj_idx < len(object_names) else f"obj{obj_idx}"
            subj_id = ids[subj_idx] if subj_idx < len(ids) else subj_idx
            obj_id = ids[obj_idx] if obj_idx < len(ids) else obj_idx
            
            text = f"{subj_name}({subj_id}) -> {interaction_name} -> {obj_name}({obj_id}): {score:.2f}"
            
            # Calculate text position (middle of the line, slightly offset)
            text_x = int((subj_center[0] + obj_center[0]) / 2)
            text_y = int((subj_center[1] + obj_center[1]) / 2) - 10
            
            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_annotated, 
                         (text_x - 5, text_y - text_height - 5), 
                         (text_x + text_width + 5, text_y + 5), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return img_annotated


def draw_hoi_list_on_frame(img, hoi_list, start_y=30):
    """
    Draw a list of HOI interactions as text overlay on the frame
    
    Args:
        img (np.ndarray): input image
        hoi_list (list): list of HOI strings to display
        start_y (int): starting Y position for text
    
    Returns:
        np.ndarray: annotated image
    """
    if len(hoi_list) == 0:
        return img
    
    img_annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)
    
    y_offset = start_y
    line_height = 25
    
    for hoi_text in hoi_list:
        if not hoi_text.strip():
            continue
            
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(hoi_text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(img_annotated, 
                     (10, y_offset - text_height - 5), 
                     (10 + text_width + 10, y_offset + 5), 
                     background_color, -1)
        
        # Draw text
        cv2.putText(img_annotated, hoi_text, (15, y_offset), font, font_scale, text_color, thickness)
        
        y_offset += line_height
        
        # Don't overflow the image
        if y_offset > img.shape[0] - 30:
            break
    
    return img_annotated


def plot_image_grids(imgs, labels, BGR=True):
    """
    TODO 

    Args:
        imgs ([type]): [description]
        labels ([type]): [description]
        BGR (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    num_imgs = len(imgs)
    num_grids = int(np.ceil(np.sqrt(num_imgs)).item())
    fig = plt.figure()
    for idx, (img, label) in enumerate(zip(imgs, labels), start=1):
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(num_grids, num_grids, idx)
        ax.imshow(img)
        ax.set_title(label)
    return fig
