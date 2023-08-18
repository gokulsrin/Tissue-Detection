import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import slideio
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import shapely 
from matplotlib.patches import Polygon as MPolygon
from matplotlib.collections import PatchCollection

class TissueDetector():
    def __init__(self, device): #init all the model stuff 
        self.DEVICE = device
        self.CHECKPOINT_PATH = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/Tissue_Detection/Tissue_Detection/sam_vit_h_4b8939.pth"
        self.MODEL_TYPE = "vit_h"
        
        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=self.DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            points_per_batch=64, 
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=int(1e5), 
        )
        self.reduced_img = None
        self.scale_ratio = None
        self.contours = []
        self.scaled_contours = []
        
    def find_tissue_contours(self, np_im):
        #reset after every call
        self.contours = []
        self.scaled_contours = []
        
        min_dim_size = 1028
        min_dim = min(np_im.shape[1], np_im.shape[0])
        max_dim = max(np_im.shape[1], np_im.shape[0])
        self.scale_ratio = min_dim_size/min_dim
        res = cv2.resize(np_im, dsize=(int(self.scale_ratio*np_im.shape[1]), int(self.scale_ratio*np_im.shape[0])), interpolation=cv2.INTER_CUBIC)
        self.reduced_img = res 
        
        sam_result = self.mask_generator.generate(res)
        
        big_regions = []
        for r in sam_result:
            if r["area"] > 5e4: 
                big_regions.append(r)
                
        big_regions = sorted(big_regions, key=lambda x: x["area"], reverse=False)
        big_regions = big_regions[:-1] #include everything but the big background segmentation
        
        for region in big_regions:
            binary_mask = (region["segmentation"] * 255).astype(np.uint8)
            contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.contours.append(contour[0]*(int(1/self.scale_ratio)))
            self.scaled_contours.append(contour[0])
                         
    def get_contours(self):
        return self.contours 
    
    def plot_annotations(self):
        # Create a figure and a single subplot
        fig, ax = plt.subplots(1)
        # Display the binary mask
        ax.imshow(self.reduced_img)

        # For each contour, create a polygon and add it to the plot
        for contour in self.scaled_contours:
            # Contour is an array of shape (N, 1, 2). We reshape it to (N, 2)
            reshaped_contour = contour.reshape(-1, 2)
            # Create a Polygon object
            polygon = MPolygon(reshaped_contour, fill=None, edgecolor='r')
            # Add the polygon to the plot
            ax.add_patch(polygon)

        plt.show()
        