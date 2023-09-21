import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import slideio
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import shapely
from shapely import Polygon, MultiPolygon
from shapely.affinity import translate 
from shapely.affinity import scale 
from matplotlib.patches import Polygon as MPolygon
from matplotlib.collections import PatchCollection


class TissueDetector():
    """
    This class is meant to help you generate a shapely polygon that contains the tissue regions in a WSI. 
    
    In particular, this class can be very useful when generating patch data for histopathology CV algorithms, where we want 
    to only generate data for those patches that are within the tissue and not in the whitespace. The SAM model with a box prompt is used to
    produce and collect these contours. 
    
    Methods:
    --------
    init() -> None
        This will just init the TissueDetector class. Pass to it the relevant device. 
        
    find_tissue_contours(np_im, erosion_val = 0.05) -> None
        This method finds the tissue regions within the WSI. First the WSI is resized down so the smallest axis is 1024 pixels in length. 
        More efficient computation is performed on this scaled down image. Then the generated contour is scaled back up. The erosion_val dicates how much smaller the box prompt
        is relative to the WSI. So the larger the erosion_val, the smaller the box prompt. 
    
    get_contours() -> Shapely Polygon/MultiPolygon
        This method returns the Polygon/MultiPolygon that describes the tissue regions. 
    
    plot_annotations() -> None
        Visualizes the generated segmentations on the WSI. Note that this method will also visualize small artifacts that are not returned as Polygons. 
    
    Examples:
    ---------
    >>> device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    >>> detector = TissueDetector(device) 
    >>> np_im = np.load("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/DH/held_out_wsi/8_A1_0_orig.npy")
    >>> detector.find_tissue_contours(np_im)
    >>> detector.plot_annotations()
    
    Image of the contours plotted on top of the WSI 
    
    >>> polygon = detector.get_contours()
        
    Shapely Polygon
    
    Notes:
    ------
    Please note that you might need to change the erosion_val to get better results depending on your task. 
    
    """
    
    def __init__(self, device): #init all the model stuff 
        self.DEVICE = device
        self.CHECKPOINT_PATH = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/Tissue_Detection/sam_vit_h_4b8939.pth"
        self.MODEL_TYPE = "vit_h"
        
        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=self.DEVICE)
    
        self.reduced_img = None
        self.scale_ratio = None
        self.contours = []
        self.small_contours = []
        
    def find_tissue_contours(self, np_im, erosion_val=0.05):
        #reset after every call
        self.contours = []
        self.small_contours = []
        
        min_dim_size = 1028
        min_dim = min(np_im.shape[1], np_im.shape[0])
        max_dim = max(np_im.shape[1], np_im.shape[0])
        self.scale_ratio = min_dim_size/min_dim
        res = cv2.resize(np_im, dsize=(int(self.scale_ratio*np_im.shape[1]), int(self.scale_ratio*np_im.shape[0])), interpolation=cv2.INTER_CUBIC)
        self.reduced_img = res 
        
        input_box = np.array([int(0+erosion_val*res.shape[1]), int(0+erosion_val*res.shape[0]), int(res.shape[1]-erosion_val*res.shape[1]), int(res.shape[0] - erosion_val*res.shape[0])])
        
        predictor = SamPredictor(self.sam)
        predictor.set_image(res)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks = masks.squeeze()

        binary_mask = (masks * 255).astype(np.uint8)
        small_contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #now extract the polygon and scale up 
        minimum_thresh = 1e4
        polygons = []
        for i in small_contours: 
            contour_coords = i.squeeze()
            if contour_coords.ndim == 1: 
                continue  
            poly = Polygon(contour_coords)
            if poly.buffer(0).area > minimum_thresh: 
                polygons.append(poly.buffer(0))
                
        multi_polygon = MultiPolygon(polygons)
        multi_polygon = scale(multi_polygon, yfact = int(1/self.scale_ratio), xfact = int(1/self.scale_ratio))
        self.contours = multi_polygon
        self.small_contours = small_contours
                         
    def get_contours(self):
        return self.contours 
    
    def plot_annotations(self):
        # Create a figure and a single subplot
        fig, ax = plt.subplots(1)
        # Display the binary mask
        ax.imshow(self.reduced_img)

        # For each contour, create a polygon and add it to the plot
        for contour in self.small_contours:
            # Contour is an array of shape (N, 1, 2). We reshape it to (N, 2)
            reshaped_contour = contour.reshape(-1, 2)
            # Create a Polygon object
            polygon = MPolygon(reshaped_contour, fill=None, edgecolor='r')
            # Add the polygon to the plot
            ax.add_patch(polygon)

        plt.show()

        