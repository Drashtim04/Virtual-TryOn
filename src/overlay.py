import cv2
import numpy as np
import math

class OverlayEngine:
    def __init__(self):
        pass

    def overlay_transparent(self, background, overlay, x, y, overlay_size=None):
        """
        Overlays a transparent PNG onto the background at (x, y).
        (x, y) is the CENTER of where the overlay should be.
        """
        bg_h, bg_w, _ = background.shape
        
        if overlay_size is not None:
            overlay = cv2.resize(overlay, overlay_size)
        
        h, w, c = overlay.shape
        
        # Calculate top-left corner
        x_tl = int(x - w / 2)
        y_tl = int(y - h / 2)

        # Clipping checks to ensure we don't go out of bounds
        if x_tl < 0: x_tl = 0
        if y_tl < 0: y_tl = 0
        if x_tl + w > bg_w: w = bg_w - x_tl
        if y_tl + h > bg_h: h = bg_h - y_tl
        
        if w <= 0 or h <= 0: return background
        
        overlay_cropped = overlay[:h, :w]
        background_roi = background[y_tl:y_tl+h, x_tl:x_tl+w]
        
        # Split channels
        if overlay_cropped.shape[2] == 4:
            b, g, r, a = cv2.split(overlay_cropped)
            overlay_rgb = cv2.merge((b, g, r))
            mask = a / 255.0
            mask_inv = 1.0 - mask
            
            for i in range(3):
                background_roi[:, :, i] = (mask * overlay_rgb[:, :, i] + mask_inv * background_roi[:, :, i])
        else:
            background_roi = overlay_cropped

        background[y_tl:y_tl+h, x_tl:x_tl+w] = background_roi
        return background

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def calculate_angle(self, p1, p2):
        """Calculates the angle between two points"""
        x1, y1 = p1
        x2, y2 = p2
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def remove_white_bg(self, image, threshold=200):
        """
        Converts white background to transparent.
        """
        # If image has no alpha channel, add one
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            
        # Define range of "white" color
        lower_white = np.array([threshold, threshold, threshold, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(image, lower_white, upper_white)
        
        # Set alpha to 0 for white pixels
        image[:, :, 3] = np.where(mask == 255, 0, image[:, :, 3])
        
        return image
