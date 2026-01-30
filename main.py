import cv2
import numpy as np
import os
from src.face_mesh import FaceMeshDetector
from src.overlay import OverlayEngine

def main():
    # Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = FaceMeshDetector(max_faces=1)
    overlay_engine = OverlayEngine()

    # Define Accessories Database
    # Type: 'glasses', 'mustache', 'hat'
    accessories_db = [
        {"file": "sunglasses.png", "type": "glasses", "scale": 2.2, "offset_y": 0.05},
        {"file": "red_glasses.png", "type": "glasses", "scale": 2.2, "offset_y": 0.05},
        {"file": "pixel_glasses.png", "type": "glasses", "scale": 2.2, "offset_y": 0.05},
        {"file": "mustache.png", "type": "mustache", "scale": 1.5, "offset_y": 0.15},
        {"file": "top_hat.png", "type": "hat", "scale": 3.0, "offset_y": -1.5}, # Pushed further up
    ]

    loaded_accessories = []
    
    for item in accessories_db:
        path = os.path.join("assets", item["file"])
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = overlay_engine.remove_white_bg(img)
            loaded_accessories.append({
                "img": img,
                "type": item["type"],
                "scale": item["scale"],
                "offset_y": item["offset_y"]
            })
        else:
            print(f"Warning: Could not find {path}")

    if not loaded_accessories:
        print("Error: No accessories loaded.")
        return

    current_item_index = 0
    print("Virtual Try-On Started.")
    print("Controls: 'n' = Next Item | 'q' = Quit")

    while True:
        success, img = cap.read()
        if not success:
            continue

        # 1. Detection
        img, faces = detector.find_face_mesh(img, draw=False)

        if faces:
            face = faces[0]
            current_item = loaded_accessories[current_item_index]
            acc_type = current_item["type"]
            acc_img = current_item["img"]

            center_x, center_y = 0, 0
            width_ref_dist = 0
            angle = 0

            # 2. Logic based on Type
            if acc_type == "glasses":
                # Landmarks: 33 (L Eye), 263 (R Eye), 168 (Nose Bridge)
                p_left = face[33]
                p_right = face[263]
                p_center = face[168]
                
                width_ref_dist = np.linalg.norm(np.array(p_left) - np.array(p_right))
                angle = overlay_engine.calculate_angle(p_left, p_right)
                center_x, center_y = p_center

            elif acc_type == "mustache":
                # Landmarks: 61 (Mouth L), 291 (Mouth R), 164 (Upper Lip Center)
                p_left = face[61]
                p_right = face[291]
                p_center = face[164]
                
                width_ref_dist = np.linalg.norm(np.array(p_left) - np.array(p_right))
                angle = overlay_engine.calculate_angle(p_left, p_right)
                center_x, center_y = p_center

            elif acc_type == "hat":
                # Landmarks: 234 (Face L), 454 (Face R), 10 (Forehead Top)
                p_left = face[234]
                p_right = face[454]
                p_center = face[10]
                
                width_ref_dist = np.linalg.norm(np.array(p_left) - np.array(p_right))
                angle = overlay_engine.calculate_angle(p_left, p_right)
                center_x, center_y = p_center

            # 3. Transform & Overlay
            # Calculate width based on scale config
            target_width = int(width_ref_dist * current_item["scale"])
            
            # Maintain Aspect Ratio
            scale_factor = target_width / acc_img.shape[1]
            target_height = int(acc_img.shape[0] * scale_factor)
            
            # Resize
            img_resized = cv2.resize(acc_img, (target_width, target_height))
            
            # Rotate
            img_rotated = overlay_engine.rotate_image(img_resized, -angle)

            # Apply Offset
            # offset_y is % of the item height. 
            # Positive = Down, Negative = Up
            center_y += int(target_height * current_item["offset_y"])

            try:
                img = overlay_engine.overlay_transparent(img, img_rotated, center_x, center_y)
            except Exception as e:
                pass 

        # UI
        cv2.putText(img, "Virtual Try-On", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, "[N] Next  [W/S] Move  [A/D] Size  [Q] Quit", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)
        
        # Show current item info
        item = loaded_accessories[current_item_index]
        info_text = f"Item: {item['type'].upper()} | Scale: {item['scale']:.1f} | Y-Off: {item['offset_y']:.2f}"
        cv2.putText(img, info_text, (20, 110), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        cv2.imshow("Virtual Try-On", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_item_index = (current_item_index + 1) % len(loaded_accessories)
        
        # Adjustments
        # W/S to move Up/Down
        elif key == ord('w'): # Up
            loaded_accessories[current_item_index]["offset_y"] -= 0.1
        elif key == ord('s'): # Down
            loaded_accessories[current_item_index]["offset_y"] += 0.1
        
        # A/D to Resize
        elif key == ord('a'): # Smaller
            loaded_accessories[current_item_index]["scale"] -= 0.1
        elif key == ord('d'): # Larger
            loaded_accessories[current_item_index]["scale"] += 0.1

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
