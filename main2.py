import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
from datetime import datetime
import random
import time
import glob
import board
import busio
import adafruit_vl53l0x
import subprocess

# I2Cバスの初期化
i2c = busio.I2C(board.SCL, board.SDA)
# VL53L0Xセンサーの初期化
vl53 = adafruit_vl53l0x.VL53L0X(i2c)
# 距離の閾値設定
distance_min = 100  # 単位はミリメートル
distance_max = 600

overlay_images_list = [
    './text/attention_yellow.png',
    './text/5.png',
    './text/4.png',
    './text/3.png',
    './text/2.png',
    './text/1.png',
    './text/arrow.png'
]

def set_camera_control(control, value):
    command = f"v4l2-ctl -c {control}={value}"
    subprocess.run(command, shell=True, check=True)

last_selected_ghost_images = []

def apply_segmentation(image):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_image)
    
    mask = results.segmentation_mask
    _, binary_mask = cv2.threshold(mask, 0.35, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 元の画像に輪郭を描画
    outlined_image = image.copy()
    cv2.drawContours(outlined_image, contours, -1, (0, 0, 255), 2)  # 赤色の線、太さは2
    
    person_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return person_image, binary_mask, outlined_image

def resize_and_place_ghost(image, ghost_image, position):
    image_height, image_width = image.shape[:2]
    
    # Resize the ghost image while keeping the aspect ratio
    ghost_aspect_ratio = ghost_image.shape[1] / ghost_image.shape[0]
    new_height = int(image_height // 1.1)  # Adjust the size to be larger
    new_width = int(new_height * ghost_aspect_ratio)
    
    ghost_image_resized = cv2.resize(ghost_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    ghost_height, ghost_width = ghost_image_resized.shape[:2]
    ghost_image_resized = rotate_image(ghost_image_resized, 270)  # Change to 270 degrees
    
    # Update ghost dimensions after rotation
    ghost_height, ghost_width = ghost_image_resized.shape[:2]
    
    max_x = image_width - ghost_width
    max_y = image_height - ghost_height
    
    if max_x < 0 or max_y < 0:
        print("Error: Ghost image is larger than the frame.")
        return image
    
    # Ensure the ghost is positioned on the right side of the frame
    ghost_x = random.randint(int(max_x * 1 / 2), max_x)
    if position == 'top':
        ghost_y = random.randint(0, max_y // 4)
    elif position == 'bottom':
        ghost_y = random.randint(3 * max_y // 4, max_y)
    else:
        ghost_y = random.randint(0, max_y)  # Default to any position

    print(f"Placing ghost at x: {ghost_x}, y: {ghost_y}, position: {position}")

    # Resize ghost to fit within the image boundaries
    def fit_within_bounds(ghost_x, ghost_y, ghost_width, ghost_height, image_width, image_height):
        if ghost_x < 0:
            ghost_x = 0
        if ghost_y < 0:
            ghost_y = 0
        if ghost_x + ghost_width > image_width:
            ghost_width = image_width - ghost_x
        if ghost_y + ghost_height > image_height:
            ghost_height = image_height - ghost_y
        return ghost_x, ghost_y, ghost_width, ghost_height

    ghost_x, ghost_y, ghost_width, ghost_height = fit_within_bounds(ghost_x, ghost_y, ghost_width, ghost_height, image.shape[1], image.shape[0])

    alpha_ghost_cropped = ghost_image_resized[:, :, 3] / 255.0
    alpha_image_cropped = 1.0 - alpha_ghost_cropped
    overlay_ghost = ghost_image_resized[:, :, :3]

    for c in range(0, 3):
        if image[ghost_y:ghost_y + ghost_height, ghost_x:ghost_x + ghost_width, c].shape == alpha_ghost_cropped.shape:
            image[ghost_y:ghost_y + ghost_height, ghost_x:ghost_x + ghost_width, c] = alpha_ghost_cropped * overlay_ghost[:, :, c] + alpha_image_cropped * image[ghost_y:ghost_y + ghost_height, ghost_x:ghost_x + ghost_width, c]

    return image

def capture_ghost_photo(camera_frame, image1_path, image2_path, ghost_image_folder, output_diff_path, output_same_path):
    global last_selected_ghost_images
    
    original_image = camera_frame.copy()
    cv2.imwrite(image1_path, original_image)
    print(f"Original image saved to {image1_path}")
    
    person_image, binary_mask = apply_segmentation(original_image)
    cv2.imwrite(image2_path, person_image)
    print(f"Person-only image saved to {image2_path}")
    
    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_same_path = os.path.join(os.path.dirname(output_same_path), f'GhostPhoto{current_time_str}.png')

    diff = cv2.absdiff(original_image, person_image)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binary_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    diff_masked = cv2.bitwise_and(original_image, original_image, mask=binary_diff)

    ghost_images = glob.glob(os.path.join(ghost_image_folder, '*.png'))
    selected_ghost_images = random.sample(ghost_images, 2)
    
    # Ensure the new selection is different from the last selection
    while selected_ghost_images == last_selected_ghost_images:
        selected_ghost_images = random.sample(ghost_images, 2)
    
    last_selected_ghost_images = selected_ghost_images
    print(selected_ghost_images)

    ghost_image1 = cv2.imread(selected_ghost_images[0], cv2.IMREAD_UNCHANGED)
    ghost_image2 = cv2.imread(selected_ghost_images[1], cv2.IMREAD_UNCHANGED)
    image_with_ghost1 = resize_and_place_ghost(diff_masked, ghost_image1, 'top')
    image_with_ghost2 = resize_and_place_ghost(image_with_ghost1, ghost_image2, 'bottom')

    diff_with_ghost = cv2.cvtColor(image_with_ghost2, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(output_diff_path, diff_with_ghost)
    overlay_images(output_diff_path, image2_path, output_same_path)
    return output_same_path


def convert_to_rgba(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image from {image_path}.")
        return None

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    black = np.all(image[:, :, :3] == 0, axis=2)
    image[black, 3] = 0

    return image

def overlay_images(background_image_path, overlay_image_path, output_path):
    background_rgba = convert_to_rgba(background_image_path)
    overlay_rgba = convert_to_rgba(overlay_image_path)

    if background_rgba is None or overlay_rgba is None:
        print("Error: Could not process one or both images.")
        return

    background_pil = Image.fromarray(cv2.cvtColor(background_rgba, cv2.COLOR_BGRA2RGBA))
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay_rgba, cv2.COLOR_BGRA2RGBA))

    combined_pil = Image.alpha_composite(background_pil, overlay_pil)
    combined_pil.save(output_path)
    print(f"Overlay image saved to {output_path}.")

def overlay_text(background, overlay, position=(0, 0)):
    bg_height, bg_width = background.shape[:2]
    ol_height, ol_width = overlay.shape[:2]
    
    x, y = position

    if x >= bg_width or y >= bg_height:
        return background

    h, w = min(bg_height - y, ol_height), min(bg_width - x, ol_width)

    overlay_text = overlay[:h, :w]
    overlay_mask = overlay_text[:, :, 3] / 255.0
    background[y:y+h, x:x+w] = (1.0 - overlay_mask)[:, :, None] * background[y:y+h, x:x+w] + overlay_mask[:, :, None] * overlay_text[:, :, :3]
    return background

def rotate_image(image, angle):
    if angle == 90 or angle == 270:
        image = np.rot90(image, k=angle//90)
    else:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    return image

def load_and_resize_images(overlay_images_list, frame_shape, scale_factor=0.6):
    resized_images = []
    for overlay_path in overlay_images_list:
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

        if overlay is None:
            continue
        
        overlay_height, overlay_width = overlay.shape[:2]
        frame_height, frame_width = frame_shape[:2]

        # Calculate scaling factor for resizing
        scaling_factor = min(frame_height / overlay_height, frame_width / overlay_width) * scale_factor
        new_overlay_width = int(overlay_width * scaling_factor)
        new_overlay_height = int(overlay_height * scaling_factor)
        
        # Resize overlay
        overlay_resized = cv2.resize(overlay, (new_overlay_width, new_overlay_height), interpolation=cv2.INTER_AREA)
        overlay_resized = rotate_image(overlay_resized, 270)  # Change to 270 degrees

        resized_images.append(overlay_resized)
    return resized_images


def display_overlay_sequence(frame, resized_images, overlay_stage, overlay_display_time):
    current_time = time.time()
    
    # 新しいオーバーレイ画像を全てのステージに追加
    new_overlay_image = resized_images[-1]

    if overlay_stage < len(resized_images) - 1:
        if current_time - overlay_display_time > (2.5 if overlay_stage == 0 else 1):
            overlay_stage += 1
            overlay_display_time = current_time
            print(f"Advancing to overlay stage {overlay_stage}")

    if overlay_stage < len(resized_images) - 1:
        overlay_resized = resized_images[overlay_stage]
        frame = overlay_text(frame, overlay_resized, position=((frame.shape[1] - overlay_resized.shape[1]) // 2, (frame.shape[0] - overlay_resized.shape[0]) // 2))

    # 常に新しいオーバーレイ画像を追加 (上下中央の左端に配置)
    frame = overlay_text(frame, new_overlay_image, position=(0, (frame.shape[0] - new_overlay_image.shape[0]) // 2))

    overlay_visible = overlay_stage < len(resized_images) - 1
    return overlay_visible, frame, overlay_stage, overlay_display_time

def play_video(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 再生開始位置をリセット
    if not cap.isOpened():
        print(f"Error: Could not open video")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC key
            break

def main():
    progress_directory = './progress'
    ghost_photo_directory = './GhostPhoto'
    video_path = './shutter1.mp4'
    
    if not os.path.exists(progress_directory):
        os.makedirs(progress_directory)
    if not os.path.exists(ghost_photo_directory):
        os.makedirs(ghost_photo_directory)

    image1_path = os.path.join(progress_directory, 'original_image.png')
    image2_path = os.path.join(progress_directory, 'person_only_image.png')
    ghost_image_folder = './yurei'
    output_diff_path = os.path.join(progress_directory, 'diff_with_ghost.png')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # ここに固定値でカメラの露出とフォーカスを設定するコードを追加
    set_camera_control("focus_automatic_continuous", 0)  # マニュアルフォーカスに設定
    set_camera_control("focus_absolute", 500)  # フォーカスを固定値に設定
    set_camera_control("auto_exposure", 1)  # マニュアル露出に設定
    set_camera_control("exposure_time_absolute", 200)  # 露出を固定値に設定

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cv2.namedWindow('Live Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    resized_images = None
    overlay_display_time = 0
    overlay_visible = False
    overlay_stage = 0
    start_time = time.time()
    firsttime = True
    photo_captured = False
    output_same_path = None

    video_cap = cv2.VideoCapture(video_path)

    while True:
        distance = vl53.range

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        if resized_images is None:
            resized_images = load_and_resize_images(overlay_images_list, frame.shape, scale_factor=0.4)

        elapsed_time = time.time() - start_time
        current_time = (elapsed_time % 3600 % 60)
        
        if current_time > 5 and firsttime:
            firsttime = False

        if distance_min < distance and distance < distance_max:
            firsttime = False
            if not overlay_visible:
                overlay_display_time = time.time()
                overlay_visible = True
                overlay_stage = 0
                print("Starting overlay sequence...")
                cv2.setWindowProperty('Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if overlay_visible:
            overlay_visible, frame, overlay_stage, overlay_display_time = display_overlay_sequence(frame, resized_images, overlay_stage, overlay_display_time)
            if overlay_stage == len(resized_images) and not photo_captured:
                ret, camera_frame = cap.read()
                if ret:
                    output_same_path = capture_ghost_photo(camera_frame, image1_path, image2_path, ghost_image_folder, output_diff_path, os.path.join(ghost_photo_directory, 'GhostPhoto.png'))
                    if output_same_path:
                        photo_captured = True
                        print("Photo captured successfully at the last overlay stage.")
                        cap.release()
                        play_video(video_cap)  # 事前に初期化されたビデオキャプチャを使用
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        if not cap.isOpened():
                            print("Error: Could not re-open camera.")
                            break
                        
        if photo_captured and output_same_path and not overlay_visible:
            ghost_image = cv2.imread(output_same_path)
            if ghost_image is not None:
                cv2.imshow('Live Feed', ghost_image)
                cv2.waitKey(5000)
                print("Ghost photo displayed...")
                photo_captured = False
                overlay_stage = 0
                firsttime = True
                resized_images = None
                start_time = time.time()
                cv2.setWindowProperty('Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                continue

        if not photo_captured or overlay_visible:
            person_image, binary_mask, outlined_image = apply_segmentation(frame)
            cv2.imshow('Live Feed', outlined_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
