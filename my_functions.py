import csv
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from scipy.interpolate import CubicSpline
from tkinter import filedialog

import my_cmap

EARTH_R=(6378.1, 6356.7)
POLAR_FLATTENING = (EARTH_R[1] / EARTH_R[0]) ** 1

def get_path_via_gui(mode='file'):
    '''
    Opens a GUI to select either a 'file' or a 'directory'.
    '''
    root = tk.Tk()
    root.withdraw()
    path = ''
    if mode == 'file':
        path = filedialog.askopenfilename(title='Select Reference Image')
    elif mode == 'dir':
        path = filedialog.askdirectory(title='Select Folder for Batch Processing')
    elif mode == 'csv':
        path = filedialog.askdirectory(title='Select csv file')
    else:
        raise ValueError('Invalid mode. Use "file" or "dir" or "csv".')
    root.destroy()
    return path

def get_multiple_files():
    '''
    Opens a dialog to select one or more specific image files.
    Returns a list of file paths.
    '''
    root = tk.Tk()
    root.withdraw()
    # Returns a tuple of strings
    files = filedialog.askopenfilenames(
        title='Select Images to Crop',
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return list(files)

def get_calibration_folder(image_folder):
    '''
    Given the image folder path, creates the calibration folder
    and returns the 'latest' and 'timestamped' CSV paths.
    '''
    # Move up one level from 'image' and into 'calibration'
    base_dir = Path(image_folder).parent
    cal_dir = base_dir / 'Calibration'

    # Create directory if it doesn't exist
    cal_dir.mkdir(parents=True, exist_ok=True)

    return cal_dir


def get_calibration_csv_paths(image_folder):
    '''
    Given the image folder path, creates the calibration folder
    and returns the 'latest' and 'timestamped' CSV paths.
    '''
    # Get calibration folder
    cal_dir = get_calibration_folder(image_folder)

    # Generate filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    latest_file = cal_dir / 'film_background_analysis_latest.csv'
    backup_file = cal_dir / f'film_background_analysis_{timestamp}.csv'

    return latest_file, backup_file

def select_input(mode='file'):
    '''
    Opens a file explorer to select a single image or a directory.
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    path = ''
    if mode == 'file':
        # Option 1: Select a single file
        path = filedialog.askopenfilename(
        title='Select a Satellite Image',
        filetypes=[('Image files', '*.png *.bmp *.jpg *.jpeg')]
    )
    else:
        path = filedialog.askdirectory(title='Select Folder for Batch Processing')

    root.destroy()
    return path

def get_image_list(folder_path, extensions=('.png', '.bmp', '.jpg', '.jpeg')):
    '''
    Returns a list of all processable images in a folder.
    '''
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(extensions)]

def change_extension(fn, new_ext, old_ext):

    if hasattr(old_ext, '__iter__') and not isinstance(old_ext, str):
        for ext in old_ext:
            fn = fn.replace(ext, new_ext)
        return fn
    else:
        return fn.replace(old_ext, new_ext)

def load_image_lossless(path):
    '''
    Reads the image as-is without any depth scaling.
    Keeps loading isolated to allow memory freeing after analysis.
    '''
    # cv2.IMREAD_UNCHANGED ensures we don't drop bit-depth (e.g., 16-bit)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Could not find or read: {path}')

    if len(img.shape) > 2:
        # Convert from BGR or BGRA to Grayscale
        # Check if there are 4 channels (BGRA)
        if img.shape[2] == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            # Convert from BGR to Grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return img_gray

# Usage
#     img = load_image_lossless(image_path)

def save_image_lossless(img, path):
    success = cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        raise Exception(f'Failed to save image: {path}')
    else:
        print(f'Saved image: {path}')


def load_coefficients(csv_path):
    '''Loads the entire CSV into memory for fast lookups by filename.'''
    all_data = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data[row['Filename']] = {
                'bright': (float(row['Coeff_A_Bright']), float(row['Coeff_B_Bright']), float(row['Coeff_C_Bright'])),
                'dark':   (float(row['Coeff_A_Dark']), float(row['Coeff_B_Dark']), float(row['Coeff_C_Dark']))
            }
    return all_data

def load_fix_parameters(csv_path):
    '''Loads the entire CSV into memory for fast lookups by filename.'''
    all_data = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data[row['Filename']] = {
                'cx': int(row['Center_X']),
                'cy': int(row['Center_Y']),
                'r': int(row['Radius']),
                'lon': float(row['Longitude']),
                'lat': float(row['Latitude'])
            }
    return all_data

def find_earth_disk(img, scale_to_width=800.0, flattening=POLAR_FLATTENING):
    '''
        Detects the Earth disk using downsampling for performance.
        Args:
            img: The input image.
            scale_to_width: The target width for downscaling. Default is 1000
        Returns:
            A tuple (x, y, r) representing the center and radius of the detected circle.
            Returns None if no circle is detected.
    '''
    # 1. Determine downsample factor (e.g., target a width of ~1000px)
    original_h, original_w = img.shape[:2]
    scale_factor = scale_to_width / original_w
    #flattening = 1.4
    new_w = round(original_w * scale_factor)
    new_h = round(original_h * scale_factor / flattening) #flat earth disk is stretched vertically

    # 2. Downsample
    # INTER_AREA is best for shrinking; it prevents aliasing
    small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Pre-process (Blurring is much faster on small_img)
    blurred = cv2.GaussianBlur(small_img, (3, 3), 1)

    # 4. Hough Circles on the small image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(new_w, new_h) // 4,
        param1=50,
        param2=30,
        minRadius=min(new_w, new_h) // 5,
        maxRadius=max(new_w, new_h) * 2 // 3
    )

    if circles is not None:
        # Get the first circle detected
        circle = circles[0, 0]  # [x, y, r]

        # 5. Scale coordinates back to original size
        # We multiply by (1 / scale_factor)
        inv_scale = 1.0 / scale_factor
        orig_x = circle[0] / scale_factor
        orig_y = circle[1] / scale_factor * flattening
        orig_r = circle[2] / scale_factor

        # Free memory of the proxy image
        del small_img, blurred

        return np.array([orig_x, orig_y, orig_r], dtype=np.float32)

    del small_img, blurred
    return None

# Usage example:
# center_x, center_y, radius = find_earth_disk('path_to_image.png')

def find_film_boundary(img):
    '''
    Finds the rectangular coordinates of the film frame within the scan.
    '''

    # Threshold to find anything brighter than the extreme outer border
    # Use a slightly larger blur to merge any internal text/noise into the cosmic block
    blurred = cv2.GaussianBlur(img, (5, 61), 0)
    # Use a medium threshold (e.g., 100) to catch the 'dark' space background
    _, thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del blurred

    if not contours:
        return None

    # Get the bounding box of the largest contour (the film frame)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h

# Usage
# rect = find_film_boundary('image.png')
# if rect:
#    x, y, w, h = rect

def get_bright_sampling_points(rect):
    '''
    Calculates 16 sampling zones 3-5% inside the edge of the rectangle.
    rect: (x, y, w, h)
    Returns a list of (top_left_x, top_left_y, patch_w, patch_h)
    '''
    x, y, w, h = rect

    # Define offsets (3% or 5% for position, 1% for patch size)
    rel_offset = 0.05
    bottom_rel_offset = 0.03
    rel_patch_size = 0.01
    x_off = int(w * rel_offset)
    y_off = int(h * rel_offset)
    bottom_y_off = int(h * bottom_rel_offset)
    pw = max(1, int(w * rel_patch_size))
    ph = max(1, int(h * rel_patch_size))

    # Relative coordinates for the 16 points (normalized 0 to 1)
    # 3 points per side: at 25%, 50%, and 75% along each edge
    positions = [0.10, 0.30, 0.70, 0.90]

    samples = []

    # Left edge (fixed X at 5%)
    for p in positions:
        # 1. Left edge (5% in)
        samples.append((x + x_off, y + int(h * p), pw, ph))

        # 2. Right edge (5% in from right)
        samples.append((x + w - x_off - pw, y + int(h * p), pw, ph))

        # 3. Bottom edge (3% up from bottom)
        samples.append((x + int(w * p), y + h - bottom_y_off - ph, pw, ph))

        # 4. Top edge (Using the deeper top_y_off to clear the bar)
        samples.append((x + int(w * p), y + y_off, pw, ph))

    return samples

def get_dark_sampling_points(rect):
    '''
    Calculates 8 sampling zones 0.5% outside the edge of the rectangle.
    rect: (x, y, w, h)
    Returns a list of (top_left_x, top_left_y, patch_w, patch_h)
    '''
    x, y, w, h = rect

    # Define offsets (3% or 5% for position, 1% for patch size)
    rel_offset = 0.01

    rel_patch_size = 0.005
    x_off = int(w * rel_offset)
    pw = max(1, int(w * rel_patch_size))
    ph = max(1, int(h * rel_patch_size))

    # Relative coordinates for the 8 points (normalized 0 to 1)
    # 4 points per side: at 10%, 30%, 70%, and 90% along each edge
    positions = [0.10, 0.30, 0.70, 0.90]

    samples = []

    # Left edge (fixed X at 5%)
    for p in positions:
        # 1. Left edge (5% in)
        samples.append((x - x_off, y + int(h * p), pw, ph))

        # 2. Right edge (5% in from right)
        samples.append((x + w + x_off - pw, y + int(h * p), pw, ph))

    return samples


def calculate_brightness(img, samples):
    '''
    Extracts the mean brightness for each of the unknown number of patches.
    '''
    results = []
    for (sx, sy, sw, sh) in samples:
        roi = img[sy:sy + sh, sx:sx + sw]
        results.append(np.mean(roi))
    return results


def fit_background_plane(samples, brightness_values):
    '''
    Fits a plane z = Ax + By + C to the sampled brightness data.

    samples: List of (x, y, w, h) for each patch.
    brightness_values: List of mean brightness values for those patches.
    Returns: (A, B, C) coefficients.
    '''
    # We use the center of each patch for the fit
    pts = []
    for (sx, sy, sw, sh) in samples:
        pts.append([sx + sw / 2, sy + sh / 2])

    pts = np.array(pts)
    z = np.array(brightness_values)

    # Create the design matrix for Ax + By + C = z
    # Column 1: x coordinates, Column 2: y coordinates, Column 3: ones (for C)
    A_matrix = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))

    # Solve for [A, B, C] using ordinary least squares
    # lstsq returns a tuple; the first element is the coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, z, rcond=None)

    return coeffs


def predict_brightness(x, y, coeffs):
    '''
    Calculates the expected background brightness at a specific (x, y), given coefficients (A,B,C) for Ax+By+C.
    '''
    A, B, C = coeffs
    return A * x + B * y + C

def visualize_sampling(img, rect, circle, path=None, save_preview=False):
    '''
    Isolated visualization function to prevent RAM congestion in the main loop.
    '''
    if path is None:
        save_preview = False
    else:
        filename = os.path.basename(path)
    # Convert to RGB for matplotlib display if it's grayscale
    if len(img.shape) == 2:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        display_img = img.copy()

    bright_samples = get_bright_sampling_points(rect)
    dark_samples = get_dark_sampling_points(rect)

    rx, ry, rw, rh = rect
    cx, cy, radius = circle

    # Draw the main film boundary in green

    cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 5)

    # Draw the full disk circle in blue
    cv2.ellipse(display_img, (int(cx), int(cy)),
               (int(radius),int(radius*POLAR_FLATTENING)), 0, 0, 360, (0, 0, 255), 5)

    # Draw the center of the film frame in cyan
    cv2.circle(display_img, (rx + rw // 2, ry + rh // 2), int(rh * 0.02), (0, 255, 255), 5)

    # Draw each sampling patch in red
    for (sx, sy, sw, sh) in bright_samples:
        cv2.rectangle(display_img, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), -1)

    # Draw each dark sampling patch in blue
    for (sx, sy, sw, sh) in dark_samples:
        cv2.rectangle(display_img, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), -1)

    plt.figure(figsize=(9, 7))
    plt.imshow(display_img)
    plt.title(f'Sampling Points')
    plt.axis('off')
    if save_preview:
        preview_fn = change_extension(filename, '.png', ('.png', '.bmp', '.jpg', '.jpeg')).replace('.png', '_sampling.png')
        preview_path = get_calibration_folder(Path(path).parent) / f'{preview_fn}'
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Explicitly delete image to free RAM immediately

    del display_img

    return None


def analyze_brightness(path=None, visualize=False, save_preview=False):
    '''
    Processes the standard image to establish target brightness.
    '''
    if path == None:
        ref_path = select_input(mode='file')
    else:
        ref_path = path

    img = load_image_lossless(ref_path)
    rect = find_film_boundary(img)
    if not rect:
        print(f'Reference boundary of {path} not found.')
        return False, None, None, None, None, None
    else:
        print(f'Rectangular frame found.')

    bright_centers = get_bright_sampling_points(rect)
    bright_values = calculate_brightness(img, bright_centers)
    bright_coeffs = fit_background_plane(bright_centers, bright_values)

    dark_centers = get_dark_sampling_points(rect)
    dark_values = calculate_brightness(img, dark_centers)
    dark_coeffs = fit_background_plane(dark_centers, dark_values)

    # Predict target brightness at the frame center
    cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
    bright_target = predict_brightness(cx, cy, bright_coeffs)
    dark_target = predict_brightness(cx, cy, dark_coeffs)

    print(f'Targets found.')

    circle = find_earth_disk(img)
    print(f'Earth disk found.')

    if visualize:
        visualize_sampling(img, rect, circle, path, save_preview)

    del img

    print(ref_path)
    print(bright_target, dark_target, circle)
    print(bright_coeffs, dark_coeffs)

    return True, ref_path, bright_target, dark_target, bright_coeffs, dark_coeffs, circle

def log_to_multiple_files(paths, header_row, data_row):
    '''
    Writes a single row of data to the list of CSV files.
    '''
    paths = paths if hasattr(paths, '__iter__') and not isinstance(paths, str) else [paths]
    for p in paths:
        file_exists = (os.path.exists(p) and os.path.isfile(p))
        with open(p, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header_row)
            writer.writerow(data_row)

def log_coefficients(file_paths, filename, coeffs_b, coeffs_d, pred_z_b, pred_z_d, circle):
    '''
    Saves image metadata and plane coefficients to a CSV file.
    coeffs_b: (A, B, C) from the plane fit to z = Ax + By + C
    coeffs_d: (A, B, C) from the plane fit to z = Ax + By + C
    circle: (center_x, center_y, radius)
    '''
    header_row = ['Filename', 'Coeff_A_Bright', 'Coeff_B_Bright', 'Coeff_C_Bright',
                             'Coeff_A_Dark', 'Coeff_B_Dark', 'Coeff_C_Dark', 'Center_Z_Bright', 'Center_Z_Dark',
                             'Center_X', 'Center_Y', 'Radius']

    # Unpack coefficients
    a_b, b_b, c_b = coeffs_b
    a_d, b_d, c_d = coeffs_d
    cx, cy, radius = circle
    data_row = [filename, a_b, b_b, c_b, a_d, b_d, c_d, pred_z_b, pred_z_d, cx, cy, radius]

    log_to_multiple_files(file_paths, header_row, data_row)


def run_batch_metadata_extraction():
    # 1. Setup
    image_folder = get_path_via_gui(mode='dir')
    if not image_folder: return

    latest_csv, backup_csv = get_calibration_csv_paths(image_folder)
    images = get_image_list(image_folder)

    print(f"Logging to: {latest_csv}")

    for path in images:
        filename = os.path.basename(path)
        if 'VS' in filename:
            print(f"Skipped: {filename} is a VIS image.")
            continue
        else:
            success, path, bright, dark, bright_coeffs, dark_coeffs, circle = analyze_brightness(path=path, visualize=True, save_preview=True)
            if success:
                # Package data and log
                log_coefficients([latest_csv, backup_csv], filename, bright_coeffs, dark_coeffs, bright, dark, circle)

                print(f"Analyzed: {filename} (Z_bright: {bright:.2f}, Z_dark: {dark:.2f})")
            else:
                print(f"Skipped: {filename} (Boundary error)")


def generate_background_planes(shape, coeffs):
    '''
    Creates a 2D array the same size as the image representing the background plane.
    coeffs: (A, B, C)
    '''
    h, w = shape
    # Create coordinate grids
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate Z = Ax + By + C in one vectorized step
    plane = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    return plane

def apply_nonlinear_calibration(img, bright_coeffs, dark_coeffs, std_bright, std_dark):
    '''
    Calibrates image using a brightness-dependent non-linear offset.
    Solves A*brightness + B*sqrt(brightness) analytically.
    '''
    h, w = img.shape

    # 1. Generate local background predictions for every pixel
    z_b = generate_background_planes((h, w), bright_coeffs)
    z_d = generate_background_planes((h, w), dark_coeffs)

    # 2. Calculate local offsets needed for 'Standard' targets
    # Note: These planes represent the 'Current' local conditions
    off_b_plane = std_bright - z_b
    off_d_plane = std_dark - z_d
    print('before sqrt')
    # 4. Analytical solution for A and B (Denominator is shared)
    # Denom = (Z_b * sqrt(Z_d)) - (Z_d * sqrt(Z_b))
    sqrt_zb = np.sqrt(z_b)
    sqrt_zd = np.sqrt(z_d)

    denominator = (z_b * sqrt_zd) - (z_d * sqrt_zb)

    # Safety check for denominator to avoid NaN
    denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)

    # Solve for A and B planes
    A_plane = (off_b_plane * sqrt_zd - off_d_plane * sqrt_zb) / denominator
    del sqrt_zb, sqrt_zd
    B_plane = (z_b * off_d_plane - z_d * off_b_plane) / denominator

    del z_b, z_d, off_b_plane, off_d_plane, denominator

    # 5. Apply non-linear offset to actual pixel values
    # offset = A*img + B*sqrt(img)
    img_f = img.astype(np.float32)
    pixel_offset = A_plane * img_f + B_plane * np.sqrt(img_f)

    del A_plane, B_plane

    print('before clip')

    # 6. Final result and clipping
    calibrated = img_f + pixel_offset
    del img_f
    del pixel_offset
    calibrated_img = np.clip(calibrated, 0, 255).astype(np.uint8)
    if len(calibrated_img.shape) > 2:
        # Convert from BGR or RGBA to Grayscale
        # If the error said CV_8UC4, it's likely BGRA
        gray_img = cv2.cvtColor(calibrated_img, cv2.COLOR_BGRA2GRAY)
    else:
        gray_img = calibrated_img

    del calibrated

    return gray_img

def run_batch_image_calibration(std_bright=221, std_dark=39, csv_path=None):
    '''

    :param std_bright: standard bright region brightness
    :param std_dark: standard dark region brightness
    :return:
    '''
    # 1. Setup
    print('start')
    image_folder = get_path_via_gui(mode='dir')
    if not image_folder: return

    if csv_path is None:
        csv_path, _ = get_calibration_csv_paths(image_folder)
    all_coeffs_dict = load_coefficients(csv_path)
    images = get_image_list(image_folder)

    for path in images:
        filename = os.path.basename(path)
        if 'VS' in filename:
            print(f"Skipped: {filename} is a VIS image.")
            continue
        else:
            coeffs = all_coeffs_dict.get(filename)
            bright_coeffs, dark_coeffs = coeffs['bright'], coeffs['dark']
            print(f'{filename} has calibration: {coeffs}.')
            img = load_image_lossless(path)

            calibrated_img = apply_nonlinear_calibration(img, bright_coeffs, dark_coeffs, std_bright, std_dark)
            del img
            print(f'{filename} calibrated, now saving')
            cal_name = change_extension(filename, '.png', ('.png', '.bmp', '.jpg', '.jpeg'))
            cal_name = cal_name.replace('.png', '_calibrated.png')
            cal_dir = get_calibration_folder(image_folder)
            cal_dir.mkdir(exist_ok=True)
            cal_im_dir = cal_dir / 'Calibrated_PNG'
            cal_im_dir.mkdir(exist_ok=True)
            cal_im_path = cal_im_dir / cal_name

            save_image_lossless(calibrated_img, cal_im_path)

            del calibrated_img



def run_batch_crop_png(mode='dir',csv_path=None):
    print('start')
    if mode=='dir':
        image_folder = get_path_via_gui(mode=mode)
        if not image_folder: return
        images = get_image_list(image_folder)
        image_folder = Path(image_folder)
    elif mode=='file':
        image = get_path_via_gui(mode=mode)
        if not image: return
        images = [image]
        image_folder = Path(image).parent
    else:
        print('Invalid mode, choose "dir" or "file".')
        return

    if csv_path is None:
        csv_path = get_path_via_gui(mode='csv')
    all_fixes_dict = load_fix_parameters(csv_path)


    for path in images:
        filename = os.path.basename(path)
        fix = all_fixes_dict.get(filename)
        cx, cy, r = fix['cx'], fix['cy'], fix['r']

        print(f'{filename} is at: ({cx}, {cy}) with radius {r}.')
        img = load_image_lossless(path)

        cropped_img = img[cy-r:cy+r, cx-r:cx+r].copy()
        del img
        print(f'{filename} cropped, now saving')
        new_name = change_extension(filename, '.png', ('.png', '.bmp', '.jpg', '.jpeg'))
        new_name = new_name.replace('.png', f'_cropped_{int(r*2)}px.png')
        new_im_dir = image_folder / 'Cropped_PNG'
        new_im_dir.mkdir(exist_ok=True)
        new_im_path = new_im_dir / new_name

        save_image_lossless(cropped_img, new_im_path)

        del cropped_img


def Film2BT(img):
    Gray_BT_matching = np.array([
        [20, 27],
        [53, 22],
        [75, 18],
        [100, 11],
        [122, 1],
        [149, -15],
        [171, -31],
        [183, -42],
        [195.5, -54],
        [204, -64],
        [209.5, -70],
        [218.2, -80],
        [227, -92],

    ]).T
    cs = CubicSpline(Gray_BT_matching[0], Gray_BT_matching[1])

    return cs(img)


def plot_BT_image(image_path, image_folder, cn = 'ir_cc_2',gauss_smooth=True, zoom_in=True, save_image=False):
    timestamp = os.path.basename(image_path).split('_')[2]
    img = load_image_lossless(image_path)
    if gauss_smooth:
        img = cv2.GaussianBlur(img, (3, 3), 1)
    display_img = np.array(img)
    del img
    img_bt = Film2BT(display_img)
    del display_img

    zoom_in = True
    if zoom_in:
        img_bt = img_bt[400:1200, 400:1200]


    if hasattr(cn, '__iter__') and not isinstance(cn, str):
        cmap_names = cn
    else:
        cmap_names = [cn]
    for cmap_name in cmap_names:
        cmap, vmin, vmax = my_cmap.cmap_fetch(cmap_name)

        if not save_image:
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(f'Rita {timestamp}')
            gs = fig.add_gridspec(1, 1)
            '''
            ax1 = fig.add_subplot(gs[0,0])
            ax1.hist(img_bt.ravel(), bins=range(-90, 30))
            '''

            ax2 = fig.add_subplot(gs[0, :])
            im = ax2.imshow(img_bt, cmap=cmap, vmin=vmin, vmax=vmax)

            cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
            plt.colorbar(im, cax=cax)
            plt.show()
        else:
            filename = os.path.basename(image_path)
            new_name = change_extension(filename, '.png', ('.png', '.bmp', '.jpg', '.jpeg'))
            new_name = new_name.replace('_calibrated', '').replace('_cropped','')
            new_name = new_name.replace('_IR_', f'_{cmap_name}_')
            new_im_dir = image_folder
            new_im_dir.mkdir(exist_ok=True)
            new_im_path = new_im_dir / new_name

            plt.imsave(new_im_path, img_bt, cmap=cmap, vmin=vmin, vmax=vmax, format='png', pil_kwargs={'compress_level': 9})
            print(f'{new_im_path} saved.')

    del img_bt

def batch_plot_BT_image(mode='file', cmap_name='ir_cc_2', gauss_smooth=True, zoom_in=True, save_image=True):
    print('start')
    if mode=='dir':
        image_folder = get_path_via_gui(mode=mode)
        if not image_folder: return
        images = get_image_list(image_folder)
        image_folder = Path(image_folder)
    elif mode=='file':
        image = get_path_via_gui(mode=mode)
        if not image: return
        images = [image]
        image_folder = Path(image).parent
    else:
        print('Invalid mode, choose "dir" or "file".')
        return

    for path in images:

        plot_BT_image(path, image_folder, cmap_name, save_image=save_image)
