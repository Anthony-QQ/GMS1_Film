import csv
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog


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
    else:
        raise ValueError('Invalid mode. Use "file" or "dir".')
    root.destroy()
    return path

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
        filetypes=[('Image files', '*.png *.bmp *.jpg')]
    )
    else:
        path = filedialog.askdirectory(title='Select Folder for Batch Processing')

    root.destroy()
    return path

def get_image_list(folder_path):
    '''
    Returns a list of all processable images in a folder.
    '''
    extensions = ('.png', '.bmp', '.jpg', '.jpeg')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(extensions)]

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
        # Convert from BGR or RGBA to Grayscale
        # If the error said CV_8UC4, it's likely BGRA
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
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

def find_earth_disk(image_path):
    '''
    Detects the Earth disk in a full-disk GEO image and returns the center
    coordinates and radius.
    '''
    # Load image in grayscale (lossless)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not load image at {image_path}')

    # Apply Gaussian Blur to reduce digitization noise/grain before edge detection
    blurred = cv2.GaussianBlur(img, (9, 9), 2)

    # Use Hough Circle Transform
    # param1: Higher threshold for Canny edge detector
    # param2: Accumulator threshold (smaller = more circles detected)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(img.shape) // 2,
        param1=50,
        param2=30,
        minRadius=min(img.shape) // 4,
        maxRadius=min(img.shape) // 2
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return the most prominent circle: [x, y, radius]
        return circles[0, 0]

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
    _, thresh = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

def visualize_sampling(img, rect, path=None, save_preview=False):
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

    # Draw the main film boundary in green
    rx, ry, rw, rh = rect
    cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 5)

    # Draw the center of the film frame in blue
    cv2.circle(display_img, (rx + rw // 2, ry + rh // 2), int(rh * 0.02), (0, 0, 255), -1)

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
        preview_path = get_calibration_folder(Path(path).parent) / f'{filename}_sampling.png'
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

    if visualize:
        visualize_sampling(img, rect, path, save_preview)

    del img

    return True, ref_path, bright_target, dark_target, bright_coeffs, dark_coeffs

def log_to_multiple_files(paths, header_row, data_row):
    '''
    Writes a single row of data to the list of CSV files.
    '''
    for p in paths:
        file_exists = (os.path.exists(p) and os.path.isfile(p))
        with open(p, 'a', newline='') as f:
            writer = csv.writer(f)
            if file_exists:
                os.remove(p)
            writer.writerow(header_row)
            writer.writerow(data_row)

def log_coefficients(file_paths, filename, coeffs_b, coeffs_d, pred_z_b, pred_z_d):
    '''
    Saves image metadata and plane coefficients to a CSV file.
    coeffs_b: (A, B, C) from the plane fit to z = Ax + By + C
    coeffs_d: (A, B, C) from the plane fit to z = Ax + By + C
    '''
    header_row = ['Filename', 'Coeff_A_Bright', 'Coeff_B_Bright', 'Coeff_C_Bright',
                             'Coeff_A_Dark', 'Coeff_B_Dark', 'Coeff_C_Dark', 'Center_Z_Bright', 'Center_Z_Dark']

    # Unpack coefficients
    a_b, b_b, c_b = coeffs_b
    a_d, b_d, c_d = coeffs_d
    data_row = [filename, a_b, b_b, c_b, a_d, b_d, c_d, pred_z_b, pred_z_d]

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
            success, path, bright, dark, bright_coeffs, dark_coeffs = analyze_brightness(path=path, visualize=True, save_preview=True)
            if success:
                # Package data and log
                log_coefficients([latest_csv, backup_csv], filename, bright_coeffs, dark_coeffs, bright, dark)

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

def run_batch_image_calibration(std_bright=221, std_dark=39):
    '''

    :param std_bright: standard bright region brightness
    :param std_dark: standard dark region brightness
    :return:
    '''
    # 1. Setup
    print('start')
    image_folder = get_path_via_gui(mode='dir')
    if not image_folder: return

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

            cal_name = filename.replace('.png', '_calibrated.png')
            cal_dir = get_calibration_folder(image_folder)
            cal_dir.mkdir(exist_ok=True)
            cal_im_dir = cal_dir / 'Calibrated_PNG'
            cal_im_dir.mkdir(exist_ok=True)
            cal_im_path = cal_im_dir / cal_name

            save_image_lossless(calibrated_img, cal_im_path)

            del calibrated_img

'''
image_path = select_input()
img = load_image_lossless(image_path)
rect = find_film_boundary(img)
if rect:
    visualize_sampling(img, rect)

reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/PNG/GMS1_IR_19781022Z2333.png'
path, bright, dark, bright_coeffs , dark_coeffs = analyze_brightness(path=reference_path, visualize=True)
print(path)
print(bright,dark)
print(bright_coeffs, dark_coeffs)

std_dark, std_bright = 39, 221
csv_path, _ = get_calibration_csv_paths(Path(reference_path).parent)
img = load_image_lossless(reference_path)
filename = os.path.basename(reference_path)

all_coeffs_dict = load_coefficients(csv_path)

coeffs = all_coeffs_dict.get(filename)
bright_coeffs, dark_coeffs = coeffs['bright'], coeffs['dark']
print(coeffs)

calibrated_img = apply_nonlinear_calibration(img, bright_coeffs, dark_coeffs, std_bright, std_dark)

print(f'image calibrated, now plotting')

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Show the first image
axes[0].imshow(img)
axes[0].set_title(f'Original')
axes[0].axis('off')

# Show the second image
axes[1].imshow(calibrated_img)
axes[1].set_title(f'Calibrated')
axes[1].axis('off')

plt.tight_layout()
plt.show()
'''
reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/PNG/GMS1_IR_19781022Z1733.png'
#reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/PNG/GMS1_IR_19781023Z0233.png'
#reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/PNG/GMS1_IR_19781024Z1533.png'
#reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/PNG/GMS1_IR_19781025Z0233.png'
#run_batch_metadata_extraction()
#success, path, bright, dark, bright_coeffs , dark_coeffs = analyze_brightness(path=reference_path, visualize=True)

std_bright, std_dark = 221, 39

#run_batch_image_calibration(std_bright, std_dark)

gc.collect()


