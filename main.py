import gc
import numpy as np
import os
import time

import my_cmap
import my_functions
from my_functions import batch_plot_BT_image

start_time = time.time()

std_bright, std_dark = 221, 39
reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/BMP_IR/GMS1_IR_19781022Z2333.bmp'
csv_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/Calibration/film_background_analysis_20260515_034804.csv'
fix_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/Calibration/Calibrated_PNG/Fix_78Rita.csv'
image_path = ('D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/'
              'Calibration/Calibrated_PNG/Cropped_PNG/GMS1_IR_19781023Z0533_calibrated_cropped_1600px.png')



'''
#success, path, bright, dark, bright_coeffs , dark_coeffs, circle = my_functions.analyze_brightness(path=reference_path, visualize=True)

img = my_functions.load_image_lossless(reference_path)
print(f'Original max before calibration: {np.max(img)}')
all_coeffs_dict = my_functions.load_coefficients(csv_path)

filename = os.path.basename(reference_path)

coeffs = all_coeffs_dict.get(filename)
bright_coeffs, dark_coeffs = coeffs['bright'], coeffs['dark']
print(f'{filename} has calibration: {coeffs}.')

new_img = my_functions.apply_nonlinear_calibration(img, bright_coeffs, dark_coeffs, std_bright, std_dark)
print(f'Original max after calibration, before saving: {np.max(new_img)}')

test_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/Test_IR_test.png'
retrieve_img = my_functions.load_image_lossless(test_path)
print(f'Max after saving: {np.max(retrieve_img)}')


my_functions.save_image_lossless(new_img, test_path)

retrieve_img = my_functions.load_image_lossless(test_path)
print(f'Max after saving again: {np.max(retrieve_img)}')'''




mode = ['dir','file'][0]
cmap_names = ['ir_bd','ir_cc','ir_zehr','ir_cc_2']
#cmap_names = ['ir_cc_2']

save_metadata = True
#my_functions.run_batch_metadata_extraction(save_csv=save_metadata,save_preview=True)

#my_functions.run_batch_image_calibration(mode, std_bright, std_dark, csv_path=csv_path, csv_fix_path=fix_path, do_crop=True)

#my_functions.run_batch_image_calibration(mode, std_bright, std_dark, csv_path=None, csv_fix_path=None, do_crop=True)

#my_functions.run_batch_crop_png(mode=mode,csv_path=fix_path)


zoom_in = True
ask_name = False

if ask_name:
    batch_name = input('Enter batch name: ')
else:
    batch_name = f'78_Olive'

my_functions.batch_plot_BT_image(mode=mode, cmap_name=cmap_names, gauss_smooth=True, zoom_in=zoom_in, batch_name=batch_name)

gc.collect()

print(f'{time.time() - start_time} seconds spent.')

