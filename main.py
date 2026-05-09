import gc


import my_cmap
import my_functions

#reference_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/BMP/GMS1_IR_19781022Z2333.bmp'
#my_functions.run_batch_metadata_extraction()
#success, path, bright, dark, bright_coeffs , dark_coeffs, circle = my_functions.analyze_brightness(path=reference_path, visualize=True)

std_bright, std_dark = 221, 39
csv_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/Calibration/film_background_analysis_20260509_192259.csv'

#my_functions.run_batch_image_calibration(std_bright, std_dark, csv_path=csv_path)

fix_path = 'D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/Calibration/Calibrated_PNG/Fix.csv'

mode = ['dir','file'][0]
cmap_names = ['ir_cc_2']

#my_functions.run_batch_crop_png(mode=mode,csv_path=fix_path)

image_path = ('D:/Documents/TC and Weather/Images/W/1900-79/78 Rita/GMS-1 Scan/'
              'Calibration/Calibrated_PNG/Cropped_PNG/GMS1_IR_19781023Z0533_calibrated_cropped_1600px.png')

my_functions.batch_plot_BT_image(mode=mode, cmap_name=cmap_names)

gc.collect()


