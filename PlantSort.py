# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:00:20 2023

@author: Sailing587
"""

import glob
import cv2
import matplotlib.pyplot as plt
from colorthief import ColorThief
import numpy as np

bad_plants = []



green_lower = np.array([30, 60, 60])
green_upper = np.array([90, 255, 255])


brown_lower = np.array([10, 20, 20])
brown_upper = np.array([20, 255, 255])
black_lower = np.array([0,0,0])
black_lower1 = np.array([20, 20, 20])
black_lower2 = np.array([20, 20, 20])
black_upper = np.array([180, 255, 30])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])


# Define the color ranges for dark dirty green
dark_green_lower = np.array([32, 50, 1])
dark_green_upper = np.array([90, 120, 80])

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 100000

# Filter by Colour (0 = black)
params.filterByColor = 1
params.blobColor = 0
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity = 1
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 1
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
  detector = cv2.SimpleBlobDetector(params)
else : 
  detector = cv2.SimpleBlobDetector_create(params)



def remove_color(image, color, threshold=50):
    # Convert RGB color to HSV (OpenCV format)
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Create a mask for pixels similar to the specified color
    lower_bound = np.array([c - threshold for c in hsv_color])
    upper_bound = np.array([c + threshold for c in hsv_color])
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)

    # Set matching pixels to white
    image[mask > 0] = [255, 255, 255]

    return image

def remove_black(image, threshold=30):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for dark pixels
    mask_black = (gray <= threshold)

    # Set dark pixels to white
    image[mask_black] = [255, 255, 255]

    return image

# Function to calculate the percentage of non-green pixels
def calculate_non_green_percentage(processed_image, white_threshold=200):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

    # Create a mask for green pixels
    mask_green = cv2.inRange(hsv_image, green_lower, green_upper)

    # Create a mask for brown pixels
    mask_brown = cv2.inRange(hsv_image, brown_lower, brown_upper)
    
    #mask_brown2 = cv2.inRange(hsv_image, brown_lower2, brown_upper2)
    
    # Create a mask for black pixels
    mask_black = cv2.inRange(hsv_image, black_lower2, black_upper)
    
    # Create a mask for yellow pixels
    mask_yellow = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    
    # Create a mask for dark green pixels
    mask_dirty_green = cv2.inRange(hsv_image, dark_green_lower, dark_green_upper)
    
    
    

    # Create a mask for white pixels
    mask_white = cv2.inRange(processed_image, (white_threshold, white_threshold, white_threshold), (255, 255, 255))

    # Exclude white pixels from the green and brown masks
    mask_brown = cv2.bitwise_and(mask_brown, cv2.bitwise_not(mask_white))
    #mask_brown2 = cv2.bitwise_and(mask_brown2, cv2.bitwise_not(mask_white))
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_white | mask_brown))
    #mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_white | mask_brown | mask_brown2))


    # Combine the masks to create a mask for non-green pixels
    mask_non_green = mask_brown + mask_black + mask_yellow
    #mask_non_green = mask_brown + mask_brown2 + mask_black + mask_yellow

    # Count the number of non-green and white pixels
    non_green_pixel_count = cv2.countNonZero(mask_non_green)
    white_pixel_count = cv2.countNonZero(mask_white)
    green_pixel_count = cv2.countNonZero(mask_green)
    brown_pixel_count = cv2.countNonZero(mask_brown)
    #brown_pixel_count2 = cv2.countNonZero(mask_brown2)
    total_brown_pixel_count = brown_pixel_count
    #total_brown_pixel_count = brown_pixel_count + brown_pixel_count2
    black_pixel_count = cv2.countNonZero(mask_black)
    yellow_pixel_count = cv2.countNonZero(mask_yellow)
    dark_green_pixel_count = cv2.countNonZero(mask_dirty_green)
    black_pixel_count2 = black_pixel_count + dark_green_pixel_count 
    black_ratio = black_pixel_count2/green_pixel_count
    
    image_green = cv2.bitwise_and(processed_image, processed_image, mask=mask_green)
    
    #Check for rusting with blob detector
    # Detect blobs
    keypoints = detector.detect(mask_non_green)
    
    # Draw detected blobs as red circles.
    im_with_keypoints = cv2.drawKeypoints(mask_brown, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
    plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Blob detection')
    plt.axis('off')  # Hide the axis
    plt.show()    
        
    
    #Check for rusting
    if len(keypoints) > 50: 
        # Calculate the total number of pixels (excluding white pixels)
        total_pixels = total_brown_pixel_count + green_pixel_count
        # Calculate the percentage of non-green pixels
        non_green_percentage = (total_brown_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0 
        
        print('Rusting on leaves! ')
        print('Remedy by using Neem oil, dusting of Sulphur or rust control fungicides. Climate change: Too much water or too little airflow and sunlight')
    
    
    
    #Check for black spots    
    elif black_ratio > 0.05: 
        # Calculate the total number of pixels (excluding white pixels)
        total_pixels = black_pixel_count2 + green_pixel_count        

        # Calculate the percentage of non-green pixels
        non_green_percentage = (black_pixel_count2 / total_pixels) * 100 if total_pixels > 0 else 0 
    
        image_green = cv2.bitwise_and(processed_image, processed_image, mask=mask_green) 
        print('Black pixel count: ', black_pixel_count2 )
        print('Black ratio: ', black_ratio )
        print('Black spot percentage: ', non_green_percentage)
        print('Can try to wash off with soap and water or increase temperature to kill off disease')
    
    #Check for browning
    elif brown_pixel_count > black_pixel_count2 & yellow_pixel_count: 

        # Calculate the total number of pixels (excluding white pixels)
        total_pixels = total_brown_pixel_count + green_pixel_count
        #total_pixels = non_green_pixel_count + green_pixel_count
    
        # Calculate the percentage of non-green pixels
        non_green_percentage = (total_brown_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0 
    
        image_green = cv2.bitwise_and(processed_image, processed_image, mask=mask_green)
        
        #Black over brown percentage
        black_brown_percentage = black_pixel_count / brown_pixel_count

        

        if non_green_percentage < 1:
            print('Trim brown edges or cut the leaf off Too little moisture in the air (lack of humidity) or Inconsistent watering habits ')
            print('Browning percentage: ', non_green_percentage)
        
        
    #Check for yellowing
    elif  yellow_pixel_count > brown_pixel_count & black_pixel_count2:
        
        # Calculate the total number of pixels (excluding white pixels)
        total_pixels = yellow_pixel_count + green_pixel_count
        #total_pixels = non_green_pixel_count + green_pixel_count
    
        # Calculate the percentage of non-green pixels
        non_green_percentage = (yellow_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0 
    
        image_green = cv2.bitwise_and(processed_image, processed_image, mask=mask_green)
        
        if non_green_percentage > 0.4:
            print('Yellowing, lack of nutrients (manganese, iron or magnesium)')
            print('Yellowing percentage: ', non_green_percentage)
        else:
            print('Plant is good')
    
    elif white_pixel_count > 5000:
        
        print('White spots')
        
    
        
    
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Without background')
    plt.axis('off')  # Hide the axis
    plt.show()    
    
    plt.imshow(cv2.cvtColor(image_green, cv2.COLOR_BGR2RGB))
    plt.title('Green Pixels')
    plt.axis('off')  # Hide the axis
    plt.show()    
    

    print("Non-green pixels:", non_green_pixel_count)
    print("Yellow pixels:", yellow_pixel_count)
    print("Green pixels:", green_pixel_count)
    print("Brown pixels:", brown_pixel_count)
    print("Black pixels:", black_pixel_count2)
    print("White pixels:", white_pixel_count)
    return non_green_percentage


# Get image from files
image_names = glob.glob('./Plant images/Image/*.JPG')
image_names.sort()


# Iterate through the image file names
for image_path in image_names:
    image = cv2.imread(image_path)
    
    # Use ColorThief to get the dominant color
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    
    # Remove the dominant color
    processed_image = remove_color(image, dominant_color)

    # Remove black color
    processed_image1 = remove_black(processed_image, threshold=30)
    
    
    plt.imshow(cv2.cvtColor(processed_image1, cv2.COLOR_BGR2RGB))
    plt.title('without black, ' )
    plt.axis('off')  # Hide the axis
    plt.show() 
    
    # Calculate non-green percentage
    non_green_percentage = calculate_non_green_percentage(processed_image1, white_threshold=200)
    


    if non_green_percentage > 0.8 :
        print('This plant image needs notice: ', image_path)
        bad_plants.append(image_path)
        
    #print('Plants with browning are: ', bad_plants)


#SORT THE IMAGES, STATE THE DISEASE, GIVE REMEDY TO USER (Yellow spots might be counted as green, so might need to use blob detection on green pixel pictures)




