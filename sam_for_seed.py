import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import pandas as pd
from skimage.measure import regionprops

# Define paths
input_directory_1 = "../processed_data/22-MU-BURKEY-SEED-SCAN_cropped_without_tags"  # Replace with your first input image directory
#input_directory_2 = "../processed_data/22-MU-BURKEY-SEED-SCAN_cropped_without_tags"  # Replace with your second input image directory
output_directory = "22-MU-BURKEY-SEED-SCAN_cropped_without_tags_without_tags_sam"  # Replace with your output directory

os.makedirs(output_directory, exist_ok=True)

# Load the SAM model
sam_checkpoint = "sam_vit_l_0b3195.pth"  # Replace with your SAM checkpoint
model_type = "vit_l"  # Replace with your desired SAM model type
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Function to process and save segmented image with histogram and filtering
def process_image(image_path, output_directory):
    # Load the image
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    output_name = os.path.splitext(image_name)[0]

    # Generate masks using SAM
    masks = mask_generator.generate(image)

    # Create a blank image for the color-coded segmentation
    color_mask = np.zeros_like(image)
    areas = []

    # Assign random colors to each mask and calculate areas
    for mask in masks:
        segmentation = mask["segmentation"]
        area = np.sum(segmentation)
        areas.append(area)

        # Generate random color for the mask
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        color_mask[segmentation] = color

    # Save the color-coded mask
    color_mask_path = os.path.join(output_directory, f"{output_name}_color_mask.png")
    cv2.imwrite(color_mask_path, color_mask)

    # Plot and save the histogram of areas
    #plt.figure(figsize=(8, 6))
    #plt.hist(areas, bins=20, color='blue', alpha=0.7)
    #plt.title("Histogram of Segment Sizes")
    #plt.xlabel("Segment Area (pixels)")
    #plt.ylabel("Frequency")
    #histogram_path = os.path.join(output_directory, f"{output_name}_histogram.png")
    #plt.savefig(histogram_path)
    #plt.close()

    # Calculate area thresholds for filtering (e.g., remove outliers)
    areas_np = np.array(areas)
    q1, q3 = np.percentile(areas_np, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a filtered mask based on area thresholds
    # Create a list to store the properties of each filtered mask
    mask_properties = []
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for mask, area in zip(masks, areas):
        if lower_bound <= area <= upper_bound:
            segmentation = mask["segmentation"]
            filtered_mask[segmentation] = 255
            # Calculate the mean hue value of the segmented region
            mean_hue = np.mean(hsv_image[segmentation, 0])

            # Calculate properties using regionprops
            label_img = segmentation.astype(np.uint8)
            props = regionprops(label_img)[0]
            major_axis_length = props.major_axis_length
            minor_axis_length = props.minor_axis_length
            aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0

            # Append the properties to the list
            mask_properties.append({
                "Area": area,
                "Major Axis Length": major_axis_length,
                "Minor Axis Length": minor_axis_length,
                "Aspect Ratio": aspect_ratio,
                "Mean Hue": mean_hue
            })

    # Convert the list to a DataFrame and save as CSV
    df = pd.DataFrame(mask_properties)
    csv_path = os.path.join(output_directory, f"{output_name}_seed_properties.csv")
    df.to_csv(csv_path, index=False)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    for mask, area in zip(masks, areas):
        if lower_bound <= area <= upper_bound:
            segmentation = mask["segmentation"]
        
            filtered_mask[segmentation] = 255

    # Save the filtered mask applied to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(filtered_mask, cv2.COLOR_BGR2GRAY))
    filtered_mask_path = os.path.join(output_directory, f"{output_name}_filtered_mask.png")
    cv2.imwrite(filtered_mask_path, filtered_image)


# Function to process images from a given directory
def process_images_from_directory(input_directory, output_directory):
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if '_5' not in file_name:
                process_image(file_path, output_directory)

# Process images from both directories
process_images_from_directory(input_directory_1, output_directory)
#process_images_from_directory(input_directory_2, output_directory)

print(f"Processed images saved in {output_directory}")