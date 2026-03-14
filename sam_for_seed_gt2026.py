import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import pandas as pd
from skimage.measure import regionprops

# Define paths
input_directory_1 = "../data/seed_gt/cropped"  # Replace with your first input image directory
#input_directory_2 = "../processed_data/22-MU-BURKEY-SEED-SCAN_cropped_without_tags"  # Replace with your second input image directory
output_directory = "seed_gt_cropped_grid"  # Replace with your output directory

os.makedirs(output_directory, exist_ok=True)

# Load the SAM model
sam_checkpoint = "sam_vit_l_0b3195.pth"  # Replace with your SAM checkpoint
model_type = "vit_l"  # Replace with your desired SAM model type
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

GRID_ROWS = 6
GRID_COLS = 10
MAX_DISTANCE_TO_GRID = 60.0

# Corner coordinates provided as (x, y) = (col, row)
TOP_LEFT = np.array([209.0, 316.0])
TOP_RIGHT = np.array([910.0, 306.0])
BOTTOM_LEFT = np.array([199.0, 705.0])
BOTTOM_RIGHT = np.array([902.0, 696.0])


def generate_reference_grid(rows, cols, top_left, top_right, bottom_left, bottom_right):
    grid_records = []

    for row_idx in range(rows):
        row_alpha = row_idx / (rows - 1) if rows > 1 else 0.0
        left_point = (1.0 - row_alpha) * top_left + row_alpha * bottom_left
        right_point = (1.0 - row_alpha) * top_right + row_alpha * bottom_right

        for col_idx in range(cols):
            col_alpha = col_idx / (cols - 1) if cols > 1 else 0.0
            point_xy = (1.0 - col_alpha) * left_point + col_alpha * right_point
            grid_number = row_idx * cols + col_idx + 1

            grid_records.append({
                "Grid Row": row_idx + 1,
                "Grid Col": col_idx + 1,
                "Grid Number": grid_number,
                "Grid X": point_xy[0],
                "Grid Y": point_xy[1]
            })

    return pd.DataFrame(grid_records)


def assign_objects_to_grid(df_objects, df_grid, max_distance_to_grid):
    if df_objects.empty:
        return df_objects

    centroid_xy = df_objects[["Centroid Col", "Centroid Row"]].to_numpy(dtype=float)
    grid_xy = df_grid[["Grid X", "Grid Y"]].to_numpy(dtype=float)

    object_count = centroid_xy.shape[0]
    grid_count = grid_xy.shape[0]

    distances = np.linalg.norm(centroid_xy[:, None, :] - grid_xy[None, :, :], axis=2)

    assigned_grid_for_object = np.full(object_count, -1, dtype=int)
    grid_is_taken = np.zeros(grid_count, dtype=bool)

    pair_order = np.argsort(distances, axis=None)
    for pair_idx in pair_order:
        object_idx = pair_idx // grid_count
        grid_idx = pair_idx % grid_count

        if assigned_grid_for_object[object_idx] == -1 and not grid_is_taken[grid_idx]:
            assigned_grid_for_object[object_idx] = grid_idx
            grid_is_taken[grid_idx] = True

            if np.all(assigned_grid_for_object != -1) or np.all(grid_is_taken):
                break

    grid_number = np.full(object_count, np.nan)
    grid_row = np.full(object_count, np.nan)
    grid_col = np.full(object_count, np.nan)
    distance_to_grid = np.full(object_count, np.nan)

    assigned_mask = assigned_grid_for_object != -1
    if np.any(assigned_mask):
        assigned_object_idx = np.where(assigned_mask)[0]
        assigned_grid_idx = assigned_grid_for_object[assigned_mask]

        grid_number[assigned_mask] = df_grid.iloc[assigned_grid_idx]["Grid Number"].to_numpy()
        grid_row[assigned_mask] = df_grid.iloc[assigned_grid_idx]["Grid Row"].to_numpy()
        grid_col[assigned_mask] = df_grid.iloc[assigned_grid_idx]["Grid Col"].to_numpy()
        distance_to_grid[assigned_mask] = distances[assigned_object_idx, assigned_grid_idx]

    df_objects["Grid Number"] = grid_number
    df_objects["Grid Row"] = grid_row
    df_objects["Grid Col"] = grid_col
    df_objects["Distance To Grid"] = distance_to_grid

    df_objects["Grid Number"] = pd.Series(df_objects["Grid Number"], dtype="Int64")
    df_objects["Grid Row"] = pd.Series(df_objects["Grid Row"], dtype="Int64")
    df_objects["Grid Col"] = pd.Series(df_objects["Grid Col"], dtype="Int64")

    within_distance = df_objects["Distance To Grid"] <= max_distance_to_grid
    df_objects = df_objects[within_distance].copy()

    if not df_objects.empty:
        df_objects = df_objects.sort_values(by=["Grid Number", "Distance To Grid"]).reset_index(drop=True)
        df_objects.insert(0, "Object Number", np.arange(1, len(df_objects) + 1, dtype=int))
    else:
        df_objects.insert(0, "Object Number", pd.Series(dtype="Int64"))

    return df_objects

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
    q1, q3 = np.percentile(areas_np, [10, 90])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a filtered mask based on area thresholds
    # Create a list to store the properties of each filtered mask
    mask_properties = []
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for mask_index, (mask, area) in enumerate(zip(masks, areas)):
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
                "Mask Index": mask_index,
                "Area": area,
                "Major Axis Length": major_axis_length,
                "Minor Axis Length": minor_axis_length,
                "Aspect Ratio": aspect_ratio,
                "Mean Hue": mean_hue,
                "Centroid Row": props.centroid[0],
                "Centroid Col": props.centroid[1]
            })

    # Convert the list to a DataFrame and assign each centroid to the nearest grid location
    df = pd.DataFrame(mask_properties)

    if not df.empty:
        df_grid = generate_reference_grid(
            GRID_ROWS,
            GRID_COLS,
            TOP_LEFT,
            TOP_RIGHT,
            BOTTOM_LEFT,
            BOTTOM_RIGHT
        )

        df = assign_objects_to_grid(df, df_grid, MAX_DISTANCE_TO_GRID)

    valid_mask_indices = set(df["Mask Index"].astype(int).tolist()) if not df.empty else set()

    csv_path = os.path.join(output_directory, f"{output_name}_seed_properties.csv")
    df.to_csv(csv_path, index=False)

    annotated_image = image.copy()
    if not df.empty:
        for _, row in df.iterrows():
            centroid_col = int(round(row["Centroid Col"]))
            centroid_row = int(round(row["Centroid Row"]))
            object_number = int(row["Object Number"])

            cv2.circle(annotated_image, (centroid_col, centroid_row), 6, (0, 255, 255), -1)
            cv2.putText(
                annotated_image,
                str(object_number),
                (centroid_col + 8, centroid_row - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    annotated_image_path = os.path.join(output_directory, f"{output_name}_object_ids.png")
    cv2.imwrite(annotated_image_path, annotated_image)

    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    for mask_index, (mask, area) in enumerate(zip(masks, areas)):
        if lower_bound <= area <= upper_bound:
            if mask_index not in valid_mask_indices:
                continue
            segmentation = mask["segmentation"]
        
            filtered_mask[segmentation] = 255

    # Save the filtered mask applied to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(filtered_mask, cv2.COLOR_BGR2GRAY))
    filtered_mask_path = os.path.join(output_directory, f"{output_name}_filtered_mask.png")
    cv2.imwrite(filtered_mask_path, filtered_image)

    # Save very transparent overlay: only thin white outlines of segmented objects
    mask_gray = cv2.cvtColor(filtered_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = (mask_gray > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    transparent_overlay = image.copy()
    cv2.drawContours(transparent_overlay, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    transparent_overlay_path = os.path.join(output_directory, f"{output_name}_transparent_overlay.png")
    cv2.imwrite(transparent_overlay_path, transparent_overlay)


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