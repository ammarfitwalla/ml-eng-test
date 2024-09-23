import cv2
import numpy as np
# import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Load the image
# image_path = 'image.png'
# image_path = 's1_a1_s1.png'
# image_path = 'a3.1_a2.1.png'
# image_path = 'A-102 .00 - 2ND  FLOOR PLAN_page-0001.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to check the image size and decide how to split it
def determine_split(image):
    height, width = image.shape
    check_split = height if height > width else width
    print(check_split)
    if check_split <= 2000:
        return 1, 1  # No splitting for images smaller than 2000x2000
    elif 2000 <= check_split <= 4000:
        return 1, 2  # Split into 2 parts for images between 2000x4000
    elif 4000 < check_split <= 6000:
        return 2, 2  # Split into 4 parts for images between 4000x6000
    elif 6000 < check_split <= 9000:
        return 2, 3  # Split into 6 parts for images between 6000x9000
    else:
        return 3, 3  # Split into 9 parts for images larger than 9000

# Split the image into a grid (rows x cols)
def split_image(image, rows, cols):
    sub_images = []
    height, width = image.shape
    sub_height = height // rows
    sub_width = width // cols
    for i in range(rows):
        for j in range(cols):
            sub_image = image[i * sub_height: (i + 1) * sub_height, j * sub_width: (j + 1) * sub_width]
            sub_images.append(sub_image)
    return sub_images, sub_height, sub_width

# Define a function to check if a line intersects a box (works for both OCR and YOLO boxes)
def does_line_intersect_box(x1, y1, x2, y2, boxes):
    for box in boxes:
        x_min = np.min(box[:, 0])  # Min x from all four points of the box
        y_min = np.min(box[:, 1])  # Min y from all four points of the box
        x_max = np.max(box[:, 0])  # Max x from all four points of the box
        y_max = np.max(box[:, 1])  # Max y from all four points of the box

        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
            return True
        
        if (x1 < x_min < x2 or x2 < x_min < x1) and (y1 < y_min < y2 or y2 < y_min < y1):
            return True
    return False

# Function to merge overlapping lines
def merge_overlapping_lines(lines, threshold=0.1):
    merged_lines = []
    used = [False] * len(lines)

    def line_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def overlap_percentage(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Check if the lines are close in terms of start and end points
        dist1 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        dist2 = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)

        # Length of both lines
        len1 = line_length(x1, y1, x2, y2)
        len2 = line_length(x3, y3, x4, y4)

        # If they are close and have similar lengths, they are considered overlapping
        if dist1 < 0.3 * len1 and dist2 < 0.3 * len2:
            overlap = min(len1, len2) / max(len1, len2)
            return overlap
        return 0

    for i in range(len(lines)):
        if not used[i]:
            x1, y1, x2, y2 = lines[i][0]
            for j in range(i + 1, len(lines)):
                if not used[j]:
                    x3, y3, x4, y4 = lines[j][0]
                    overlap = overlap_percentage([x1, y1, x2, y2], [x3, y3, x4, y4])
                    if overlap >= threshold:
                        # Merge the lines by taking the smallest x and y for start and the largest for end
                        x1, y1 = min(x1, x3), min(y1, y3)
                        x2, y2 = max(x2, x4), max(y2, y4)
                        used[j] = True
            merged_lines.append([[x1, y1, x2, y2]])
            used[i] = True
    return merged_lines

# Check if two boxes overlap or touch on any of the four sides
def is_box_touching_or_overlapping(box1, box2):
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)
    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)

    # Check if the boxes overlap or touch horizontally or vertically
    horizontal_overlap = x_min1 <= x_max2 and x_max1 >= x_min2
    vertical_overlap = y_min1 <= y_max2 and y_max1 >= y_min2

    return horizontal_overlap and vertical_overlap

# Merge overlapping or touching text bounding boxes
def merge_text_boxes(text_boxes):
    merged_boxes = []
    used = [False] * len(text_boxes)

    for i in range(len(text_boxes)):
        if not used[i]:
            x_min, y_min = np.min(text_boxes[i], axis=0)
            x_max, y_max = np.max(text_boxes[i], axis=0)

            for j in range(i + 1, len(text_boxes)):
                if not used[j]:
                    x_min_j, y_min_j = np.min(text_boxes[j], axis=0)
                    x_max_j, y_max_j = np.max(text_boxes[j], axis=0)

                    # Merge if the boxes touch or overlap on any of the 4 sides
                    if is_box_touching_or_overlapping(text_boxes[i], text_boxes[j]):
                        x_min = min(x_min, x_min_j)
                        y_min = min(y_min, y_min_j)
                        x_max = max(x_max, x_max_j)
                        y_max = max(y_max, y_max_j)
                        used[j] = True

            merged_boxes.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
            used[i] = True
    return merged_boxes

# Draw red dots at the start and end of the merged line
def draw_line_with_red_dots(output_image, x1, y1, x2, y2, line_color=(0, 255, 0), dot_color=(0, 0, 255), line_thickness=2, dot_radius=5):
    # Draw the green merged line
    cv2.line(output_image, (x1, y1), (x2, y2), line_color, line_thickness)
    
    # Draw the red dots at the start and end of the merged line
    cv2.circle(output_image, (x1, y1), dot_radius, dot_color, -1)  # Start dot
    cv2.circle(output_image, (x2, y2), dot_radius, dot_color, -1)  # End dot

# Process each sub-image
def process_sub_image(sub_image):
    # Apply Canny edge detection
    edges = cv2.Canny(sub_image, threshold1=100, threshold2=150)

    # Apply Hough Line Transformation to detect lines (walls)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    # Create a copy of the original sub-image to draw on
    output_image = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR)

    # Initialize PaddleOCR to detect room names
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the sub-image
    ocr_results = ocr.ocr(sub_image)

    # Convert OCR results into bounding boxes for room names
    text_boxes = []
    for result in ocr_results:
        for line in result:
            box = np.array(line[0], dtype=int)
            text_boxes.append(box)

    # Merge overlapping or touching text boxes
    merged_text_boxes = merge_text_boxes(text_boxes)

    # YOLO detected object coordinates (replace with actual YOLO detections)
    yolo_boxes = []  # Add YOLO bounding boxes if needed

    # Combine OCR and YOLO boxes into one list
    all_boxes = merged_text_boxes + yolo_boxes

    # Merge overlapping lines
    if lines is not None:
        merged_lines = merge_overlapping_lines(lines)

        # Draw the merged green lines and red dots at their boundaries
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            if not does_line_intersect_box(x1, y1, x2, y2, all_boxes):
                draw_line_with_red_dots(output_image, x1, y1, x2, y2)

    # Draw only merged text boxes (without redundant drawing)
    for box in merged_text_boxes:
        cv2.polylines(output_image, [box], isClosed=True, color=(255, 0, 0), thickness=2)

    return output_image

# Stitch the processed sub-images back together
def stitch_images(sub_images, rows, cols, sub_height, sub_width):
    stitched_image = np.zeros((rows * sub_height, cols * sub_width, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            stitched_image[i * sub_height: (i + 1) * sub_height, j * sub_width: (j + 1) * sub_width] = sub_images[i * cols + j]
    return stitched_image

# # Determine how many parts to split the image into based on its size
# rows, cols = determine_split(image)

# # Split the image into sub-images if necessary
# if rows > 1 or cols > 1:
#     sub_images, sub_height, sub_width = split_image(image, rows, cols)
#     # Process each sub-image individually
#     processed_sub_images = [process_sub_image(sub_image) for sub_image in sub_images]
#     # Stitch the processed sub-images back together
#     stitched_image = stitch_images(processed_sub_images, rows, cols, sub_height, sub_width)
# else:
#     # Process the whole image if no splitting is necessary
#     stitched_image = process_sub_image(image)

# Show the final stitched image (or the whole image if no splitting occurred)
# plt.figure(figsize=(12, 12))
# plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
# plt.title('Processed Image with Merged Text Boxes and Red Dots at Merged Line Boundaries')
# plt.show()
