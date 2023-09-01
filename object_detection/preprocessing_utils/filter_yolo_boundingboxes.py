import os
import argparse

image_size = 250
print_csv = False

# Define a function to read YOLO-formatted annotation file
def read_yolo_annotation_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

def get_pos_and_size(x, y, w, h):
    x_pos = x * image_size
    y_pos = y * image_size
    width = w * image_size
    height = h * image_size
    return x_pos, y_pos, width, height

def check_near_extent(x, y, w, h):
    close_to = 2
    max_size = 30

    x_pos, y_pos, width, height = get_pos_and_size(x, y, w, h)
    if not print_csv: print(f"(chk_ext) Checking box with x={x_pos}, y={y_pos}, width={width}, height={height}")
  

    if(width > max_size):
        if not print_csv: print(f"(chk_ext) Large width {width}")    
    if(height > max_size):
        if not print_csv: print(f"(chk_ext) Large height {height}")    

    # Check X pos
    distance = image_size
    left_dist = (x_pos - width/2)
    right_dist = (image_size-(x_pos + width/2))
    bottom_dist = (y_pos - height/2)
    top_dist = (image_size-(y_pos + height/2))

    if not print_csv: print(f"(chk_ext) left {left_dist}, right {right_dist}, bottom {bottom_dist}, top {top_dist}")
    if left_dist<close_to:
        if not print_csv: print(f'!! Object is close to left side of the image, {left_dist} pixels')
    if right_dist<close_to:
        if not print_csv: print(f'!! Object is close to right side of the image, {right_dist} pixels')
    if bottom_dist<close_to:
        if not print_csv: print(f'!! Object is close to bottom side of the image, {bottom_dist} pixels')
    if top_dist < close_to:
        if not print_csv: print(f'!! Object is close to top side of the image, {top_dist} pixels')

    return distance

# Define a function to filter bounding boxes with correct aspect ratios
def filter_bounding_boxes(annotation_lines, filename, shrink):
    correct_boxes = []
    for line in annotation_lines:
        line = line.strip()
        if len(line.split()) == 5:
            x, y, w, h = map(float, line.split()[1:])
            class_id = line.split()[0]

            x_pos, y_pos, width, height = get_pos_and_size(x, y, w, h)
            if print_csv:
                print(f"{filename},{x_pos},{y_pos},{width},{height}")

            if not print_csv: print(f"---- Checking box from file {filename} ----")
            if (h != 0):
                aspect_ratio = w / h
                
                if shrink: 
                    # >22 to 17 for 0.5m data
                    '''
                    if(width > 22): 
                        w = 17.0/image_size
                    if(height > 22):
                        h = 17.0/image_size
                    '''
                    # >11 to 11 for 1m data
                    if(width > 11): 
                        w = 11.0/image_size
                    if(height > 11):
                        h = 11.0/image_size

                    correct_boxes.append(f'{class_id} {x} {y} {w} {h}')
                else:
                    #if(distance > 2):
                    #    print(f"Boundingbox close to extext, {distance} pixels.")                
                    if aspect_ratio < 0.9 or aspect_ratio > 1.1111:
                        if not print_csv: print("! Bounding box in file {} has unusual aspect ratio: {} from annotation {}".format(filename, aspect_ratio, line))
                        distance = check_near_extent(x,y,w,h)
                    else:
                        correct_boxes.append(line)
            else:
                if not print_csv: print("! Bounding box in file {} has ZERO height! Annotation {}".format(filename, line))
        else:
            correct_boxes.append(line)
    return correct_boxes

# Define a function to write the filtered bounding boxes back to the file
def write_filtered_boxes(file_path, filtered_boxes):
    with open(file_path, "w") as f:
        for box in filtered_boxes:
            f.write(box + '\n')

# Define the directory path containing the annotation files
#annotations_dir = "datasets/final_data_05m_normalized/training/labels"

def main(annotations_dir, dry_run, shink_boxes):
    # Loop over the annotation files in the directory
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(annotations_dir, filename)
            annotation_lines = read_yolo_annotation_file(file_path)
            filtered_boxes = filter_bounding_boxes(annotation_lines, filename=filename, shrink=shink_boxes)
            if not dry_run:
                write_filtered_boxes(file_path, filtered_boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove bounding boxes with unusual aspect ratios from YOLO annotation files.")
    parser.add_argument("annotations_dir", help="Path to the directory containing the annotation files.")
    parser.add_argument("--dry_run", help="If dry_run is True the new filtered annotations will not be written to disk.", action='store_true')
    parser.add_argument("--print_csv", help="Print position och size of all boxes in CSV format.", action='store_true')
    parser.add_argument("--shrink_boxes", help="Shrink all large boxes to smaller ons, >22 px -> 17 px.", action='store_true')
    
    args = parser.parse_args()
    print(f"dry_run = {args.dry_run}")
    print_csv = args.print_csv
    main(args.annotations_dir, args.dry_run, args.shrink_boxes)