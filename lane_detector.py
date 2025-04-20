import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque

# Configuration parameters
HISTORY_LENGTH = 5  # Number of frames to average over
MIN_SLOPE = 0.5     # Minimum slope to consider as lane
MAX_DIFF = 0.2      # Maximum allowed difference from average

# Lane tracking buffers
left_lane_buffer = deque(maxlen=HISTORY_LENGTH)
right_lane_buffer = deque(maxlen=HISTORY_LENGTH)

def calculate_line_parameters(line):
    """Convert line endpoints to slope-intercept form"""
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return None, None  # Avoid division by zero
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def average_lines(buffer):
    """Calculate average line from buffer using slope-intercept form"""
    if not buffer:
        return None
    
    # Average slopes and intercepts separately
    avg_slope = np.mean([s for s, i in buffer])
    avg_intercept = np.mean([i for s, i in buffer])
    
    # Create line endpoints based on average parameters
    y1 = 480  # Bottom of the image
    y2 = int(480 * 0.6)  # Top of ROI
    x1 = int((y1 - avg_intercept) / avg_slope)
    x2 = int((y2 - avg_intercept) / avg_slope)
    
    return (x1, y1, x2, y2)

def detect_lanes(frame):
    global left_lane_buffer, right_lane_buffer
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Region of interest mask
    height, width = edges.shape
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (width*0.1, height),
        (width*0.45, height*0.6),
        (width*0.55, height*0.6),
        (width*0.9, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )
    
    # Create blank image to draw lines
    line_image = np.zeros_like(frame)
    
    current_left = []
    current_right = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, intercept = calculate_line_parameters(line[0])
            
            if slope is None:
                continue
                
            # Filter by slope and direction
            if slope < -MIN_SLOPE:  # Left lane
                current_left.append((slope, intercept))
            elif slope > MIN_SLOPE:  # Right lane
                current_right.append((slope, intercept))

    # Process left lane with moving average
    if current_left:
        # Calculate current average
        avg_slope = np.mean([s for s, i in current_left])
        avg_intercept = np.mean([i for s, i in current_left])
        
        # Check against previous average if buffer isn't empty
        if left_lane_buffer:
            prev_avg_s = np.mean([s for s, i in left_lane_buffer])
            prev_avg_i = np.mean([i for s, i in left_lane_buffer])
            if abs(avg_slope - prev_avg_s) > MAX_DIFF:
                # Use previous values if difference too big
                avg_slope = prev_avg_s
                avg_intercept = prev_avg_i
        
        left_lane_buffer.append((avg_slope, avg_intercept))
    
    # Process right lane with moving average
    if current_right:
        avg_slope = np.mean([s for s, i in current_right])
        avg_intercept = np.mean([i for s, i in current_right])
        
        if right_lane_buffer:
            prev_avg_s = np.mean([s for s, i in right_lane_buffer])
            prev_avg_i = np.mean([i for s, i in right_lane_buffer])
            if abs(avg_slope - prev_avg_s) > MAX_DIFF:
                avg_slope = prev_avg_s
                avg_intercept = prev_avg_i
        
        right_lane_buffer.append((avg_slope, avg_intercept))
    
    # Get averaged lines
    left_line = average_lines(left_lane_buffer)
    right_line = average_lines(right_lane_buffer)
    
    # Draw lines if available
    if left_line:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
    if right_line:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Combine with original image
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

def create_lane_video(input_dir, output_dir, fps=20):
    global left_lane_buffer, right_lane_buffer
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(input_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    if not image_files:
        print("No images found in the directory!")
        return

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writer parameters
    frame_size = (640, 480)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"lane_detection_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Process all images
    for idx, img_file in enumerate(image_files):
        # Reset buffers for new video
        left_lane_buffer.clear()
        right_lane_buffer.clear()
        
        img_path = os.path.join(input_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Skipping unreadable file: {img_file}")
            continue
            
        # Resize and process frame
        frame = cv2.resize(frame, frame_size)
        processed = detect_lanes(frame)
        
        # Write to video
        out.write(processed)
        
        # Show progress
        print(f"Processed {idx+1}/{len(image_files)}: {img_file}")
        cv2.imshow('Processing Preview', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")
    
    # Play the created video
    cap = cv2.VideoCapture(output_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Final Video Output', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_directory = r"C:\Users\devas\vboxsf\New\photos\photo2"
    output_directory = r"C:\Users\devas\vboxsf\New\processed_videos"
    
    # Create video with 20 FPS (adjust as needed)
    create_lane_video(input_directory, output_directory, fps=20)