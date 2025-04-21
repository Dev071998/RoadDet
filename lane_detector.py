import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque

# Configuration parameters
HISTORY_LENGTH = 10  # Increased for better stability
MIN_SLOPE = 0.5      # Minimum slope to consider as lane
MAX_DIFF = 0.15      # Reduced for stricter lane consistency
SHARPEN_KERNEL = np.array([[-1, -1, -1], 
                          [-1, 9, -1],
                          [-1, -1, -1]])  # Sharpening kernel

# Lane tracking buffers
left_lane_buffer = deque(maxlen=HISTORY_LENGTH)
right_lane_buffer = deque(maxlen=HISTORY_LENGTH)

def sharpen_image(image):
    """Apply sharpening to enhance lane markings"""
    return cv2.filter2D(image, -1, SHARPEN_KERNEL)

def calculate_line_parameters(line):
    """Convert line endpoints to slope-intercept form with stability checks"""
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-5:  # Avoid division by zero with epsilon
        return None, None
    slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Add small epsilon
    intercept = y1 - slope * x1
    return slope, intercept

def average_lines(buffer):
    """Calculate average line from buffer with outlier rejection"""
    if not buffer:
        return None
    
    # Extract slopes and intercepts
    slopes = [s for s, i in buffer]
    intercepts = [i for s, i in buffer]
    
    # Remove outliers using IQR
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    valid_indices = [i for i, s in enumerate(slopes) 
                    if lower_bound <= s <= upper_bound]
    
    if not valid_indices:
        return None
    
    # Calculate robust average
    avg_slope = np.mean([slopes[i] for i in valid_indices])
    avg_intercept = np.mean([intercepts[i] for i in valid_indices])
    
    # Create line endpoints
    y1 = 480  # Bottom of the image
    y2 = int(480 * 0.6)  # Top of ROI
    x1 = int((y1 - avg_intercept) / (avg_slope + 1e-5))  # Prevent division by zero
    x2 = int((y2 - avg_intercept) / (avg_slope + 1e-5))
    
    return (x1, y1, x2, y2)

def detect_lanes(frame):
    global left_lane_buffer, right_lane_buffer
    
    # 1. Image Enhancement
    sharpened = sharpen_image(frame)
    
    # 2. Convert to grayscale with CLAHE for better contrast
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. Adaptive Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Edge detection with adaptive thresholds
    median_intensity = np.median(blur)
    lower_thresh = int(max(0, 0.7 * median_intensity))
    upper_thresh = int(min(255, 1.3 * median_intensity))
    edges = cv2.Canny(blur, lower_thresh, upper_thresh)
    
    # 5. Region of interest with dynamic masking
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
    
    # 6. Improved Hough Line Transform
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
                
            # Filter by slope and direction with dynamic threshold
            if slope < -MIN_SLOPE:  # Left lane
                current_left.append((slope, intercept))
            elif slope > MIN_SLOPE:  # Right lane
                current_right.append((slope, intercept))

    # Process left lane with moving average and stability checks
    if current_left:
        avg_slope = np.median([s for s, i in current_left])  # Using median for robustness
        avg_intercept = np.median([i for s, i in current_left])
        
        if left_lane_buffer:
            prev_avg_s = np.median([s for s, i in left_lane_buffer])
            prev_avg_i = np.median([i for s, i in left_lane_buffer])
            
            # Dynamic threshold based on previous values
            slope_diff = abs(avg_slope - prev_avg_s)
            intercept_diff = abs(avg_intercept - prev_avg_i)
            
            if slope_diff > MAX_DIFF or intercept_diff > 100:  # Pixel threshold for intercept
                # Use weighted average if difference is large
                avg_slope = 0.7*prev_avg_s + 0.3*avg_slope
                avg_intercept = 0.7*prev_avg_i + 0.3*avg_intercept
        
        left_lane_buffer.append((avg_slope, avg_intercept))
    
    # Process right lane
    if current_right:
        avg_slope = np.median([s for s, i in current_right])
        avg_intercept = np.median([i for s, i in current_right])
        
        if right_lane_buffer:
            prev_avg_s = np.median([s for s, i in right_lane_buffer])
            prev_avg_i = np.median([i for s, i in right_lane_buffer])
            
            slope_diff = abs(avg_slope - prev_avg_s)
            intercept_diff = abs(avg_intercept - prev_avg_i)
            
            if slope_diff > MAX_DIFF or intercept_diff > 100:
                avg_slope = 0.7*prev_avg_s + 0.3*avg_slope
                avg_intercept = 0.7*prev_avg_i + 0.3*avg_intercept
        
        right_lane_buffer.append((avg_slope, avg_intercept))
    
    # Get averaged lines with outlier rejection
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
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    # Display processing info
    fps_text = f"Buffer: L{len(left_lane_buffer)}/R{len(right_lane_buffer)}"
    cv2.putText(result, fps_text, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result

def create_lane_video(input_dir, output_dir, fps=20):
    global left_lane_buffer, right_lane_buffer
    
    # Get all image files with multiple sorting attempts
    try:
        image_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )
    except:
        image_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
    
    if not image_files:
        print(f"No images found in directory: {input_dir}")
        return

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writer parameters with multiple codec attempts
    frame_size = (640, 480)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"lane_detection_{timestamp}.mp4")
    
    # Try different video codecs
    for fourcc in [cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'avc1')]:
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if out.isOpened():
            break
    else:
        print("Error: Could not create video writer")
        return

    # Process all images
    for idx, img_file in enumerate(image_files):
        # Reset buffers for new video
        if idx == 0:
            left_lane_buffer.clear()
            right_lane_buffer.clear()
        
        img_path = os.path.join(input_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Skipping unreadable file: {img_file}")
            continue
            
        try:
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
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

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
    input_directory = r"C:\Users\devas\vboxsf\New\photos\photos"
    output_directory = r"C:\Users\devas\vboxsf\New\processed_videos"
    
    # Create video with 20 FPS (adjust as needed)
    create_lane_video(input_directory, output_directory, fps=20)