import cv2 as cv
import os

def save_video_frames(video_path, output_dir, frame_rate=30):
    """
    Read a video and save its frames to a specified directory.
    
    Parameters:
    - video_path: str, path to the input video.
    - output_dir: str, directory where frames will be saved.
    - how many frames you want to save "1" means everything
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video_capture = cv.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")
    
    frame_number = 0
    count = 0
    
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        
        # If the frame was not read correctly, exit the loop
        if not ret:
            break
        
        if frame_number % frame_rate == 0:
            # Construct the output filename
            frame_filename = os.path.join(output_dir, f"{count}.jpg")
                
            # Save the frame as a JPEG file
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = frame[150:,:] # Crop the image to remove the top part
            cv.imwrite(frame_filename, frame)
            print(f"Saved frame {frame_number} to {frame_filename}")
            count += 1
            
        # Increment the frame number
        frame_number += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Frames have been saved to: {output_dir}")

if __name__ == "__main__":
    video_path = "samples.mp4"
    output_dir = "samples"
    save_video_frames(video_path, output_dir, frame_rate=20)