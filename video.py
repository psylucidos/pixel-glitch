"""Video creation and processing functions."""

import cv2
import numpy as np
from PIL import Image


def images_to_mp4(images, output_path, fps=30):
  """
  Convert list of images to MP4 video.
  
  @param {list} images - List of PIL Images or numpy arrays
  @param {str} output_path - Output video path
  @param {int} fps - Frames per second
  @return {bool} True if successful, False otherwise
  """
  try:
    if isinstance(images[0], Image.Image):
      images = [np.array(img.convert("RGB")) for img in images]
    
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in images:
      if img.shape[2] == 3:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      else:
        raise ValueError("Images must be RGB (3 channels).")
      out.write(frame)
    
    out.release()
    print(f"Saved: {output_path}")
    return True
    
  except Exception as e:
    print(f"Error creating video: {e}")
    return False

def get_video_info(video_path):
  """
  Get video metadata.
  
  @param {str} video_path - Path to video file
  @return {dict|None} Dictionary with total_frames, fps, width, height or None on error
  """
  try:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
      print(f"Error: Could not open video '{video_path}'")
      return None
    
    info = {
      'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      'fps': cap.get(cv2.CAP_PROP_FPS),
      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return info
    
  except Exception as e:
    print(f"Error reading video info: {e}")
    return None

def read_video_frame(cap, frame_index):
  """
  Read a specific frame from video capture.
  
  @param {cv2.VideoCapture} cap - OpenCV video capture object
  @param {int} frame_index - Frame index to read
  @return {np.ndarray|None} Frame as numpy array (BGR) or None on error
  """
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
  ret, frame = cap.read()
  
  if not ret:
    return None
  
  return frame

def bgr_to_pil(frame):
  """
  Convert BGR frame to PIL Image.
  
  @param {np.ndarray} frame - Frame in BGR format (OpenCV)
  @return {Image.Image} PIL Image in RGB format
  """
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return Image.fromarray(rgb_frame)

def pil_to_bgr(image):
  """
  Convert PIL Image to BGR frame.
  
  @param {Image.Image} image - PIL Image in RGB format
  @return {np.ndarray} Frame in BGR format (OpenCV)
  """
  rgb_array = np.array(image.convert("RGB"))
  return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
