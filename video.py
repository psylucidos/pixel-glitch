"""Video creation and processing functions with color space management."""

import cv2
import numpy as np
import os
from PIL import Image

try:
    from pymediainfo import MediaInfo
    MEDIAINFO_AVAILABLE = True
except ImportError:
    MEDIAINFO_AVAILABLE = False
    MediaInfo = None
def detect_video_color_profile(video_path):
  """
  Detect actual color profile from video metadata using MediaInfo.
  This reads the actual color space information, not just file extension.
  
  @param {str} video_path - Path to video file
  @return {dict|None} Dictionary with color profile info or None on error
  """
  if not MEDIAINFO_AVAILABLE:
    print("Warning: pymediainfo not available, falling back to extension-based detection")
    return detect_video_color_space_fallback(video_path)
  
  try:
    media_info = MediaInfo.parse(video_path)
    
    # Find video track
    video_track = None
    for track in media_info.tracks:
      if track.track_type == "Video":
        video_track = track
        break
    
    if not video_track:
      print("Error: No video track found in file")
      return None
    
    # Extract color information
    color_primaries = getattr(video_track, 'color_primaries', None)
    transfer_characteristics = getattr(video_track, 'transfer_characteristics', None)
    matrix_coefficients = getattr(video_track, 'matrix_coefficients', None)
    color_range = getattr(video_track, 'color_range', None)
    
    # Get codec for additional context
    codec = getattr(video_track, 'format', 'Unknown')
    
    return {
      'path': video_path,
      'color_primaries': color_primaries,
      'transfer_characteristics': transfer_characteristics,
      'matrix_coefficients': matrix_coefficients,
      'color_range': color_range,
      'codec': codec,
      'width': video_track.width,
      'height': video_track.height,
      'frame_rate': video_track.frame_rate,
      'frame_count': video_track.frame_count or 0
    }
    
  except Exception as e:
    print(f"Error reading video color profile: {e}")
    return None
def detect_video_color_space_fallback(video_path):
  """
  Fallback color space detection based on file extension.
  WARNING: This is unreliable and can cause color issues!
  
  @param {str} video_path - Path to video file
  @return {dict|None} Basic color space info or None
  """
  file_ext = os.path.splitext(video_path)[1].lower()
  
  if file_ext == '.mov':
    return {
      'path': video_path,
      'color_primaries': 'BT.709 (assumed)',
      'transfer_characteristics': 'BT.709 (assumed)',
      'matrix_coefficients': None,
      'color_range': None,
      'codec': 'Unknown',
      'warning': 'Using extension-based detection - may be inaccurate!'
    }
  elif file_ext == '.mp4':
    return {
      'path': video_path,
      'color_primaries': 'Unknown',
      'transfer_characteristics': 'Unknown',
      'matrix_coefficients': None,
      'color_range': None,
      'codec': 'Unknown',
      'warning': 'Using extension-based detection - may be inaccurate!'
    }
  else:
    return None
def is_supported_color_profile(profile_info, allow_override=False):
  """
  Check if video color profile is supported for processing.
  Only Rec.709 (SDR) is officially supported. Other profiles can be
  processed with --type-override but may produce incorrect colors.
  
  @param {dict} profile_info - Color profile information from detect_video_color_profile()
  @param {bool} allow_override - If True, allow unsupported profiles with warning
  @return {tuple} (is_supported, color_space, message)
  """
  if not profile_info:
    return False, None, "Could not read color profile"
  
  primaries = profile_info.get('color_primaries', '')
  transfer = profile_info.get('transfer_characteristics', '')
  
  # Normalize names (MediaInfo can return variations)
  if primaries:
    primaries = primaries.replace('BT.709', 'BT709').replace('BT709', 'BT.709')
    primaries = primaries.replace('BT.2020', 'BT2020').replace('BT2020', 'BT.2020')
  
  if transfer:
    transfer = transfer.replace('BT.709', 'BT709').replace('BT709', 'BT.709')
  
  # Check for Rec.709 (the only officially supported profile)
  is_rec709 = (
    primaries and 'BT.709' in primaries and
    transfer and 'BT.709' in transfer
  )
  
  if is_rec709:
    return True, 'rec709', "Rec.709 (SDR) - Supported"
  
  # Check for HDR profiles (explicitly not supported)
  is_hdr = (
    transfer and any(hdr in transfer for hdr in ['HLG', 'PQ', 'SMPTE ST 2084', 'ARIB STD-B67'])
  )
  
  if is_hdr:
    if allow_override:
      return True, 'hdr_override', f"HDR ({transfer}) - Processing with override (colors may be incorrect!)"
    else:
      return False, None, f"HDR video not supported: {primaries} / {transfer}\nUse --type-override to process anyway (not recommended)"
  
  # Check for SLOG3 or other log profiles
  is_log = (
    transfer and any(log in transfer.lower() for log in ['log', 'slog', 's-log'])
  )
  
  if is_log:
    if allow_override:
      return True, 'slog3', f"Log profile ({transfer}) - Processing with override"
    else:
      return False, None, f"Log profile not fully supported: {transfer}\nUse --type-override to process anyway"
  
  # Unknown or unsupported profile
  if allow_override:
    profile_desc = f"{primaries}/{transfer}" if primaries and transfer else "Unknown"
    return True, 'unknown_override', f"Unknown profile ({profile_desc}) - Processing with override (colors may be incorrect!)"
  else:
    return False, None, f"Unsupported color profile: {primaries} / {transfer}\nOnly Rec.709 (SDR) is supported. Use --type-override to process anyway."
def validate_video_file(video_path, allow_override=False):
  """
  Validate video file format and color space.
  Reads actual color profile metadata to ensure compatibility.
  
  @param {str} video_path - Path to video file
  @param {bool} allow_override - Allow unsupported color profiles with warning
  @return {dict|None} Dictionary with validation results or None on error
  """
  # Check file extension
  file_ext = os.path.splitext(video_path)[1].lower()
  
  if file_ext not in ['.mov', '.mp4', '.avi', '.mkv']:
    print(f"Error: Unsupported video format '{file_ext}'")
    print("Supported container formats: .mov, .mp4, .avi, .mkv")
    print("Note: Only Rec.709 color space is officially supported")
    return None
  
  # Check if file exists
  if not os.path.exists(video_path):
    print(f"Error: Video file not found: {video_path}")
    return None
  
  # Detect actual color profile from metadata
  print(f"Reading color profile metadata...")
  profile_info = detect_video_color_profile(video_path)
  
  if not profile_info:
    print("Error: Could not read video metadata")
    return None
  
  # Check if color profile is supported
  is_supported, color_space, message = is_supported_color_profile(profile_info, allow_override)
  
  if not is_supported:
    print(f"\nError: {message}")
    print(f"\nVideo color profile:")
    print(f"  Color Primaries: {profile_info.get('color_primaries', 'Unknown')}")
    print(f"  Transfer Characteristics: {profile_info.get('transfer_characteristics', 'Unknown')}")
    print(f"  Codec: {profile_info.get('codec', 'Unknown')}")
    return None
  
  # Show warning if using override
  if allow_override and color_space != 'rec709':
    print(f"\n⚠️  WARNING: {message}")
    print(f"Processing will continue but colors may be incorrect!")
    print(f"The tool is designed for Rec.709 (SDR) content only.\n")
  
  # Open video to validate it's readable
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    return None
  
  # Get and validate metadata
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  cap.release()
  
  # Validate metadata
  if total_frames <= 0:
    print(f"Error: Invalid frame count: {total_frames}")
    return None
  
  if fps <= 0 or fps > 1000:
    print(f"Error: Invalid FPS: {fps}")
    return None
  
  if width <= 0 or height <= 0:
    print(f"Error: Invalid dimensions: {width}x{height}")
    return None
  
  # Print color profile info
  print(f"Color profile: {message}")
  if profile_info.get('color_primaries'):
    print(f"  Primaries: {profile_info['color_primaries']}")
  if profile_info.get('transfer_characteristics'):
    print(f"  Transfer: {profile_info['transfer_characteristics']}")
  
  return {
    'path': video_path,
    'color_space': color_space,
    'color_profile': profile_info,
    'total_frames': total_frames,
    'fps': fps,
    'width': width,
    'height': height,
    'extension': file_ext
  }
def get_video_info(video_path, allow_override=False):
  """
  Get video metadata with validation.
  
  @param {str} video_path - Path to video file
  @param {bool} allow_override - Allow unsupported color profiles
  @return {dict|None} Dictionary with total_frames, fps, width, height, color_space or None on error
  """
  return validate_video_file(video_path, allow_override)

def get_output_path_for_color_space(output_path, color_space):
  """
  Adjust output path extension based on color space.
  
  @param {str} output_path - Original output path
  @param {str} color_space - Color space identifier
  @return {str} Output path with appropriate extension
  """
  base, ext = os.path.splitext(output_path)
  
  # Default to .mov for Rec.709, keep original or .mp4 for others
  if color_space == 'rec709':
    return base + '.mov'
  elif color_space in ['slog3', 'hdr_override', 'unknown_override']:
    # Keep original extension or default to .mp4
    if ext.lower() in ['.mov', '.mp4']:
      return output_path
    return base + '.mp4'
  else:
    return base + '.mov'

def get_video_codec_for_color_space(color_space):
  """
  Get appropriate codec for color space.
  
  @param {str} color_space - Color space identifier
  @return {str} Four-character codec code
  """
  # For Rec.709, use avc1 (H.264) for better quality
  # For others, use mp4v for compatibility
  if color_space == 'rec709':
    return 'avc1'
  else:
    return 'mp4v'
def images_to_mp4(images, output_path, fps=30, color_space='rec709'):
  """
  Convert list of images to MP4 video with color space preservation.
  
  @param {list} images - List of PIL Images or numpy arrays
  @param {str} output_path - Output video path
  @param {int} fps - Frames per second
  @param {str} color_space - 'rec709' or 'slog3' for color space handling
  @return {bool} True if successful, False otherwise
  """
  if not images:
    print("Error: No images provided")
    return False
  
  try:
    # Convert PIL images to numpy arrays if needed
    if isinstance(images[0], Image.Image):
      numpy_images = []
      for img in images:
        # Ensure RGB mode without unnecessary conversions
        if img.mode == 'RGB':
          numpy_images.append(np.array(img))
        else:
          numpy_images.append(np.array(img.convert('RGB')))
      images = numpy_images
    
    # Validate all frames have same dimensions
    height, width, channels = images[0].shape
    for i, img in enumerate(images):
      if img.shape != (height, width, channels):
        print(f"Error: Frame {i} has mismatched dimensions. Expected {width}x{height}, got {img.shape[1]}x{img.shape[0]}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    
    # Adjust output path for color space
    output_path = get_output_path_for_color_space(output_path, color_space)
    
    # Get codec for color space
    codec_code = get_video_codec_for_color_space(color_space)
    
    # Try primary codec, fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*codec_code)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If primary codec failed, try fallback
    if not out.isOpened():
      print(f"Warning: Codec '{codec_code}' not available, falling back to 'mp4v'")
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      # Ensure .mp4 extension for mp4v codec
      if not output_path.endswith('.mp4'):
        output_path = os.path.splitext(output_path)[0] + '.mp4'
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
      print("Error: Failed to create video writer")
      return False
    
    # Write frames
    for img in images:
      if img.shape[2] != 3:
        print(f"Error: Image must have 3 channels (RGB), got {img.shape[2]}")
        out.release()
        return False
      
      # Convert RGB to BGR for OpenCV
      frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      out.write(frame)
    
    out.release()
    print(f"Saved: {output_path} ({color_space.upper()})")
    return True
    
  except Exception as e:
    print(f"Error creating video: {e}")
    return False
def bgr_to_pil(frame):
  """
  Convert BGR frame to PIL Image in RGB format.
  
  @param {np.ndarray} frame - Frame in BGR format (OpenCV)
  @return {Image.Image} PIL Image in RGB format
  """
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return Image.fromarray(rgb_frame)
def pil_to_bgr(image):
  """
  Convert PIL Image to BGR frame without unnecessary conversions.
  
  @param {Image.Image} image - PIL Image
  @return {np.ndarray} Frame in BGR format (OpenCV)
  """
  # Only convert if not already RGB
  if image.mode == 'RGB':
    rgb_array = np.array(image)
  else:
    rgb_array = np.array(image.convert('RGB'))
  
  return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
