"""Utility functions for pixel sorter."""

import os
from PIL import Image
def count_files(folder_path, extensions=None):
  """
  Count files in a folder, optionally filtered by extensions.
  
  @param {str} folder_path - Path to folder
  @param {list|None} extensions - List of extensions to filter (e.g., ['.jpg', '.png']), or None for all files
  @return {int} Number of files, or -1 on error
  """
  try:
    if extensions is None:
      return sum(1 for item in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, item)))
    else:
      # Normalize extensions to lowercase for comparison
      extensions = [ext.lower() for ext in extensions]
      return sum(1 for item in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, item)) 
                 and os.path.splitext(item)[1].lower() in extensions)
  except Exception as e:
    print(f"Error: {e}")
    return -1

def load_image(image_path):
  """
  Load an image from path with validation.
  
  @param {str} image_path - Path to image file
  @return {Image.Image|None} PIL Image or None on error
  """
  # Supported image formats
  SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
  
  try:
    # Validate file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
      print(f"Error: Unsupported image format '{file_ext}'. Supported formats: {SUPPORTED_IMAGE_FORMATS}")
      return None
    
    # Load image
    img = Image.open(image_path)
    
    # Validate image dimensions
    if img.width <= 0 or img.height <= 0:
      print(f"Error: Invalid image dimensions ({img.width}x{img.height})")
      return None
    
    # Convert to RGB for consistent processing (handles RGBA, P, L, etc.)
    if img.mode not in ['RGB', 'RGBA']:
      img = img.convert('RGB')
    elif img.mode == 'RGBA':
      # Composite RGBA onto white background to preserve visual appearance
      background = Image.new('RGB', img.size, (255, 255, 255))
      background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
      img = background
    
    return img
    
  except FileNotFoundError:
    print(f"Error: Image not found at '{image_path}'")
    return None
  except Exception as e:
    print(f"Error loading image: {e}")
    return None

def load_images_from_folder(folder_path, start_index=1):
  """
  Load all numbered images from folder with validation.
  
  @param {str} folder_path - Path to folder
  @param {int} start_index - Starting file number
  @return {list} List of PIL Images
  """
  SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
  images = []
  
  # Build list of numbered image files that exist
  found_files = []
  image_count = count_files(folder_path, SUPPORTED_EXTENSIONS)
  
  if image_count <= 0:
    print("Error: No image files found in folder")
    return images
  
  # Scan for numbered images (try up to a reasonable limit)
  max_scan = start_index + image_count + 100  # Scan a bit beyond expected count
  for i in range(start_index, max_scan):
    found = False
    for ext in SUPPORTED_EXTENSIONS:
      image_path = os.path.join(folder_path, f"{i}{ext}")
      if os.path.exists(image_path):
        found_files.append((i, image_path))
        found = True
        break
    
    # Stop if we've found all images and hit a gap of 10 missing files
    if len(found_files) >= image_count and not found:
      # Check if there's a gap
      if len(found_files) > 0 and i - found_files[-1][0] > 10:
        break
  
  # Warn about sequence gaps
  if len(found_files) > 1:
    expected_indices = list(range(found_files[0][0], found_files[-1][0] + 1))
    actual_indices = [idx for idx, _ in found_files]
    missing = set(expected_indices) - set(actual_indices)
    if missing:
      print(f"Warning: Missing {len(missing)} image(s) in sequence: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
  
  # Load images
  for idx, path in found_files:
    img = load_image(path)
    if img:
      images.append(img)
  
  print(f"Loaded {len(images)} images from folder")
  return images

def prompt_input(text, default, value_type=str):
  """
  Prompt for user input with default value.
  
  @param {str} text - Prompt text
  @param {any} default - Default value
  @param {type} value_type - Type to convert input to
  @return {any} User input or default value
  """
  try:
    user_input = input(f"{text} [{default}]: ").strip()
    if not user_input:
      return default
    return value_type(user_input)
  except ValueError:
    print(f"Invalid input, using default: {default}")
    return default
