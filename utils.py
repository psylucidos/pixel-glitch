"""Utility functions for pixel sorter."""

import os
from PIL import Image


def count_files(folder_path):
  """
  Count files in a folder.
  
  @param {str} folder_path - Path to folder
  @return {int} Number of files, or -1 on error
  """
  try:
    return sum(1 for item in os.listdir(folder_path) 
               if os.path.isfile(os.path.join(folder_path, item)))
  except Exception as e:
    print(f"Error: {e}")
    return -1

def load_image(image_path):
  """
  Load an image from path.
  
  @param {str} image_path - Path to image file
  @return {Image.Image|None} PIL Image or None on error
  """
  try:
    return Image.open(image_path)
  except FileNotFoundError:
    print(f"Error: Image not found at '{image_path}'")
    return None
  except Exception as e:
    print(f"Error loading image: {e}")
    return None

def load_images_from_folder(folder_path, start_index=1):
  """
  Load all numbered images from folder.
  
  @param {str} folder_path - Path to folder
  @param {int} start_index - Starting file number
  @return {list} List of PIL Images
  """
  images = []
  file_count = count_files(folder_path)
  
  if file_count <= 0:
    return images
  
  for i in range(start_index, file_count + start_index):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
      image_path = os.path.join(folder_path, f"{i}{ext}")
      if os.path.exists(image_path):
        img = load_image(image_path)
        if img:
          images.append(img)
        break
  
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
