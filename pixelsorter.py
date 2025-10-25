#!/usr/bin/env python3
"""
Pixel Sorter - Create videos with progressive pixel sorting effects
Supports single image mode, image stitching mode, and video mode.
"""

import argparse
import sys
import time
import cv2
from utils import load_image, load_images_from_folder, count_files, prompt_input
from video import images_to_mp4, get_video_info, bgr_to_pil, pil_to_bgr
from sorter import calculate_increments, generate_sorted_frames, generate_stitched_frames, process_video_with_sorting


def process_single_image(image_path, output_path, num_frames, num_normal_frames,
                         start_upper, upper_buffer, start_lower, lower_buffer,
                         reuse_frames, fps, use_multiprocessing=True):
  """
  Process single image into progressively pixel-sorted video.
  
  @param {str} image_path - Path to input image
  @param {str} output_path - Path to output video
  @param {int} num_frames - Number of sorted frames to generate
  @param {int} num_normal_frames - Number of static frames at start
  @param {float} start_upper - Starting upper threshold
  @param {float} upper_buffer - Upper threshold buffer
  @param {float} start_lower - Starting lower threshold
  @param {float} lower_buffer - Lower threshold buffer
  @param {bool} reuse_frames - Sort previous frame instead of source
  @param {int} fps - Frames per second
  @param {bool} use_multiprocessing - Enable parallel processing (default: True)
  @return {bool} True if successful, False otherwise
  """
  print(f"\nProcessing: {image_path}")
  start_time = time.time()
  
  img = load_image(image_path)
  if img is None:
    return False
  
  increment_up, increment_lower = calculate_increments(
    start_upper, upper_buffer, start_lower, lower_buffer, num_frames
  )
  
  frames = []
  
  # Add static frames
  if num_normal_frames > 0:
    frames.extend([img] * num_normal_frames)
  
  # Add sorted frames
  print(f"Generating {num_frames} sorted frames...")
  sorted_frames = generate_sorted_frames(
    img, num_frames, increment_up, increment_lower, reuse_frames, use_multiprocessing=use_multiprocessing
  )
  frames.extend(sorted_frames)
  
  # Create video
  success = images_to_mp4(frames, output_path, fps=fps)
  
  elapsed = time.time() - start_time
  print(f"Completed in {elapsed:.1f}s ({len(frames)} frames)")
  
  return success


def process_image_folder(folder_path, output_path, num_normal_frames,
                         start_upper, upper_buffer, start_lower, lower_buffer, fps, use_multiprocessing=True):
  """
  Process folder of images into stitched, progressively pixel-sorted video.
  
  @param {str} folder_path - Path to folder containing numbered images
  @param {str} output_path - Path to output video
  @param {int} num_normal_frames - Number of static frames at start
  @param {float} start_upper - Starting upper threshold
  @param {float} upper_buffer - Upper threshold buffer
  @param {float} start_lower - Starting lower threshold
  @param {float} lower_buffer - Lower threshold buffer
  @param {int} fps - Frames per second
  @param {bool} use_multiprocessing - Enable parallel processing (default: True)
  @return {bool} True if successful, False otherwise
  """
  print(f"\nProcessing folder: {folder_path}")
  start_time = time.time()
  
  file_count = count_files(folder_path)
  if file_count <= 0:
    print("Error: No images found")
    return False
  
  print(f"Loading {file_count} images...")
  images = load_images_from_folder(folder_path)
  if not images:
    print("Error: Failed to load images")
    return False
  
  increment_up, increment_lower = calculate_increments(
    start_upper, upper_buffer, start_lower, lower_buffer, len(images)
  )
  
  print(f"Generating frames...")
  frames = generate_stitched_frames(
    images, num_normal_frames, increment_up, increment_lower, use_multiprocessing=use_multiprocessing
  )
  
  success = images_to_mp4(frames, output_path, fps=fps)
  
  elapsed = time.time() - start_time
  print(f"Completed in {elapsed:.1f}s ({len(frames)} frames)")
  
  return success

def process_video(input_path, output_path, num_normal_frames,
                  start_upper, upper_buffer, start_lower, lower_buffer, use_multiprocessing=True):
  """
  Process video with progressive pixel sorting on each frame.
  
  @param {str} input_path - Path to input video
  @param {str} output_path - Path to output video
  @param {int} num_normal_frames - Number of initial frames to keep unchanged
  @param {float} start_upper - Starting upper threshold
  @param {float} upper_buffer - Upper threshold buffer
  @param {float} start_lower - Starting lower threshold
  @param {float} lower_buffer - Lower threshold buffer
  @param {bool} use_multiprocessing - Enable parallel processing (default: True)
  @return {bool} True if successful, False otherwise
  """
  print(f"\nProcessing video: {input_path}")
  start_time = time.time()
  
  # Get video info
  info = get_video_info(input_path)
  if info is None:
    return False
  
  total_frames = info['total_frames']
  fps = info['fps']
  width = info['width']
  height = info['height']
  
  print(f"Video info: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
  
  # Calculate sorting parameters
  frames_to_sort = total_frames - num_normal_frames
  if frames_to_sort <= 0:
    print("Error: num_normal_frames must be less than total video frames")
    return False
  
  increment_up, increment_lower = calculate_increments(
    start_upper, upper_buffer, start_lower, lower_buffer, frames_to_sort
  )
  
  # Open video capture and writer
  cap = cv2.VideoCapture(input_path)
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
  if not cap.isOpened():
    print("Error: Could not open video")
    return False
  
  # Process frames
  print(f"Processing {total_frames} frames ({num_normal_frames} normal, {frames_to_sort} sorted)...")
  success = process_video_with_sorting(
    cap, out, total_frames, num_normal_frames,
    increment_up, increment_lower, bgr_to_pil, pil_to_bgr,
    use_multiprocessing=use_multiprocessing
  )
  
  # Cleanup
  cap.release()
  out.release()
  
  if success:
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    print(f"Saved: {output_path}")
  
  return success


def interactive_mode():
  """
  Run in interactive mode with prompts.
  
  @return {bool} True if successful, False otherwise
  """
  print("\n=== Pixel Sorter ===\n")
  
  print("Select mode:")
  print("  1. Single image")
  print("  2. Image stitcher")
  print("  3. Video")
  mode = prompt_input("Mode (1, 2, or 3)", "1", str)
  
  if mode == "1":
    image_path = prompt_input("Input image", "input.jpg", str)
    output_path = prompt_input("Output video", "output.mp4", str)
    num_frames = prompt_input("Sorted frames", 90, int)
    num_normal_frames = prompt_input("Static frames", 5, int)
    reuse_frames = prompt_input("Reuse frames (true/false)", "false", str).lower() == "true"
    start_upper = prompt_input("Start upper (0-1)", 0.5, float)
    upper_buffer = prompt_input("Upper buffer", 0, float)
    start_lower = prompt_input("Start lower (0-1)", 0.5, float)
    lower_buffer = prompt_input("Lower buffer", 0, float)
    fps = prompt_input("FPS", 30, int)
    
    return process_single_image(
      image_path, output_path, num_frames, num_normal_frames,
      start_upper, upper_buffer, start_lower, lower_buffer,
      reuse_frames, fps, use_multiprocessing=True
    )
  
  elif mode == "2":
    folder_path = prompt_input("Input folder", "./images_to_stitch/", str)
    output_path = prompt_input("Output video", "stitched.mp4", str)
    num_normal_frames = prompt_input("Static frames", 42, int)
    start_upper = prompt_input("Start upper (0-1)", 0.5, float)
    upper_buffer = prompt_input("Upper buffer", 0, float)
    start_lower = prompt_input("Start lower (0-1)", 0.5, float)
    lower_buffer = prompt_input("Lower buffer", 0, float)
    fps = prompt_input("FPS", 30, int)
    
    return process_image_folder(
      folder_path, output_path, num_normal_frames,
      start_upper, upper_buffer, start_lower, lower_buffer, fps, use_multiprocessing=True
    )
  
  elif mode == "3":
    input_path = prompt_input("Input video", "input.mp4", str)
    output_path = prompt_input("Output video", "output.mp4", str)
    num_normal_frames = prompt_input("Normal frames at start", 30, int)
    start_upper = prompt_input("Start upper (0-1)", 0.5, float)
    upper_buffer = prompt_input("Upper buffer", 0, float)
    start_lower = prompt_input("Start lower (0-1)", 0.5, float)
    lower_buffer = prompt_input("Lower buffer", 0, float)
    
    return process_video(
      input_path, output_path, num_normal_frames,
      start_upper, upper_buffer, start_lower, lower_buffer, use_multiprocessing=True
    )
  
  else:
    print("Invalid mode")
    return False


def main():
  """
  Main entry point.
  
  @return {int} Exit code (0 for success, 1 for failure)
  """
  parser = argparse.ArgumentParser(
    description="Create videos with progressive pixel sorting effects"
  )
  
  parser.add_argument('-m', '--mode', choices=['single', 'stitch', 'video'])
  parser.add_argument('-i', '--input', help='Input image path (single mode) or video path (video mode)')
  parser.add_argument('-d', '--directory', help='Input directory (stitch mode)')
  parser.add_argument('-o', '--output', help='Output video path')
  parser.add_argument('-f', '--frames', type=int, help='Number of sorted frames (single mode)')
  parser.add_argument('-s', '--static', type=int, help='Number of static/normal frames')
  parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30, single/stitch only)')
  parser.add_argument('--start-upper', type=float, default=0.5, help='Start upper threshold (default: 0.5)')
  parser.add_argument('--upper-buffer', type=float, default=0, help='Upper buffer (default: 0)')
  parser.add_argument('--start-lower', type=float, default=0.5, help='Start lower threshold (default: 0.5)')
  parser.add_argument('--lower-buffer', type=float, default=0, help='Lower buffer (default: 0)')
  parser.add_argument('--reuse-frames', action='store_true', help='Sort previous frame (single mode)')
  parser.add_argument('--no-multiprocessing', action='store_true', help='Disable parallel processing')
  
  args = parser.parse_args()
  
  if not args.mode:
    success = interactive_mode()
    return 0 if success else 1
  
  if args.mode == 'single':
    if not args.input or not args.output:
      print("Error: Single mode requires --input and --output")
      return 1
    
    num_frames = args.frames if args.frames else 90
    num_static = args.static if args.static else 5
    
    success = process_single_image(
      args.input, args.output, num_frames, num_static,
      args.start_upper, args.upper_buffer, 
      args.start_lower, args.lower_buffer,
      args.reuse_frames, args.fps, use_multiprocessing=not args.no_multiprocessing
    )
    
  elif args.mode == 'stitch':
    if not args.directory or not args.output:
      print("Error: Stitch mode requires --directory and --output")
      return 1
    
    num_static = args.static if args.static else 42
    
    success = process_image_folder(
      args.directory, args.output, num_static,
      args.start_upper, args.upper_buffer,
      args.start_lower, args.lower_buffer, args.fps, use_multiprocessing=not args.no_multiprocessing
    )
  
  elif args.mode == 'video':
    if not args.input or not args.output:
      print("Error: Video mode requires --input and --output")
      return 1
    
    num_static = args.static if args.static else 30
    
    success = process_video(
      args.input, args.output, num_static,
      args.start_upper, args.upper_buffer,
      args.start_lower, args.lower_buffer, use_multiprocessing=not args.no_multiprocessing
    )
  
  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())
