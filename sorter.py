"""Pixel sorting functions."""

from pixelsort import pixelsort
from multiprocessing import Pool, cpu_count


def calculate_increments(start_upper, upper_buffer, start_lower, lower_buffer, num_frames):
  """
  Calculate threshold increments for progressive sorting.
  
  @param {float} start_upper - Starting upper threshold (0-1)
  @param {float} upper_buffer - Upper threshold buffer
  @param {float} start_lower - Starting lower threshold (0-1)
  @param {float} lower_buffer - Lower threshold buffer
  @param {int} num_frames - Number of frames to sort
  @return {tuple} (increment_up, increment_lower)
  """
  increment_up = (start_upper - upper_buffer) / num_frames
  increment_lower = ((1 - start_lower) - lower_buffer) / num_frames
  return increment_up, increment_lower

def apply_sort(image, frame_index, increment_up, increment_lower):
  """
  Apply pixel sorting with progressive thresholds.
  
  @param {Image.Image} image - PIL Image to sort
  @param {int} frame_index - Current frame index for threshold calculation
  @param {float} increment_up - Upper threshold increment
  @param {float} increment_lower - Lower threshold increment
  @return {Image.Image} Sorted PIL Image
  """
  lower_threshold = 0.5 - (frame_index * increment_up)
  upper_threshold = 0.5 + (frame_index * increment_lower)
  
  return pixelsort(
    image=image,
    sorting_function="lightness",
    interval_function="threshold",
    lower_threshold=lower_threshold,
    upper_threshold=upper_threshold
  )

def _sort_frame_worker(args):
  """
  Worker function for parallel frame sorting.
  Note: Multi-threaded worker function - called by multiprocessing.Pool
  
  @param {tuple} args - Tuple of (source_image, frame_index, increment_up, increment_lower)
  @return {tuple} Tuple of (frame_index, sorted_image)
  """
  source_image, frame_index, increment_up, increment_lower = args
  sorted_frame = apply_sort(source_image, frame_index, increment_up, increment_lower)
  return (frame_index, sorted_frame)

def generate_sorted_frames(source_image, num_frames, increment_up, increment_lower, reuse_frames=False, use_multiprocessing=True):
  """
  Generate progressively sorted frames from single image.
  Note: Multi-threaded when use_multiprocessing=True
  
  @param {Image.Image} source_image - Source PIL Image
  @param {int} num_frames - Number of frames to generate
  @param {float} increment_up - Upper threshold increment
  @param {float} increment_lower - Lower threshold increment
  @param {bool} reuse_frames - If True, sort previous frame instead of source
  @param {bool} use_multiprocessing - If True, use parallel processing (default: True)
  @return {list} List of sorted PIL Images
  """
  # Reuse frames mode cannot be parallelized (sequential dependency)
  if reuse_frames or not use_multiprocessing or num_frames < 4:
    frames = []
    for i in range(num_frames):
      base_image = frames[-1] if (i > 0 and reuse_frames) else source_image
      sorted_frame = apply_sort(base_image, i, increment_up, increment_lower)
      frames.append(sorted_frame)
      
      if (i + 1) % 10 == 0 or i == num_frames - 1:
        print(f"Progress: {i + 1}/{num_frames}")
    
    return frames
  
  # Parallel processing for independent frames
  print(f"Using {cpu_count()} CPU cores for parallel processing")
  
  # Prepare arguments for all frames
  frame_args = [
    (source_image, i, increment_up, increment_lower)
    for i in range(num_frames)
  ]
  
  # Process frames in parallel
  with Pool(cpu_count()) as pool:
    results = []
    for i, result in enumerate(pool.imap(_sort_frame_worker, frame_args)):
      results.append(result)
      if (i + 1) % 10 == 0 or i == num_frames - 1:
        print(f"Progress: {i + 1}/{num_frames}")
  
  # Sort results by frame index and extract frames
  results.sort(key=lambda x: x[0])
  frames = [frame for _, frame in results]
  
  return frames

def _sort_image_worker(args):
  """
  Worker function for parallel image sorting in stitch mode.
  Note: Multi-threaded worker function - called by multiprocessing.Pool
  
  @param {tuple} args - Tuple of (image, image_index, increment_up, increment_lower)
  @return {tuple} Tuple of (image_index, sorted_image)
  """
  image, image_index, increment_up, increment_lower = args
  sorted_frame = apply_sort(image, image_index, increment_up, increment_lower)
  return (image_index, sorted_frame)

def generate_stitched_frames(images, num_normal_frames, increment_up, increment_lower, use_multiprocessing=True):
  """
  Generate frames from multiple images with progressive sorting.
  Note: Multi-threaded when use_multiprocessing=True
  
  @param {list} images - List of PIL Images
  @param {int} num_normal_frames - Number of initial unsorted frames
  @param {float} increment_up - Upper threshold increment
  @param {float} increment_lower - Lower threshold increment
  @param {bool} use_multiprocessing - If True, use parallel processing (default: True)
  @return {list} List of PIL Images (normal + sorted)
  """
  frames = []
  
  # Add initial static frames
  for i in range(min(num_normal_frames, len(images))):
    frames.append(images[i])
  
  # Check if parallel processing is beneficial
  if not use_multiprocessing or len(images) < 4:
    # Sequential processing
    for i, image in enumerate(images):
      sorted_frame = apply_sort(image, i, increment_up, increment_lower)
      frames.append(sorted_frame)
      
      if (i + 1) % 10 == 0 or i == len(images) - 1:
        print(f"Progress: {i + 1}/{len(images)}")
    
    return frames
  
  # Parallel processing
  print(f"Using {cpu_count()} CPU cores for parallel processing")
  
  # Prepare arguments for all images
  image_args = [
    (image, i, increment_up, increment_lower)
    for i, image in enumerate(images)
  ]
  
  # Process images in parallel
  with Pool(cpu_count()) as pool:
    results = []
    for i, result in enumerate(pool.imap(_sort_image_worker, image_args)):
      results.append(result)
      if (i + 1) % 10 == 0 or i == len(images) - 1:
        print(f"Progress: {i + 1}/{len(images)}")
  
  # Sort results by image index and extract sorted frames
  results.sort(key=lambda x: x[0])
  sorted_frames = [frame for _, frame in results]
  frames.extend(sorted_frames)
  
  return frames

def _sort_video_frame_worker(args):
  """
  Worker function for parallel video frame sorting.
  Note: Multi-threaded worker function - called by multiprocessing.Pool
  
  @param {tuple} args - Tuple of (frame_bgr, frame_index, num_normal_frames, increment_up, increment_lower, bgr_to_pil_func, pil_to_bgr_func)
  @return {tuple} Tuple of (frame_index, processed_bgr_frame)
  """
  frame_bgr, frame_index, num_normal_frames, increment_up, increment_lower, bgr_to_pil_func, pil_to_bgr_func = args
  
  # Convert to PIL
  pil_image = bgr_to_pil_func(frame_bgr)
  
  # Apply sorting if past normal frames
  if frame_index >= num_normal_frames:
    sort_index = frame_index - num_normal_frames
    pil_image = apply_sort(pil_image, sort_index, increment_up, increment_lower)
  
  # Convert back to BGR
  bgr_frame = pil_to_bgr_func(pil_image)
  
  return (frame_index, bgr_frame)

def process_video_with_sorting(video_cap, video_writer, total_frames, num_normal_frames, 
                                increment_up, increment_lower, bgr_to_pil_func, pil_to_bgr_func,
                                use_multiprocessing=True, batch_size=None):
  """
  Process video frames with progressive pixel sorting.
  Note: Multi-threaded when use_multiprocessing=True
  
  @param {cv2.VideoCapture} video_cap - OpenCV video capture object
  @param {cv2.VideoWriter} video_writer - OpenCV video writer object
  @param {int} total_frames - Total number of frames in video
  @param {int} num_normal_frames - Number of initial unsorted frames
  @param {float} increment_up - Upper threshold increment
  @param {float} increment_lower - Lower threshold increment
  @param {function} bgr_to_pil_func - Function to convert BGR to PIL
  @param {function} pil_to_bgr_func - Function to convert PIL to BGR
  @param {bool} use_multiprocessing - If True, use parallel processing (default: True)
  @param {int|None} batch_size - Number of frames to process in parallel (default: cpu_count())
  @return {bool} True if successful, False otherwise
  """
  # Sequential processing fallback
  if not use_multiprocessing or total_frames < 4:
    try:
      for i in range(total_frames):
        ret, frame = video_cap.read()
        
        if not ret:
          print(f"Warning: Could not read frame {i}")
          break
        
        # Convert to PIL for processing
        pil_image = bgr_to_pil_func(frame)
        
        # Apply sorting if past normal frames
        if i >= num_normal_frames:
          sort_index = i - num_normal_frames
          pil_image = apply_sort(pil_image, sort_index, increment_up, increment_lower)
        
        # Convert back to BGR and write
        bgr_frame = pil_to_bgr_func(pil_image)
        video_writer.write(bgr_frame)
        
        if (i + 1) % 10 == 0 or i == total_frames - 1:
          print(f"Progress: {i + 1}/{total_frames}")
      
      return True
      
    except Exception as e:
      print(f"Error processing video frames: {e}")
      return False
  
  # Parallel processing with batching
  try:
    if batch_size is None:
      batch_size = cpu_count()
    
    print(f"Using {cpu_count()} CPU cores for parallel processing (batch size: {batch_size})")
    
    frame_buffer = []
    frames_processed = 0
    
    # Read and process frames in batches
    for i in range(total_frames):
      ret, frame = video_cap.read()
      
      if not ret:
        print(f"Warning: Could not read frame {i}")
        break
      
      # Add frame to buffer
      frame_buffer.append((frame, i, num_normal_frames, increment_up, increment_lower, bgr_to_pil_func, pil_to_bgr_func))
      
      # Process batch when full or at end
      if len(frame_buffer) >= batch_size or i == total_frames - 1:
        with Pool(cpu_count()) as pool:
          results = pool.map(_sort_video_frame_worker, frame_buffer)
        
        # Sort results by frame index and write in order
        results.sort(key=lambda x: x[0])
        for _, bgr_frame in results:
          video_writer.write(bgr_frame)
          frames_processed += 1
          
          if frames_processed % 10 == 0 or frames_processed == total_frames:
            print(f"Progress: {frames_processed}/{total_frames}")
        
        frame_buffer = []
    
    return True
    
  except Exception as e:
    print(f"Error processing video frames: {e}")
    return False
