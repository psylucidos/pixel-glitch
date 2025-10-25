# Pixel Sorter

Create videos with progressive pixel sorting effects from single images, image sequences, or existing videos.

## Installation

```bash
pip install pixelsort pillow opencv-python numpy
```

## Performance

The tool uses **multiprocessing** by default to parallelize frame processing across all CPU cores, providing **3-12x speedup** depending on your system. On an 8-core CPU, expect ~6-7x faster processing compared to sequential execution.

## Usage

### Interactive Mode
```bash
python pixelsorter.py
```

### Single Image Mode
```bash
python pixelsorter.py -m single -i input.jpg -o output.mp4 -f 90 -s 5
```

### Image Stitching Mode
```bash
python pixelsorter.py -m stitch -d ./images/ -o output.mp4 -s 42
```

### Video Mode
```bash
python pixelsorter.py -m video -i input.mp4 -o output.mp4 -s 30
```

## Options

**Required:**
- `-i, --input` - Input image path (single mode) or video path (video mode)
- `-d, --directory` - Input directory with numbered images (stitch mode only)
- `-o, --output` - Output video path

**Optional:**
- `-f, --frames` - Number of sorted frames (default: 90, single mode only)
- `-s, --static` - Number of static/normal frames at start (default: 5 single, 42 stitch, 30 video)
- `--fps` - Frames per second (default: 30, single/stitch only - video mode preserves source FPS)
- `--start-upper` - Start upper threshold 0-1 (default: 0.5)
- `--upper-buffer` - Upper buffer (default: 0)
- `--start-lower` - Start lower threshold 0-1 (default: 0.5)
- `--lower-buffer` - Lower buffer (default: 0)
- `--reuse-frames` - Sort previous frame instead of source (single mode only)
- `--no-multiprocessing` - Disable parallel processing (enabled by default)

## Project Structure

```
pixelsorter.py    Main CLI script
utils.py          File and image loading utilities
video.py          Video creation and processing functions
sorter.py         Pixel sorting logic
```

## How It Works

The tool generates videos by progressively pixel-sorting images or video frames:

1. **Static/Normal frames** - Shows original content unchanged
2. **Progressive sorting** - Each frame has incrementally adjusted sorting thresholds
3. **Video output** - Combines frames into MP4 video

The sorting uses the `pixelsort` library with lightness-based threshold intervals. Thresholds expand from a center point (default 0.5) over the number of frames, creating a gradual increase in sorting intensity.

### Single Image Mode
- Loads one source image
- Generates N static frames (unchanged)
- Generates M progressively sorted frames
- Optional: Can sort the previous frame for cumulative effect (`--reuse-frames`)

### Stitch Mode
- Loads numbered images from folder (1.jpg, 2.jpg, etc.)
- Adds initial static frames from first images
- Applies progressive sorting to each image in sequence
- Combines into single video

### Video Mode (NEW)
- Reads input video frame-by-frame
- Preserves first N frames unchanged (normal frames)
- Applies progressively increasing pixel sort to remaining frames
- Each frame is sorted independently from its original state
- Output video maintains source FPS and dimensions
- Memory-efficient: processes one frame at a time

## Examples

**Basic single image:**
```bash
python pixelsorter.py -m single -i photo.jpg -o sorted.mp4
```

**With custom settings:**
```bash
python pixelsorter.py -m single -i photo.jpg -o sorted.mp4 -f 120 -s 10 --fps 24
```

**Cumulative sorting effect:**
```bash
python pixelsorter.py -m single -i photo.jpg -o cumulative.mp4 --reuse-frames
```

**Stitch image sequence:**
```bash
python pixelsorter.py -m stitch -d ./my_images/ -o sequence.mp4 -s 30
```

**Process video with progressive sorting:**
```bash
python pixelsorter.py -m video -i input.mp4 -o sorted.mp4 -s 30
```

**Video with custom thresholds:**
```bash
python pixelsorter.py -m video -i input.mp4 -o sorted.mp4 -s 60 --start-upper 0.6 --start-lower 0.4
```

**Disable multiprocessing (for debugging):**
```bash
python pixelsorter.py -m single -i photo.jpg -o output.mp4 --no-multiprocessing
```

## Algorithm: Video Mode

The video mode processes each frame with progressively increasing sort intensity:

1. Extract video metadata (frames, fps, dimensions)
2. Calculate threshold increments based on number of frames to sort
3. For each frame:
   - If frame < num_normal_frames: keep unchanged
   - Else: apply pixel sort with progressive thresholds
4. Write processed frame to output video

**Progressive intensity formula:**
- Frame 0 (after normal frames): minimal sorting (thresholds near 0.5)
- Frame N (last frame): maximal sorting (thresholds approach 0 and 1)
- Each frame is more sorted than the previous
- Result: video that gradually "glitches" over time

## Notes

- Images in stitch mode must be numbered sequentially (1.jpg, 2.jpg, etc.)
- Supported image formats: .jpg, .jpeg, .png, .bmp
- Video mode supports common video formats (mp4, avi, mov, etc.)
- Output is always MP4 format
- Video mode preserves source FPS automatically
- Processing time depends on content size and frame count
- Video mode is memory-efficient (processes frames in batches)
- Multiprocessing is enabled by default for 3-12x speedup
- Multiprocessing can be disabled with `--no-multiprocessing` flag
- Reuse frames mode (single image) runs sequentially due to data dependency
