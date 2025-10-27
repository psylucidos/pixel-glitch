# Pixel Sorter

Create videos with progressive pixel sorting effects from single images, image sequences, or existing videos.

**Supports strict color space handling for video production workflows:**
- Rec.709 color graded video (.mov files)
- SLOG3 raw footage (.mp4 files)

## Installation

```bash
pip install pixelsort pillow opencv-python numpy pymediainfo
```

**Note:** `pymediainfo` requires MediaInfo library:
- Linux: `sudo apt-get install libmediainfo-dev`
- macOS: `brew install media-info`
- Windows: Included with pymediainfo package

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
- `--no-multiprocessing` - Disable parallel processing (enabled by default)
- `--type-override` - **Advanced:** Allow unsupported color profiles (video mode only, not recommended)

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
- Loads one source image (supports .jpg, .jpeg, .png, .bmp)
- Generates N static frames (unchanged)
- Generates M progressively sorted frames
- Output: Rec.709 color space (.mov)

### Stitch Mode
- Loads numbered images from folder (1.jpg, 2.jpg, etc.)
- Warns about missing sequence numbers
- Adds initial static frames from first images
- Applies progressive sorting to remaining images in sequence
- Output: Rec.709 color space (.mov)

### Video Mode
- **Automatic color profile detection:**
  - Reads actual color metadata from video (not just file extension)
  - **Supported:** Rec.709 (SDR) only
  - **Rejected:** HDR (HLG, PQ/HDR10), SLOG3, DCI-P3, and other profiles
  - Use `--type-override` to bypass restrictions (not recommended - colors will be wrong)
- Reads input video frame-by-frame
- Preserves first N frames unchanged (normal frames)
- Applies progressively increasing pixel sort to remaining frames
- Each frame is sorted independently from its original state
- Output video maintains source FPS and dimensions
- Memory-efficient: processes frames in batches

## Examples

**Basic single image:**
```bash
python pixelsorter.py -m single -i photo.jpg -o sorted.mp4
```

**With custom settings:**
```bash
python pixelsorter.py -m single -i photo.jpg -o sorted.mp4 -f 120 -s 10 --fps 24
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

## Color Space Management

This tool is designed for professional video production workflows with **strict color profile validation**:

**Supported Video Color Profiles:**
- **Rec.709 (SDR)** ✅ - Only officially supported profile
- **BT.2020 HLG** ❌ - HDR, will cause overexposure and color shifts
- **BT.2020 PQ (HDR10)** ❌ - HDR, will cause overexposure and color shifts
- **SLOG3** ❌ - Log profile, requires special handling
- **DCI-P3, BT.601, etc.** ❌ - Other gamuts not supported

**Why Only Rec.709?**
The pixel sorting algorithm processes pixel values directly. Processing HDR or log profiles as SDR causes:
- Overexposure and clipping
- Loss of shadow and highlight detail
- Color shifts and incorrect hue
- Loss of color grading

**Automatic Detection:**
The tool reads actual color metadata from video files using MediaInfo:
- Color primaries (e.g., BT.709, BT.2020)
- Transfer characteristics (e.g., BT.709, HLG, PQ)
- Prevents processing of incompatible profiles

**--type-override Flag:**
Advanced users can bypass restrictions with `--type-override`, but this is **not recommended**:
```bash
python pixelsorter.py -m video -i hdr_video.mov -o output.mov --type-override
```
⚠️ **Warning:** Colors will likely be incorrect. Convert to Rec.709 first for proper results.

**Image Processing:**
- Supported formats: .jpg, .jpeg, .png, .bmp
- RGBA images composited onto white background
- All outputs use consistent RGB color space
- Single image and stitch modes output in Rec.709 (.mov)

**Color Consistency:**
- No unnecessary color space conversions
- Color profile validated before processing
- All frames validated for dimension consistency
- Prevents accidental color grading disruption

## Notes

- Images in stitch mode must be numbered sequentially (1.jpg, 2.jpg, etc.)
- Tool warns about missing sequence numbers
- Video mode only accepts .mov (Rec.709) or .mp4 (SLOG3)
- Video mode preserves source FPS and color space automatically
- Processing time depends on content size and frame count
- Video mode is memory-efficient (processes frames in batches)
- Multiprocessing is enabled by default for 3-12x speedup
- Multiprocessing can be disabled with `--no-multiprocessing` flag
