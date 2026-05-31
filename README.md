# Aerial-Guardian

Input - VisDrone Dataset (uav0000086_00000_v - Persons playing basketball)
Input link - https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing   https://github.com/VisDrone/VisDrone-Dataset  

## Results:
OUTPUT - https://drive.google.com/drive/folders/1UkLNH1VVUUYqYOEcstlH1QHKjias8QoT
DETECTION + TRACKING FPS - 14-17 FPS for online detection and tracking on T4 Collab GPU, and 25-27 FPS for offline detection and tracking due to batch processing

METRICS for detection and tracking -

MOTA - 45 %
IDF1 - 62 %
Prcn - 80 %
Rcll - 60 %

Model size - 6 MB

## Instructions:

### Clone the detection and tracking framework
git clone https://github.com/highquanglity/LEAF-YOLO.git
git clone https://github.com/ifzhang/ByteTrack.git

### Install requirements
pip install torch torchvision opencv-python numpy pandas matplotlib thop

### Open main.py and set up your input paths, ground truth paths and model ckpt to mathc your setup and run python main.py

## Summary

1. Choice of Architecture & Small Object Detection
For the base detector, I went with LEAF-YOLO (Lightweight Efficient Attention-based Feature YOLO). The assignment required a model under 300MB that didn't rely on heavy GPU compute. LEAF-YOLO is specifically designed to be lightweight while using attention mechanisms to pull better features out of the image without blowing up the parameter count.

To handle the small scale of vehicles from high drone altitudes, I passed the frames through at a 640x640 resolution using a dynamic stride letterbox so no pixel density was lost to bad resizing. More importantly, I dropped the confidence threshold down to 0.15. Standard thresholds (like 0.5) ignore tiny or distant person, but lowering it allows the pipeline to catch them.

2. Handling ID Switching, Ego-Motion, and Occlusions
For tracking, I implemented ByteTrack. Unlike older trackers that just discard low-confidence bounding boxes, ByteTrack keeps them and tries to associate them with existing tracks. This is critical for drone footage where a vehicle might temporarily look like a "blob" due to distance or partial occlusion. I set the track_buffer to 80 frames to keep IDs alive if a car goes under a tree or bridge.

To handle the drone's ego-motion (camera movement), I wrote a Global Motion Compensation (GMC) function using OpenCV's Optical Flow (calcOpticalFlowPyrLK). Before the tracker updates, it calculates how much the background shifted between the previous and current frame, and corrects the spatial locations of the bounding boxes so the tracker doesn't get confused by sudden drone movements.

3. Edge Hardware Adaptation (NVIDIA Jetson)
I developed and tested this pipeline using a cloud GPU (NVIDIA T4). While I haven't had the opportunity to flash this onto a physical Jetson device yet, I specifically designed the pipeline with edge deployment in mind.

If I were taking this project to production on a Jetson Nano or Orin, I wouldn't run the raw PyTorch (.pt) script. Instead, my exact next steps would be:

Drop the PyTorch Overhead: Export the LEAF-YOLO model to ONNX, then compile it into a TensorRT engine. This is basically mandatory to get real-time FPS on Jetson architecture.

Quantize to FP16: I already tested converting the model to half-precision (.half()) during development. It nearly doubles the inference speed and saves massive amounts of VRAM, which is exactly what a memory-constrained drone needs.

Hardware Acceleration: I would swap out the CPU-based OpenCV Optical Flow I used for camera compensation and replace it with the Jetson's native hardware-accelerated NVidia Optical Flow (NVOF) to keep the pipeline completely bottleneck-free.
