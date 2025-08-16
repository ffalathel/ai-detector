import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, List, Optional
import logging
import mimetypes
import io
import tempfile
import re

logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace('..', '')
    # Limit length
    return filename[:255]

# File size limits (in bytes)
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

# Allowed file types
ALLOWED_IMAGE_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png',
    'image/gif', 'image/webp', 'image/bmp'
}

ALLOWED_VIDEO_TYPES = {
    'video/mp4', 'video/avi', 'video/mov',
    'video/wmv', 'video/flv', 'video/webm',
    'video/quicktime'
}

# Valid extensions
VALID_EXTENSIONS = {
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'},
    'video': {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'}
}

# Cached face cascade classifier
_face_cascade: Optional[cv2.CascadeClassifier] = None

def validate_file_bytes(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    file_type: str = "image"
) -> Dict[str, Any]:
    """
    Validate file size, type, and extension from raw bytes.

    Args:
        file_bytes: Raw file content.
        filename: Original filename.
        content_type: MIME content type.
        file_type: 'image' or 'video'.

    Returns:
        Dict with 'valid': bool, 'error': str if invalid, 'size': int bytes if valid.
    """
    try:
        if file_type == "image":
            max_size = MAX_IMAGE_SIZE
            allowed_types = ALLOWED_IMAGE_TYPES
        else:
            max_size = MAX_VIDEO_SIZE
            allowed_types = ALLOWED_VIDEO_TYPES

        file_size = len(file_bytes)
        if file_size > max_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            }

        if content_type not in allowed_types:
            return {
                "valid": False,
                "error": f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            }

        if not filename:
            return {
                "valid": False,
                "error": "Filename is required"
            }
        
        filename = sanitize_filename(filename)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in VALID_EXTENSIONS[file_type]:
            return {
                "valid": False,
                "error": f"Invalid file extension: {file_ext}"
            }

        return {"valid": True, "size": file_size}

    except Exception as e:
        logger.error(f"File validation error: {e}")
        return {
            "valid": False,
            "error": "File validation failed"
        }

def get_file_info(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract file info such as size in MB and MIME type.

    Args:
        file_content: Raw file bytes.
        filename: Filename string.

    Returns:
        Dict with filename, size (str MB), size_bytes (int), and MIME type (str).
    """
    return {
        "filename": filename,
        "size": f"{len(file_content) / (1024*1024):.2f} MB",
        "size_bytes": len(file_content),
        "type": mimetypes.guess_type(filename)[0] or "unknown"
    }

def process_image(image_content: bytes) -> Image.Image:
    """
    Load and validate image from bytes.

    Args:
        image_content: Raw image bytes.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError if image is invalid or too small.
    """
    try:
        image = Image.open(io.BytesIO(image_content))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        if width < 32 or height < 32:
            raise ValueError("Image too small (minimum 32x32 pixels)")

        if width > 4096 or height > 4096:
            logger.warning("Large image detected (>%dx%d), consider resizing", 4096, 4096)

        return image

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")

def process_video(video_content: bytes) -> Dict[str, Any]:
    """
    Save video content to a secure temp file and extract basic properties.

    Args:
        video_content: Raw video bytes.

    Returns:
        Dict with fps, total_frames, width, height, duration (seconds), temp_path.

    Raises:
        ValueError if video cannot be opened or is too short.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    try:
        temp_file.write(video_content)
        temp_file.close()

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            if duration < 0.5:
                raise ValueError("Video too short (minimum 0.5 seconds)")

            if duration > 300:
                logger.warning("Long video detected (>5 minutes), processing may take time")

        finally:
            cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
            "temp_path": temp_path
        }

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise ValueError(f"Failed to process video: {str(e)}")

def extract_video_frames(video_path: str, max_frames: int = 50) -> List[np.ndarray]:
    """
    Extract up to max_frames evenly spaced frames from video.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of frames as RGB numpy arrays.
    """
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)

        frame_count = 0
        extracted_count = 0
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
            frame_count += 1
        logger.info(f"Extracted {len(frames)} frames from video")
    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
    finally:
        if cap is not None:
            cap.release()

    return frames

def preprocess_frames(
    frames: List[np.ndarray], 
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess frames for model input.

    Args:
        frames: List of RGB numpy frames.
        target_size: Target image size (width, height).

    Returns:
        Torch tensor of shape (1, seq_len, C, H, W).
    """
    if not frames:
        raise ValueError("No frames to process")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    processed_frames = []
    for idx, frame in enumerate(frames):
        try:
            processed_frame = transform(frame)
            processed_frames.append(processed_frame)
        except Exception as e:
            logger.warning(f"Failed to process frame {idx}: {e}")

    if not processed_frames:
        raise ValueError("No frames could be processed")

    frame_tensor = torch.stack(processed_frames)  # (seq_len, C, H, W)
    return frame_tensor.unsqueeze(0)  # (1, seq_len, C, H, W)

def get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """
    Load and cache OpenCV Haar Cascade for face detection.

    Returns:
        Loaded CascadeClassifier or None if failed.
    """
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            logger.warning("Face cascade file not found at %s", cascade_path)
            return None
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade

def detect_face_regions(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an RGB frame.

    Args:
        frame: RGB image as numpy array.

    Returns:
        List of bounding boxes (x, y, w, h).
    """
    cascade = get_face_cascade()
    if cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return [(x, y, w, h) for (x, y, w, h) in faces]

def crop_face_region(frame: np.ndarray, face_coords: Tuple[int, int, int, int], padding: int = 20) -> np.ndarray:
    """
    Crop face region with padding from frame.

    Args:
        frame: RGB numpy array.
        face_coords: (x, y, w, h).
        padding: pixels to pad around face.

    Returns:
        Cropped face region as numpy array.
    """
    try:
        x, y, w, h = face_coords
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        return frame[y1:y2, x1:x2]
    except Exception as e:
        logger.warning(f"Face cropping error: {e}")
        return frame

def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Delete temporary files safely.

    Args:
        file_paths: List of file paths to delete.
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Deleted temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")

def format_confidence_score(confidence: float) -> str:
    """
    Format confidence to human-readable string.

    Args:
        confidence: Float between 0 and 1.

    Returns:
        Formatted string with descriptive label.
    """
    percentage = confidence * 100
    if percentage >= 90:
        return f"Very High ({percentage:.1f}%)"
    elif percentage >= 70:
        return f"High ({percentage:.1f}%)"
    elif percentage >= 50:
        return f"Moderate ({percentage:.1f}%)"
    else:
        return f"Low ({percentage:.1f}%)"

def get_system_info() -> Dict[str, Any]:
    """
    Collect system and environment information.

    Returns:
        Dict with system details.
    """
    import psutil
    import platform

    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=True),
        'memory_total_bytes': psutil.virtual_memory().total,
        'memory_available_bytes': psutil.virtual_memory().available,
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
def validate_file(file, file_type):
    """Validate uploaded file - wrapper for async usage"""
    if not hasattr(file, 'content_type') or not hasattr(file, 'filename'):
        return {"valid": False, "error": "Invalid file object"}
    
    content_type = file.content_type
    filename = file.filename
    
    if file_type == "image":
        max_size = MAX_IMAGE_SIZE
        allowed_types = ALLOWED_IMAGE_TYPES
    else:
        max_size = MAX_VIDEO_SIZE
        allowed_types = ALLOWED_VIDEO_TYPES
    
    if content_type not in allowed_types:
        return {"valid": False, "error": f"Invalid file type. Allowed: {', '.join(allowed_types)}"}
    
    if not filename:
        return {"valid": False, "error": "Filename is required"}
    
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in VALID_EXTENSIONS[file_type]:
        return {"valid": False, "error": f"Invalid extension: {file_ext}"}
    
    return {"valid": True, "error": None}