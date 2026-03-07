import os
import cv2
import numpy as np
import onnxruntime as ort

from scripts.reactor_logger import logger
from scripts.reactor_helpers import get_Device


_OCCLUDER_SESSION = None
_OCCLUDER_MODEL_PATH = None


def _get_providers():
    device = get_Device()
    if device == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_occluder_session(model_path: str) -> ort.InferenceSession:
    global _OCCLUDER_SESSION, _OCCLUDER_MODEL_PATH
    if _OCCLUDER_SESSION is None or _OCCLUDER_MODEL_PATH != model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Occluder model not found: {model_path}")
        logger.status("Loading Occluder model: %s", model_path)
        _OCCLUDER_MODEL_PATH = model_path
        _OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=_get_providers())
    return _OCCLUDER_SESSION


def detect_occlusion(
    face_image: np.ndarray,
    model_path: str,
    threshold: float = 0.5,
    dilate_kernel: int = 5,
    dilate_iterations: int = 2,
) -> np.ndarray:
    """Detect occluded regions on a face crop using the occluder ONNX model.

    Parameters
    ----------
    face_image : np.ndarray
        Face crop in BGR format (as used throughout reactor).
    model_path : str
        Absolute path to occluder.onnx.
    threshold : float
        Probability threshold above which a pixel is considered occluded (default 0.5).
    dilate_kernel : int
        Kernel size for dilation (safety margin). Default 5.
    dilate_iterations : int
        Number of dilation iterations. Default 2.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) at the same size as *face_image*,
        where 255 = occluded pixel.
    """
    orig_h, orig_w = face_image.shape[:2]

    # Prepare input: resize to 256x256, RGB, float32, NCHW
    resized = cv2.resize(face_image, (256, 256))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)    # CHW -> NCHW

    session = _get_occluder_session(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})

    # The model outputs a probability map; squeeze to 2D
    prob_map = output[0].squeeze()
    if prob_map.ndim == 3:
        # If multi-channel, take the occlusion channel (last one)
        prob_map = prob_map[-1]

    # Binarize
    occlusion_mask = (prob_map > threshold).astype(np.uint8) * 255

    # Dilate for safety margin
    if dilate_kernel > 0 and dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=dilate_iterations)

    # Resize back to original face crop size
    occlusion_mask = cv2.resize(occlusion_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return occlusion_mask
