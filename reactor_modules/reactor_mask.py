import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from torchvision.transforms.functional import to_pil_image

from scripts.reactor_logger import logger
from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from scripts.reactor_entities.face import FaceArea
from scripts.reactor_entities.rect import Rect
from scripts.reactor_globals import BASE_PATH


def _get_mask_generator(mask_engine: str = "BiSeNet"):
    """Get the appropriate mask generator. FaRL/FaceXFormer are lazy-imported to avoid loading if unused."""
    if mask_engine == "FaRL":
        try:
            from scripts.reactor_inferencers.farl_mask_generator import FaRLMaskGenerator
            return FaRLMaskGenerator()
        except Exception as e:
            logger.warning("Failed to load FaRL engine: %s. Falling back to BiSeNet.", e)
            return BiSeNetMaskGenerator()
    if mask_engine == "FaceXFormer":
        try:
            from scripts.reactor_inferencers.facexformer_mask_generator import FaceXFormerMaskGenerator
            return FaceXFormerMaskGenerator()
        except Exception as e:
            logger.warning("Failed to load FaceXFormer engine: %s. Falling back to BiSeNet.", e)
            return BiSeNetMaskGenerator()
    return BiSeNetMaskGenerator()


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (0, 128, 128),
]

def color_generator(colors):
    while True:
        for color in colors:
            yield color


def process_face_image(
        face: FaceArea,
        **kwargs,
    ) -> Image:
        image = np.array(face.image)
        overlay = image.copy()
        color_iter = color_generator(colors)
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
        l, t, r, b = face.face_area_on_image
        cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
        if face.landmarks_on_image is not None:
            for landmark in face.landmarks_on_image:
                cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
        alpha = 0.3
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return Image.fromarray(output)


def _color_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Transfer LAB color statistics from target face region to source (swapped) face region."""
    mask_gray = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bool = mask_gray > 128
    if mask_bool.sum() < 100:
        return source
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        s = src_lab[:, :, ch][mask_bool]
        t = tgt_lab[:, :, ch][mask_bool]
        s_mean, s_std = s.mean(), max(s.std(), 1e-6)
        t_mean, t_std = t.mean(), max(t.std(), 1e-6)
        src_lab[:, :, ch][mask_bool] = (s - s_mean) * (t_std / s_std) + t_mean
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _analyze_scene(bisenet_classes: np.ndarray) -> dict:
    """Analyze BiSeNet segmentation output to detect accessories and occlusions."""
    total_face_pixels = np.sum((bisenet_classes >= 1) & (bisenet_classes <= 13))
    has_glasses = np.sum(bisenet_classes == 6) > 50
    has_hat = np.sum(bisenet_classes == 18) > 50
    has_earring = np.sum(bisenet_classes == 9) > 20
    skin_pixels = np.sum(bisenet_classes == 1)
    occlusion_ratio = 1.0 - (skin_pixels / max(total_face_pixels, 1))
    return {
        "has_glasses": has_glasses,
        "has_hat": has_hat,
        "has_earring": has_earring,
        "occlusion_ratio": occlusion_ratio,
        "total_face_pixels": total_face_pixels,
    }


def _compute_edge_contrast(swapped: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Measure color contrast at mask boundary to decide blending aggressiveness."""
    mask_gray = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Create boundary band: dilate - erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask_gray, kernel, iterations=1)
    eroded = cv2.erode(mask_gray, kernel, iterations=1)
    boundary = ((dilated > 127) & (eroded < 128))
    if boundary.sum() < 10:
        return 0.0
    diff = np.abs(swapped.astype(float) - target.astype(float))
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)
    return float(diff[boundary].mean()) / 255.0


def _compute_adaptive_params(scene: dict, edge_contrast: float, face_size: int, extra_analysis: dict = None) -> dict:
    """Compute optimal mask parameters based on scene analysis.
    extra_analysis: optional dict from FaceXFormer with 'headpose', 'visibility', etc."""
    # Erosion: more with accessories to avoid halo
    erosion = 0
    if scene["has_glasses"]:
        erosion = max(erosion, 3)
    if scene["occlusion_ratio"] > 0.4:
        erosion = max(erosion, 2)

    # Blur kernel: larger when edge contrast is high
    base_kernel = max(5, int(face_size * 0.03))
    contrast_boost = int(edge_contrast * 10)
    kernel_size = base_kernel + contrast_boost
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Color correction: stronger when LAB distance is large
    color_factor = min(1.0, edge_contrast * 3.0)
    apply_color = color_factor > 0.15

    # Seamless clone: use when contrast is noticeable and mask doesn't touch border
    use_seamless = edge_contrast > 0.08

    # FaceXFormer extra data adjustments
    if extra_analysis is not None:
        headpose = extra_analysis.get('headpose')
        if headpose is not None:
            try:
                if isinstance(headpose, dict):
                    yaw = abs(float(headpose.get('yaw', headpose.get(0, 0))))
                elif hasattr(headpose, '__getitem__'):
                    yaw = abs(float(headpose[0]))
                else:
                    yaw = abs(float(headpose))
            except (TypeError, IndexError, ValueError, KeyError):
                yaw = 0
            # Extreme profile angles: reduce seamless clone aggressiveness
            if yaw > 30:
                use_seamless = False
                logger.info("Extended mask - headpose yaw=%.1f, disabling seamlessClone", yaw)
            # Moderate angles: increase blur for smoother transition
            if yaw > 15:
                kernel_size = max(kernel_size, base_kernel + 3)

    return {
        "erosion": erosion,
        "kernel_size": kernel_size,
        "use_gaussian": True,
        "apply_color": apply_color,
        "use_seamless": use_seamless,
    }


def apply_face_mask(swapped_image:np.ndarray,target_image:np.ndarray,target_face,entire_mask_image:np.array,mouth_mask:bool=False,mask_face_mode:int=1,mask_engine:str="BiSeNet",use_occluder:bool=False)->np.ndarray:
    logger.status("Correcting Face Mask%s [%s]", " (Extended)" if mask_face_mode == 2 else "", mask_engine)
    mask_generator = _get_mask_generator(mask_engine)
    face = FaceArea(target_image,Rect.from_ndarray(np.array(target_face.bbox)),1.6,512,"")
    face_image = np.array(face.image)
    process_face_image(face)
    face_area_on_image = face.face_area_on_image
    affected_areas = ["Face"]
    if mouth_mask:
        affected_areas.append("MouthExclude")
    mask = mask_generator.generate_mask(
        face_image,
        face_area_on_image=face_area_on_image,
        affected_areas=affected_areas,
        mask_size=0,
        use_minimal_area=True
    )

    # Occlusion detection: subtract occluded regions from the mask
    if use_occluder:
        try:
            from reactor_modules.reactor_occluder import detect_occlusion
            occluder_model_path = os.path.join(BASE_PATH, "models", "occluder.onnx")
            if os.path.exists(occluder_model_path):
                logger.status("Applying Occlusion Detection")
                occlusion_mask = detect_occlusion(face_image, occluder_model_path)
                mask[occlusion_mask > 127] = 0
            else:
                logger.warning("Occluder model not found at %s, skipping occlusion detection", occluder_model_path)
        except Exception as e:
            logger.warning("Occlusion detection failed: %s", e)

    face_size = max(face.width, face.height)

    if mask_face_mode == 2:
        # === EXTENDED MODE: adaptive auto-parameters ===

        # Get BiSeNet raw classes for scene analysis
        bisenet_classes = mask_generator.get_raw_classes(face_image, face_area_on_image)
        scene = _analyze_scene(bisenet_classes)
        logger.info("Extended mask - scene: glasses=%s, hat=%s, occlusion=%.2f",
                     scene["has_glasses"], scene["has_hat"], scene["occlusion_ratio"])

        # Preliminary blur to compute edge contrast
        k_pre = max(5, int(face_size * 0.03))
        mask_pre = cv2.blur(mask.copy(), (k_pre, k_pre))
        larger_mask_pre = cv2.resize(mask_pre, dsize=(face.width, face.height))
        temp_mask = np.zeros_like(entire_mask_image)
        temp_mask[face.top:face.bottom, face.left:face.right] = larger_mask_pre
        edge_contrast = _compute_edge_contrast(swapped_image, target_image, temp_mask)

        # Retrieve FaceXFormer extra analysis if available
        extra_analysis = getattr(mask_generator, 'last_analysis', None)
        params = _compute_adaptive_params(scene, edge_contrast, face_size, extra_analysis)
        logger.info("Extended mask - params: erosion=%d, kernel=%d, gaussian=%s, color=%s, seamless=%s",
                     params["erosion"], params["kernel_size"], params["use_gaussian"],
                     params["apply_color"], params["use_seamless"])

        # Apply erosion
        if params["erosion"] > 0:
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["erosion"], params["erosion"]))
            mask = cv2.erode(mask, ek, iterations=1)

        # Apply blur (Gaussian for Extended)
        ks = params["kernel_size"]
        if ks % 2 == 0:
            ks += 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)

        # Place mask in full image
        larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
        entire_mask_image[face.top:face.bottom, face.left:face.right] = larger_mask

        # Color correction
        if params["apply_color"]:
            swapped_image = _color_transfer(swapped_image, target_image, entire_mask_image)

        # Blending: seamless clone with fallback
        if params["use_seamless"]:
            try:
                mask_gray = entire_mask_image if len(entire_mask_image.shape) == 2 else cv2.cvtColor(entire_mask_image, cv2.COLOR_BGR2GRAY)
                clone_mask = np.where(mask_gray > 127, 255, 0).astype(np.uint8)
                bbox = target_face.bbox.astype(int)
                h, w = target_image.shape[:2]
                center = (
                    int(np.clip((bbox[0] + bbox[2]) // 2, 1, w - 2)),
                    int(np.clip((bbox[1] + bbox[3]) // 2, 1, h - 2)),
                )
                # Check mask doesn't touch image border (seamlessClone fails)
                if (clone_mask[0, :].any() or clone_mask[-1, :].any() or
                        clone_mask[:, 0].any() or clone_mask[:, -1].any()):
                    logger.info("Extended mask - mask touches border, skipping seamlessClone")
                else:
                    result = cv2.seamlessClone(swapped_image, target_image, clone_mask, center, cv2.NORMAL_CLONE)
                    return result
            except cv2.error as e:
                logger.warning("seamlessClone failed (%s), falling back to composite", e)

        # Fallback: standard composite
        result = Image.composite(
            Image.fromarray(swapped_image), Image.fromarray(target_image),
            Image.fromarray(entire_mask_image).convert("L"))
        return np.array(result)

    else:
        # === STANDARD MODE (Yes): original behavior ===
        kernel_size = max(5, int(face_size * 0.03))
        mask = cv2.blur(mask, (kernel_size, kernel_size))

        larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
        entire_mask_image[face.top:face.bottom, face.left:face.right] = larger_mask

        result = Image.composite(
            Image.fromarray(swapped_image), Image.fromarray(target_image),
            Image.fromarray(entire_mask_image).convert("L"))
        return np.array(result)


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def rotate_image(image: Image, angle: float) -> Image:
    if angle == 0:
        return image
    return Image.fromarray(rotate_array(np.array(image), angle))


def correct_face_tilt(angle: float) -> bool:
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle > 40


def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)


def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)

def mask_to_pil(masks, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]

def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks
