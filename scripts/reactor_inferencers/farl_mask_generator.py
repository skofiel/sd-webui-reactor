from typing import List, Tuple

import cv2
import numpy as np
import torch
import modules.shared as shared

from scripts.reactor_logger import logger
from scripts.reactor_inferencers.mask_generator import MaskGenerator

# Singleton cache for FaRL models (lazy loaded)
_face_detector = None
_face_parser = None
_farl_device = None

# FaRL CelebM/448 class mapping:
# 0: background, 1: neck, 2: face (skin), 3: cloth, 4: rr (right ear), 5: lr (left ear),
# 6: rb (right brow), 7: lb (left brow), 8: re (right eye), 9: le (left eye),
# 10: nose, 11: imouth (inner mouth), 12: llip (lower lip), 13: ulip (upper lip),
# 14: hair, 15: eyeg (eyeglasses), 16: hat, 17: earr (earring), 18: neckl (necklace)

# Classes included in face mask
FARL_FACE_CLASSES = {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}
# Excluded: 0 (bg), 1 (neck), 3 (cloth), 14 (hair), 16 (hat), 17 (earring), 18 (necklace)

# Mapping from BiSeNet-style affected_areas to FaRL class exclusions
# BiSeNet class 11 = inner mouth; FaRL class 11 = inner mouth
FARL_MOUTH_CLASS = 11
FARL_NECK_CLASS = 1
FARL_HAIR_CLASS = 14
FARL_HAT_CLASS = 16


def _get_farl_models(device):
    """Lazy-load FaRL face detector and parser (singleton)."""
    global _face_detector, _face_parser, _farl_device
    if _face_detector is not None and _face_parser is not None and _farl_device == str(device):
        return _face_detector, _face_parser
    try:
        import facer
        logger.status("Loading FaRL models on %s...", device)
        _face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        _face_parser = facer.face_parser('farl/celebm/448', device=device)
        _farl_device = str(device)
        logger.status("FaRL models loaded successfully")
        return _face_detector, _face_parser
    except Exception as e:
        logger.error("Failed to load FaRL models: %s", e)
        _face_detector = None
        _face_parser = None
        raise


class FaRLMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        self._device = shared.device

    def name(self):
        return "FaRL"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        affected_areas: List[str],
        mask_size: int,
        use_minimal_area: bool,
        fallback_ratio: float = 0.25,
        **kwargs,
    ) -> np.ndarray:
        import facer

        face_image_input = face_image.copy()
        h, w, _ = face_image_input.shape

        if use_minimal_area:
            face_image_input = MaskGenerator.mask_non_face_areas(face_image_input, face_area_on_image)

        try:
            face_detector, face_parser = _get_farl_models(self._device)

            # facer expects RGB uint8 HWC -> BCHW tensor
            # face_image is already RGB (comes from FaceArea which provides RGB)
            image_rgb = face_image_input[:, :, ::-1]  # BGR -> RGB (same as bisenet does)
            image_rgb = image_rgb[:, :, ::-1]  # back to BGR... actually face_image comes as BGR from numpy(face.image)
            # Let's be explicit: face_image from FaceArea is BGR (opencv default)
            # facer needs RGB
            image_rgb = cv2.cvtColor(face_image_input, cv2.COLOR_BGR2RGB)

            image_tensor = facer.hwc2bchw(torch.from_numpy(image_rgb).to(dtype=torch.uint8)).to(device=self._device)

            with torch.inference_mode():
                faces = face_detector(image_tensor)

            if faces is None or len(faces.get('rects', [])) == 0:
                logger.warning("FaRL: No faces detected, returning empty mask")
                return np.zeros((h, w, 3), dtype=np.uint8)

            with torch.inference_mode():
                faces = face_parser(image_tensor, faces)

            seg_logits = faces['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)
            seg_map = seg_probs.argmax(dim=1)  # nfaces x h x w

            # Use first detected face
            class_map = seg_map[0].cpu().numpy().astype(np.uint8)

            mask = self.__to_mask(class_map, affected_areas)

            if mask_size > 0:
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

            # Resize back if needed (seg_map may differ from input size)
            mask_h, mask_w = mask.shape[:2]
            if mask_w != w or mask_h != h:
                mask = cv2.resize(mask, dsize=(w, h))

            return mask

        except Exception as e:
            logger.warning("FaRL mask generation failed: %s. Falling back to BiSeNet.", e)
            from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
            fallback = BiSeNetMaskGenerator()
            return fallback.generate_mask(
                face_image, face_area_on_image, affected_areas, mask_size, use_minimal_area, fallback_ratio
            )

    def get_raw_classes(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return raw per-pixel class labels from FaRL.
        Maps FaRL classes to BiSeNet-equivalent classes for scene analysis compatibility."""
        import facer

        face_image_input = face_image.copy()
        face_image_input = MaskGenerator.mask_non_face_areas(face_image_input, face_area_on_image)

        try:
            face_detector, face_parser = _get_farl_models(self._device)

            image_rgb = cv2.cvtColor(face_image_input, cv2.COLOR_BGR2RGB)
            image_tensor = facer.hwc2bchw(torch.from_numpy(image_rgb).to(dtype=torch.uint8)).to(device=self._device)

            with torch.inference_mode():
                faces = face_detector(image_tensor)

            if faces is None or len(faces.get('rects', [])) == 0:
                return np.zeros(face_image.shape[:2], dtype=np.uint8)

            with torch.inference_mode():
                faces = face_parser(image_tensor, faces)

            seg_logits = faces['seg']['logits']
            seg_map = seg_logits.softmax(dim=1).argmax(dim=1)
            class_map = seg_map[0].cpu().numpy().astype(np.uint8)

            # Map FaRL classes to BiSeNet-equivalent for _analyze_scene compatibility:
            # BiSeNet: 1=skin, 2-3=brows, 4-5=eyes, 6=glasses, 7-8=ears, 9=earring,
            #          10=nose, 11=mouth, 12=ulip, 13=llip, 14=neck, 17=hair, 18=hat
            # FaRL:    2=skin, 6-7=brows, 8-9=eyes, 15=glasses, 4-5=ears, 17=earring,
            #          10=nose, 11=mouth, 12=llip, 13=ulip, 1=neck, 14=hair, 16=hat
            farl_to_bisenet = {
                0: 0,   # background
                1: 14,  # neck
                2: 1,   # face/skin
                3: 0,   # cloth -> background
                4: 7,   # right ear
                5: 8,   # left ear
                6: 2,   # right brow
                7: 3,   # left brow
                8: 4,   # right eye
                9: 5,   # left eye
                10: 10, # nose
                11: 11, # inner mouth
                12: 13, # lower lip
                13: 12, # upper lip
                14: 17, # hair
                15: 6,  # eyeglasses
                16: 18, # hat
                17: 9,  # earring
                18: 0,  # necklace -> background
            }
            mapped = np.zeros_like(class_map)
            for farl_cls, bisenet_cls in farl_to_bisenet.items():
                mapped[class_map == farl_cls] = bisenet_cls

            return mapped

        except Exception as e:
            logger.warning("FaRL get_raw_classes failed: %s. Falling back to BiSeNet.", e)
            from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
            fallback = BiSeNetMaskGenerator()
            return fallback.get_raw_classes(face_image, face_area_on_image)

    def __to_mask(self, face: np.ndarray, affected_areas: List[str]) -> np.ndarray:
        """Convert FaRL class map to binary mask, matching BiSeNet __to_mask behavior."""
        keep_face = "Face" in affected_areas
        keep_neck = "Neck" in affected_areas
        keep_hair = "Hair" in affected_areas
        keep_hat = "Hat" in affected_areas
        exclude_mouth = "MouthExclude" in affected_areas

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)

        if keep_face:
            for cls in FARL_FACE_CLASSES:
                if exclude_mouth and cls == FARL_MOUTH_CLASS:
                    continue
                idx = np.where(face == cls)
                if len(idx[0]) > 0:
                    mask[idx[0], idx[1], :] = [255, 255, 255]

        if keep_neck:
            idx = np.where(face == FARL_NECK_CLASS)
            if len(idx[0]) > 0:
                mask[idx[0], idx[1], :] = [255, 255, 255]

        if keep_hair:
            idx = np.where(face == FARL_HAIR_CLASS)
            if len(idx[0]) > 0:
                mask[idx[0], idx[1], :] = [255, 255, 255]

        if keep_hat:
            idx = np.where(face == FARL_HAT_CLASS)
            if len(idx[0]) > 0:
                mask[idx[0], idx[1], :] = [255, 255, 255]

        # Dilate mouth exclusion so it survives the blur pass (same as BiSeNet)
        if exclude_mouth:
            mouth_region = np.zeros(face.shape[:2], dtype=np.uint8)
            idx = np.where(face == FARL_MOUTH_CLASS)
            if len(idx[0]) > 0:
                mouth_region[idx[0], idx[1]] = 255
            if mouth_region.any():
                dilation_size = max(11, int(face.shape[0] * 0.03))
                if dilation_size % 2 == 0:
                    dilation_size += 1
                mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
                mouth_region = cv2.dilate(mouth_region, mouth_kernel, iterations=1)
                mouth_exclude_idx = np.where(mouth_region > 0)
                mask[mouth_exclude_idx[0], mouth_exclude_idx[1], :] = [0, 0, 0]

        return mask
