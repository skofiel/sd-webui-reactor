from typing import List, Tuple

import cv2
import numpy as np

from scripts.reactor_logger import logger
from scripts.reactor_inferencers.mask_generator import MaskGenerator

# Singleton cache for FaceXFormer pipeline (lazy loaded)
_pipeline = None

# FaceXFormer uses CelebAMask-HQ classes — identical to BiSeNet:
# 0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
# 6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: earring, 10: nose,
# 11: mouth (inner), 12: u_lip, 13: l_lip, 14: neck, 15: necklace,
# 16: cloth, 17: hair, 18: hat

# Classes included in face mask (same as BiSeNet)
FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13}
# Excluded: 0 (bg), 9 (earring), 14 (neck), 15 (necklace), 16 (cloth), 17 (hair), 18 (hat)

MOUTH_CLASS = 11
NECK_CLASS = 14
HAIR_CLASS = 17
HAT_CLASS = 18


def _get_pipeline():
    """Lazy-load FaceXFormer pipeline (singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from facexformer_pipeline import FacexformerPipeline
        logger.status("Loading FaceXFormer pipeline...")
        _pipeline = FacexformerPipeline(
            debug=False,
            tasks=['faceparsing', 'visibility', 'landmark', 'headpose']
        )
        logger.status("FaceXFormer pipeline loaded successfully")
        return _pipeline
    except Exception as e:
        logger.error("Failed to load FaceXFormer pipeline: %s", e)
        _pipeline = None
        raise


class FaceXFormerMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        # Store last analysis results for Extended mode to use
        self.last_analysis = None
        self._last_classes = None

    def name(self):
        return "FaceXFormer"

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
        face_image_input = face_image.copy()
        h, w, _ = face_image_input.shape

        if use_minimal_area:
            face_image_input = MaskGenerator.mask_non_face_areas(face_image_input, face_area_on_image)

        try:
            pipeline = _get_pipeline()

            # FaceXFormer pipeline expects BGR or RGB numpy array
            # face_image comes as BGR from FaceArea (opencv default)
            # Convert to RGB for the pipeline
            image_rgb = cv2.cvtColor(face_image_input, cv2.COLOR_BGR2RGB)

            results = pipeline.run_model(image_rgb)

            if results is None or results.get('faceparsing_mask') is None:
                logger.warning("FaceXFormer: No face parsing result, returning empty mask")
                return np.zeros((h, w, 3), dtype=np.uint8)

            class_map = results['faceparsing_mask']
            if not isinstance(class_map, np.ndarray):
                class_map = np.array(class_map.cpu() if hasattr(class_map, "cpu") else class_map)
            class_map = class_map.astype(np.uint8)
            self._last_classes = class_map.copy()

            # Store extra analysis data for Extended mode
            self.last_analysis = {
                'visibility': results.get('visibility'),
                'headpose': results.get('headpose'),
                'landmarks': results.get('landmarks'),
                'face_coordinates': results.get('face_coordinates'),
            }

            mask = self.__to_mask(class_map, affected_areas)

            if mask_size > 0:
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

            # Resize back if needed
            mask_h, mask_w = mask.shape[:2]
            if mask_w != w or mask_h != h:
                mask = cv2.resize(mask, dsize=(w, h))

            return mask

        except Exception as e:
            logger.warning("FaceXFormer mask generation failed: %s. Falling back to BiSeNet.", e)
            from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
            fallback = BiSeNetMaskGenerator()
            return fallback.generate_mask(
                face_image, face_area_on_image, affected_areas, mask_size, use_minimal_area, fallback_ratio
            )

    def get_cached_classes(self):
        return self._last_classes

    def get_raw_classes(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return raw per-pixel class labels from FaceXFormer.
        Classes are already BiSeNet-compatible (same CelebAMask-HQ labeling)."""
        face_image_input = face_image.copy()
        face_image_input = MaskGenerator.mask_non_face_areas(face_image_input, face_area_on_image)

        try:
            pipeline = _get_pipeline()

            image_rgb = cv2.cvtColor(face_image_input, cv2.COLOR_BGR2RGB)
            results = pipeline.run_model(image_rgb)

            if results is None or results.get('faceparsing_mask') is None:
                return np.zeros(face_image.shape[:2], dtype=np.uint8)

            class_map = results['faceparsing_mask']
            if not isinstance(class_map, np.ndarray):
                class_map = np.array(class_map.cpu() if hasattr(class_map, "cpu") else class_map)

            # Store extra data for Extended mode
            self.last_analysis = {
                'visibility': results.get('visibility'),
                'headpose': results.get('headpose'),
                'landmarks': results.get('landmarks'),
                'face_coordinates': results.get('face_coordinates'),
            }

            # FaceXFormer uses same class IDs as BiSeNet — no mapping needed
            return class_map.astype(np.uint8)

        except Exception as e:
            logger.warning("FaceXFormer get_raw_classes failed: %s. Falling back to BiSeNet.", e)
            from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
            fallback = BiSeNetMaskGenerator()
            return fallback.get_raw_classes(face_image, face_area_on_image)

    def __to_mask(self, face: np.ndarray, affected_areas: List[str]) -> np.ndarray:
        """Convert FaceXFormer class map to binary mask.
        Same logic as BiSeNet since classes are identical."""
        keep_face = "Face" in affected_areas
        keep_neck = "Neck" in affected_areas
        keep_hair = "Hair" in affected_areas
        keep_hat = "Hat" in affected_areas
        exclude_mouth = "MouthExclude" in affected_areas

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if len(index[0]) == 0:
                continue
            if i < 14 and keep_face:
                if exclude_mouth and i == MOUTH_CLASS:
                    continue
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == NECK_CLASS and keep_neck:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == HAIR_CLASS and keep_hair:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == HAT_CLASS and keep_hat:
                mask[index[0], index[1], :] = [255, 255, 255]

        # Dilate mouth exclusion so it survives the blur pass
        if exclude_mouth:
            mouth_region = np.zeros(face.shape[:2], dtype=np.uint8)
            idx = np.where(face == MOUTH_CLASS)
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
