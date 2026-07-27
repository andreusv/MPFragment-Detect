import os
import json
import csv
import math
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from skimage.measure import regionprops, label as label_np
from ensemble_boxes import weighted_boxes_fusion

class ScaleBarDetector:
    """
    Utility to detect scale bars in microscope/scanner image overlays
    and convert pixels to micrometers (um).
    """
    @staticmethod
    def detect_scalebar(image_bgr):
        """
        Attempts to detect scale bar line in top-right or bottom-right corner.
        Returns: (pixel_width, detected_flag)
        """
        h, w = image_bgr.shape[:2]

        regions = [
            ("top_right", image_bgr[:int(h*0.25), int(w*0.75):]),
            ("bottom_right", image_bgr[int(h*0.75):, int(w*0.75):])
        ]

        for name, crop in regions:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if cw > 80 and 5 < ch < 60:
                    return float(cw), True

        return None, False


class ColorClassifier:
    """
    Extracts dominant color from particle mask and maps to standard color names.
    Adapted and enhanced from process_images.py.
    """
    SIMPLE_COLORS = {
        "Red": (220, 38, 38),
        "Orange": (234, 88, 12),
        "Yellow": (234, 179, 8),
        "Green": (22, 163, 74),
        "Cyan/Blue": (14, 165, 233),
        "Blue": (37, 99, 235),
        "Purple": (147, 51, 234),
        "Pink": (236, 72, 153),
        "Black/Dark": (30, 41, 59),
        "White/Light": (241, 245, 249),
        "Grey": (100, 116, 139),
        "Brown": (120, 53, 15)
    }

    @classmethod
    def get_color_name(cls, requested_rgb):
        min_dist = float('inf')
        closest_name = "Unknown"
        req = np.array(requested_rgb, dtype=np.float32)

        for name, rgb in cls.SIMPLE_COLORS.items():
            color_arr = np.array(rgb, dtype=np.float32)
            dist = np.linalg.norm(req - color_arr)
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name

    @classmethod
    def extract_particle_color(cls, image_bgr, mask):
        """
        Extracts dominant color name, RGB tuple, and HEX code for a particle mask.
        """
        if mask is None or np.sum(mask) == 0:
            return "Unknown", [128, 128, 128], "#808080"

        # Apply mask to image BGR
        masked_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask.astype(np.uint8))
        pixels_bgr = masked_bgr[mask > 0]

        if len(pixels_bgr) == 0:
            return "Unknown", [128, 128, 128], "#808080"

        # Convert to HLS color space to filter out black/shadow artifacts
        pixels_hls = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HLS).reshape(-1, 3)

        # Filter out extreme dark background shadows (L < 15 or L > 245)
        valid_mask = (pixels_hls[:, 1] > 15) & (pixels_hls[:, 1] < 245)
        if np.sum(valid_mask) > 5:
            pixels_bgr = pixels_bgr[valid_mask]

        if len(pixels_bgr) == 0:
            return "Unknown", [128, 128, 128], "#808080"

        # Compute median BGR color
        median_bgr = np.median(pixels_bgr, axis=0).astype(int)
        median_rgb = [int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])]

        color_name = cls.get_color_name(median_rgb)
        hex_code = f"#{median_rgb[0]:02x}{median_rgb[1]:02x}{median_rgb[2]:02x}"

        return color_name, median_rgb, hex_code


class MPFragmentEngine:
    """
    High-Performance ONNX Sliced Inference, Color Characterization & Morphological Engine
    for Microplastics Fragment Detection & Quantification Software.
    """

    def __init__(
        self,
        model_path,
        box_score_thresh=0.6,
        wbf_iou_thresh=0.4,
        tile_size=1024,
        overlap_pct=0.2,
        mask_threshold=0.5,
        providers=None
    ):
        self.model_path = model_path
        self.box_score_thresh = box_score_thresh
        self.wbf_iou_thresh = wbf_iou_thresh
        self.tile_size = tile_size
        self.overlap_pct = overlap_pct
        self.mask_threshold = mask_threshold

        if providers is None:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        print(f"[MPFragmentEngine] Loading ONNX model from {model_path}...")
        print(f"[MPFragmentEngine] Using Execution Providers: {providers}")
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"[MPFragmentEngine] Provider warning ({e}), fallback to CPUExecutionProvider.")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def preprocess_patch(self, patch_bgr):
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        img_float = rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_float, (2, 0, 1))
        normalized = (chw - self.mean) / self.std
        return normalized.astype(np.float32)

    def get_slice_coordinates(self, w, h):
        stride = int(self.tile_size * (1.0 - self.overlap_pct))
        stride = max(1, stride)
        boxes = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = min(x, w - self.tile_size) if x + self.tile_size > w else x
                y1 = min(y, h - self.tile_size) if y + self.tile_size > h else y
                x2 = min(x1 + self.tile_size, w)
                y2 = min(y1 + self.tile_size, h)
                boxes.append((x1, y1, x2, y2))
                if x2 == w:
                    break
            if y2 == h:
                break

        return list(set(boxes))

    def predict_image(self, image_input, pixel_to_um=None):
        """
        Runs sliced inference on a high-resolution image, merges tile predictions,
        and extracts morphological + color measurements for all microplastic fragments.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Failed to read image at {image_input}")
            img_name = os.path.basename(image_input)
        else:
            image = image_input
            img_name = "image_input"

        h_orig, w_orig = image.shape[:2]

        scalebar_detected = False
        if pixel_to_um is None:
            detected_px, scalebar_detected = ScaleBarDetector.detect_scalebar(image)
            if scalebar_detected and detected_px > 0:
                pixel_to_um = 500.0 / detected_px

        slice_coords = self.get_slice_coordinates(w_orig, h_orig)

        tile_boxes, tile_scores, tile_labels, tile_masks = [], [], [], []

        for (x1, y1, x2, y2) in slice_coords:
            patch = image[y1:y2, x1:x2]
            patch_h, patch_w = patch.shape[:2]

            if patch_h < self.tile_size or patch_w < self.tile_size:
                padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded[:patch_h, :patch_w] = patch
                patch_tensor = self.preprocess_patch(padded)
            else:
                patch_tensor = self.preprocess_patch(patch)

            outputs = self.session.run(None, {self.input_name: patch_tensor})
            out_dict = {name: val for name, val in zip(self.output_names, outputs)}

            b = out_dict["boxes"]
            s = out_dict["scores"]
            l = out_dict["labels"]
            m = out_dict.get("masks", None)

            if len(b) == 0:
                continue

            keep = s >= self.box_score_thresh
            if not np.any(keep):
                continue

            b = b[keep]
            s = s[keep]
            l = l[keep]
            if m is not None:
                m = m[keep]

            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1

            tile_boxes.append(b)
            tile_scores.append(s)
            tile_labels.append(l)

            if m is not None:
                for idx in range(len(b)):
                    full_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
                    mask_crop = m[idx, 0, :patch_h, :patch_w]
                    full_mask[y1:y1+patch_h, x1:x1+patch_w] = mask_crop
                    tile_masks.append(full_mask)

        if len(tile_boxes) > 0:
            c_boxes = np.vstack(tile_boxes)
            c_scores = np.concatenate(tile_scores)
            c_labels = np.concatenate(tile_labels)

            boxes_norm = c_boxes.astype(np.float32).copy()
            boxes_norm[:, [0, 2]] /= w_orig
            boxes_norm[:, [1, 3]] /= h_orig
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                [boxes_norm.tolist()],
                [c_scores.tolist()],
                [c_labels.tolist()],
                weights=None,
                iou_thr=self.wbf_iou_thresh,
                skip_box_thr=self.box_score_thresh
            )

            f_boxes = np.array(f_boxes, dtype=np.float32)
            if len(f_boxes) > 0:
                f_boxes[:, [0, 2]] *= w_orig
                f_boxes[:, [1, 3]] *= h_orig
                f_scores = np.array(f_scores, dtype=np.float32)
                f_labels = np.array(f_labels, dtype=np.int64)
            else:
                f_boxes = np.empty((0, 4), dtype=np.float32)
                f_scores = np.empty((0,), dtype=np.float32)
                f_labels = np.empty((0,), dtype=np.int64)
        else:
            f_boxes = np.empty((0, 4), dtype=np.float32)
            f_scores = np.empty((0,), dtype=np.float32)
            f_labels = np.empty((0,), dtype=np.int64)

        fused_masks = []
        for idx in range(len(f_boxes)):
            box = f_boxes[idx]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            mask = np.zeros((h_orig, w_orig), dtype=bool)
            if len(tile_masks) > 0 and x2 > x1 and y2 > y1:
                box_mask_prob = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
                count = 0
                for tm in tile_masks:
                    crop = tm[y1:y2, x1:x2]
                    if np.max(crop) > self.mask_threshold:
                        box_mask_prob += crop
                        count += 1
                if count > 0:
                    box_mask_prob /= count
                    mask[y1:y2, x1:x2] = box_mask_prob >= self.mask_threshold
                else:
                    mask[y1:y2, x1:x2] = True
            elif x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

            fused_masks.append(mask)

        if len(fused_masks) > 0:
            fused_masks = np.stack(fused_masks, axis=0)
        else:
            fused_masks = np.empty((0, h_orig, w_orig), dtype=bool)

        # Analyze morphological fragment measurements + Color Characterization
        fragments = self.analyze_fragments(image, f_boxes, f_scores, f_labels, fused_masks, pixel_to_um, img_name)

        return {
            "image_name": img_name,
            "image": image,
            "fragments": fragments,
            "fused_boxes": f_boxes,
            "fused_scores": f_scores,
            "fused_labels": f_labels,
            "fused_masks": fused_masks,
            "pixel_to_um": pixel_to_um,
            "scalebar_detected": scalebar_detected
        }

    def predict_batch(self, image_inputs, pixel_to_um=None):
        """
        Runs batch prediction on a list of image paths or image arrays.
        Returns a list of dict results per image.
        """
        batch_results = []
        for img_input in image_inputs:
            res = self.predict_image(img_input, pixel_to_um=pixel_to_um)
            batch_results.append(res)
        return batch_results

    def analyze_fragments(self, image_bgr, boxes, scores, labels, masks, pixel_to_um=None, image_name="image"):
        fragments = []

        for idx in range(len(boxes)):
            box = boxes[idx].tolist()
            score = float(scores[idx])
            label_id = int(labels[idx])
            mask = masks[idx]

            # Extract color classification using ColorClassifier
            color_name, rgb_tuple, hex_code = ColorClassifier.extract_particle_color(image_bgr, mask)

            labeled_mask = label_np(mask)
            props = regionprops(labeled_mask)

            if len(props) > 0:
                prop = props[0]
                area_px = float(prop.area)
                perimeter_px = float(prop.perimeter)
                eq_diameter_px = float(prop.equivalent_diameter_area)
                major_axis_px = float(prop.major_axis_length)
                minor_axis_px = float(prop.minor_axis_length)
                solidity = float(prop.solidity)
            else:
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                area_px = w_box * h_box
                perimeter_px = 2 * (w_box + h_box)
                eq_diameter_px = 2 * math.sqrt(area_px / math.pi)
                major_axis_px = max(w_box, h_box)
                minor_axis_px = min(w_box, h_box)
                solidity = 1.0

            aspect_ratio = (major_axis_px / minor_axis_px) if minor_axis_px > 0 else 1.0
            circularity = (4.0 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
            circularity = min(1.0, circularity)

            frag_dict = {
                "image_name": image_name,
                "id": idx + 1,
                "label_id": label_id,
                "label_name": "Microplastics",
                "score": round(score, 4),
                "color_name": color_name,
                "rgb": rgb_tuple,
                "hex_code": hex_code,
                "bbox": [round(c, 2) for c in box],
                "area_px": round(area_px, 2),
                "perimeter_px": round(perimeter_px, 2),
                "eq_diameter_px": round(eq_diameter_px, 2),
                "major_axis_px": round(major_axis_px, 2),
                "minor_axis_px": round(minor_axis_px, 2),
                "aspect_ratio": round(aspect_ratio, 2),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3)
            }

            if pixel_to_um is not None and pixel_to_um > 0:
                um_per_px = float(pixel_to_um)
                frag_dict["area_um2"] = round(area_px * (um_per_px ** 2), 2)
                frag_dict["perimeter_um"] = round(perimeter_px * um_per_px, 2)
                frag_dict["eq_diameter_um"] = round(eq_diameter_px * um_per_px, 2)
                frag_dict["major_axis_um"] = round(major_axis_px * um_per_px, 2)
                frag_dict["minor_axis_um"] = round(minor_axis_px * um_per_px, 2)

            fragments.append(frag_dict)

        return fragments

    def render_visualization(self, pred_dict, save_path=None):
        image = pred_dict["image"].copy()
        h, w = image.shape[:2]
        fragments = pred_dict["fragments"]
        masks = pred_dict["fused_masks"]
        pixel_to_um = pred_dict["pixel_to_um"]

        mask_color = np.array([255, 165, 0], dtype=np.uint8)
        bbox_color = (0, 255, 127)

        overlay = image.copy()

        for idx, frag in enumerate(fragments):
            if idx < len(masks):
                mask = masks[idx]
                overlay[mask] = (overlay[mask] * 0.4 + mask_color * 0.6).astype(np.uint8)

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        for frag in fragments:
            box = [int(c) for c in frag["bbox"]]
            x1, y1, x2, y2 = box

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

            color_str = frag.get("color_name", "")
            if "area_um2" in frag:
                tag = f"#{frag['id']} MP [{color_str}] ({frag['score']:.2f}) | {frag['area_um2']:.0f} um²"
            else:
                tag = f"#{frag['id']} MP [{color_str}] ({frag['score']:.2f}) | {frag['area_px']:.0f} px²"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), (0, 0, 0), -1)
            cv2.putText(image, tag, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Summary Header Panel
        cv2.rectangle(image, (20, 20), (460, 110), (0, 0, 0), -1)
        cv2.rectangle(image, (20, 20), (460, 110), (0, 255, 127), 2)

        total_frags = len(fragments)
        if total_frags > 0 and "area_um2" in fragments[0]:
            tot_area = sum(f["area_um2"] for f in fragments)
            unit_str = "um²"
        else:
            tot_area = sum(f["area_px"] for f in fragments)
            unit_str = "px²"

        cv2.putText(image, "MPFragment Studio - Color & Particle Analysis", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Detected Fragments: {total_frags}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Total Fragment Area: {tot_area:,.1f} {unit_str}", (30, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if pixel_to_um is not None and pixel_to_um > 0:
            scale_um = 500.0
            bar_px = int(scale_um / pixel_to_um)
            sb_x2 = w - 40
            sb_x1 = max(40, sb_x2 - bar_px)
            sb_y = h - 40

            cv2.rectangle(image, (sb_x1 - 10, sb_y - 30), (sb_x2 + 10, sb_y + 15), (0, 0, 0), -1)
            cv2.line(image, (sb_x1, sb_y), (sb_x2, sb_y), (255, 255, 255), 4)
            cv2.putText(image, f"{int(scale_um)} um", (sb_x1 + (bar_px // 4), sb_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

        return image

    @staticmethod
    def export_to_csv(fragments, save_path):
        if len(fragments) == 0:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        keys = list(fragments[0].keys())

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for frag in fragments:
                writer.writerow(frag)

    @staticmethod
    def export_to_json(pred_dict, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        export_data = {
            "summary": {
                "image_name": pred_dict.get("image_name", ""),
                "total_fragments": len(pred_dict["fragments"]),
                "pixel_to_um": pred_dict["pixel_to_um"],
                "scalebar_detected": pred_dict["scalebar_detected"]
            },
            "fragments": pred_dict["fragments"]
        }
        with open(save_path, "w") as f:
            json.dump(export_data, f, indent=2)
