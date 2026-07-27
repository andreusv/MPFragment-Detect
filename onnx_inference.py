import os
import glob
import json
import argparse
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from torchvision.ops import box_iou
import torch

from mp_fragment_engine import MPFragmentEngine

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def compute_confusion_matrix(pred_boxes, pred_labels, gt_boxes, gt_labels, num_classes=2, iou_thresh=0.5):
    """
    Computes confusion matrix between predicted boxes and ground-truth boxes.
    cm[gt_class][pred_class]:
      cm[0][1] = False Positive (Background predicted as Class 1)
      cm[1][1] = True Positive (Class 1 predicted as Class 1)
      cm[1][0] = False Negative (Class 1 predicted as Background)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return cm
    if len(pred_boxes) == 0:
        for gl in gt_labels:
            cm[gl][0] += 1
        return cm
    if len(gt_boxes) == 0:
        for pl in pred_labels:
            cm[0][pl] += 1
        return cm

    p_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32)
    g_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
    iou_matrix = box_iou(p_boxes_t, g_boxes_t).numpy()

    matched_preds, matched_gts = set(), set()
    ious_flat = np.argsort(iou_matrix.flatten())[::-1]

    for idx in ious_flat:
        p_idx = idx // len(gt_boxes)
        g_idx = idx % len(gt_boxes)
        val = iou_matrix[p_idx, g_idx]
        if val < iou_thresh:
            break

        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            cm[gt_labels[g_idx]][pred_labels[p_idx]] += 1

    for p_idx, pl in enumerate(pred_labels):
        if p_idx not in matched_preds:
            cm[0][pl] += 1
    for g_idx, gl in enumerate(gt_labels):
        if g_idx not in matched_gts:
            cm[gl][0] += 1

    return cm

def evaluate_coco_dataset(engine, image_dir, coco_ann_path, args):
    """
    Runs ONNX inference across dataset images and evaluates against COCO ground-truth.
    """
    print(f"\n==================================================")
    print(f"[COCO Evaluation] Loading annotations from {coco_ann_path}...")
    with open(coco_ann_path, "r") as f:
        coco_data = json.load(f)

    img_map = {img["file_name"]: img for img in coco_data["images"]}
    ann_map = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        ann_map.setdefault(img_id, []).append(ann)

    all_image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")))
    if args.max_images is not None and args.max_images > 0:
        all_image_files = all_image_files[:args.max_images]
    print(f"[COCO Evaluation] Evaluating {len(all_image_files)} test images...")


    tp_total, fp_total, fn_total = 0, 0, 0
    per_image_results = []

    for img_path in tqdm(all_image_files, desc="Running ONNX Inference & Evaluation"):
        fname = os.path.basename(img_path)
        if fname not in img_map:
            continue

        coco_img = img_map[fname]
        gt_anns = ann_map.get(coco_img["id"], [])

        # Parse ground-truth boxes [x1, y1, x2, y2]
        gt_boxes, gt_labels = [], []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            # Default mapping: COCO category_id -> label 1
            gt_labels.append(1)

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.empty((0, 4), dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64) if len(gt_labels) > 0 else np.empty((0,), dtype=np.int64)

        # Run ONNX prediction
        pred_dict = engine.predict_image(img_path, pixel_to_um=args.pixel_to_um)
        p_boxes = pred_dict["fused_boxes"]
        p_labels = pred_dict["fused_labels"]

        # Compute confusion matrix
        cm = compute_confusion_matrix(p_boxes, p_labels, gt_boxes, gt_labels, num_classes=2, iou_thresh=0.5)

        fp = int(cm[0][1])
        tp = int(cm[1][1])
        fn = int(cm[1][0])

        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Optionally save visualization for sample images
        if len(per_image_results) < 10:
            out_vis_path = os.path.join(args.output_dir, "visualizations", f"vis_{fname}")
            engine.render_visualization(pred_dict, save_path=out_vis_path)
            out_csv_path = os.path.join(args.output_dir, "reports", f"report_{fname}.csv")
            engine.export_to_csv(pred_dict["fragments"], save_path=out_csv_path)

        per_image_results.append({
            "filename": fname,
            "predictions_count": len(p_boxes),
            "ground_truth_count": len(gt_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    # Summary metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n==================================================")
    print(f"       ONNX INFERENCE EVALUATION SUMMARY           ")
    print(f"==================================================")
    print(f" Evaluated Images    : {len(per_image_results)}")
    print(f" True Positives (TP)  : {tp_total}")
    print(f" False Positives (FP) : {fp_total}")
    print(f" False Negatives (FN) : {fn_total}")
    print(f" Precision            : {precision:.4f} ({precision*100:.2f}%)")
    print(f" Recall               : {recall:.4f} ({recall*100:.2f}%)")
    print(f" F1-Score             : {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"==================================================\n")

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "metrics": {
                "TP": tp_total,
                "FP": fp_total,
                "FN": fn_total,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "parameters": {
                "box_score_thresh": engine.box_score_thresh,
                "wbf_iou_thresh": engine.wbf_iou_thresh,
                "tile_size": engine.tile_size,
                "overlap_pct": engine.overlap_pct
            },
            "per_image": per_image_results
        }, f, indent=2)
    print(f"[COCO Evaluation] Full evaluation metrics saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference & Evaluation Script for MPFragment Software")
    parser.add_argument("--model_path", type=str, default="final_maskrcnn_fragments_model.onnx", help="Path to ONNX model file")
    parser.add_argument("--config", type=str, default="fragment-Config.yaml", help="Path to config YAML")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image file for inference")
    parser.add_argument("--image_dir", type=str, default="test/images", help="Directory of test images")
    parser.add_argument("--coco_ann", type=str, default="test/_annotations.coco.json", help="Path to ground-truth COCO annotations JSON for evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for visualizations and reports")
    parser.add_argument("--tile_size", type=int, default=None, help="Sliding window tile size (e.g. 1024)")
    parser.add_argument("--overlap", type=float, default=None, help="Sliding window overlap fraction (e.g. 0.2)")
    parser.add_argument("--box_score_thresh", type=float, default=None, help="Detection score threshold (e.g. 0.6)")
    parser.add_argument("--wbf_iou_thresh", type=float, default=None, help="WBF IoU threshold for prediction fusion (e.g. 0.4)")
    parser.add_argument("--pixel_to_um", type=float, default=None, help="Spatial resolution scale (micrometers per pixel)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to evaluate (for fast testing)")
    args = parser.parse_args()


    # Load parameters from YAML config if available
    cfg = load_config(args.config)
    hp = cfg.get("hyperparameters", {})
    val_cfg = cfg.get("validation", {})

    tile_size = args.tile_size if args.tile_size is not None else val_cfg.get("tile_size", 1024)
    overlap_pct = args.overlap if args.overlap is not None else val_cfg.get("overlap_pct", 0.2)
    box_score_thresh = args.box_score_thresh if args.box_score_thresh is not None else hp.get("box_score_thresh", 0.6)
    wbf_iou_thresh = args.wbf_iou_thresh if args.wbf_iou_thresh is not None else val_cfg.get("wbf_iou_thresh", 0.4)

    # Initialize MPFragmentEngine
    engine = MPFragmentEngine(
        model_path=args.model_path,
        box_score_thresh=box_score_thresh,
        wbf_iou_thresh=wbf_iou_thresh,
        tile_size=tile_size,
        overlap_pct=overlap_pct
    )

    # Mode 1: Single image inference
    if args.image_path is not None:
        print(f"\n[Single Image] Running ONNX inference on {args.image_path}...")
        pred_dict = engine.predict_image(args.image_path, pixel_to_um=args.pixel_to_um)

        out_vis = os.path.join(args.output_dir, f"vis_{os.path.basename(args.image_path)}")
        out_csv = os.path.join(args.output_dir, f"report_{os.path.splitext(os.path.basename(args.image_path))[0]}.csv")
        out_json = os.path.join(args.output_dir, f"data_{os.path.splitext(os.path.basename(args.image_path))[0]}.json")

        engine.render_visualization(pred_dict, save_path=out_vis)
        engine.export_to_csv(pred_dict["fragments"], save_path=out_csv)
        engine.export_to_json(pred_dict, save_path=out_json)

    # Mode 2: Directory evaluation with COCO ground-truth annotations
    elif os.path.exists(args.coco_ann) and os.path.exists(args.image_dir):
        evaluate_coco_dataset(engine, args.image_dir, args.coco_ann, args)

    # Mode 3: Directory inference without evaluation
    elif os.path.exists(args.image_dir):
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
        print(f"\n[Directory Inference] Processing {len(image_files)} images from {args.image_dir}...")

        for img_p in tqdm(image_files, desc="Processing Images"):
            fname = os.path.basename(img_p)
            pred_dict = engine.predict_image(img_p, pixel_to_um=args.pixel_to_um)

            out_vis = os.path.join(args.output_dir, "visualizations", f"vis_{fname}")
            out_csv = os.path.join(args.output_dir, "reports", f"report_{os.path.splitext(fname)[0]}.csv")
            engine.render_visualization(pred_dict, save_path=out_vis)
            engine.export_to_csv(pred_dict["fragments"], save_path=out_csv)

if __name__ == "__main__":
    main()
