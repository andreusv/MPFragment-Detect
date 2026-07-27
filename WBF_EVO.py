import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.ops import box_iou
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from src.utils.coco_instance import CocoInstanceDataset

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_val_transforms():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_slice_coordinates(w, h, tile_size=1024, overlap_pct=0.2):
    stride = int(tile_size * (1.0 - overlap_pct))
    boxes = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = min(x, w - tile_size) if x + tile_size > w else x
            y1 = min(y, h - tile_size) if y + tile_size > h else y
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            boxes.append((x1, y1, x2, y2))
            if x2 == w: break
        if y2 == h: break
    return list(set(boxes))

def apply_wbf(boxes, scores, labels, w, h, iou_threshold):
    from ensemble_boxes import weighted_boxes_fusion

    if len(boxes) == 0: return boxes, scores, labels

    boxes_norm = boxes.clone().float()
    boxes_norm[:, [0, 2]] /= w
    boxes_norm[:, [1, 3]] /= h
    boxes_norm = boxes_norm.clamp(0.0, 1.0).tolist()

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        [boxes_norm], [scores.tolist()], [labels.tolist()], 
        weights=None, iou_thr=iou_threshold, skip_box_thr=0.0
    )

    fused_boxes = torch.tensor(f_boxes, dtype=torch.float32)
    fused_boxes[:, [0, 2]] *= w
    fused_boxes[:, [1, 3]] *= h
    fused_scores = torch.tensor(f_scores, dtype=torch.float32)
    fused_labels = torch.tensor(f_labels, dtype=torch.int64)

    return fused_boxes, fused_scores, fused_labels

def compute_confusion_matrix(pred_boxes, pred_labels, gt_boxes, gt_labels, num_classes, iou_thresh=0.5):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    if len(pred_boxes) == 0 and len(gt_boxes) == 0: return cm
    if len(pred_boxes) == 0:
        for gl in gt_labels: cm[gl.item()][0] += 1
        return cm
    if len(gt_boxes) == 0:
        for pl in pred_labels: cm[0][pl.item()] += 1
        return cm

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    matched_preds, matched_gts = set(), set()
    
    ious_flat = iou_matrix.flatten().argsort(descending=True)
    for idx in ious_flat:
        p_idx = (idx // len(gt_boxes)).item()
        g_idx = (idx % len(gt_boxes)).item()
        val = iou_matrix[p_idx, g_idx].item()
        if val < iou_thresh: break

        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            cm[gt_labels[g_idx].item()][pred_labels[p_idx].item()] += 1

    for p_idx, pl in enumerate(pred_labels):
        if p_idx not in matched_preds: cm[0][pl.item()] += 1
    for g_idx, gl in enumerate(gt_labels):
        if g_idx not in matched_gts: cm[gl.item()][0] += 1
    return cm

@torch.inference_mode()
def study_wbf_thresholds(model, loader, device, cfg, tile_size, overlap_pct):
    model.eval()
    num_classes = cfg["data"]["num_classes"]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {thr: {"TP": 0, "FP": 0, "FN": 0} for thr in thresholds}

    for images, targets in tqdm(loader, desc="Sweeping WBF Thresholds"):
        full_img = images[0].to(device)
        target = targets[0]
        _, h_orig, w_orig = full_img.shape
        
        all_boxes, all_scores, all_labels = [], [], []

        slice_coords = get_slice_coordinates(w_orig, h_orig, tile_size, overlap_pct)
        for (x1, y1, x2, y2) in slice_coords:
            patch = full_img[:, y1:y2, x1:x2]
            patch_pred = model([patch])[0]

            if len(patch_pred["boxes"]) == 0: continue

            p_boxes = patch_pred["boxes"].cpu()
            p_boxes[:, [0, 2]] += x1
            p_boxes[:, [1, 3]] += y1
            
            all_boxes.append(p_boxes)
            all_scores.append(patch_pred["scores"].cpu())
            all_labels.append(patch_pred["labels"].cpu())

        if len(all_boxes) > 0:
            c_boxes  = torch.cat(all_boxes, dim=0)
            c_scores = torch.cat(all_scores, dim=0)
            c_labels = torch.cat(all_labels, dim=0)
        else:
            c_boxes, c_scores, c_labels = torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64)

        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        for thr in thresholds:
            f_b, f_s, f_l = apply_wbf(c_boxes, c_scores, c_labels, w_orig, h_orig, iou_threshold=thr)
            
            cm = compute_confusion_matrix(f_b, f_l, gt_boxes, gt_labels, num_classes, iou_thresh=0.5)
            
            # cm[1][0] = Class 1 predicted as Background (False Negative)
            results[thr]["FP"] += cm[0][1]
            results[thr]["TP"] += cm[1][1]
            results[thr]["FN"] += cm[1][0]

    # --- 3. GENERATE THE PLOT ---
    plot_evolution(thresholds, results)

def plot_evolution(thresholds, results):
    fps = [results[thr]["FP"] for thr in thresholds]
    tps = [results[thr]["TP"] for thr in thresholds]
    fns = [results[thr]["FN"] for thr in thresholds]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('WBF IoU Threshold')
    ax1.set_ylabel('False Positives (FPs)', color=color)
    line1, = ax1.plot(thresholds, fps, marker='o', color=color, linewidth=2, label='FPs')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('True Positives (TPs)', color=color)
    line2, = ax2.plot(thresholds, tps, marker='s', color=color, linewidth=2, label='TPs')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('WBF Performance by IoU thresh')
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    os.makedirs("visualizations", exist_ok=True)
    save_path = "visualizations/wbf_threshold_study.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Study complete! Plot saved to: {save_path}")
    
    print("\n--- RAW DATA ---")
    print("Threshold | True Positives | False Positives | False Negatives")
    print("-" * 62)
    for thr in thresholds:
        print(f"   {thr:.1f}    |      {results[thr]['TP']:<9} |      {results[thr]['FP']:<9} |      {results[thr]['FN']:<9}")

def main(args):
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VAL_DATASET = CocoInstanceDataset(
        "/home/robotin/AllMPs/data/trainLongFormat/fragments/test/images", 
        "/home/robotin/AllMPs/data/trainLongFormat/fragments/test/_annotations.coco.json", 
        transforms=get_val_transforms()
    )
    val_loader = DataLoader(VAL_DATASET, batch_size=1, shuffle=False, num_workers=min(2, cfg["data"]["num_workers"]), collate_fn=collate_fn)

    hp, ma = cfg["hyperparameters"], cfg["model_architecture"]
    ANCHOR_SIZES = tuple(tuple(x) for x in ma["anchor_sizes"])
    ASPECT_RATIOS = tuple(tuple(x) for x in (ma["aspect_ratios"] * len(ANCHOR_SIZES)))
    anchor_generator = AnchorGenerator(sizes=ANCHOR_SIZES, aspect_ratios=ASPECT_RATIOS)

    model = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=cfg["data"]["num_classes"],
        box_score_thresh=0.6, box_nms_thresh=hp["box_nms_thresh"], 
        box_detections_per_img=hp["box_detections_per_img"], rpn_nms_thresh=hp["rpn_nms_thresh"], 
        rpn_score_thresh=hp["rpn_score_thresh"], rpn_batch_size_per_image=hp["rpn_batch_size_per_image"], 
        rpn_positive_fraction=hp["rpn_positive_fraction"], box_batch_size_per_image=hp["box_batch_size_per_image"], 
        box_positive_fraction=hp["box_positive_fraction"]
    )
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    study_wbf_thresholds(model, val_loader, device, cfg, args.tile_size, args.overlap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
