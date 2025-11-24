import os
import cv2
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from ultralytics import SAM
from transformers import AutoImageProcessor, AutoModel
import supervision as sv

# ========================== LOAD MODELS ==========================
print("Đang load MobileSAM + DINOv2-small + float16...")
sam = SAM("mobile_sam.pt")  

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()

# ========================== EMBEDDING BATCH ==========================
@torch.inference_mode()
def batch_embed(crops):
    if len(crops) == 0:
        return torch.empty((0, 384), device="cuda")
    inputs = processor(images=crops, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = dinov2(**inputs)
    patches = out.last_hidden_state[:, 1:]
    topk = patches.topk(16, dim=1).values.mean(dim=1)
    return torch.nn.functional.normalize(topk, dim=1)

# ========================== REFERENCE TEMPLATE ==========================
def get_template(ref_paths, sam_model):
    imgs = []
    for p in ref_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop = img_rgb

        results = sam_model(img_rgb, imgsz=768, device="cuda", verbose=False)[0]

        if results.masks is not None and len(results.masks) > 0:
            masks = results.masks.data.cpu().numpy()

            scores = None
            if results.boxes is not None and results.boxes.conf is not None:
                scores = results.boxes.conf.detach().cpu().numpy()
            elif hasattr(results.masks, "scores") and results.masks.scores is not None:
                scores = results.masks.scores.detach().cpu().numpy()

            if scores is not None and len(scores) == len(masks):
                best_idx = int(np.argmax(scores))
            else:
                areas = masks.sum(axis=(1, 2))
                best_idx = int(np.argmax(areas))

            best_mask = masks[best_idx]
            pos = np.where(best_mask)

            if pos[0].size > 100:
                y1, y2 = pos[0].min(), pos[0].max()
                x1, x2 = pos[1].min(), pos[1].max()

                h, w = img_rgb.shape[:2]
                pad_y = max(1, int((y2 - y1 + 1) * 0.05))
                pad_x = max(1, int((x2 - x1 + 1) * 0.05))
                y1 = max(0, y1 - pad_y)
                y2 = min(h - 1, y2 + pad_y)
                x1 = max(0, x1 - pad_x)
                x2 = min(w - 1, x2 + pad_x)

                crop = img_rgb[y1:y2+1, x1:x2+1]
        # Fallback: crop 20% trung tâm khi không có mask dùng được
        if crop.size == 0:
            crop = img_rgb
        if crop is img_rgb:
            h, w = img_rgb.shape[:2]
            margin_h, margin_w = max(1, int(h * 0.2)), max(1, int(w * 0.2))
            crop = img_rgb[margin_h:h-margin_h, margin_w:w-margin_w]
            if crop.size == 0:
                crop = img_rgb

        crop_resized = cv2.resize(crop, (224, 224))
        imgs.append(crop_resized)
    
    if len(imgs) == 0:
        raise ValueError("Không tạo được crop nào từ ảnh tham chiếu.")

    embs = batch_embed(imgs)
    template = embs.mean(dim=0)
    return torch.nn.functional.normalize(template, dim=0)

# ========================== MAIN PROCESS ==========================
@torch.inference_mode()
def process_video(folder: Path):
    video_path = folder / "drone_video.mp4"
    refs = sorted((folder / "object_images").glob("*.jpg"))
    video_id = folder.name

    template = get_template(refs, sam)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo tracker riêng cho mỗi video
    tracker = sv.ByteTrack(
        track_activation_threshold=0.8,
        lost_track_buffer=100,
        minimum_matching_threshold=0.6,
        frame_rate=25
    )

    detections = []
    last_bbox = None
    target_track_id = None  # Track ID của object cần tìm
    skip_counter = 0

    pbar = tqdm(total=total_frames, desc=video_id, leave=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(pbar.n)

        # SKIP FRAME: chỉ detect mỗi 3 frame
        if skip_counter % 2 == 0 or last_bbox is None:
            results = sam(frame, imgsz=768, device="cuda", verbose=False)[0]
            
            if results.masks is None:
                # Không có mask → update tracker với empty
                tracks = tracker.update_with_detections(sv.Detections.empty())
                if len(tracks) > 0 and target_track_id is not None:
                    for i, tid in enumerate(tracks.tracker_id):
                        if tid == target_track_id:
                            bx = tracks.xyxy[i].astype(int)
                            last_bbox = {"x1": int(bx[0]), "y1": int(bx[1]), 
                                        "x2": int(bx[2]), "y2": int(bx[3])}
                            break
                if last_bbox is not None:
                    detections.append({"frame": frame_idx, **last_bbox})
                pbar.update(1)
                skip_counter += 1
                continue

            masks = results.masks.data.cpu().numpy()

            # Lọc masks theo pixel
            min_pixels = 70
            max_pixels = 4000
            valid_masks = []
            valid_areas = []

            for mask in masks:
                pos = np.where(mask)
                num_pixels = len(pos[0])
                if min_pixels <= num_pixels <= max_pixels:
                    valid_masks.append(mask)
                    valid_areas.append(num_pixels)

            # Lấy top 10 lớn nhất
            if len(valid_masks) > 24:
                topk_idx = np.argsort(valid_areas)[-24:]
                masks = [valid_masks[i] for i in topk_idx]
                areas = [valid_areas[i] for i in topk_idx]
            else:
                masks = valid_masks
                areas = valid_areas

            # Lọc bbox quá lớn (theo kích thước) - trước khi crop và embed
            img_h, img_w = frame.shape[:2]
            max_bbox_width = img_w * 0.2   # Tối đa 10% chiều rộng ảnh
            max_bbox_height = img_h * 0.2  # Tối đa 10% chiều cao ảnh
            max_bbox_area = (img_w * img_h) * 0.05  # Tối đa 10% diện tích ảnh
            
            size_filtered_masks = []
            for mask in masks:
                pos = np.where(mask)
                if pos[0].size == 0:
                    continue
                y1, y2 = pos[0].min(), pos[0].max()
                x1, x2 = pos[1].min(), pos[1].max()
                
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_area = bbox_w * bbox_h
                
                # Loại bỏ bbox quá lớn
                if bbox_w > max_bbox_width or bbox_h > max_bbox_height or bbox_area > max_bbox_area:
                    continue
                
                size_filtered_masks.append(mask)
            
            masks = size_filtered_masks

            # Crop và lấy bbox
            crops_rgb = []
            bboxes = []
            for mask in masks:
                pos = np.where(mask)
                if pos[0].size == 0:
                    continue
                y1, y2 = pos[0].min(), pos[0].max()
                x1, x2 = pos[1].min(), pos[1].max()
                
                crop = frame[y1:y2+1, x1:x2+1]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops_rgb.append(cv2.resize(crop_rgb, (224, 224)))
                bboxes.append([x1, y1, x2, y2])

            # So khớp với template
            best_sim = 0.0
            if crops_rgb:
                embs = batch_embed(crops_rgb)
                sims = torch.nn.functional.cosine_similarity(embs, template, dim=1)
                best_idx = sims.argmax().item()
                best_sim = sims[best_idx].item()
                
                if best_sim > 0.8:
                    x1, y1, x2, y2 = bboxes[best_idx]
                    # Bbox đã được lọc kích thước ở trên, không cần kiểm tra lại
                    detections_sv = sv.Detections(
                        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        confidence=np.array([best_sim])
                    )
                else:
                    detections_sv = sv.Detections.empty()
            else:
                detections_sv = sv.Detections.empty()

            # Update tracker
            tracks = tracker.update_with_detections(detections_sv)

            if len(tracks) > 0:
                # Lần đầu detect → lưu track_id
                if target_track_id is None and best_sim > 0.8:
                    target_track_id = tracks.tracker_id[0]
                    idx = 0
                else:
                    # Tìm track đúng target_track_id
                    idx = None
                    for i, tid in enumerate(tracks.tracker_id):
                        if tid == target_track_id:
                            idx = i
                            break
                    
                    # Không tìm thấy target track
                    if idx is None:
                        if best_sim > 0.8:
                            # Có detection mới tốt → chuyển sang track mới
                            target_track_id = tracks.tracker_id[0]
                            idx = 0
                
                if idx is not None:
                    bx = tracks.xyxy[idx].astype(int)
                    last_bbox = {"x1": int(bx[0]), "y1": int(bx[1]), 
                                "x2": int(bx[2]), "y2": int(bx[3])}
            
            # Lưu detection
            if last_bbox is not None:
                detections.append({"frame": frame_idx, **last_bbox})

        else:
            # Skip frame → dùng bbox cũ
            if last_bbox is not None:
                detections.append({"frame": frame_idx, **last_bbox})

        skip_counter += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # INTERPOLATION
    if len(detections) > 0:
        full_dets = []
        prev_f = -999
        prev_b = None
        for det in sorted(detections, key=lambda x: x["frame"]):
            if det["frame"] - prev_f > 1 and prev_b is not None:
                for f in range(prev_f + 1, det["frame"]):
                    alpha = (f - prev_f) / (det["frame"] - prev_f)
                    bbox = [
                        int(prev_b["x1"] * (1-alpha) + det["x1"] * alpha),
                        int(prev_b["y1"] * (1-alpha) + det["y1"] * alpha),
                        int(prev_b["x2"] * (1-alpha) + det["x2"] * alpha),
                        int(prev_b["y2"] * (1-alpha) + det["y2"] * alpha),
                    ]
                    full_dets.append({"frame": f, "x1": bbox[0], "y1": bbox[1], 
                                     "x2": bbox[2], "y2": bbox[3]})
            full_dets.append(det)
            prev_f = det["frame"]
            prev_b = {"x1": det["x1"], "y1": det["y1"], 
                     "x2": det["x2"], "y2": det["y2"]}
        detections = full_dets

    # Format output
    if len(detections) > 0:
        final_output_detections = [{"bboxes": detections}]
    else:
        final_output_detections = []

    return {"video_id": video_id, "detections": final_output_detections}

# ========================== MAIN ==========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission.json")
    args = parser.parse_args()

    results = []
    for folder in sorted(Path(args.data_dir).iterdir()):
        if (folder / "drone_video.mp4").exists():
            print(f"Processing {folder.name} ...")
            res = process_video(folder)
            results.append(res)

    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nHoàn tất! Output: {args.output}")