import os
import sys
import time
import base64
import numpy as np
import cv2
import uvicorn
import webview
import requests
from typing import List
from threading import Thread
from collections import Counter
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from mp_fragment_engine import MPFragmentEngine

app = FastAPI(title="MPFragment Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "final_maskrcnn_fragments_model.onnx")
engine = None

def get_engine(box_score_thresh=0.6, wbf_iou_thresh=0.4, tile_size=1024):
    global engine
    if engine is None or engine.box_score_thresh != box_score_thresh or engine.wbf_iou_thresh != wbf_iou_thresh or engine.tile_size != tile_size:
        engine = MPFragmentEngine(
            model_path=MODEL_PATH,
            box_score_thresh=box_score_thresh,
            wbf_iou_thresh=wbf_iou_thresh,
            tile_size=tile_size
        )
    return engine

@app.get("/api/health")
def api_health():
    return {"status": "ok", "model_path": MODEL_PATH, "model_exists": os.path.exists(MODEL_PATH)}

@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(None),
    file_path: str = Form(None),
    box_score_thresh: float = Form(0.6),
    wbf_iou_thresh: float = Form(0.4),
    tile_size: int = Form(1024),
    pixel_to_um: float = Form(None)
):
    try:
        current_engine = get_engine(
            box_score_thresh=box_score_thresh,
            wbf_iou_thresh=wbf_iou_thresh,
            tile_size=tile_size
        )

        if file_path and os.path.exists(file_path):
            img_input = file_path
        elif file is not None:
            contents = await file.read()
            if len(contents) == 0:
                return JSONResponse({"error": "Uploaded image file is empty"}, status_code=400)
            nparr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if nparr is None:
                return JSONResponse({"error": "Could not decode image file"}, status_code=400)
            img_input = nparr
        else:
            return JSONResponse({"error": "No image file provided"}, status_code=400)

        pred_dict = current_engine.predict_image(img_input, pixel_to_um=pixel_to_um)
        if file and file.filename:
            pred_dict["image_name"] = file.filename

        vis_image = current_engine.render_visualization(pred_dict)
        _, buffer = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "image_name": pred_dict.get("image_name", "image.jpg"),
            "fragments": pred_dict["fragments"],
            "pixel_to_um": pred_dict["pixel_to_um"],
            "scalebar_detected": pred_dict["scalebar_detected"],
            "visualization_base64": vis_b64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/predict_batch")
async def api_predict_batch(
    files: List[UploadFile] = File(...),
    box_score_thresh: float = Form(0.6),
    wbf_iou_thresh: float = Form(0.4),
    tile_size: int = Form(1024),
    pixel_to_um: float = Form(None)
):
    try:
        current_engine = get_engine(
            box_score_thresh=box_score_thresh,
            wbf_iou_thresh=wbf_iou_thresh,
            tile_size=tile_size
        )

        batch_results = []
        all_fragments = []
        color_counts = Counter()

        for file in files:
            contents = await file.read()
            if len(contents) == 0:
                continue
            nparr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if nparr is None:
                continue

            pred_dict = current_engine.predict_image(nparr, pixel_to_um=pixel_to_um)
            pred_dict["image_name"] = file.filename

            vis_image = current_engine.render_visualization(pred_dict)
            _, buffer = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            vis_b64 = base64.b64encode(buffer).decode("utf-8")

            # Collect color statistics
            for f in pred_dict["fragments"]:
                color_counts[f.get("color_name", "Unknown")] += 1
                all_fragments.append(f)

            batch_results.append({
                "image_name": file.filename,
                "fragments": pred_dict["fragments"],
                "pixel_to_um": pred_dict["pixel_to_um"],
                "scalebar_detected": pred_dict["scalebar_detected"],
                "visualization_base64": vis_b64
            })

        total_area = sum(f.get("area_um2", f.get("area_px", 0)) for f in all_fragments)

        return {
            "batch_summary": {
                "total_images": len(batch_results),
                "total_fragments": len(all_fragments),
                "total_area": round(total_area, 2),
                "color_distribution": dict(color_counts)
            },
            "results": batch_results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static_root")

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def wait_for_server(url="http://127.0.0.1:8000/api/health", timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.1)
    return False

class PyWebViewApi:
    def select_file(self):
        window = webview.windows[0]
        result = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False)
        if result and len(result) > 0:
            return result[0]
        return None

def main():
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    print("[MPFragment Studio] Waiting for backend server startup...")
    if not wait_for_server():
        print("[MPFragment Studio] Error: Backend server failed to start.")
        sys.exit(1)
    print("[MPFragment Studio] Backend server ready on http://127.0.0.1:8000")

    api = PyWebViewApi()

    webview.create_window(
        title="MPFragment Studio - Microplastics Detection & Color Analysis",
        url="http://127.0.0.1:8000",
        width=1400,
        height=920,
        resizable=True,
        js_api=api
    )
    webview.start()

if __name__ == "__main__":
    main()
