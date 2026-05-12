from flask import Flask, request, jsonify
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime
import uuid
import os
import urllib.request
import torch
import torch.nn.functional as F
from torchvision import models, transforms

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "resnet50_garbage_last2.pth"

# 这里不写死，后面在 Render 环境变量里配置
MODEL_URL = os.getenv("MODEL_URL", "")

CLASS_NAMES = ["hazardous", "kitchen", "other", "recyclable"]
NAME_MAP = {
    "hazardous": "有害垃圾",
    "kitchen": "厨余垃圾",
    "other": "其他垃圾",
    "recyclable": "可回收物",
}
DESC_MAP = {
    "hazardous": "含有有毒有害成分，需要专门回收处理。",
    "kitchen": "易腐烂的生活有机垃圾，适合分类投放和资源化处理。",
    "other": "不属于可回收物、厨余垃圾和有害垃圾的普通生活垃圾。",
    "recyclable": "具有回收利用价值，适合进入再生资源回收体系。",
}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model_exists():
    if MODEL_PATH.exists():
        print(f"模型文件已存在：{MODEL_PATH}")
        return

    if not MODEL_URL:
        raise FileNotFoundError(
            "模型文件不存在，且未配置 MODEL_URL 环境变量，无法自动下载模型。"
        )

    print(f"开始下载模型：{MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"模型下载完成：{MODEL_PATH}")
    except Exception as exc:
        raise RuntimeError(f"模型下载失败：{exc}") from exc


def build_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, 4)
    )
    return model


def load_model():
    ensure_model_exists()

    model = build_model()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image_path: Path):
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("图片文件无效或已损坏，无法识别。") from exc

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].detach().cpu()
        topk = torch.topk(probs, 2)

    top_items = [(CLASS_NAMES[idx], float(probs[idx])) for idx in topk.indices.tolist()]
    label_key = top_items[0][0]
    confidence = top_items[0][1]
    raw_label = NAME_MAP[label_key]
    second_name = NAME_MAP[top_items[1][0]] if len(top_items) > 1 else "无"

    warning = ""
    is_uncertain = False
    show_label = raw_label
    show_desc = DESC_MAP[label_key]

    if confidence < 0.60:
        is_uncertain = True
        show_label = "识别不确定"
        show_desc = "当前图片在多个类别之间区分不明显，建议结合材质、污染情况和使用场景人工判断。"
        warning = f"模型倾向于：{raw_label}；同时与“{second_name}”存在混淆。"
    elif confidence < 0.80:
        warning = "当前结果可信度中等，建议参考次优类别进行辅助判断。"

    return {
        "label": show_label,
        "raw_label": raw_label,
        "next_label": second_name,
        "confidence": confidence,
        "description": show_desc,
        "warning": warning,
        "is_uncertain": is_uncertain,
    }


@app.route("/")
def index():
    return "Flask backend is running."


@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "未上传文件"
        }), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "error": "文件名为空"
        }), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        return jsonify({
            "success": False,
            "error": "仅支持 jpg、jpeg、png、bmp、webp 格式"
        }), 400

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    save_path = UPLOAD_DIR / save_name
    file.save(save_path)

    try:
        result = predict_image(save_path)
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400

    image_url = f"/static/uploads/{save_name}"

    return jsonify({
        "success": True,
        "label": result["label"],
        "raw_label": result["raw_label"],
        "confidence": result["confidence"],
        "next_label": result["next_label"],
        "description": result["description"],
        "warning": result["warning"],
        "is_uncertain": result["is_uncertain"],
        "image_url": image_url
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)