"""
基準物検出ロジック（モック実装）
後からYOLO実装に差し替え可能な構造にする
"""

from PIL import Image
from typing import Optional, Dict, Any
from ultralytics  import YOLO
import numpy as np


YOLO_MODEL = YOLO("best.pt")
# YOLOモデルから実際のクラス名を取得
if hasattr(YOLO_MODEL, 'names') and YOLO_MODEL.names:
    # モデルのクラス名をそのまま使用（モデルのクラス順序に合わせる）
    model_classes = YOLO_MODEL.names
    
    # クラス名のマッピング（モデルのクラス名を標準名に変換）
    class_name_mapping = {
        "PO": "POST",
        "POST": "POST",
        "INTERCOM": "INTERCOM",
        "INTER": "INTERCOM",
        "IC": "INTERCOM"
    }
    
    # YOLOモデルのnamesは辞書形式 {0: "class0", 1: "class1"} の可能性がある
    if isinstance(model_classes, dict):
        # 辞書の場合、クラスID順にソートしてリストを作成
        max_class_id = max(model_classes.keys()) if model_classes else -1
        raw_classes = [model_classes.get(i, f"CLASS_{i}").upper() for i in range(max_class_id + 1)]
    else:
        # リストの場合
        raw_classes = [model_classes[i].upper() for i in range(len(model_classes))]
    
    # クラス名をマッピングして標準名に変換
    classes = [class_name_mapping.get(cls, cls) for cls in raw_classes]
else:
    # フォールバック: 手動定義
    classes = ["POST","INTERCOM"]

def yolo_detect_reference_object(image: Image.Image) -> Optional[Dict[str, Any]]:
    """
    YOLOモデルを使って基準物（postbox）を検出して返す。

    Args:
        image: PIL Image

    Returns:
        {
            "type": "postbox",
            "x": int,
            "y": int,
            "width": int,
            "height": int,
            "confidence": float
        }
        または None
    """

    # Ensure 3-channel RGB
    image = image.convert("RGB")

    # PIL → numpy
    img_np = np.array(image)

    # YOLO推論（信頼度の閾値を上げる）
    results = YOLO_MODEL.predict(img_np, verbose=False, conf=0.5)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    # 最も信頼度の高い1つを採用
    box = results[0].boxes[0]
    # クラスIDを取得（results[0].boxes.clsは全ボックスのクラスIDを含むテンソル）
    # 最初のボックスのクラスIDを取得
    classId = int(results[0].boxes.cls[0].item())
    # xyxy形式で取り出す
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0].item())
    
    rawClassType = classes[classId] if 0 <= classId < len(classes) else "UNKNOWN"
    
    # クラス名のマッピング（念のため再度適用）
    class_name_mapping = {
        "POST": "POST",
        "INTERCOM": "INTERCOM",
    }
    classType = class_name_mapping.get(rawClassType, rawClassType)

    width = int(x2 - x1)
    height = int(y2 - y1)

    return {
        "type": classType,   # ← あなたのクラス名（1クラス学習なら固定でOK）
        "x": int(x1),
        "y": int(y1),
        "width": width,
        "height": height,
        "confidence": conf
    }

def mock_detect_reference_object(image: Image.Image) -> Optional[Dict[str, Any]]:
    return yolo_detect_reference_object(image)


# 後からYOLO実装に差し替える場合は、この関数を実装して
# main.pyで import を変更するだけでOK
def detect_reference_object(image: Image.Image) -> Optional[Dict[str, Any]]:
    """
    実際のYOLO検出関数（将来実装用）
    
    現時点では mock_detect_reference_object を呼び出すだけ
    """
    return mock_detect_reference_object(image)

