"""
基準物検出ロジック（モック実装）
後からYOLO実装に差し替え可能な構造にする
"""

from PIL import Image
from typing import Optional, Dict, Any
from ultralytics  import YOLO
import numpy as np


YOLO_MODEL = YOLO("best.pt")
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

    # YOLO推論
    results = YOLO_MODEL.predict(img_np, verbose=False, conf=0.25)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    # 最も信頼度の高い1つを採用
    box = results[0].boxes[0]
    classType = classes[int(results[0].boxes.cls)]
    # xyxy形式で取り出す
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0].item())

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

