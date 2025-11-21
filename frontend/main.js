/**
 * 寸法測定PoC - メインJavaScript
 * 状態管理、Canvas操作、API呼び出し、スケール計算を担当
 */

// ==================== 状態管理 ====================
// 状態: "idle" | "analyzing" | "confirm_reference" | "set_reference_size" | "select_rect" | "rect_done"
let appState = "idle";

// アプリケーションの内部状態
const state = {
    // 画像関連
    imageFile: null,
    imageElement: null,
    canvas: null,
    ctx: null,
    canvasScale: 1.0,  // 画像表示時のスケール（元画像サイズ / Canvas表示サイズ）
    
    // 基準物情報
    referenceObject: null,  // APIから返された基準物情報
    referenceWidthMm: null,  // ユーザーが入力した基準物の幅(mm)
    referenceHeightMm: null,  // ユーザーが入力した基準物の高さ(mm)
    mmPerPx: null,  // スケール係数（mm/px）
    
    // 矩形選択関連
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    currentRect: null,  // {x, y, width, height} in px
    
    // カメラ関連
    cameraStream: null,  // MediaStreamオブジェクト
};

// 基準物のデフォルトサイズ（mm）
// バックエンドと統一して大文字固定
const DEFAULT_SIZES = {
    POST: { width: 340, height: 150 },
    INTERCOM: { width: 115, height: 150 },
    BLOCK: { width: 390, height: 190 }
};

// 基準物タイプの日本語表示名
// バックエンドと統一して大文字固定
const TYPE_NAMES = {
    POST: "ポスト",
    INTERCOM: "インターフォン",
    BLOCK: "ブロック"
};

/**
 * タイプから日本語表示名を取得
 * @param {string} type - バックエンドが返すタイプ（例: "POST", "INTERCOM"）
 * @returns {string} - 日本語表示名
 */
function getTypeDisplayName(type) {
    // クラス名のマッピング（モデルのクラス名を標準名に変換）
    const classNameMapping = {
        "PO": "POST",
        "POST": "POST",
        "INTERCOM": "INTERCOM",
        "INTER": "INTERCOM",
        "IC": "INTERCOM"
    };
    
    // マッピングを適用
    const mappedType = classNameMapping[type] || type;
    return TYPE_NAMES[mappedType] || mappedType;
}

// APIのベースURL
const API_BASE_URL = "http://127.0.0.1:8000";

// ==================== DOM要素の取得 ====================
const elements = {
    // 接続状態関連
    statusIcon: document.getElementById("statusIcon"),
    statusText: document.getElementById("statusText"),
    connectionInfo: document.getElementById("connectionInfo"),
    testConnectionBtn: document.getElementById("testConnectionBtn"),
    // その他
    imageFileInput: document.getElementById("imageFileInput"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    analyzeStatus: document.getElementById("analyzeStatus"),
    canvas: document.getElementById("mainCanvas"),
    confirmSection: document.getElementById("confirmSection"),
    confirmMessage: document.getElementById("confirmMessage"),
    confirmYesBtn: document.getElementById("confirmYesBtn"),
    confirmNoBtn: document.getElementById("confirmNoBtn"),
    referenceSizeSection: document.getElementById("referenceSizeSection"),
    referenceTypeDisplay: document.getElementById("referenceTypeDisplay"),
    referenceWidthInput: document.getElementById("referenceWidthInput"),
    referenceHeightInput: document.getElementById("referenceHeightInput"),
    confirmSizeBtn: document.getElementById("confirmSizeBtn"),
    selectRectSection: document.getElementById("selectRectSection"),
    rectStatus: document.getElementById("rectStatus"),
    resultSection: document.getElementById("resultSection"),
    resultWidth: document.getElementById("resultWidth"),
    resultHeight: document.getElementById("resultHeight"),
    resultAspectRatio: document.getElementById("resultAspectRatio"),
    // カメラ関連
    cameraBtn: document.getElementById("cameraBtn"),
    cameraModal: document.getElementById("cameraModal"),
    cameraVideo: document.getElementById("cameraVideo"),
    cameraCanvas: document.getElementById("cameraCanvas"),
    cameraError: document.getElementById("cameraError"),
    captureBtn: document.getElementById("captureBtn"),
    cancelCameraBtn: document.getElementById("cancelCameraBtn"),
    closeCameraBtn: document.getElementById("closeCameraBtn"),
};

// Canvasコンテキストの初期化
state.canvas = elements.canvas;
state.ctx = state.canvas.getContext("2d");

// ==================== イベントリスナーの設定 ====================

// 画像ファイル選択
elements.imageFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        state.imageFile = file;
        loadImageToCanvas(file);
        elements.analyzeBtn.disabled = false;
        updateStatus("analyzeStatus", "画像が選択されました。解析ボタンをクリックしてください。", "info");
    }
});

// 基準物解析ボタン
elements.analyzeBtn.addEventListener("click", async () => {
    if (!state.imageFile) {
        alert("画像を選択してください");
        return;
    }
    await analyzeImage();
});

// 基準物確認: はい
elements.confirmYesBtn.addEventListener("click", () => {
    changeState("set_reference_size");
});

// 基準物確認: いいえ（PoCではアラート表示）
elements.confirmNoBtn.addEventListener("click", () => {
    alert("枠の修正機能はPoCでは未実装です。\n将来的には、ユーザーが手動で枠の位置を調整できる機能を追加予定です。");
});

// 基準寸法確定ボタン
elements.confirmSizeBtn.addEventListener("click", () => {
    confirmReferenceSize();
});

// Canvas上でのマウス操作（矩形選択）
elements.canvas.addEventListener("mousedown", handleMouseDown);
elements.canvas.addEventListener("mousemove", handleMouseMove);
elements.canvas.addEventListener("mouseup", handleMouseUp);
elements.canvas.addEventListener("mouseleave", handleMouseUp);  // マウスがCanvas外に出た場合も終了

// 接続テストボタン
elements.testConnectionBtn.addEventListener("click", testConnection);

// カメラボタン
elements.cameraBtn.addEventListener("click", openCamera);

// カメラモーダルの閉じるボタン
elements.closeCameraBtn.addEventListener("click", closeCamera);
elements.cancelCameraBtn.addEventListener("click", closeCamera);

// 撮影ボタン
elements.captureBtn.addEventListener("click", capturePhoto);

// ==================== 状態遷移関数 ====================

function changeState(newState) {
    appState = newState;
    
    // 各状態に応じてUIを更新
    switch (newState) {
        case "idle":
            // 初期状態
            break;
        case "analyzing":
            elements.analyzeBtn.disabled = true;
            updateStatus("analyzeStatus", "解析中...", "info");
            break;
        case "confirm_reference":
            elements.confirmSection.style.display = "block";
            break;
        case "set_reference_size":
            elements.confirmSection.style.display = "none";
            elements.referenceSizeSection.style.display = "block";
            setupReferenceSizeInputs();
            break;
        case "select_rect":
            elements.referenceSizeSection.style.display = "none";
            elements.selectRectSection.style.display = "block";
            updateStatus("rectStatus", "画像上でドラッグして矩形を選択してください", "info");
            break;
        case "rect_done":
            elements.resultSection.style.display = "block";
            break;
    }
}

// ==================== 画像読み込み ====================

function loadImageToCanvas(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            state.imageElement = img;
            drawImageToCanvas(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function drawImageToCanvas(img) {
    // Canvasの最大サイズを設定（表示用）
    const maxWidth = 800;
    const maxHeight = 600;
    
    let displayWidth = img.width;
    let displayHeight = img.height;
    
    // アスペクト比を保ちながらリサイズ
    if (displayWidth > maxWidth) {
        displayHeight = (displayHeight * maxWidth) / displayWidth;
        displayWidth = maxWidth;
    }
    if (displayHeight > maxHeight) {
        displayWidth = (displayWidth * maxHeight) / displayHeight;
        displayHeight = maxHeight;
    }
    
    // Canvasサイズを設定
    state.canvas.width = displayWidth;
    state.canvas.height = displayHeight;
    
    // スケール係数を計算（元画像サイズ / Canvas表示サイズ）
    state.canvasScale = img.width / displayWidth;
    
    // 画像を描画
    state.ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
}

// ==================== API呼び出し ====================

async function analyzeImage() {
    changeState("analyzing");
    
    try {
        const formData = new FormData();
        formData.append("pillar_image", state.imageFile);
        
        const response = await fetch(`${API_BASE_URL}/api/v1/pillar/analyze`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.reference_object) {
            // 基準物が検出された
            // クラス名のマッピングを適用（バックエンドから返された値を正規化）
            const classNameMapping = {
                "POST": "POST",
                "INTERCOM": "INTERCOM",
            };
            const originalType = data.reference_object.type;
            const mappedType = classNameMapping[originalType] || originalType;
            if (originalType !== mappedType) {
                data.reference_object.type = mappedType;
            }
            
            state.referenceObject = data.reference_object;
            drawReferenceBox(data.reference_object);
            changeState("confirm_reference");
            updateConfirmMessage(data.reference_object);
        } else {
            // 基準物が検出されなかった
            updateStatus("analyzeStatus", "基準物が検出されませんでした。別の画像を試してください。", "error");
            changeState("idle");
            elements.analyzeBtn.disabled = false;
        }
    } catch (error) {
        console.error("解析エラー:", error);
        updateStatus("analyzeStatus", `解析エラー: ${error.message}`, "error");
        changeState("idle");
        elements.analyzeBtn.disabled = false;
        alert(`解析エラーが発生しました: ${error.message}\nバックエンドが起動しているか確認してください。`);
    }
}

// ==================== 描画関数 ====================

function drawReferenceBox(refObj) {
    // Canvas上に再描画（画像 + 基準物の枠）
    if (state.imageElement) {
        drawImageToCanvas(state.imageElement);
    }
    
    // 基準物の枠を描画（Canvas表示サイズに合わせてスケール調整）
    const x = refObj.x / state.canvasScale;
    const y = refObj.y / state.canvasScale;
    const width = refObj.width / state.canvasScale;
    const height = refObj.height / state.canvasScale;
    
    state.ctx.strokeStyle = "#e74c3c";
    state.ctx.lineWidth = 3;
    state.ctx.strokeRect(x, y, width, height);
    
    // ラベルを描画
    state.ctx.fillStyle = "#e74c3c";
    state.ctx.font = "bold 16px Arial";
    const label = getTypeDisplayName(refObj.type);
    state.ctx.fillText(label, x, y - 5);
}

function drawCurrentRect() {
    if (!state.currentRect) return;
    
    // 画像と基準物の枠を再描画
    if (state.imageElement) {
        drawImageToCanvas(state.imageElement);
    }
    if (state.referenceObject) {
        drawReferenceBox(state.referenceObject);
    }
    
    // 現在選択中の矩形を描画
    state.ctx.strokeStyle = "#3498db";
    state.ctx.lineWidth = 2;
    state.ctx.setLineDash([5, 5]);
    state.ctx.strokeRect(
        state.currentRect.x,
        state.currentRect.y,
        state.currentRect.width,
        state.currentRect.height
    );
    state.ctx.setLineDash([]);
}

// ==================== マウス操作ハンドラ ====================

function handleMouseDown(e) {
    if (appState !== "select_rect") return;
    
    const rect = state.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    state.isDragging = true;
    state.dragStartX = x;
    state.dragStartY = y;
    state.currentRect = { x, y, width: 0, height: 0 };
}

function handleMouseMove(e) {
    if (!state.isDragging || appState !== "select_rect") return;
    
    const rect = state.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // 矩形の座標とサイズを計算
    state.currentRect.x = Math.min(state.dragStartX, x);
    state.currentRect.y = Math.min(state.dragStartY, y);
    state.currentRect.width = Math.abs(x - state.dragStartX);
    state.currentRect.height = Math.abs(y - state.dragStartY);
    
    // 矩形を描画
    drawCurrentRect();
}

function handleMouseUp(e) {
    if (!state.isDragging) return;
    
    state.isDragging = false;
    
    if (state.currentRect && state.currentRect.width > 5 && state.currentRect.height > 5) {
        // 矩形が有効なサイズの場合、実寸を計算して表示
        calculateAndDisplayResult();
        changeState("rect_done");
    } else {
        // 矩形が小さすぎる場合は無視
        state.currentRect = null;
        if (state.imageElement) {
            drawImageToCanvas(state.imageElement);
        }
        if (state.referenceObject) {
            drawReferenceBox(state.referenceObject);
        }
    }
}

// ==================== 基準物サイズ入力 ====================

function setupReferenceSizeInputs() {
    if (!state.referenceObject) return;
    
    const type = state.referenceObject.type;
    // クラス名のマッピング（モデルのクラス名を標準名に変換）
    const classNameMapping = {
        "PO": "POST",
        "POST": "POST",
        "INTERCOM": "INTERCOM",
        "INTER": "INTERCOM",
        "IC": "INTERCOM"
    };
    const mappedType = classNameMapping[type] || type;
    // マッピング後のタイプを使用
    const defaults = DEFAULT_SIZES[mappedType] || { width: 100, height: 100 };
    
    // タイプ表示
    const displayName = getTypeDisplayName(type);
    elements.referenceTypeDisplay.textContent = displayName;
    
    // デフォルト値を設定
    elements.referenceWidthInput.value = defaults.width;
    elements.referenceHeightInput.value = defaults.height;
}

function confirmReferenceSize() {
    const width = parseFloat(elements.referenceWidthInput.value);
    const height = parseFloat(elements.referenceHeightInput.value);
    
    if (!width || !height || width <= 0 || height <= 0) {
        alert("有効な幅と高さを入力してください");
        return;
    }
    
    if (!state.referenceObject) {
        alert("基準物情報がありません");
        return;
    }
    
    // 基準物の実寸を保存
    state.referenceWidthMm = width;
    state.referenceHeightMm = height;
    
    // スケール係数を計算（幅を使用）
    // ref_width_mm / bbox_width_px = mm_per_px
    const bboxWidthPx = state.referenceObject.width;
    state.mmPerPx = width / bboxWidthPx;
    
    // 矩形選択モードに遷移
    changeState("select_rect");
}

// ==================== 実寸計算と結果表示 ====================

function calculateAndDisplayResult() {
    if (!state.currentRect || !state.mmPerPx) {
        return;
    }
    
    // Canvas上の矩形サイズ（px）を取得
    const rectWidthPx = state.currentRect.width;
    const rectHeightPx = state.currentRect.height;
    
    // 元画像サイズに合わせてスケール調整
    const rectWidthPxOriginal = rectWidthPx * state.canvasScale;
    const rectHeightPxOriginal = rectHeightPx * state.canvasScale;
    
    // 実寸を計算（mm）
    // rect_width_mm = rect_width_px * mm_per_px
    const rectWidthMm = rectWidthPxOriginal * state.mmPerPx;
    const rectHeightMm = rectHeightPxOriginal * state.mmPerPx;
    
    // 結果を表示（小数第1位で四捨五入）
    elements.resultWidth.textContent = Math.round(rectWidthMm * 10) / 10;
    elements.resultHeight.textContent = Math.round(rectHeightMm * 10) / 10;
    
    // アスペクト比を計算
    const aspectRatio = rectWidthMm / rectHeightMm;
    elements.resultAspectRatio.textContent = `${Math.round(aspectRatio * 10) / 10} : 1.0`;
}

// ==================== 接続テスト関数 ====================

/**
 * バックエンドへの接続をテストする
 */
async function testConnection() {
    // 接続確認中の状態を表示
    updateConnectionStatus("checking", "接続確認中...", "");
    
    try {
        const startTime = Date.now();
        const response = await fetch(`${API_BASE_URL}/`, {
            method: "GET",
            headers: {
                "Accept": "application/json"
            }
        });
        const endTime = Date.now();
        const responseTime = endTime - startTime;
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // 接続成功
        const info = `接続成功！<br>
            API: ${API_BASE_URL}<br>
            応答時間: ${responseTime}ms<br>
            バージョン: ${data.version || "不明"}`;
        
        updateConnectionStatus("connected", "バックエンドに接続済み", info);
        
        return true;
        
    } catch (error) {
        console.error("接続テストエラー:", error);
        
        // 接続失敗
        const errorMessage = error.message || "不明なエラー";
        const info = `接続に失敗しました。<br>
            API: ${API_BASE_URL}<br>
            エラー: ${errorMessage}<br>
            <br>
            バックエンドが起動しているか確認してください:<br>
            <code>cd backend && uvicorn main:app --reload --host 127.0.0.1 --port 8000</code>`;
        
        updateConnectionStatus("disconnected", "バックエンドに接続できません", info);
        return false;
    }
}

/**
 * 接続状態のUIを更新する
 */
function updateConnectionStatus(status, text, info) {
    // アイコンとテキストのクラスをリセット
    elements.statusIcon.className = "status-icon";
    elements.statusText.className = "status-text";
    elements.connectionInfo.className = "connection-info";
    
    // 状態に応じてクラスを設定
    if (status === "connected") {
        elements.statusIcon.classList.add("connected");
        elements.statusText.classList.add("connected");
        elements.connectionInfo.classList.add("connected");
        elements.statusIcon.textContent = "●";
    } else if (status === "disconnected") {
        elements.statusIcon.classList.add("disconnected");
        elements.statusText.classList.add("disconnected");
        elements.connectionInfo.classList.add("disconnected");
        elements.statusIcon.textContent = "●";
    } else if (status === "checking") {
        elements.statusIcon.classList.add("checking");
        elements.statusText.classList.add("checking");
        elements.statusIcon.textContent = "●";
    }
    
    // テキストと情報を更新
    elements.statusText.textContent = text;
    elements.connectionInfo.innerHTML = info;
}

// ==================== カメラ機能 ====================

/**
 * カメラを開く
 */
async function openCamera() {
    // エラーメッセージを非表示
    elements.cameraError.style.display = "none";
    elements.cameraError.textContent = "";
    
    // モーダルを表示
    elements.cameraModal.style.display = "block";
    
    try {
        // カメラにアクセス
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",  // 背面カメラを優先（スマホの場合）
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        
        // ストリームを保存
        state.cameraStream = stream;
        
        // ビデオ要素にストリームを設定
        elements.cameraVideo.srcObject = stream;
        
        // 撮影ボタンを有効化
        elements.captureBtn.disabled = false;
        
    } catch (error) {
        console.error("カメラアクセスエラー:", error);
        
        // エラーメッセージを表示
        let errorMessage = "カメラにアクセスできませんでした。";
        
        if (error.name === "NotAllowedError") {
            errorMessage = "カメラの使用が許可されていません。ブラウザの設定でカメラへのアクセスを許可してください。";
        } else if (error.name === "NotFoundError") {
            errorMessage = "カメラが見つかりませんでした。カメラが接続されているか確認してください。";
        } else if (error.name === "NotReadableError") {
            errorMessage = "カメラが他のアプリケーションで使用中の可能性があります。";
        } else {
            errorMessage = `エラー: ${error.message}`;
        }
        
        elements.cameraError.textContent = errorMessage;
        elements.cameraError.style.display = "block";
        elements.captureBtn.disabled = true;
    }
}

/**
 * カメラを閉じる
 */
function closeCamera() {
    // ストリームを停止
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
        state.cameraStream = null;
    }
    
    // ビデオ要素をクリア
    elements.cameraVideo.srcObject = null;
    
    // モーダルを非表示
    elements.cameraModal.style.display = "none";
}

/**
 * 写真を撮影する
 */
function capturePhoto() {
    try {
        const video = elements.cameraVideo;
        const canvas = elements.cameraCanvas;
        const ctx = canvas.getContext("2d");
        
        // Canvasサイズをビデオのサイズに合わせる
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // ビデオの現在のフレームをCanvasに描画
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // CanvasからBlobに変換
        canvas.toBlob((blob) => {
            if (!blob) {
                alert("画像の取得に失敗しました");
                return;
            }
            
            // BlobをFileオブジェクトに変換
            const file = new File([blob], `camera_${Date.now()}.jpg`, {
                type: "image/jpeg",
                lastModified: Date.now()
            });
            
            // 既存の画像選択フローと同じ処理
            state.imageFile = file;
            loadImageToCanvas(file);
            elements.analyzeBtn.disabled = false;
            updateStatus("analyzeStatus", "カメラから画像を取得しました。解析ボタンをクリックしてください。", "success");
            
            // カメラを閉じる
            closeCamera();
        }, "image/jpeg", 0.95);  // JPEG形式、品質95%
        
    } catch (error) {
        console.error("撮影エラー:", error);
        alert(`撮影エラー: ${error.message}`);
    }
}

// モーダルの外側をクリックしたら閉じる
elements.cameraModal.addEventListener("click", (e) => {
    if (e.target === elements.cameraModal) {
        closeCamera();
    }
});

// ==================== ユーティリティ関数 ====================

function updateStatus(elementId, message, type = "info") {
    const element = elements[elementId];
    if (!element) return;
    
    element.textContent = message;
    element.className = `status-message ${type}`;
}

function updateConfirmMessage(refObj) {
    const typeName = getTypeDisplayName(refObj.type);
    elements.confirmMessage.textContent = `この枠は${typeName}の位置で正しいですか？`;
}

// ==================== 初期化 ====================

// ページ読み込み時に接続テストを実行
window.addEventListener("DOMContentLoaded", () => {
    testConnection();
    
    // 30秒ごとに自動的に接続状態をチェック（オプション）
    // setInterval(testConnection, 30000);
});

