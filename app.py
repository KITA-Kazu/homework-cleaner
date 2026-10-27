import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ページ基本設定
st.set_page_config(page_title="宿題スキャナー (手書き消し)", layout="wide")
st.title("📝 宿題プリント 手書き文字消しアプリ")

# --- 画像処理パイプライン関数群 ---

def normalize_lighting(img_bgr, target_color_bgr):
    """Step 1: 照明ムラ除去・紙色の均一化"""
    h, w = img_bgr.shape[:2]
    k_size = int(max(h, w) * 0.03) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    illum_map = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        channel = img_bgr[:, :, c]
        dilated = cv2.dilate(channel, kernel)
        blurred = cv2.GaussianBlur(dilated, (k_size, k_size), 0)
        illum_map[:, :, c] = np.maximum(blurred.astype(np.float32), 1.0)
        
    orig_f = img_bgr.astype(np.float32)
    normalized = (orig_f / illum_map) * target_color_bgr
    return np.clip(normalized, 0, 255).astype(np.uint8)

def auto_erase_pipeline(img_bgr, target_color_bgr):
    """Step 1〜4を一気通貫で実行する自動消去処理"""
    # 1. 均一化
    norm = normalize_lighting(img_bgr, target_color_bgr)
    h, w = norm.shape[:2]
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    
    # 2. 二値化
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 色ペン（赤）抽出
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    color_handwriting = cv2.bitwise_or(m1, m2)
    
    # 3. 活字保護と罫線分離
    # 水平モルフォロジーで行を特定
    line_kw = int(w * 0.015) | 1
    dilated_lines = cv2.dilate(binary_inv, cv2.getStructuringElement(cv2.MORPH_RECT, (line_kw, 1)), iterations=2)
    contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    protect_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > line_kw * 2 and bh > 5:
            protect_mask[max(0, y-2):min(h, y+bh+2), max(0, x-2):min(w, x+bw+2)] = 255
            
    # 罫線抽出
    min_line_len = int(w * 0.04) | 1
    h_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1)))
    v_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len)))
    clean_lines = cv2.bitwise_or(h_lines, v_lines)
    
    # 手書きマスクの算出
    black_hw = cv2.bitwise_and(binary_inv, cv2.bitwise_not(protect_mask))
    total_hw = cv2.bitwise_or(black_hw, color_handwriting)
    
    # 罫線中心を除外して少し膨張
    eroded_lines = cv2.erode(clean_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    stroke_mask = cv2.bitwise_and(total_hw, cv2.bitwise_not(eroded_lines))
    stroke_mask = cv2.dilate(stroke_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # 4. Inpainting
    result = cv2.inpaint(norm, stroke_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result

# --- UIセッション状態の管理 ---
if "step1_done" not in st.session_state:
    st.session_state.step1_done = False
if "current_img" not in st.session_state:
    st.session_state.current_img = None

# --- サイドバー操作 ---
st.sidebar.header("⚙️ 設定パネル")
uploaded_file = st.sidebar.file_uploader("プリント画像をアップロード", type=["jpg", "png", "jpeg"])

bg_color_hex = st.sidebar.color_picker("紙の地色（スポイト指定）", "#FFFFFF")
# HEXからBGRに変換
bg_b = int(bg_color_hex[5:7], 16)
bg_g = int(bg_color_hex[3:5], 16)
bg_r = int(bg_color_hex[1:3], 16)
target_bgr = np.array([bg_b, bg_g, bg_r], dtype=np.float32)

brush_size = st.sidebar.slider("追加消しゴムの太さ", min_value=5, max_value=50, value=15)

# --- メイン画面 ---
if uploaded_file is not None:
    # 初回読み込み
    if st.session_state.current_img is None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # 処理が重くなりすぎないよう、長辺を最大1600pxにリサイズ
        h, w = img_bgr.shape[:2]
        max_side = 1600
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        st.session_state.current_img = img_bgr

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. 自動消去の実行")
        if st.button("🚀 自動消去を実行する", type="primary"):
            with st.spinner("手書き文字を自動検出・消去中..."):
                st.session_state.current_img = auto_erase_pipeline(st.session_state.current_img, target_bgr)
                st.session_state.step1_done = True
                st.success("自動消去が完了しました！")

        # 現在の画像プレビュー
        curr_rgb = cv2.cvtColor(st.session_state.current_img, cv2.COLOR_BGR2RGB)
        st.image(curr_rgb, caption="現在の画像状態", use_container_width=True)

    with col2:
        st.subheader("2. 残った箇所の追加消しゴム")
        st.info("💡 拡大・縮小ツールでズームし、消し残し部分を赤ペンでなぞってください。")

        # キャンバス設定
        curr_pil = Image.fromarray(curr_rgb)
        c_w, c_h = curr_pil.size

        # 描画キャンバス（ズーム・パン機能付き）
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.4)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.6)",
            background_image=curr_pil,
            update_streamlit=True,
            height=600,
            width=min(700, c_w),
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("✨ なぞった箇所を消去する"):
            if canvas_result.image_data is not None:
                # なぞられた領域のマスク（アルファチャンネルから取得）
                drawn_mask = canvas_result.image_data[:, :, 3]
                if np.any(drawn_mask > 0):
                    with st.spinner("なぞった部分を修復中..."):
                        # 表示サイズから元画像サイズへマスクをリサイズ
                        mask_resized = cv2.resize(drawn_mask, (c_w, c_h), interpolation=cv2.INTER_NEAREST)
                        _, mask_binary = cv2.threshold(mask_resized, 10, 255, cv2.THRESH_BINARY)
                        
                        # 局所インペイント
                        inpainted = cv2.inpaint(st.session_state.current_img, mask_binary, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                        st.session_state.current_img = inpainted
                        st.rerun()

    # ダウンロードボタン
    st.markdown("---")
    out_rgb = cv2.cvtColor(st.session_state.current_img, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out_rgb)
    st.download_button(
        label="📥 完成した画像をダウンロード",
        data=cv2.imencode('.png', st.session_state.current_img)[1].tobytes(),
        file_name="cleaned_homework.png",
        mime="image/png"
    )
