import os
import sys
import logging
import cv2
import numpy as np

# ログ設定の初期化
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# マウス・ズーム制御用のグローバル変数
drawing = False
current_stroke = []  # 元画像座標系での座標リスト
scale = 1.0
offset_x = 0
offset_y = 0
mouse_x = 0
mouse_y = 0

# ウィンドウサイズ用の変数は画像読み込み後に動的に設定
WIN_W, WIN_H = 0, 0

def mouse_callback(event, x, y, flags, param):
    """マウス移動の監視と、表示座標から元画像座標への逆変換・描画プロット"""
    global drawing, current_stroke, mouse_x, mouse_y, scale, offset_x, offset_y

    # 現在のマウスウィンドウ座標を保持（ズーム時の中心計算用）
    mouse_x, mouse_y = x, y

    # ウィンドウ上の座標 (x, y) を、現在のズーム・オフセットを考慮して元画像座標へ逆変換
    orig_x = int(x / scale) + offset_x
    orig_y = int(y / scale) + offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_stroke.append((orig_x, orig_y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_stroke.append((orig_x, orig_y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def extract_object_by_freehand_zoom(obj_path: str, output_path: str, target_size: int = 250) -> bool:
    """ユーザーがズーム操作を交えながらフリーハンドで囲んだ領域の内側のみを透過抽出する"""
    global scale, offset_x, offset_y, mouse_x, mouse_y, current_stroke

    if not os.path.exists(obj_path):
        logging.error(f"物体画像が見つかりません: {obj_path}")
        return False

    img_obj = cv2.imread(obj_path)
    if img_obj is None:
        logging.error("画像の読み込みに失敗しました。")
        return False

    global WIN_W, WIN_H
    img_h, img_w = img_obj.shape[:2]
    
    # ウィンドウサイズを元の画像サイズと完全に一致させる
    WIN_W, WIN_H = img_w, img_h
    
    # 初期状態の表示スケールを等倍（1.0）に設定
    scale = 1.0
    offset_x = 0
    offset_y = 0
    current_stroke = []

    # 元画像のサイズに合わせて自動でウィンドウが生成されるようにフラグを変更
    cv2.namedWindow("Freehand Tracing (Zoom: Up/Down Arrow)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Freehand Tracing (Zoom: Up/Down Arrow)", mouse_callback)

    print("【操作方法】")
    print("1. [左クリック長押し（ドラッグ）] で物体の周りをなぞって囲んでください。")
    print("2. [十字キーの上 / 下] を押すと、マウスポインタの位置を中心に拡大・縮小します。")
    print("3. キーボードの 'e' を押すと、囲んだ内側を透過PNGとして切り出します。")
    print("4. キーボードの 'r' を押すと、描いた線をリセットします。")
    print("5. 中断する場合は 'q' を押してください。")

    while True:
        # 表示用ベース画像の生成（元画像をコピーして座標系に蓄積された線を元画像側に描画）
        img_canvas = img_obj.copy()
        if len(current_stroke) > 1:
            pts = np.array(current_stroke, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_canvas, [pts], False, (0, 255, 0), max(2, int(2/scale)))

        # 現在のズーム・オフセット状態に合わせて元画像からROIを切り出して表示用にリサイズ
        # クロップ範囲の計算
        crop_w = int(WIN_W / scale)
        crop_h = int(WIN_H / scale)
        
        # 範囲外制御
        offset_x = max(0, min(offset_x, img_w - crop_w))
        offset_y = max(0, min(offset_y, img_h - crop_h))

        img_crop = img_canvas[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]
        img_display = cv2.resize(img_crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

        # 解像度に比例した動的スケーリング (基準幅 1000px)
        scale_ratio = max(0.5, WIN_W / 1000.0)
        font_scale = 0.6 * scale_ratio
        thickness = max(1, int(2 * scale_ratio))
        banner_h = int(70 * scale_ratio)
        y1 = int(28 * scale_ratio)
        y2 = int(55 * scale_ratio)
        x_off = int(15 * scale_ratio)

        # 画面上部に操作説明用の半透明バナーを描画
        overlay = img_display.copy()
        cv2.rectangle(overlay, (0, 0), (WIN_W, banner_h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.7, img_display, 0.3, 0, img_display)

        # 操作説明テキストの描画 (動的にスケーリングされたサイズを使用)
        msg_trace_zoom = "Drag Mouse: Trace  |  W/S or Up/Down Arrow: Zoom"
        msg_confirm_reset = "E: Confirm & Save  |  R: Reset Line  |  Q: Cancel"
        cv2.putText(img_display, msg_trace_zoom, (x_off, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(img_display, msg_confirm_reset, (x_off, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imshow("Freehand Tracing (Zoom: Up/Down Arrow)", img_display)
        
        # 十字キーなどの拡張キーコードを取得するため、& 0xFF は行わずに待機
        key = cv2.waitKey(20)
        
        if key == -1:
            continue

        # 'e' キーで確定
        if key == ord('e'):
            if len(current_stroke) < 3:
                logging.warning("領域を確定するには最低3点以上のプロットが必要です。")
                continue
            cv2.destroyWindow("Freehand Tracing (Zoom: Up/Down Arrow)")
            break
        # 'r' キーでリセット
        elif key == ord('r'):
            current_stroke = []
            logging.info("描画リセット")
        # 'q' キーで中断
        elif key == ord('q'):
            cv2.destroyAllWindows()
            logging.info("中断されました。")
            return False

        # --- ズームロジック（十字キー or W/S キー） ---
        # 環境依存の矢印キーコードに加え、確実に動作する W/S キーを判定に追加
        is_up = key in [63232, 82, 2490368, 0x260000, ord('w'), 3, 0]
        is_down = key in [63233, 84, 2621440, 0x280000, ord('s'), 4, 1]

        if is_up or is_down:
            # ズーム前のマウスポインタ下の元画像座標を特定
            orig_mx = (mouse_x / scale) + offset_x
            orig_my = (mouse_y / scale) + offset_y

            # スケールの更新
            if is_up:
                scale = min(scale * 1.2, 10.0)  # 最大10倍
            elif is_down:
                scale = max(scale / 1.2, min(WIN_W / img_w, WIN_H / img_h))

            # 新しいスケールにおいて、マウスポインタ下の元画像座標がズレないようにオフセットを逆算
            offset_x = int(orig_mx - (mouse_x / scale))
            offset_y = int(orig_my - (mouse_y / scale))

    try:
        # 1. 確定した多角形（クローズポリゴン）から純粋なマスクを生成
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array(current_stroke, dtype=np.int32).reshape((-1, 1, 2))
        
        # ユーザーがなぞった内側のみを255(白)にする（GrabCutは実行しない）
        cv2.fillPoly(mask, [pts], 255)

        # 2. ダイレクトアルファマスク適用（指定範囲外を100%透過）
        b_ch, g_ch, r_ch = cv2.split(img_obj)
        img_rgba = cv2.merge([b_ch, g_ch, r_ch, mask])

        # 3. 指定された多角形の外接矩形で最小トリミング
        x, y, w_orig, h_orig = cv2.boundingRect(pts)
        img_cropped = img_rgba[y:y+h_orig, x:x+w_orig]

        # 4. 公平性を保つ一律リサイズ
        s = target_size / max(w_orig, h_orig)
        w_target = int(w_orig * s)
        h_target = int(h_orig * s)
        img_resized = cv2.resize(img_cropped, (w_target, h_target), interpolation=cv2.INTER_AREA)

        # 5. 保存
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite(output_path, img_resized)
        logging.info(f"フリーハンド透過抽出成功: {output_path} ({w_target}x{h_target})")
        return True

    except Exception as e:
        logging.error(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    OBJ_FILE = "captured_images/object.png"
    OUTPUT_FILE = "captured_images/extracted_object.png"

    extract_object_by_freehand_zoom(OBJ_FILE, OUTPUT_FILE, target_size=150)