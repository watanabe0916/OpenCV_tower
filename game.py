import os
import sys
import math
import time
import shutil
import random
import subprocess
import cv2
import numpy as np
import pygame
import pymunk

# -------------------------------------------------------------
# 定数とカラー定義 (リッチで洗練されたダークテーマ)
# -------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GAME_WIDTH = 600

# カラーパレット
COLOR_BG_GAME = (20, 24, 33)        # 深みのあるスペースグレー
COLOR_BG_SIDE = (30, 37, 48)        # 少し明るいサイドパネル
COLOR_STAGE = (0, 184, 212)         # ネオンシアンの土台
COLOR_ACCENT = (0, 229, 255)        # 明るいシアン (アクティブ要素)
COLOR_TEXT = (240, 244, 248)        # クリーンな白
COLOR_TEXT_MUTED = (150, 160, 175)  # 暗めの文字
COLOR_RED = (255, 82, 82)           # アラート・ゲームオーバー用

# コリジョンタイプ (PyMunkの衝突検知用)
COLLISION_TYPE_OBJECT = 1
COLLISION_TYPE_SENSOR = 2

# ディレクトリパス
STOCK_DIR = "captured_images/stock"
os.makedirs(STOCK_DIR, exist_ok=True)

# -------------------------------------------------------------
# スライダークラス (PygameによるカスタムGUI)
# -------------------------------------------------------------
class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, 8)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.grabbed = False
        
        # つまみ（ハンドル）のサイズ
        self.handle_radius = 8
        self.handle_rect = pygame.Rect(0, 0, self.handle_radius * 2, self.handle_radius * 2)
        self.update_handle_pos()

    def update_handle_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.center = (
            self.rect.x + ratio * self.rect.width,
            self.rect.centery
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                distance = math.hypot(mouse_pos[0] - self.handle_rect.centerx, mouse_pos[1] - self.handle_rect.centery)
                if distance <= self.handle_radius + 4 or self.rect.inflate(10, 20).collidepoint(mouse_pos):
                    self.grabbed = True
                    self.update_value(mouse_pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.grabbed = False
        elif event.type == pygame.MOUSEMOTION:
            if self.grabbed:
                self.update_value(event.pos[0])

    def update_value(self, mouse_x):
        mouse_x = max(self.rect.x, min(mouse_x, self.rect.right))
        ratio = (mouse_x - self.rect.x) / self.rect.width
        self.val = self.min_val + ratio * (self.max_val - self.min_val)
        self.update_handle_pos()

    def draw(self, surface, font):
        label_text = font.render(f"{self.label}", True, COLOR_TEXT)
        val_text = font.render(f"{self.val:.2f}", True, COLOR_ACCENT)
        surface.blit(label_text, (self.rect.x, self.rect.y - 20))
        surface.blit(val_text, (self.rect.right - val_text.get_width(), self.rect.y - 20))
        
        pygame.draw.rect(surface, (50, 60, 75), self.rect, border_radius=4)
        
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, self.handle_rect.centerx - self.rect.x, self.rect.height)
        pygame.draw.rect(surface, COLOR_ACCENT, filled_rect, border_radius=4)
        
        color = (255, 255, 255) if self.grabbed else COLOR_ACCENT
        if self.grabbed:
            pygame.draw.circle(surface, (0, 229, 255, 100), self.handle_rect.center, self.handle_radius + 4, 2)
        pygame.draw.circle(surface, color, self.handle_rect.center, self.handle_radius)


# -------------------------------------------------------------
# コライダー頂点の自動生成およびパディング処理
# -------------------------------------------------------------
def load_object_vertices(image_path: str) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[int, int]]:
    """
    透過PNG画像を読み込み、そのアルファチャンネルから凸包（Convex Hull）の頂点を抽出して返す。
    """
    if not os.path.exists(image_path):
        return None, (0.0, 0.0), (0, 0)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        h, w = img.shape[:2]
        half_w, half_h = w / 2.0, h / 2.0
        vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        return vertices, (half_w, half_h), (w, h)

    alpha = img[:, :, 3]
    h, w = img.shape[:2]

    _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        half_w, half_h = w / 2.0, h / 2.0
        vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        return vertices, (half_w, half_h), (w, h)

    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    hull = cv2.convexHull(approx, clockwise=False)

    M = cv2.moments(hull)
    if M['m00'] != 0.0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        x_b, y_b, w_b, h_b = cv2.boundingRect(hull)
        cx = x_b + w_b / 2.0
        cy = y_b + h_b / 2.0

    vertices = []
    for pt in hull:
        px, py = pt[0]
        vertices.append((px - cx, py - cy))

    return vertices, (cx, cy), (w, h)


def center_image_on_centroid(image: pygame.Surface, cx: float, cy: float) -> pygame.Surface:
    """
    画像の重心がSurfaceの中心に一致するように、透過余白を追加した新しいSurfaceを生成する。
    """
    w, h = image.get_size()
    max_dist_x = max(cx, w - cx)
    max_dist_y = max(cy, h - cy)
    
    new_w = int(max_dist_x * 2)
    new_h = int(max_dist_y * 2)
    
    new_surf = pygame.Surface((new_w, new_h), pygame.SRCALPHA)
    dest_x = new_w // 2 - int(cx)
    dest_y = new_h // 2 - int(cy)
    new_surf.blit(image, (dest_x, dest_y))
    
    return new_surf


# -------------------------------------------------------------
# フォールバック画像と頂点作成
# -------------------------------------------------------------
def create_fallback_object() -> tuple[pygame.Surface, list[tuple[float, float]], tuple[float, float]]:
    surf = pygame.Surface((160, 160), pygame.SRCALPHA)
    points = [
        (80, 15),   # 上
        (105, 60),
        (150, 65),  # 右
        (115, 95),
        (130, 145), # 右下
        (80, 120),  # 下中央
        (30, 145),  # 左下
        (45, 95),
        (10, 65),   # 左
        (55, 60)
    ]
    pygame.draw.polygon(surf, (255, 128, 171, 220), points)
    pygame.draw.polygon(surf, (255, 255, 255, 255), points, 4)
    
    fallback_vertices = [
        (0.0, -65.0),
        (70.0, -15.0),
        (50.0, 65.0),
        (-50.0, 65.0),
        (-70.0, -15.0)
    ]
    return surf, fallback_vertices, (80.0, 80.0)


# -------------------------------------------------------------
# サブプロセスを走らせて撮影と切り抜きを行う
# -------------------------------------------------------------
def run_capture_and_extract_processes(screen, font) -> bool:
    """
    capture.py と extractor.py を外部プロセスとして順次呼び出す。
    Pygameの Cocoa イベントループを阻害しないように実行中は待機画面を描画する。
    """
    # 待機画面の描画
    screen.fill(COLOR_BG_GAME)
    title_font = pygame.font.SysFont("Helvetica", 28, bold=True)
    msg_font = pygame.font.SysFont("Helvetica", 18)
    
    txt_title = title_font.render("CAPTURING BLOCK...", True, COLOR_ACCENT)
    txt_msg = msg_font.render("Please shoot on phone & crop on the PC window.", True, COLOR_TEXT)
    screen.blit(txt_title, (GAME_WIDTH // 2 - txt_title.get_width() // 2, 250))
    screen.blit(txt_msg, (GAME_WIDTH // 2 - txt_msg.get_width() // 2, 300))
    pygame.display.flip()
    
    try:
        # 1. 撮影プロセスの同期呼び出し
        print("[System] Launching capture.py...")
        res_cap = subprocess.run([sys.executable, "capture.py"], check=False)
        if res_cap.returncode != 0:
            print("[System] Capture process was cancelled or failed.")
            return False

        # 2. 抽出プロセスの同期呼び出し
        print("[System] Launching extractor.py...")
        res_ext = subprocess.run([sys.executable, "extractor.py"], check=False)
        if res_ext.returncode != 0:
            print("[System] Extraction process was cancelled or failed.")
            return False

        return True
    except Exception as e:
        print(f"[System] Failed to invoke subprocesses: {e}")
        return False


# -------------------------------------------------------------
# 物理オブジェクトの生成
# -------------------------------------------------------------
def spawn_physics_object(space, vertices, x, y, angle_deg, friction, elasticity, mass=1.0) -> tuple[pymunk.Body, pymunk.Shape]:
    moment = pymunk.moment_for_poly(mass, vertices)
    body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)
    body.angle = math.radians(-angle_deg)
    
    shape = pymunk.Poly(body, vertices)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = COLLISION_TYPE_OBJECT
    
    space.add(body, shape)
    return body, shape


# -------------------------------------------------------------
# メインシステム
# -------------------------------------------------------------
def main():
    pygame.init()
    # IME干渉を完全に無効化
    if hasattr(pygame.key, "stop_text_input"):
        pygame.key.stop_text_input()
        
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Real Object Tower Battle - Modes & Stock")
    clock = pygame.time.Clock()
    
    # 英語フォントのロード
    try:
        font_main = pygame.font.SysFont("Helvetica", 16)
        font_large = pygame.font.SysFont("Helvetica", 24)
        font_title = pygame.font.SysFont("Helvetica", 32, bold=True)
    except:
        font_main = pygame.font.Font(None, 24)
        font_large = pygame.font.Font(None, 32)
        font_title = pygame.font.Font(None, 40)

    # UIスライダーの定義 (Friction / Elasticity / Stage Width)
    slider_friction = Slider(620, 180, 160, 0.0, 1.0, 0.6, "Friction")
    slider_elasticity = Slider(620, 240, 160, 0.0, 1.0, 0.2, "Elasticity")
    slider_stage_width = Slider(620, 300, 160, 100.0, 500.0, 300.0, "Stage Width")
    sliders = [slider_friction, slider_elasticity, slider_stage_width]

    # 物理空間 (PyMunk) の定義
    space = pymunk.Space()
    space.gravity = (0.0, 900.0)

    # 土台 (静的ボディ) の配置
    stage_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    stage_body.position = (300, 480)
    stage_shape = pymunk.Poly.create_box(stage_body, (300.0, 20))
    stage_shape.friction = slider_friction.val
    stage_shape.elasticity = slider_elasticity.val
    space.add(stage_body, stage_shape)

    # 静的センサー判定ライン (ゲームオーバー用デッドライン)
    sensor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    sensor_body.position = (300, 580)
    sensor_shape = pymunk.Poly.create_box(sensor_body, (20000, 10))
    sensor_shape.sensor = True
    sensor_shape.collision_type = COLLISION_TYPE_SENSOR
    space.add(sensor_body, sensor_shape)

    # ゲーム状態管理定数
    STATE_MODE_SELECT = 0
    STATE_AIMING = 1
    STATE_FALLING = 2
    STATE_GAMEOVER = 3
    STATE_STOCK_MANAGER = 4
    
    game_state = STATE_MODE_SELECT
    current_mode = 1  # 1: Live, 2: Random Stock
    score = 0

    # 土台の長さ追跡用
    last_stage_width = 300.0

    # 落下前のエイミングオブジェクト操作パラメータ
    obj_x = GAME_WIDTH // 2
    obj_y = 80
    obj_angle = 0.0
    move_speed = 5

    # 物理空間内のオブジェクト
    active_body = None
    active_shape = None
    falling_frames = 0

    # 現在ターンのオブジェクトデータ
    current_object_image = None
    current_object_vertices = None
    current_object_centroid = (0.0, 0.0)
    object_prepared = False

    # ストックマネージャー用の状態変数
    stock_page = 0
    stock_files = []

    # コリジョンハンドラー (センサー衝突時にゲームオーバーをトリガー)
    def on_sensor_collision(arbiter, space_ref, data):
        nonlocal game_state
        if game_state in [STATE_AIMING, STATE_FALLING]:
            game_state = STATE_GAMEOVER
            print("GAME OVER triggered via sensor.")
        return True
        
    space.on_collision(COLLISION_TYPE_OBJECT, COLLISION_TYPE_SENSOR, begin=on_sensor_collision)

    # 物理クリア
    def reset_game_physics():
        nonlocal active_body, active_shape, falling_frames, object_prepared
        for b in [body for body in space.bodies if body != stage_body and body != sensor_body]:
            for s in b.shapes:
                space.remove(s)
            space.remove(b)
        active_body = None
        active_shape = None
        falling_frames = 0
        object_prepared = False

    # モード3用のストックリスト読み込み
    def refresh_stock_list():
        nonlocal stock_files
        if os.path.exists(STOCK_DIR):
            stock_files = sorted([f for f in os.listdir(STOCK_DIR) if f.endswith(".png")])
        else:
            stock_files = []

    # ターンごとのオブジェクト準備処理
    def prepare_next_object() -> bool:
        nonlocal current_object_image, current_object_vertices, current_object_centroid, object_prepared
        
        if current_mode == 1:
            # モード1: 毎回撮影・抽出
            success = run_capture_and_extract_processes(screen, font_main)
            if not success:
                return False
            
            target_path = "captured_images/extracted_object.png"
            if os.path.exists(target_path):
                try:
                    raw_img = pygame.image.load(target_path).convert_alpha()
                    vertices, (cx, cy), (w, h) = load_object_vertices(target_path)
                    current_object_image = center_image_on_centroid(raw_img, cx, cy)
                    current_object_vertices = vertices
                    current_object_centroid = (cx, cy)
                    object_prepared = True
                    return True
                except Exception as e:
                    print(f"Failed to load captured image: {e}")
                    return False
            return False

        elif current_mode == 2:
            # モード2: ストックからランダム
            refresh_stock_list()
            if not stock_files:
                # ストック空ならフォールバックの星
                raw_img, vertices, (cx, cy) = create_fallback_object()
                current_object_image = center_image_on_centroid(raw_img, cx, cy)
                current_object_vertices = vertices
                current_object_centroid = (cx, cy)
                object_prepared = True
                return True
            
            selected_file = random.choice(stock_files)
            selected_path = os.path.join(STOCK_DIR, selected_file)
            try:
                raw_img = pygame.image.load(selected_path).convert_alpha()
                vertices, (cx, cy), (w, h) = load_object_vertices(selected_path)
                current_object_image = center_image_on_centroid(raw_img, cx, cy)
                current_object_vertices = vertices
                current_object_centroid = (cx, cy)
                object_prepared = True
                return True
            except Exception as e:
                print(f"Failed to load stock file {selected_file}: {e}")
                # 失敗時は星
                raw_img, vertices, (cx, cy) = create_fallback_object()
                current_object_image = center_image_on_centroid(raw_img, cx, cy)
                current_object_vertices = vertices
                current_object_centroid = (cx, cy)
                object_prepared = True
                return True

        return False

    running = True
    while running:
        # -------------------------------------------------------------
        # イベント処理
        # -------------------------------------------------------------
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                
            # スライダーイベント処理
            for slider in sliders:
                slider.handle_event(event)

            # キーボード入力
            if event.type == pygame.KEYDOWN:
                # プレイ中にmキーでモード選択に戻る
                if event.key == pygame.K_m:
                    reset_game_physics()
                    game_state = STATE_MODE_SELECT
                    score = 0
                    print("Returned to Mode Select Screen.")
                    continue

                if game_state == STATE_MODE_SELECT:
                    if event.key == pygame.K_1:
                        current_mode = 1
                        reset_game_physics()
                        game_state = STATE_AIMING
                    elif event.key == pygame.K_2:
                        current_mode = 2
                        reset_game_physics()
                        game_state = STATE_AIMING
                    elif event.key == pygame.K_3:
                        current_mode = 3
                        refresh_stock_list()
                        stock_page = 0
                        game_state = STATE_STOCK_MANAGER

                elif game_state == STATE_GAMEOVER:
                    if event.key == pygame.K_r:
                        reset_game_physics()
                        game_state = STATE_AIMING
                        score = 0

                elif game_state == STATE_AIMING:
                    if event.key == pygame.K_DOWN and object_prepared:
                        # 落下開始
                        game_state = STATE_FALLING
                        falling_frames = 0
                        
                        friction = slider_friction.val
                        elasticity = slider_elasticity.val
                        
                        stage_shape.friction = friction
                        stage_shape.elasticity = elasticity
                        
                        active_body, active_shape = spawn_physics_object(
                            space, current_object_vertices, obj_x, obj_y, obj_angle,
                            friction, elasticity
                        )

            # マウスクリック判定 (STATE_MODE_SELECT と STATE_STOCK_MANAGER のみ)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                
                if game_state == STATE_MODE_SELECT:
                    # モードボタン1: (100, 150, 400, 90)
                    if pygame.Rect(100, 150, 400, 90).collidepoint(mouse_pos):
                        current_mode = 1
                        reset_game_physics()
                        game_state = STATE_AIMING
                    # モードボタン2: (100, 260, 400, 90)
                    elif pygame.Rect(100, 260, 400, 90).collidepoint(mouse_pos):
                        current_mode = 2
                        reset_game_physics()
                        game_state = STATE_AIMING
                    # モードボタン3: (100, 370, 400, 90)
                    elif pygame.Rect(100, 370, 400, 90).collidepoint(mouse_pos):
                        current_mode = 3
                        refresh_stock_list()
                        stock_page = 0
                        game_state = STATE_STOCK_MANAGER

                elif game_state == STATE_STOCK_MANAGER:
                    # 新規登録ボタン: (100, 60, 400, 40)
                    if pygame.Rect(100, 60, 400, 40).collidepoint(mouse_pos):
                        success = run_capture_and_extract_processes(screen, font_main)
                        if success:
                            # タイムスタンプ名でストックに保存
                            temp_path = "captured_images/extracted_object.png"
                            if os.path.exists(temp_path):
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                dest_path = os.path.join(STOCK_DIR, f"stock_{timestamp}.png")
                                shutil.copy(temp_path, dest_path)
                                print(f"Saved to stock: {dest_path}")
                                refresh_stock_list()
                                
                    # 戻るボタン: (200, 535, 200, 40)
                    elif pygame.Rect(200, 535, 200, 40).collidepoint(mouse_pos):
                        game_state = STATE_MODE_SELECT
                        
                    # ページネーション PREV: (50, 480, 120, 35)
                    elif pygame.Rect(50, 480, 120, 35).collidepoint(mouse_pos):
                        if stock_page > 0:
                            stock_page -= 1
                    # ページネーション NEXT: (430, 480, 120, 35)
                    elif pygame.Rect(430, 480, 120, 35).collidepoint(mouse_pos):
                        max_pages = math.ceil(len(stock_files) / 6)
                        if stock_page < max_pages - 1:
                            stock_page += 1
                            
                    # グリッドの削除ボタン検知
                    # 各グリッド枠: (x, y, 150, 140)
                    # 削除ボタン: (x + 25, y + 105, 100, 24)
                    cols = 3
                    x_coords = [50, 225, 400]
                    y_coords = [135, 305]
                    start_idx = stock_page * 6
                    
                    for i in range(6):
                        idx = start_idx + i
                        if idx >= len(stock_files):
                            break
                        
                        col = i % cols
                        row = i // cols
                        btn_rect = pygame.Rect(x_coords[col] + 25, y_coords[row] + 105, 100, 24)
                        if btn_rect.collidepoint(mouse_pos):
                            # ファイル削除
                            target_file = stock_files[idx]
                            target_path = os.path.join(STOCK_DIR, target_file)
                            try:
                                os.remove(target_path)
                                print(f"Deleted stock file: {target_path}")
                                refresh_stock_list()
                                # ページ調整
                                max_pages = max(1, math.ceil(len(stock_files) / 6))
                                if stock_page >= max_pages:
                                    stock_page = max_pages - 1
                            except Exception as e:
                                print(f"Failed to delete file: {e}")
                            break

        # -------------------------------------------------------------
        # 状態更新
        # -------------------------------------------------------------
        # 土台の長さ・物理パラメータのリアルタイム更新 (操作中またはゲームオーバー時のみ変更可能)
        if game_state in [STATE_AIMING, STATE_GAMEOVER]:
            stage_shape.friction = slider_friction.val
            stage_shape.elasticity = slider_elasticity.val
            
            current_stage_w = slider_stage_width.val
            if abs(current_stage_w - last_stage_width) > 1.0:
                space.remove(stage_shape)
                stage_shape = pymunk.Poly.create_box(stage_body, (current_stage_w, 20))
                stage_shape.friction = slider_friction.val
                stage_shape.elasticity = slider_elasticity.val
                space.add(stage_shape)
                last_stage_width = current_stage_w

        # AIMING時のオブジェクトデータ自動準備
        if game_state == STATE_AIMING and not object_prepared:
            # 準備実行。キャンセルされた場合はモード選択に戻る
            success = prepare_next_object()
            if not success:
                game_state = STATE_MODE_SELECT
                score = 0
                continue

        # エイミング中の手動移動・回転操作
        keys = pygame.key.get_pressed()
        if game_state == STATE_AIMING and object_prepared:
            if keys[pygame.K_LEFT]:
                obj_x = max(40, obj_x - move_speed)
            if keys[pygame.K_RIGHT]:
                obj_x = min(GAME_WIDTH - 40, obj_x + move_speed)
            if keys[pygame.K_SPACE]:
                obj_angle = (obj_angle - 3) % 360

        # 物理シミュレーションの更新
        dt = 1.0 / 60.0
        substeps = 10
        for _ in range(substeps):
            space.step(dt / substeps)

        # 落下中の静止判定
        if game_state == STATE_FALLING and active_body is not None:
            falling_frames += 1
            if falling_frames > 60:
                vel = active_body.velocity.length
                ang = abs(active_body.angular_velocity)
                if vel < 1.5 and ang < 0.05:
                    # 積み上げ成功: 次のターンへ
                    score += 1
                    game_state = STATE_AIMING
                    obj_x = GAME_WIDTH // 2
                    obj_y = 80
                    obj_angle = 0.0
                    active_body = None
                    active_shape = None
                    falling_frames = 0
                    object_prepared = False # 次の画像を要求

        # -------------------------------------------------------------
        # 描画処理 (リッチな黒基調レイアウト)
        # -------------------------------------------------------------
        screen.fill(COLOR_BG_GAME)

        # A. MODE_SELECT 画面
        if game_state == STATE_MODE_SELECT:
            # タイトル
            txt_mode_title = font_title.render("SELECT GAME MODE", True, COLOR_TEXT)
            screen.blit(txt_mode_title, (GAME_WIDTH // 2 - txt_mode_title.get_width() // 2, 60))
            
            # モード1 カード
            card1 = pygame.Rect(100, 150, 400, 90)
            pygame.draw.rect(screen, (40, 50, 65), card1, border_radius=8)
            pygame.draw.rect(screen, COLOR_ACCENT, card1, 2, border_radius=8)
            t1 = font_large.render("1. Live Capture Mode", True, COLOR_ACCENT)
            d1 = font_main.render("Shoot on phone & drop the block every turn!", True, COLOR_TEXT)
            screen.blit(t1, (120, 165))
            screen.blit(d1, (120, 195))

            # モード2 カード
            card2 = pygame.Rect(100, 260, 400, 90)
            pygame.draw.rect(screen, (40, 50, 65), card2, border_radius=8)
            pygame.draw.rect(screen, COLOR_ACCENT, card2, 2, border_radius=8)
            t2 = font_large.render("2. Random Stock Mode", True, COLOR_ACCENT)
            d2 = font_main.render("Drop registered stock blocks randomly.", True, COLOR_TEXT)
            screen.blit(t2, (120, 275))
            screen.blit(d2, (120, 305))

            # モード3 カード
            card3 = pygame.Rect(100, 370, 400, 90)
            pygame.draw.rect(screen, (40, 50, 65), card3, border_radius=8)
            pygame.draw.rect(screen, COLOR_ACCENT, card3, 2, border_radius=8)
            t3 = font_large.render("3. Manage Stock Library", True, COLOR_ACCENT)
            d3 = font_main.render("Register new blocks or delete stock files.", True, COLOR_TEXT)
            screen.blit(t3, (120, 385))
            screen.blit(d3, (120, 415))
            
            # フッターヒント
            txt_hint = font_main.render("Press 1, 2, or 3 on keyboard, or click the card.", True, COLOR_TEXT_MUTED)
            screen.blit(txt_hint, (GAME_WIDTH // 2 - txt_hint.get_width() // 2, 490))

        # B. STOCK_MANAGER 画面
        elif game_state == STATE_STOCK_MANAGER:
            # 登録ボタン
            reg_btn = pygame.Rect(100, 60, 400, 40)
            pygame.draw.rect(screen, (0, 150, 136), reg_btn, border_radius=5)
            pygame.draw.rect(screen, (255, 255, 255), reg_btn, 1, border_radius=5)
            txt_reg = font_large.render("+ REGISTER NEW OBJECT", True, COLOR_TEXT)
            screen.blit(txt_reg, (GAME_WIDTH // 2 - txt_reg.get_width() // 2, 68))
            
            # グリッド描画 (最大6つ)
            cols = 3
            x_coords = [50, 225, 400]
            y_coords = [135, 305]
            start_idx = stock_page * 6
            
            for i in range(6):
                idx = start_idx + i
                if idx >= len(stock_files):
                    # 空き枠枠線
                    col = i % cols
                    row = i // cols
                    pygame.draw.rect(screen, (40, 48, 60), (x_coords[col], y_coords[row], 150, 140), border_radius=6)
                    txt_empty = font_main.render("[Empty]", True, (70, 80, 95))
                    screen.blit(txt_empty, (x_coords[col] + 75 - txt_empty.get_width() // 2, y_coords[row] + 70 - txt_empty.get_height() // 2))
                    continue
                
                col = i % cols
                row = i // cols
                x, y = x_coords[col], y_coords[row]
                
                # 外枠
                pygame.draw.rect(screen, (45, 55, 70), (x, y, 150, 140), border_radius=6)
                pygame.draw.rect(screen, (80, 95, 115), (x, y, 150, 140), 1, border_radius=6)
                
                # サムネイル画像の読み込みと縮小描画
                try:
                    f_name = stock_files[idx]
                    f_img = pygame.image.load(os.path.join(STOCK_DIR, f_name)).convert_alpha()
                    # アスペクト比を維持しつつ 80x80 以内にリサイズ
                    img_w, img_h = f_img.get_size()
                    scale_factor = min(80 / img_w, 80 / img_h)
                    thumb_w = int(img_w * scale_factor)
                    thumb_h = int(img_h * scale_factor)
                    thumb_img = pygame.transform.scale(f_img, (thumb_w, thumb_h))
                    screen.blit(thumb_img, (x + 75 - thumb_w // 2, y + 55 - thumb_h // 2))
                except Exception as e:
                    txt_err = font_main.render("Error", True, COLOR_RED)
                    screen.blit(txt_err, (x + 75 - txt_err.get_width() // 2, y + 45))

                # 削除ボタン
                del_btn = pygame.Rect(x + 25, y + 105, 100, 24)
                pygame.draw.rect(screen, COLOR_RED, del_btn, border_radius=3)
                txt_del = font_main.render("Delete", True, COLOR_TEXT)
                screen.blit(txt_del, (del_btn.centerx - txt_del.get_width() // 2, del_btn.centery - txt_del.get_height() // 2))

            # ページネーション描画
            max_pages = max(1, math.ceil(len(stock_files) / 6))
            
            # PREV
            prev_btn = pygame.Rect(50, 480, 120, 35)
            pygame.draw.rect(screen, (40, 50, 65) if stock_page > 0 else (25, 30, 40), prev_btn, border_radius=4)
            txt_prev = font_main.render("PREV", True, COLOR_TEXT if stock_page > 0 else COLOR_TEXT_MUTED)
            screen.blit(txt_prev, (prev_btn.centerx - txt_prev.get_width() // 2, prev_btn.centery - txt_prev.get_height() // 2))
            
            # NEXT
            next_btn = pygame.Rect(430, 480, 120, 35)
            pygame.draw.rect(screen, (40, 50, 65) if stock_page < max_pages - 1 else (25, 30, 40), next_btn, border_radius=4)
            txt_next = font_main.render("NEXT", True, COLOR_TEXT if stock_page < max_pages - 1 else COLOR_TEXT_MUTED)
            screen.blit(txt_next, (next_btn.centerx - txt_next.get_width() // 2, next_btn.centery - txt_next.get_height() // 2))

            # ページインジケータ
            txt_pg = font_main.render(f"Page {stock_page + 1} / {max_pages}", True, COLOR_TEXT)
            screen.blit(txt_pg, (GAME_WIDTH // 2 - txt_pg.get_width() // 2, 488))

            # メニューへ戻るボタン
            back_btn = pygame.Rect(200, 535, 200, 40)
            pygame.draw.rect(screen, (50, 60, 75), back_btn, border_radius=5)
            txt_back = font_large.render("Back to Menu", True, COLOR_TEXT)
            screen.blit(txt_back, (back_btn.centerx - txt_back.get_width() // 2, back_btn.centery - txt_back.get_height() // 2))

        # C. プレイ中 (AIMING / FALLING / GAMEOVER)
        else:
            # 土台 (ステージ) の描画
            stage_w = int(last_stage_width)
            pygame.draw.rect(screen, COLOR_STAGE, (300 - stage_w // 2, 470, stage_w, 20), border_radius=5)
            pygame.draw.rect(screen, (0, 96, 100), (300 - stage_w // 2, 490, stage_w, 5), border_radius=2)
            
            # 照準ガイド線 (AIMING時のみ)
            if game_state == STATE_AIMING:
                dash_y = obj_y + 40
                while dash_y < 470:
                    pygame.draw.line(screen, (60, 70, 85), (obj_x, dash_y), (obj_x, dash_y + 8), 2)
                    dash_y += 16

            # 物理空間内の動的ボディを全描画
            for body in space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC:
                    pos = body.position
                    angle_deg = -math.degrees(body.angle)
                    
                    rotated_image = pygame.transform.rotate(current_object_image, angle_deg)
                    rotated_rect = rotated_image.get_rect(center=(int(pos.x), int(pos.y)))
                    
                    screen.blit(rotated_image, rotated_rect.topleft)

            # 落下前に操作中のエイミングオブジェクト描画
            if game_state == STATE_AIMING and object_prepared:
                rotated_image = pygame.transform.rotate(current_object_image, obj_angle)
                rotated_rect = rotated_image.get_rect(center=(obj_x, int(obj_y)))
                
                # うっすら発光するグロー効果
                glow_surf = pygame.Surface((rotated_rect.width + 12, rotated_rect.height + 12), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, (0, 229, 255, 30), glow_surf.get_rect())
                screen.blit(glow_surf, glow_surf.get_rect(center=rotated_rect.center))
                
                screen.blit(rotated_image, rotated_rect.topleft)

            # デッドラインの視覚化 (薄い赤色の破線)
            for dash_x in range(0, GAME_WIDTH, 15):
                pygame.draw.line(screen, (255, 82, 82, 100), (dash_x, 580), (dash_x + 8, 580), 1)

        # -------------------------------------------------------------
        # 2. サイドパネル (ダッシュボードUI) の描画
        # -------------------------------------------------------------
        side_panel_rect = pygame.Rect(GAME_WIDTH, 0, SCREEN_WIDTH - GAME_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, COLOR_BG_SIDE, side_panel_rect)
        pygame.draw.line(screen, (50, 60, 75), (GAME_WIDTH, 0), (GAME_WIDTH, SCREEN_HEIGHT), 2)

        # タイトル
        title_text = font_title.render("TOWER", True, COLOR_TEXT)
        title_text2 = font_title.render("BATTLE", True, COLOR_ACCENT)
        screen.blit(title_text, (620, 30))
        screen.blit(title_text2, (620, 70))
        
        # スコア表示 (プレイ中のみ表示)
        if game_state in [STATE_AIMING, STATE_FALLING, STATE_GAMEOVER]:
            score_label = font_main.render("OBJECTS PLACED:", True, COLOR_TEXT_MUTED)
            score_val = font_large.render(f"{score}", True, (255, 235, 59))
            screen.blit(score_label, (620, 110))
            screen.blit(score_val, (620, 130))
        else:
            # メニューやマネージャー時はタイトル説明
            desc_label = font_main.render("Select menu options", True, COLOR_TEXT_MUTED)
            desc_label2 = font_main.render("to start playing.", True, COLOR_TEXT_MUTED)
            screen.blit(desc_label, (620, 110))
            screen.blit(desc_label2, (620, 130))

        # スライダー描画
        for slider in sliders:
            slider.draw(screen, font_main)

        # ステータス表示
        status_label = font_main.render("STATUS:", True, COLOR_TEXT_MUTED)
        if game_state == STATE_MODE_SELECT:
            status_val = font_main.render("MODE SELECT", True, COLOR_ACCENT)
        elif game_state == STATE_STOCK_MANAGER:
            status_val = font_main.render("STOCK MANAGE", True, COLOR_ACCENT)
        elif game_state == STATE_AIMING:
            status_val = font_main.render("AIMING...", True, (76, 175, 80))
        elif game_state == STATE_FALLING:
            status_val = font_main.render("FALLING!", True, COLOR_ACCENT)
        else:
            status_val = font_main.render("GAME OVER", True, COLOR_RED)
            
        screen.blit(status_label, (620, 360))
        screen.blit(status_val, (620, 380))

        # 操作方法説明
        controls_title = font_main.render("CONTROLS:", True, COLOR_TEXT_MUTED)
        ctrl_left_right = font_main.render("<- / -> : Move", True, COLOR_TEXT)
        ctrl_space = font_main.render("SPACE : Rotate (Hold)", True, COLOR_TEXT)
        ctrl_down = font_main.render("DOWN : Drop Object", True, COLOR_TEXT)
        ctrl_menu = font_main.render("M Key : Mode Menu", True, COLOR_TEXT)
        screen.blit(controls_title, (620, 430))
        screen.blit(ctrl_left_right, (620, 455))
        screen.blit(ctrl_space, (620, 475))
        screen.blit(ctrl_down, (620, 495))
        screen.blit(ctrl_menu, (620, 515))

        # 3. ゲームオーバー時のオーバーレイUIの描画
        if game_state == STATE_GAMEOVER:
            overlay = pygame.Surface((GAME_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 12, 16, 200))
            screen.blit(overlay, (0, 0))
            
            go_text = font_title.render("GAME OVER", True, COLOR_RED)
            restart_text = font_large.render("Press 'R' to Retry", True, COLOR_TEXT)
            back_menu_text = font_main.render("or press 'M' for Mode Selection", True, COLOR_TEXT_MUTED)
            
            screen.blit(go_text, (GAME_WIDTH // 2 - go_text.get_width() // 2, 240))
            screen.blit(restart_text, (GAME_WIDTH // 2 - restart_text.get_width() // 2, 300))
            screen.blit(back_menu_text, (GAME_WIDTH // 2 - back_menu_text.get_width() // 2, 340))

        # 画面更新
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
