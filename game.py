import os
import sys
import math
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
        # ラベルと値の描画
        label_text = font.render(f"{self.label}", True, COLOR_TEXT)
        val_text = font.render(f"{self.val:.2f}", True, COLOR_ACCENT)
        surface.blit(label_text, (self.rect.x, self.rect.y - 20))
        surface.blit(val_text, (self.rect.right - val_text.get_width(), self.rect.y - 20))
        
        # スライダー背景線の描画
        pygame.draw.rect(surface, (50, 60, 75), self.rect, border_radius=4)
        
        # 選択済みのゲージの描画
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, self.handle_rect.centerx - self.rect.x, self.rect.height)
        pygame.draw.rect(surface, COLOR_ACCENT, filled_rect, border_radius=4)
        
        # つまみの描画 (ホバーやドラッグで発光エフェクトを追加)
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
    重心 (cx, cy) を原点とした相対座標リストと、重心位置を返す。
    """
    if not os.path.exists(image_path):
        return None, (0.0, 0.0), (0, 0)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        # 透過チャンネルがない場合は四角形の頂点にする
        h, w = img.shape[:2]
        half_w, half_h = w / 2.0, h / 2.0
        vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        return vertices, (half_w, half_h), (w, h)

    alpha = img[:, :, 3]
    h, w = img.shape[:2]

    # 不透明領域を抽出
    _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        half_w, half_h = w / 2.0, h / 2.0
        vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        return vertices, (half_w, half_h), (w, h)

    # 最大エリアの輪郭を取得
    main_contour = max(contours, key=cv2.contourArea)

    # 多角形近似で頂点数を最適化
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    # 凸包化 (PyMunkの凸形状ルールに合わせるため反時計回りに指定)
    hull = cv2.convexHull(approx, clockwise=False)

    # 重心の算出
    M = cv2.moments(hull)
    if M['m00'] != 0.0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        x_b, y_b, w_b, h_b = cv2.boundingRect(hull)
        cx = x_b + w_b / 2.0
        cy = y_b + h_b / 2.0

    # 重心を原点とする相対座標に変換
    vertices = []
    for pt in hull:
        px, py = pt[0]
        vertices.append((px - cx, py - cy))

    return vertices, (cx, cy), (w, h)


def center_image_on_centroid(image: pygame.Surface, cx: float, cy: float) -> pygame.Surface:
    """
    画像の重心がSurfaceの中心に一致するように、透過余白を追加した新しいSurfaceを生成する。
    これにより、Pygame側で画像の回転(中心座標指定)と物理の重心運動が一致する。
    """
    w, h = image.get_size()
    
    # 重心から外枠までの最大距離を基準にサイズ決定
    max_dist_x = max(cx, w - cx)
    max_dist_y = max(cy, h - cy)
    
    new_w = int(max_dist_x * 2)
    new_h = int(max_dist_y * 2)
    
    new_surf = pygame.Surface((new_w, new_h), pygame.SRCALPHA)
    
    # 新しい中心に元の画像の重心が重なるように配置
    dest_x = new_w // 2 - int(cx)
    dest_y = new_h // 2 - int(cy)
    new_surf.blit(image, (dest_x, dest_y))
    
    return new_surf


# -------------------------------------------------------------
# フォールバック画像と頂点作成
# -------------------------------------------------------------
def create_fallback_object() -> tuple[pygame.Surface, list[tuple[float, float]], tuple[float, float]]:
    """型抜き画像が存在しない場合に使用する、フォールバック星型画像と頂点情報"""
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
    # グラデーションを意識した色合い
    pygame.draw.polygon(surf, (255, 128, 171, 220), points) # 薄いピンク
    pygame.draw.polygon(surf, (255, 255, 255, 255), points, 4) # 白枠
    
    # 星型の頂点を凸包化した時の頂点 (重心 80, 80 基準の相対座標)
    # PyMunk用 (反時計回り)
    fallback_vertices = [
        (0.0, -65.0),    # 上 (80, 15)
        (70.0, -15.0),   # 右 (150, 65)
        (50.0, 65.0),    # 右下 (130, 145)
        (-50.0, 65.0),   # 左下 (30, 145)
        (-70.0, -15.0)   # 左 (10, 65)
    ]
    return surf, fallback_vertices, (80.0, 80.0)


# -------------------------------------------------------------
# 物理オブジェクトの生成
# -------------------------------------------------------------
def spawn_physics_object(space, vertices, x, y, angle_deg, friction, elasticity, mass=1.0) -> tuple[pymunk.Body, pymunk.Shape]:
    """物理空間に動的ボディを追加し、凸包コライダーをバインドして返す"""
    # 慣性モーメントの計算
    moment = pymunk.moment_for_poly(mass, vertices)
    
    body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)
    body.angle = math.radians(-angle_deg)  # Pygame(時計回り正)からPyMunk(反時計回り正)への変換
    
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
    # IME(日本語入力)の干渉によるキー入力テキストのバッファリングを無効化
    if hasattr(pygame.key, "stop_text_input"):
        pygame.key.stop_text_input()
        
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Real Object Tower Battle - Step 3 (Physics)")
    clock = pygame.time.Clock()
    
    # フォントロード
    try:
        font_main = pygame.font.SysFont("Helvetica", 16)
        font_large = pygame.font.SysFont("Helvetica", 24)
        font_title = pygame.font.SysFont("Helvetica", 32, bold=True)
    except:
        font_main = pygame.font.Font(None, 24)
        font_large = pygame.font.Font(None, 32)
        font_title = pygame.font.Font(None, 40)
    
    # 1. 画像と物理コライダー頂点データの読み込み
    image_path = "captured_images/extracted_object.png"
    has_image = os.path.exists(image_path)
    
    if has_image:
        try:
            raw_img = pygame.image.load(image_path).convert_alpha()
            vertices, (cx, cy), (w, h) = load_object_vertices(image_path)
            # 重心を中心にパディング補正した画像を作成
            object_image = center_image_on_centroid(raw_img, cx, cy)
            print(f"物理コライダー生成完了: 頂点数={len(vertices)}, 重心=({cx:.1f}, {cy:.1f})")
        except Exception as e:
            print(f"画像解析に失敗しました。フォールバックを使用します: {e}")
            raw_img, vertices, (cx, cy) = create_fallback_object()
            object_image = center_image_on_centroid(raw_img, cx, cy)
    else:
        print(f"警告: {image_path} が見つからないため、フォールバックの星型オブジェクトを使用します。")
        raw_img, vertices, (cx, cy) = create_fallback_object()
        object_image = center_image_on_centroid(raw_img, cx, cy)

    # 2. UIスライダーの定義 (Friction / Elasticity / Stage Width)
    slider_friction = Slider(620, 180, 160, 0.0, 1.0, 0.6, "Friction")
    slider_elasticity = Slider(620, 240, 160, 0.0, 1.0, 0.2, "Elasticity")
    slider_stage_width = Slider(620, 300, 160, 100.0, 500.0, 300.0, "Stage Width")
    sliders = [slider_friction, slider_elasticity, slider_stage_width]

    # 3. 物理空間 (PyMunk) の定義
    space = pymunk.Space()
    space.gravity = (0.0, 900.0)  # 現実的な下向き重力

    # 土台 (静的ボディ) の配置
    stage_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    stage_body.position = (300, 480)
    stage_shape = pymunk.Poly.create_box(stage_body, (300.0, 20))
    stage_shape.friction = slider_friction.val
    stage_shape.elasticity = slider_elasticity.val
    space.add(stage_body, stage_shape)

    # 静的センサー判定ライン (ゲームオーバー用デッドライン)
    # 左右に激しく吹き飛んだ場合でも検知できるよう、幅を十分に広く設定 (20000px)
    sensor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    sensor_body.position = (300, 580)
    sensor_shape = pymunk.Poly.create_box(sensor_body, (20000, 10))
    sensor_shape.sensor = True
    sensor_shape.collision_type = COLLISION_TYPE_SENSOR
    space.add(sensor_body, sensor_shape)

    # ゲーム状態管理
    STATE_AIMING = 0
    STATE_FALLING = 1
    STATE_GAMEOVER = 2
    
    game_state = STATE_AIMING
    score = 0
    
    # 土台の長さ追跡用
    last_stage_width = 300.0

    # 落下前のエイミングオブジェクト操作パラメータ
    obj_x = GAME_WIDTH // 2
    obj_y = 80
    obj_angle = 0.0
    move_speed = 5

    # 物理エンジンに追加されているアクティブ(落下中)なオブジェクト情報
    active_body = None
    active_shape = None
    
    # 落下開始してからの経過フレームカウンター (即時静止判定を回避するため)
    falling_frames = 0

    # コリジョンハンドラー (センサー衝突時にゲームオーバーをトリガー)
    def on_sensor_collision(arbiter, space_ref, data):
        nonlocal game_state
        if game_state != STATE_GAMEOVER:
            game_state = STATE_GAMEOVER
            print("【判定】オブジェクトがデッドラインに接触しました。ゲームオーバー！")
        return True
        
    space.on_collision(COLLISION_TYPE_OBJECT, COLLISION_TYPE_SENSOR, begin=on_sensor_collision)

    # 物理空間からすべての動的オブジェクトを削除するリセット用関数
    def reset_game():
        nonlocal game_state, score, obj_x, obj_y, obj_angle, active_body, active_shape, falling_frames
        
        # stage_body, sensor_body 以外の全ボディをクリア
        for b in [body for body in space.bodies if body != stage_body and body != sensor_body]:
            for s in b.shapes:
                space.remove(s)
            space.remove(b)
            
        game_state = STATE_AIMING
        score = 0
        obj_x = GAME_WIDTH // 2
        obj_y = 80
        obj_angle = 0.0
        active_body = None
        active_shape = None
        falling_frames = 0
        print("物理ゲームリセット完了")

    running = True
    while running:
        # -------------------------------------------------------------
        # イベント処理
        # -------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # スライダーイベント処理
            for slider in sliders:
                slider.handle_event(event)

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

            # キーボード単発入力
            if event.type == pygame.KEYDOWN:
                if game_state == STATE_GAMEOVER:
                    if event.key == pygame.K_r:
                        reset_game()
                
                elif game_state == STATE_AIMING:
                    if event.key == pygame.K_DOWN:
                        # 落下開始: 物理空間へ動的オブジェクトを追加
                        game_state = STATE_FALLING
                        falling_frames = 0
                        
                        # スライダーから摩擦力と反発係数を動的バインド
                        friction = slider_friction.val
                        elasticity = slider_elasticity.val
                        
                        # 土台側にもスライダーの物理パラメータを適用
                        stage_shape.friction = friction
                        stage_shape.elasticity = elasticity
                        
                        active_body, active_shape = spawn_physics_object(
                            space, vertices, obj_x, obj_y, obj_angle,
                            friction, elasticity
                        )

        # -------------------------------------------------------------
        # 状態更新
        # -------------------------------------------------------------
        keys = pygame.key.get_pressed()
        
        # エイミング中の手動移動・回転操作
        if game_state == STATE_AIMING:
            if keys[pygame.K_LEFT]:
                obj_x = max(40, obj_x - move_speed)
            if keys[pygame.K_RIGHT]:
                obj_x = min(GAME_WIDTH - 40, obj_x + move_speed)
                
            if keys[pygame.K_SPACE]:
                # 押し続けている間、時計回りに回転
                obj_angle = (obj_angle - 3) % 360

        # 物理シミュレーションの更新
        # シミュレーションが安定するようにステップを分割して進める
        dt = 1.0 / 60.0
        substeps = 10
        for _ in range(substeps):
            space.step(dt / substeps)

        # 落下中の静止判定 (静止したら次のオブジェクトの操作フェーズへ)
        if game_state == STATE_FALLING and active_body is not None:
            falling_frames += 1
            
            # 最低60フレーム (約1秒) 経過後、かつ速度・回転速度が静止閾値以下の場合
            if falling_frames > 60:
                vel = active_body.velocity.length
                ang = abs(active_body.angular_velocity)
                
                if vel < 1.5 and ang < 0.05:
                    # 積み上げ成功: スコア加算してエイミング状態に戻る
                    score += 1
                    game_state = STATE_AIMING
                    obj_x = GAME_WIDTH // 2
                    obj_y = 80
                    obj_angle = 0.0
                    active_body = None
                    active_shape = None
                    falling_frames = 0
                    print(f"積み上げ成功！現在のスコア: {score}")

        # -------------------------------------------------------------
        # 描画処理 (リッチな黒基調レイアウト)
        # -------------------------------------------------------------
        screen.fill(COLOR_BG_GAME)

        # 1. ゲームエリアの描画
        # 土台 (ステージ) の描画
        stage_w = int(last_stage_width)
        pygame.draw.rect(screen, COLOR_STAGE, (300 - stage_w // 2, 470, stage_w, 20), border_radius=5)
        pygame.draw.rect(screen, (0, 96, 100), (300 - stage_w // 2, 490, stage_w, 5), border_radius=2) # 陰影
        
        # 照準ガイド線 (AIMING時のみ)
        if game_state == STATE_AIMING:
            dash_y = obj_y + 40
            while dash_y < 470:
                pygame.draw.line(screen, (60, 70, 85), (obj_x, dash_y), (obj_x, dash_y + 8), 2)
                dash_y += 16

        # 物理空間内の動的ボディを全描画 (積み上がったものも含めて全て描画)
        for body in space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                # 物理座標と角度を取得
                pos = body.position
                # PyMunk(反時計回り正)からPygame(時計回り正)にアングルを変換して回転
                angle_deg = -math.degrees(body.angle)
                
                # アフィン回転
                rotated_image = pygame.transform.rotate(object_image, angle_deg)
                rotated_rect = rotated_image.get_rect(center=(int(pos.x), int(pos.y)))
                
                screen.blit(rotated_image, rotated_rect.topleft)

                # デバッグ用：頂点コライダーをワイヤーフレームで描画 (必要に応じて視覚化)
                # shape = list(body.shapes)[0]
                # pts = [body.position + v.rotated(body.angle) for v in shape.get_vertices()]
                # pygame.draw.polygon(screen, (255, 255, 0), [(p.x, p.y) for p in pts], 1)

        # 落下前に操作中のエイミングオブジェクト描画 (まだ物理空間には追加されていない)
        if game_state == STATE_AIMING:
            rotated_image = pygame.transform.rotate(object_image, obj_angle)
            rotated_rect = rotated_image.get_rect(center=(obj_x, int(obj_y)))
            
            # うっすら発光するグロー効果
            glow_surf = pygame.Surface((rotated_rect.width + 12, rotated_rect.height + 12), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, (0, 229, 255, 30), glow_surf.get_rect())
            screen.blit(glow_surf, glow_surf.get_rect(center=rotated_rect.center))
            
            screen.blit(rotated_image, rotated_rect.topleft)

        # デッドライン（センサー判定ライン）の視覚化 (薄い赤色の破線)
        for dash_x in range(0, GAME_WIDTH, 15):
            pygame.draw.line(screen, (255, 82, 82, 100), (dash_x, 580), (dash_x + 8, 580), 1)

        # 2. サイドパネル (ダッシュボードUI) の描画
        side_panel_rect = pygame.Rect(GAME_WIDTH, 0, SCREEN_WIDTH - GAME_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, COLOR_BG_SIDE, side_panel_rect)
        pygame.draw.line(screen, (50, 60, 75), (GAME_WIDTH, 0), (GAME_WIDTH, SCREEN_HEIGHT), 2)

        # タイトル
        title_text = font_title.render("TOWER", True, COLOR_TEXT)
        title_text2 = font_title.render("BATTLE", True, COLOR_ACCENT)
        screen.blit(title_text, (620, 30))
        screen.blit(title_text2, (620, 70))
        
        # スコア表示
        score_label = font_main.render("OBJECTS PLACED:", True, COLOR_TEXT_MUTED)
        score_val = font_large.render(f"{score}", True, (255, 235, 59))
        screen.blit(score_label, (620, 110))
        screen.blit(score_val, (620, 130))

        # スライダー描画
        for slider in sliders:
            slider.draw(screen, font_main)

        # ステータス表示
        status_label = font_main.render("STATUS:", True, COLOR_TEXT_MUTED)
        if game_state == STATE_AIMING:
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
        screen.blit(controls_title, (620, 440))
        screen.blit(ctrl_left_right, (620, 465))
        screen.blit(ctrl_space, (620, 490))
        screen.blit(ctrl_down, (620, 515))

        # 3. ゲームオーバー時のオーバーレイUIの描画
        if game_state == STATE_GAMEOVER:
            overlay = pygame.Surface((GAME_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 12, 16, 200))
            screen.blit(overlay, (0, 0))
            
            go_text = font_title.render("GAME OVER", True, COLOR_RED)
            restart_text = font_large.render("Press 'R' to Retry", True, COLOR_TEXT)
            
            screen.blit(go_text, (GAME_WIDTH // 2 - go_text.get_width() // 2, 240))
            screen.blit(restart_text, (GAME_WIDTH // 2 - restart_text.get_width() // 2, 300))

        # 画面更新
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
