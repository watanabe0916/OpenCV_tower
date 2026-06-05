import os
import sys
import time
import logging
import subprocess
import cv2
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ManualADBCamera:
    def __init__(self):
        # 監視対象の主要フォルダリスト
        self.camera_dirs = [
            "/sdcard/DCIM/Camera/",
            "/sdcard/DCIM/",
            "/storage/emulated/0/DCIM/Camera/",
            "/storage/emulated/0/DCIM/",
            "/sdcard/Pictures/",
        ]
        self.active_dirs = []
        self._check_adb()
        self._detect_active_dirs()

    def _check_adb(self):
        """
        ADB接続と認可状態の確認
        """
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            devices = [line.split()[0] for line in lines[1:] if "device" in line and "unauthorized" not in line]
            
            if not devices:
                logging.error("認可されたAndroid端末が見つかりません。USBデバッグが許可されているか確認してください。")
                sys.exit(1)
            logging.info(f"Androidデバイス接続完了: {devices[0]}")
        except Exception as e:
            logging.error(f"adbコマンド実行失敗。PATH設定を確認してください: {e}")
            sys.exit(1)

    def _detect_active_dirs(self):
        """
        デバイス上に存在する有効なディレクトリを特定
        """
        for directory in self.camera_dirs:
            result = subprocess.run(
                ["adb", "shell", f"[ -d {directory} ] && echo 'exists'"],
                capture_output=True, text=True
            )
            if "exists" in result.stdout:
                self.active_dirs.append(directory)
        
        if not self.active_dirs:
            self.active_dirs = ["/sdcard/DCIM/Camera/"]
            
        logging.info(f"監視対象ディレクトリ: {self.active_dirs}")

    def get_all_latest_files(self) -> dict:
        """
        各監視ディレクトリの最新ファイル名を取得し、辞書で返す
        """
        latest_dict = {}
        for directory in self.active_dirs:
            result = subprocess.run(
                ["adb", "shell", f"ls -t {directory} | head -n 1"],
                capture_output=True, text=True
            )
            filename = result.stdout.strip().replace('\r', '').replace('\n', '')
            if filename and not filename.endswith('/') and '.' in filename:
                latest_dict[directory] = filename
        return latest_dict

    def pull_and_delete_image(self, remote_path: str, local_path: str):
        """
        AndroidからPCへ転送し、PNGに変換後、スマホ側のオリジナルファイルを削除します。
        """
        ext = os.path.splitext(remote_path)[1]
        temp_local = "temp_captured" + ext
        
        logging.info(f"画像をPCへ転送中: {remote_path}")
        try:
            subprocess.run(["adb", "pull", remote_path, temp_local], check=True, stdout=subprocess.DEVNULL)
            img = cv2.imread(temp_local)
            if img is None:
                raise ValueError("転送された画像ファイルが正常に読み込めません。")
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            cv2.imwrite(local_path, img)
            logging.info(f"PCに保存完了: {local_path}")
            
            # スマホ側のファイルを即時削除
            logging.info(f"スマホ側から削除中: {remote_path}")
            subprocess.run(["adb", "shell", "rm", remote_path], check=True)
            logging.info("スマホ側のファイルを正常に削除しました。")
            
        finally:
            if os.path.exists(temp_local):
                os.remove(temp_local)


def main():
    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)
    
    bg_path = os.path.join(save_dir, "background.png")
    obj_path = os.path.join(save_dir, "object.png")
    
    camera = ManualADBCamera()
    
    # 状態管理定数
    STATE_WAITING_BG = 0
    STATE_WAITING_OBJ = 1
    STATE_CAPTURED_BOTH = 2
    
    state = STATE_WAITING_BG
    
    # 前回のファイルを念のためクリア
    if os.path.exists(bg_path): os.remove(bg_path)
    if os.path.exists(obj_path): os.remove(obj_path)
    
    # 撮影前のフォルダの初期状態を取得
    before_state = camera.get_all_latest_files()
    
    print("\n=======================================================")
    print("【操作方法】")
    print("  1. スマホでカメラアプリを起動し、構図を合わせます。")
    print("  2. スマホでシャッターを押して撮影してください:")
    print("     - 1枚目: 背景のみの画像を撮影")
    print("     - 2枚目: 物体を置いた画像を撮影")
    print("  3. 両方撮影後、PC側でキーを押します:")
    print("     - E キー : 2枚の画像で確定して終了 (物体抽出へ)")
    print("     - R キー : 画像をクリアし、最初から撮り直し")
    print("     - Q キー : プログラムの終了 (中断)")
    print("  ※ 撮影された画像はPC転送後、スマホから自動的に削除されます。")
    print("=======================================================")
    
    # ダミー画像の作成 (案内表示用)
    def make_instruction_image(msg):
        img = np.zeros((480, 640, 3), np.uint8)
        # 暗い藍色の背景
        img[:] = (35, 25, 25)
        # テキストの描画
        cv2.putText(img, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Press Q to Quit", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return img

    cv2.imshow("Captured Photo Preview", make_instruction_image("1. Shoot BACKGROUND on phone..."))
    
    try:
        while True:
            # OpenCVのイベント処理 (0.1秒スリープ)
            key = cv2.waitKey(100) & 0xFF
            
            # Qキーで終了 (いつでも可能)
            if key == ord('q') or key == ord('Q'):
                print("\nプログラムを終了します。")
                break
                
            # Rキーでリセット (撮り直し)
            elif key == ord('r') or key == ord('R'):
                print("\n画像をクリアし、最初から撮り直します。")
                if os.path.exists(bg_path): os.remove(bg_path)
                if os.path.exists(obj_path): os.remove(obj_path)
                state = STATE_WAITING_BG
                before_state = camera.get_all_latest_files()
                cv2.imshow("Captured Photo Preview", make_instruction_image("1. Shoot BACKGROUND on phone..."))
                
            # Eキーで確定して次のステップへ (両方撮影済みの場合のみ)
            elif key == ord('e') or key == ord('E'):
                if state == STATE_CAPTURED_BOTH:
                    print("\n2枚の画像を確定しました。撮影ステップを完了します。")
                    sys.exit(0) # 正常終了
                else:
                    print("\n警告: まだ撮影が完了していません。Eキーによる確定はできません。")
            
            # スマホ撮影の自動監視
            if state == STATE_WAITING_BG:
                current_state = camera.get_all_latest_files()
                new_file = None
                for directory, current_file in current_state.items():
                    if directory not in before_state or before_state[directory] != current_file:
                        new_file = os.path.join(directory, current_file)
                        break
                
                if new_file:
                    try:
                        print(f"\n【検知】背景画像の撮影を確認しました。PCへ転送中...")
                        camera.pull_and_delete_image(new_file, bg_path)
                        state = STATE_WAITING_OBJ
                        
                        # プレビューの更新
                        img = cv2.imread(bg_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            resized = cv2.resize(img, (640, int(640 * h / w)))
                            cv2.putText(resized, "BACKGROUND OK. Next: Shoot OBJECT on phone...", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.imshow("Captured Photo Preview", resized)
                            
                        # 物体撮影に向けて撮影前の状態を再取得
                        before_state = camera.get_all_latest_files()
                    except Exception as e:
                        logging.error(f"背景画像の処理中にエラーが発生しました: {e}")
                        before_state = camera.get_all_latest_files() # 状態を元に戻す
                        
            elif state == STATE_WAITING_OBJ:
                current_state = camera.get_all_latest_files()
                new_file = None
                for directory, current_file in current_state.items():
                    if directory not in before_state or before_state[directory] != current_file:
                        new_file = os.path.join(directory, current_file)
                        break
                        
                if new_file:
                    try:
                        print(f"\n【検知】物体画像の撮影を確認しました。PCへ転送中...")
                        camera.pull_and_delete_image(new_file, obj_path)
                        state = STATE_CAPTURED_BOTH
                        
                        # プレビューの更新
                        img = cv2.imread(obj_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            resized = cv2.resize(img, (640, int(640 * h / w)))
                            cv2.putText(resized, "BOTH CAPTURED. Press E to Confirm, R to Retry", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.imshow("Captured Photo Preview", resized)
                    except Exception as e:
                        logging.error(f"物体画像の処理中にエラーが発生しました: {e}")
                        before_state = camera.get_all_latest_files()
                        
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
