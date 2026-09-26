"""動画から手のランドマークを検出し、点・骨組み・左右ラベルを描画して output.mp4 に保存する。"""

import cv2

# STEP 1: 必要なモジュールを読み込む
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VisionRunningMode = vision.RunningMode

# STEP 2: HandLandmarker（手の検出器）の設定を作る
# - num_hands=2: 最大2つの手を検出する
# - running_mode=VIDEO: 前のフレームの結果を使って手を追跡する動画モード
#   （このモードでは detect() ではなく detect_for_video() を使う）
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=2, running_mode=VisionRunningMode.VIDEO
)

# STEP 3: 入力動画を開く
cap = cv2.VideoCapture("video.webm")

if not cap.isOpened():
    print("エラー：動画ファイルを開けません")
    exit()

# 入力動画の情報を取得する（全フレーム共通なのでループの前に1回だけ）
# fps はタイムスタンプ計算の誤差を防ぐため小数のまま持つ
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 出力動画の書き出し先を用意する
# - fourcc: コーデックを表す4文字を整数に変換したもの（mp4v は .mp4 用の定番）
# - fps とサイズを入力に合わせることで、同じ速さ・大きさの動画になる
# - サイズは (幅, 高さ) の順。書き込むフレームと一致しないと何も書かれない
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# VideoWriter は失敗してもエラーを出さないことがあるので、開けたかを確認する
if not out.isOpened():
    print(
        "エラー：動画ファイルを開けませんでした。設定（解像度やコーデック）を確認してください"
    )
    cap.release()
    exit()


# STEP 4: 検出器を作り、1フレームずつ検出・描画・書き出しを行う
# 検出器の作成は重いので、with はループの外で1回だけにする
with vision.HandLandmarker.create_from_options(options) as detector:
    # タイムスタンプ計算用のフレーム番号
    frame_num = 0

    while cap.isOpened():
        # 1フレーム読み込む。最後まで読み終えると ret が False になる
        ret, frame = cap.read()

        if not ret:
            print("動画の読み込みが終了しました")
            break

        # OpenCV のフレームは BGR 順、MediaPipe は RGB 順を期待するので変換する
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # numpy 配列を MediaPipe の画像オブジェクトに包み、検出する
        # タイムスタンプ（ミリ秒）は「フレーム番号 × 1000 ÷ fps」で求める
        # 前のフレームより必ず大きい整数である必要がある
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect_for_video(
            mp_image, int(frame_num * 1000 / fps)
        )

        frame_num += 1

        # 描画（検出された手ごとに処理する）
        # handedness[i] と hand_landmarks[i] は同じ手の情報なので zip で組にする
        for handedness, landmarks in zip(
            detection_result.handedness, detection_result.hand_landmarks
        ):
            # 21点のランドマークを点として描き、ピクセル座標を x_y_list にためる
            # 座標は 0〜1 に正規化されているので、幅・高さを掛けてピクセルに直す
            x_y_list = []
            for coordinate in landmarks:
                format_x, format_y = (
                    int(coordinate.x * width),
                    int(coordinate.y * height),
                )
                cv2.circle(frame, (format_x, format_y), 5, (0, 0, 255), thickness=-1)
                x_y_list.append((format_x, format_y))

            # ラベルの位置を決めるため、手を囲む枠の左上（x・y の最小値）を求める
            min_x, min_y = x_y_list[0]
            for x, y in x_y_list:
                min_x = min(min_x, x)
                min_y = min(min_y, y)

            # 左右ラベル（Left / Right）を手の左上に描く
            # handedness[0] は最も確からしい判定結果
            # org は文字の左下の位置。画面外に出ないよう max で下限をつける
            # （y の下限 30 は文字が上端で切れないようにするため）
            cv2.putText(
                frame,
                text=handedness[0].category_name,
                org=(max(min_x - 20, 0), max(min_y - 20, 30)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(0, 0, 255),
                thickness=3,
            )

            # 骨組みの線を描く
            # HAND_CONNECTIONS は「何番と何番の点をつなぐか」の公式定義
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    x_y_list[connection.start],
                    x_y_list[connection.end],
                    (0, 255, 0),
                    thickness=2,
                )

        # 描画済みのフレームを出力動画に書き込む
        out.write(frame)

# 後片付け。VideoWriter を release しないと再生できないファイルになることがある
cap.release()
out.release()
