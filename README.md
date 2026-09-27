# MediaPipe Hand Landmarker

MediaPipe で動画（`video.webm`）から手のランドマークを検出し、点・骨組み・左右ラベルを描画して `output_<入力名>.mp4` に保存する。
検出結果（左右・スコア・21点の正規化座標）は `output_<入力名>.json` に保存する（例: `video.webm` → `output_video.mp4` / `output_video.json`）。

## モデルとサンプル動画のダウンロード

モデルファイルとサンプル動画はリポジトリに含めていないため、初回に取得する。

```sh
# モデル
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# サンプル動画（アルメニア手話 "house blessing"、約6秒）
curl -L -A "Mozilla/5.0" -o video.webm \
  "https://upload.wikimedia.org/wikipedia/commons/9/99/Armenian_Sign_Language_%28ArSL%29_-_%D5%8F%D5%B6%D6%85%D6%80%D5%B0%D5%B6%D5%A5%D6%84_-_house_blessing.webm"
```

モデルは Google の MediaPipe が [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) で提供している。
サンプル動画は Wikimedia Commons の [Armenian Sign Language (ArSL) - house blessing](https://commons.wikimedia.org/wiki/File:Armenian_Sign_Language_(ArSL)_-_%D5%8F%D5%B6%D6%85%D6%80%D5%B0%D5%B6%D5%A5%D6%84_-_house_blessing.webm)（Wikimedia Armenia、[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）。

## ローカルで実行（uv）

macOS では mediapipe 1.0.x の HandLandmarker がクラッシュするため（[google-ai-edge/mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)）、`pyproject.toml` で macOS のみ 0.10.35 に固定している。

```sh
uv sync
uv run python hand_landmarker.py video.webm
```

## Docker で実行

```sh
# イメージのビルド
docker build -t mediapipe-hand .

# カレントディレクトリを /app にマウントして実行（出力ファイルはホスト側に書き出される）
docker run --rm -v "$PWD":/app mediapipe-hand python hand_landmarker.py video.webm

# コンテナに入って作業する場合（画面がないため cv2.imshow は使えない）
docker run --rm -it -v "$PWD":/app mediapipe-hand
```

## フォーマット

```sh
uvx ruff format .
```
