# MediaPipe Hand Landmarker

MediaPipe で画像から手のランドマークを検出し、点・骨組み・左右ラベルを描画して `output.jpg` に保存する。

## モデルとサンプル画像のダウンロード

モデルファイルとサンプル画像はリポジトリに含めていないため、初回に取得する。

```sh
# モデル
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# サンプル画像（MediaPipe 公式サンプルと同じもの）
curl -L -o image.jpg \
  https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
```

モデルは Google の MediaPipe が [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) で提供している。
サンプル画像は MediaPipe 公式サンプルコードで使われているもの（元は Unsplash の写真）。

## ローカルで実行（uv）

```sh
uv sync
uv run python hand_landmarker.py
```

## Docker で実行

```sh
# イメージのビルド
docker build -t mediapipe-hand .

# カレントディレクトリを /app にマウントして実行
docker run --rm -v "$PWD":/app mediapipe-hand python hand_landmarker.py

# コンテナに入って作業する場合
docker run --rm -it -v "$PWD":/app mediapipe-hand
```

## フォーマット

```sh
uvx ruff format .
```
