---
title: Vid2Prompt
emoji: 👀
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
short_description: 'Vid2Prompt is a tool for analyzing AI-generated videos. '
---

# Vid2Prompt - AI Video Prompt Analysis Tool

**Vid2Prompt** is a tool designed to analyze AI-generated videos. It extracts keyframes from the video and uses the CLIP model to generate possible prompt descriptions. It also attempts to infer which AI model (like Sora or Runway Gen-2) was used.

## Features

- Automatically extracts 3–5 keyframes from the video

- Uses CLIP-Interrogator to generate detailed text for each frame

- Aggregates the results to form a comprehensive prompt

- Infers the likely AI model based on visual and textual features

- Simple and intuitive Streamlit interface

## Installation

bash

复制编辑

`git clone https://github.com/yourusername/Vid2Prompt.git cd Vid2Prompt pip install -r requirements.txt`

## Usage

bash

复制编辑

`streamlit run app.py`

Then open the displayed address in your browser (usually [http://localhost:8501](http://localhost:8501)).

## Requirements

- Python 3.8+

- GPU recommended for faster processing

- Minimum 4GB RAM (8GB+ recommended with GPU)

## Tech Stack

- Streamlit

- OpenCV

- CLIP-Interrogator

- PyTorch

- NumPy

---

<details>
<summary>🌏 简体中文</summary>

### Vid2Prompt - AI生成视频提示词分析工具

**Vid2Prompt** 是一个用于分析 AI 生成视频的工具，能够从视频中提取关键帧，并使用 CLIP 模型生成可能的提示词，同时推测视频可能使用的 AI 生成模型（如 Sora、Runway Gen-2 等）。

#### 功能特点

- 自动从视频中提取3~5帧关键图像

- 使用 CLIP-Interrogator 分析每一帧并生成详细文本描述

- 汇总分析结果，生成综合性提示词

- 根据图像特征与文本，推测可能使用的生成模型

- 提供简洁友好的 Streamlit 界面

#### 安装方法

bash

复制编辑

`git clone https://github.com/yourusername/Vid2Prompt.git cd Vid2Prompt pip install -r requirements.txt`

#### 使用方法

bash

复制编辑

`streamlit run app.py`

在浏览器中访问显示的地址（通常是 [http://localhost:8501）。](http://localhost:8501%EF%BC%89%E3%80%82)

#### 系统要求

- Python 3.8+

- 建议使用 GPU 加速

- 至少 4GB 内存（GPU 推荐 8GB 以上）

#### 技术栈

- Streamlit

- OpenCV

- CLIP-Interrogator

- PyTorch

- NumPy

</details>

---

<details>
<summary>🌸 日本語</summary>

### Vid2Prompt - AI生成動画プロンプト解析ツール

**Vid2Prompt** は、AIが生成した動画を解析するツールです。動画からキーフレームを抽出し、CLIPモデルを使ってプロンプトの説明文を生成します。また、使用された可能性のあるAIモデル（例：Sora、Runway Gen-2）も推測します。

#### 特徴

- 動画から3～5枚のキーフレームを自動抽出

- CLIP-Interrogatorで各フレームを分析し、詳細なテキストを生成

- 結果をまとめて総合的なプロンプトを生成

- 画像とテキストの特徴から生成モデルを推測

- 直感的なStreamlitインターフェースを提供

#### インストール手順

bash

复制编辑

`git clone https://github.com/yourusername/Vid2Prompt.git cd Vid2Prompt pip install -r requirements.txt`

#### 使用方法

bash

复制编辑

`streamlit run app.py`

その後、表示されたURL（通常 [http://localhost:8501）をブラウザで開きます。](http://localhost:8501%EF%BC%89%E3%82%92%E3%83%96%E3%83%A9%E3%82%A6%E3%82%B6%E3%81%A7%E9%96%8B%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82)

#### 動作環境

- Python 3.8以上

- CLIPの処理を高速化するためにGPU推奨

- メモリ4GB以上（GPU使用時は8GB以上推奨）

#### 技術スタック

- Streamlit

- OpenCV

- CLIP-Interrogator

- PyTorch

- NumPy

</details>

</details>
