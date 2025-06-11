# Vid2Prompt - AI生成视频提示词分析工具

Vid2Prompt是一个用于分析AI生成视频的工具，能够从视频中提取关键帧，并使用CLIP模型生成可能的提示词，同时推测视频可能使用的AI生成模型。

## 功能特点

- 自动从视频中等间隔提取3-5帧关键图像
- 使用CLIP-Interrogator分析每一帧，生成详细的文本描述
- 汇总分析结果，生成综合性的提示词
- 根据图像特征和提示词，推测可能使用的生成模型（如Sora、Runway Gen-2等）
- 提供友好的Streamlit界面，展示分析结果

## 安装方法

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/Vid2Prompt.git
cd Vid2Prompt
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

启动Streamlit应用：
```bash
streamlit run app.py
```

然后在浏览器中访问显示的地址（通常是 http://localhost:8501）。

## 使用流程

1. 上传AI生成的视频文件（支持mp4、avi、mov、mkv格式）
2. 系统自动分析视频，提取关键帧
3. 使用CLIP-Interrogator生成每一帧的描述
4. 汇总分析结果，生成综合提示词
5. 推测可能使用的AI生成模型

## 系统要求

- Python 3.8+
- 推荐使用GPU以加速CLIP模型处理
- 至少4GB内存（使用GPU时推荐8GB+）

## 技术栈

- Streamlit：用户界面
- OpenCV：视频处理
- CLIP-Interrogator：图像分析和提示词生成
- PyTorch：深度学习框架
- NumPy：数据处理

## 注意事项

- 首次运行时会下载CLIP和BLIP模型，可能需要一些时间
- 分析结果仅供参考，不能保证100%准确
- 对于复杂视频，推测的提示词可能不完全匹配原始提示词 