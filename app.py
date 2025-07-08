import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import torch

from src.video_processor import extract_frames
from src.prompt_generator import analyze_frames, generate_combined_prompt
from src.model_detector import guess_generation_model

st.set_page_config(page_title="视频Prompt分析器", layout="wide")

st.title("AI生成视频Prompt分析器")
st.write("上传AI生成的视频，分析可能使用的提示词和生成模型")

uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # 保存上传的视频到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    # 显示视频
    st.video(video_path)
    
    with st.spinner('正在处理视频...'):
        # 提取帧
        frames, timestamps = extract_frames(video_path, num_frames=5)
        
        # 分析每一帧
        frame_prompts = analyze_frames(frames)
        
        # 生成综合提示词
        combined_prompt = generate_combined_prompt(frame_prompts)
        
        # 猜测生成模型
        model_guess = guess_generation_model(frames, combined_prompt)
    
    # 显示结果
    st.header("分析结果")
    
    # 显示关键帧
    st.subheader("关键帧")
    cols = st.columns(len(frames))
    for i, (frame, timestamp, prompt) in enumerate(zip(frames, timestamps, frame_prompts)):
        with cols[i]:
            st.image(frame, caption=f"时间: {timestamp:.2f}秒")
            with st.expander("查看描述"):
                st.write(prompt)
    
    # 显示推测的提示词
    st.subheader("推测的提示词")
    st.text_area("综合提示词", combined_prompt, height=150)
    
    # 显示猜测的生成模型
    st.subheader("可能使用的生成模型")
    st.info(f"推测模型: {model_guess['model']}")
    st.write(f"置信度: {model_guess['confidence']:.2f}")
    st.write("模型特征:")
    for feature, score in model_guess['features'].items():
        st.write(f"- {feature}: {score:.2f}")
    
    # 清理临时文件
    os.unlink(video_path) 