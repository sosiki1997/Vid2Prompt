#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vid2Prompt 启动脚本
"""

import os
import sys
import subprocess

def main():
    """启动Streamlit应用"""
    print("正在启动Vid2Prompt应用...")
    
    # 检查依赖是否已安装
    try:
        import streamlit
        import torch
        import cv2
        import numpy
        from clip_interrogator import Interrogator
    except ImportError as e:
        print(f"缺少必要依赖: {e}")
        print("请先运行: pip install -r requirements.txt")
        return 1
    
    # 检查CUDA是否可用
    import torch
    if torch.cuda.is_available():
        device = "CUDA"
        device_name = torch.cuda.get_device_name(0)
        print(f"检测到GPU: {device_name}")
    else:
        device = "CPU"
        print("未检测到GPU，将使用CPU运行（处理速度可能较慢）")
    
    print(f"使用设备: {device}")
    print("正在启动界面...")
    
    # 启动Streamlit应用
    streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path])
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 