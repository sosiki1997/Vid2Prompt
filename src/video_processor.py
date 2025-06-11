import cv2
import numpy as np
from typing import List, Tuple
import moviepy.editor as mp

def extract_frames(video_path: str, num_frames: int = 5) -> Tuple[List[np.ndarray], List[float]]:
    """
    从视频中等间隔提取指定数量的帧
    
    参数:
        video_path: 视频文件路径
        num_frames: 要提取的帧数量
        
    返回:
        frames: 提取的帧列表
        timestamps: 每帧对应的时间戳（秒）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # 计算采样间隔
    if num_frames > total_frames:
        num_frames = total_frames
    
    if num_frames <= 1:
        frame_indices = [0]
    else:
        # 确保第一帧和最后一帧被包含，其余帧等间隔分布
        frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # 提取帧
    frames = []
    timestamps = []
    
    for idx in frame_indices:
        # 设置帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # 计算时间戳
            timestamp = idx / fps
            timestamps.append(timestamp)
    
    # 释放视频
    cap.release()
    
    return frames, timestamps

def get_video_info(video_path: str) -> dict:
    """
    获取视频的基本信息
    
    参数:
        video_path: 视频文件路径
        
    返回:
        包含视频信息的字典
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # 获取视频编解码器信息
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    # 尝试获取音频信息
    try:
        video = mp.VideoFileClip(video_path)
        has_audio = video.audio is not None
        audio_fps = video.audio.fps if has_audio else None
        video.close()
    except:
        has_audio = False
        audio_fps = None
    
    cap.release()
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": fourcc_str,
        "has_audio": has_audio,
        "audio_fps": audio_fps
    } 