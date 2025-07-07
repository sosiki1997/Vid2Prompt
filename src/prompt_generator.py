import numpy as np
from PIL import Image
from typing import List, Dict, Any
import torch
from clip_interrogator import Config, Interrogator
import re
import string

# 初始化CLIP Interrogator
def get_interrogator():
    """获取CLIP Interrogator实例，使用单例模式避免重复加载"""
    if not hasattr(get_interrogator, "instance"):
        config = Config()
        config.clip_model_name = "ViT-L-14/openai"  # 使用OpenAI的CLIP模型
        config.chunk_size = 2048  # 减小此值可以减少内存使用
        config.blip_num_beams = 8  # 减小此值可以减少内存使用
        config.blip_max_length = 64  # 生成的描述最大长度
        
        # 检查是否有GPU可用
        if torch.cuda.is_available():
            config.device = "cuda"
        else:
            config.device = "cpu"
        
        # 创建Interrogator实例
        get_interrogator.instance = Interrogator(config)
    
    return get_interrogator.instance

def analyze_frame(frame: np.ndarray) -> str:
    """
    分析单帧图像，生成描述
    
    参数:
        frame: 图像数据，RGB格式的numpy数组
        
    返回:
        生成的描述文本
    """
    # 转换numpy数组为PIL图像
    image = Image.fromarray(frame)
    
    # 获取Interrogator实例
    ci = get_interrogator()
    
    # 分析图像
    try:
        # 使用CLIP Interrogator生成描述
        prompt = ci.interrogate(image)
        # 清理和规范化提示词
        prompt = clean_prompt(prompt)
        return prompt
    except Exception as e:
        print(f"分析图像时出错: {str(e)}")
        return "无法分析图像"

def clean_prompt(prompt: str) -> str:
    """
    清理和规范化提示词，移除非英语词汇和不相关内容
    
    参数:
        prompt: 原始提示词
        
    返回:
        清理后的提示词
    """
    # 转换为小写
    prompt = prompt.lower()
    
    # 移除非英语词汇和常见噪声词
    noise_words = {
        "très", "détaillé", "megalophoba", "midlands", "grape", "8 h", "tiktok",
        "megalophobia", "detailed", "highly detailed", "ultra detailed",
        "trending on artstation", "trending", "artstation", "instagram", "pinterest",
        "facebook", "twitter", "deviantart", "unsplash", "flickr", "500px",
        "award winning", "award-winning", "hd", "4k", "8k", "uhd", "hdr",
        "unreal engine", "unity", "blender", "octane render", "cycles", "ray tracing"
    }
    
    # 分割提示词
    parts = [p.strip() for p in prompt.split(",")]
    
    # 过滤掉噪声词和非英语词
    filtered_parts = []
    for part in parts:
        words = part.split()
        if not any(word in noise_words for word in words) and all(c in string.printable for c in part):
            filtered_parts.append(part)
    
    # 重新组合提示词
    cleaned_prompt = ", ".join(filtered_parts)
    
    # 确保提示词不为空
    if not cleaned_prompt:
        # 如果过滤后为空，至少保留第一部分
        cleaned_prompt = parts[0] if parts else "image"
    
    return cleaned_prompt

def analyze_frames(frames: List[np.ndarray]) -> List[str]:
    """
    分析多个视频帧，生成描述列表
    
    参数:
        frames: 图像帧列表
        
    返回:
        每帧对应的描述列表
    """
    prompts = []
    for frame in frames:
        prompt = analyze_frame(frame)
        prompts.append(prompt)
    return prompts

def extract_keywords(prompts: List[str]) -> Dict[str, int]:
    """
    从多个提示词中提取关键词及其频率
    
    参数:
        prompts: 提示词列表
        
    返回:
        关键词及其出现频率的字典
    """
    # 合并所有提示词
    all_text = " ".join(prompts).lower()
    
    # 移除常见的修饰词和连接词
    stop_words = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
        "by", "of", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
        "may", "might", "must", "can", "could"
    }
    
    # 提取所有单词
    words = re.findall(r'\b\w+\b', all_text)
    
    # 统计词频
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:  # 忽略停用词和过短的词
            word_freq[word] = word_freq.get(word, 0) + 1
    
    return word_freq

def generate_combined_prompt(prompts: List[str]) -> str:
    """
    根据多个帧的提示词生成一个综合提示词
    
    参数:
        prompts: 每帧对应的提示词列表
        
    返回:
        综合后的提示词
    """
    if not prompts:
        return ""
    
    # 提取关键词及频率
    keywords = extract_keywords(prompts)
    
    # 按频率排序关键词
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    
    # 提取风格关键词（通常在CLIP-Interrogator输出的末尾）
    style_words = []
    for prompt in prompts:
        # CLIP-Interrogator通常会在描述后添加艺术风格关键词
        parts = prompt.split(", ")
        if len(parts) > 3:
            # 取最后几个部分作为风格词
            style_words.extend(parts[-3:])
    
    # 统计风格词频率
    style_freq = {}
    for word in style_words:
        # 确保只包含英文单词
        if all(c in string.printable for c in word):
            style_freq[word] = style_freq.get(word, 0) + 1
    
    # 按频率排序风格词
    sorted_styles = sorted(style_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 构建综合提示词
    # 1. 从第一帧获取基本场景描述
    base_description = prompts[0].split(",")[0]
    
    # 2. 添加出现频率最高的关键词（最多15个）
    top_keywords = [word for word, freq in sorted_keywords[:15]]
    
    # 3. 添加最常见的风格词（最多5个）
    top_styles = [style for style, freq in sorted_styles[:5] if all(c in string.printable for c in style)]
    
    # 组合最终提示词
    combined = f"{base_description}, {', '.join(top_keywords)}"
    if top_styles:
        combined += f", {', '.join(top_styles)}"
    
    return combined 