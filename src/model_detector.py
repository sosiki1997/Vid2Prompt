import numpy as np
from typing import List, Dict, Any
import re

# 定义不同AI生成模型的特征
MODEL_FEATURES = {
    "Sora": {
        "description": "OpenAI的视频生成模型",
        "features": {
            "高真实感": 0.9,
            "复杂运动": 0.9,
            "长时间连贯性": 0.85,
            "物理准确性": 0.8,
            "高分辨率": 0.8
        },
        "keywords": ["photorealistic", "realistic", "cinematic", "detailed", "4k", "8k"]
    },
    "Runway Gen-2": {
        "description": "Runway的视频生成模型",
        "features": {
            "艺术风格": 0.8,
            "创意转场": 0.7,
            "风格多样性": 0.85,
            "长时间连贯性": 0.6,
            "分辨率": 0.7
        },
        "keywords": ["stylized", "artistic", "creative", "transition", "animation"]
    },
    "Pika Labs": {
        "description": "Pika Labs的视频生成模型",
        "features": {
            "动画风格": 0.8,
            "卡通效果": 0.75,
            "创意表现": 0.8,
            "短视频": 0.7,
            "风格一致性": 0.75
        },
        "keywords": ["animation", "cartoon", "stylized", "creative", "character"]
    },
    "AnimateDiff": {
        "description": "基于Stable Diffusion的动画生成模型",
        "features": {
            "动画风格": 0.9,
            "二次元": 0.85,
            "角色动画": 0.8,
            "短循环": 0.9,
            "风格一致性": 0.7
        },
        "keywords": ["anime", "animation", "character", "loop", "2d", "cartoon"]
    },
    "Stable Video Diffusion": {
        "description": "Stability AI的视频扩散模型",
        "features": {
            "图像到视频": 0.9,
            "短视频": 0.8,
            "风格一致性": 0.75,
            "有限运动": 0.7,
            "分辨率": 0.65
        },
        "keywords": ["short", "consistent", "image-to-video", "diffusion"]
    }
}

def analyze_video_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    分析视频的特征
    
    参数:
        frames: 视频帧列表
        
    返回:
        视频特征评分字典
    """
    features = {}
    
    # 分析分辨率
    if frames:
        height, width = frames[0].shape[:2]
        if width >= 1920 or height >= 1080:
            features["高分辨率"] = 0.8
        elif width >= 1280 or height >= 720:
            features["高分辨率"] = 0.6
        else:
            features["高分辨率"] = 0.4
    
    # 分析帧间一致性
    if len(frames) > 1:
        frame_diffs = []
        for i in range(len(frames) - 1):
            # 计算相邻帧的差异
            diff = np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float)))
            frame_diffs.append(diff)
        
        avg_diff = np.mean(frame_diffs)
        if avg_diff < 20:  # 低差异表示高一致性
            features["风格一致性"] = 0.9
        elif avg_diff < 40:
            features["风格一致性"] = 0.7
        else:
            features["风格一致性"] = 0.5
    
    # 分析色彩丰富度
    if frames:
        color_variance = np.mean([np.var(frame.reshape(-1, 3), axis=0).mean() for frame in frames])
        if color_variance > 2000:
            features["色彩丰富度"] = 0.8
        elif color_variance > 1000:
            features["色彩丰富度"] = 0.6
        else:
            features["色彩丰富度"] = 0.4
    
    return features

def analyze_prompt_features(prompt: str) -> Dict[str, float]:
    """
    分析提示词中的特征
    
    参数:
        prompt: 提示词
        
    返回:
        提示词特征评分字典
    """
    features = {}
    prompt_lower = prompt.lower()
    
    # 检查是否包含真实感相关词汇
    realism_keywords = ["realistic", "photorealistic", "real", "photograph", "cinematic", "4k", "8k", "hdr"]
    realism_score = sum(1 for kw in realism_keywords if kw in prompt_lower) / len(realism_keywords)
    if realism_score > 0:
        features["高真实感"] = min(realism_score * 1.5, 1.0)
    
    # 检查是否包含动画相关词汇
    animation_keywords = ["anime", "animation", "cartoon", "animated", "stylized", "2d", "character"]
    animation_score = sum(1 for kw in animation_keywords if kw in prompt_lower) / len(animation_keywords)
    if animation_score > 0:
        features["动画风格"] = min(animation_score * 1.5, 1.0)
    
    # 检查是否包含艺术风格词汇
    art_keywords = ["painting", "artwork", "illustration", "digital art", "concept art", "artistic"]
    art_score = sum(1 for kw in art_keywords if kw in prompt_lower) / len(art_keywords)
    if art_score > 0:
        features["艺术风格"] = min(art_score * 1.5, 1.0)
    
    return features

def calculate_model_match(video_features: Dict[str, float], prompt_features: Dict[str, float], 
                         prompt: str) -> Dict[str, Dict[str, Any]]:
    """
    计算视频与各个模型的匹配度
    
    参数:
        video_features: 视频特征
        prompt_features: 提示词特征
        prompt: 原始提示词
        
    返回:
        各模型的匹配分数及详情
    """
    results = {}
    prompt_lower = prompt.lower()
    
    # 合并视频和提示词特征
    combined_features = {**video_features, **prompt_features}
    
    for model_name, model_info in MODEL_FEATURES.items():
        score = 0
        total_weight = 0
        matching_features = {}
        
        # 计算特征匹配度
        for feature, model_score in model_info["features"].items():
            if feature in combined_features:
                feature_score = combined_features[feature]
                # 特征匹配度 = 特征存在度 * 模型特征重要性
                match_score = feature_score * model_score
                score += match_score
                total_weight += model_score
                matching_features[feature] = feature_score
        
        # 检查关键词匹配
        keyword_matches = []
        for keyword in model_info["keywords"]:
            if keyword in prompt_lower:
                keyword_matches.append(keyword)
        
        # 关键词匹配加分
        keyword_score = len(keyword_matches) / len(model_info["keywords"]) if model_info["keywords"] else 0
        score += keyword_score * 0.3  # 关键词匹配占30%的权重
        total_weight += 0.3
        
        # 归一化分数
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0
        
        results[model_name] = {
            "score": final_score,
            "matching_features": matching_features,
            "keyword_matches": keyword_matches
        }
    
    return results

def guess_generation_model(frames: List[np.ndarray], prompt: str) -> Dict[str, Any]:
    """
    根据视频帧和提示词推测可能使用的生成模型
    
    参数:
        frames: 视频帧列表
        prompt: 推测的提示词
        
    返回:
        包含推测结果的字典
    """
    # 分析视频特征
    video_features = analyze_video_features(frames)
    
    # 分析提示词特征
    prompt_features = analyze_prompt_features(prompt)
    
    # 计算各模型匹配度
    model_matches = calculate_model_match(video_features, prompt_features, prompt)
    
    # 找出最匹配的模型
    best_model = max(model_matches.items(), key=lambda x: x[1]["score"])
    model_name = best_model[0]
    match_info = best_model[1]
    
    # 构建结果
    result = {
        "model": model_name,
        "confidence": match_info["score"],
        "description": MODEL_FEATURES[model_name]["description"],
        "features": match_info["matching_features"],
        "keyword_matches": match_info["keyword_matches"]
    }
    
    return result 