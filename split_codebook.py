#!/usr/bin/env python3
"""
将残差VQ码本分层保存为独立文件
"""

import torch
import sys

def split_and_save(model_path, output_dir="."):
    """将模型中的每个码本层级保存为独立文件"""
    
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 找到所有 centroids 层级
    centroids = {}
    for key, value in checkpoint.items():
        if key.startswith('centroids.'):
            layer_idx = key.split('.')[1]
            centroids[int(layer_idx)] = value
    
    num_layers = len(centroids)
    print(f"检测到 {num_layers} 个码本层级\n")
    
    # 保存每个层级
    for layer_idx in sorted(centroids.keys()):
        output_path = f"{output_dir}/centroids_{layer_idx}.pt"
        torch.save(centroids[layer_idx], output_path)
        
        shape = centroids[layer_idx].shape
        num_params = centroids[layer_idx].numel()
        print(f"✅ 已保存: centroids_{layer_idx}.pt")
        print(f"   Shape: {shape}")
        print(f"   参数量: {num_params:,} ({num_params/1e6:.2f}M)")
        print()
    
    print(f"✅ 完成！共保存 {num_layers} 个文件到 {output_dir}/")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    split_and_save(model_path, output_dir)