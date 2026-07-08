#!/usr/bin/env python3
"""
PyTorch模型检查工具 - 用于查看.pt文件结构
"""

import torch
import sys
from pathlib import Path

def inspect_model(model_path):
    """检查PyTorch模型文件的结构"""
    
    print("=" * 80)
    print(f"正在加载模型: {model_path}")
    print("=" * 80)
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 判断是 state_dict 还是完整的 checkpoint
        if isinstance(checkpoint, dict):
            # 检查常见的 checkpoint 格式
            if 'state_dict' in checkpoint:
                print("\n📦 检测到完整 checkpoint (包含 'state_dict' 键)")
                state_dict = checkpoint['state_dict']
                print_keys = ['epoch', 'step', 'best_loss', 'optimizer', 'scheduler']
                for key in print_keys:
                    if key in checkpoint:
                        if key == 'state_dict':
                            continue
                        elif key == 'epoch':
                            print(f"  - {key}: {checkpoint[key]}")
                        elif isinstance(checkpoint[key], (int, float)):
                            print(f"  - {key}: {checkpoint[key]:.4f}" if isinstance(checkpoint[key], float) else f"  - {key}: {checkpoint[key]}")
                        else:
                            print(f"  - {key}: {type(checkpoint[key]).__name__}")
            elif any(k.startswith('encoder') or k.startswith('decoder') or k.startswith('model') 
                     for k in checkpoint.keys() if isinstance(k, str)):
                print("\n📦 检测到 state_dict 格式")
                state_dict = checkpoint
            else:
                print("\n📦 检测到普通字典格式")
                state_dict = checkpoint
        else:
            print(f"\n⚠️ 模型类型: {type(checkpoint)}")
            return
        
        # 统计信息
        total_params = 0
        layer_types = {}
        
        print(f"\n📊 模型包含 {len(state_dict)} 个参数张量")
        print("-" * 80)
        
        # 按模块分组显示
        modules = {}
        for key, value in state_dict.items():
            # 提取模块名（第一部分）
            module = key.split('.')[0] if '.' in key else 'root'
            if module not in modules:
                modules[module] = []
            modules[module].append((key, value))
        
        # 显示每个模块的参数
        for module in sorted(modules.keys()):
            items = modules[module]
            print(f"\n🔹 模块: {module} ({len(items)} 参数)")
            
            for key, value in items:
                num_params = value.numel()
                total_params += num_params
                
                # 统计层类型
                if 'weight' in key:
                    layer_type = 'weight'
                elif 'bias' in key:
                    layer_type = 'bias'
                else:
                    layer_type = 'other'
                
                dtype = str(value.dtype)
                shape = ' x '.join(str(s) for s in value.shape)
                
                print(f"    {key:<60} | shape: [{shape:<30}] | params: {num_params:>10,} | dtype: {dtype}")
        
        print("\n" + "=" * 80)
        print(f"📈 总参数数量: {total_params:,}")
        print(f"📈 约 {total_params / 1e6:.2f}M 参数量")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 默认检查当前目录下的 model.pt
        model_path = "model.pt"
    
    if not Path(model_path).exists():
        print(f"❌ 文件不存在: {model_path}")
        sys.exit(1)
    
    inspect_model(model_path)