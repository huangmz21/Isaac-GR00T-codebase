#!/usr/bin/env python3
"""验证8卡训练环境是否就绪"""

import os
import sys

def check_environment():
    """检查训练环境"""
    print("=" * 70)
    print("RoboCasa GR00T 8卡训练环境检查")
    print("=" * 70)

    all_checks_passed = True

    # 1. 检查 PyTorch 和 GPU
    print("\n[1/6] 检查 PyTorch 和 GPU...")
    try:
        import torch
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        print(f"  ✅ CUDA 可用: {torch.cuda.is_available()}")
        print(f"  ✅ GPU 数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() != 8:
            print(f"  ⚠️  警告: 检测到 {torch.cuda.device_count()} 个GPU, 预期 8 个")
    except Exception as e:
        print(f"  ❌ PyTorch 检查失败: {e}")
        all_checks_passed = False

    # 2. 检查 flash-attn
    print("\n[2/6] 检查 flash-attn...")
    try:
        import flash_attn
        print(f"  ✅ flash-attn 版本: {flash_attn.__version__}")
    except Exception as e:
        print(f"  ❌ flash-attn 未安装: {e}")
        all_checks_passed = False

    # 3. 检查数据集注册表
    print("\n[3/6] 检查数据集注册表...")
    try:
        from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
        if "atomic_seen" in DATASET_SOUP_REGISTRY:
            num_tasks = len(DATASET_SOUP_REGISTRY["atomic_seen"])
            print(f"  ✅ atomic_seen 数据集: {num_tasks} 个任务")
            if num_tasks != 18:
                print(f"  ⚠️  警告: 检测到 {num_tasks} 个任务, 预期 18 个")
        else:
            print(f"  ❌ 未找到 atomic_seen 数据集")
            all_checks_passed = False
    except Exception as e:
        print(f"  ❌ 数据集注册表加载失败: {e}")
        all_checks_passed = False

    # 4. 检查数据集路径
    print("\n[4/6] 检查数据集路径...")
    try:
        from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
        missing_paths = []
        for dataset in DATASET_SOUP_REGISTRY.get("atomic_seen", []):
            path = dataset["path"]
            if not os.path.exists(path):
                missing_paths.append(path)

        if not missing_paths:
            print(f"  ✅ 所有 18 个数据集路径存在")
        else:
            print(f"  ❌ {len(missing_paths)} 个数据集路径不存在:")
            for path in missing_paths[:3]:
                print(f"      - {path}")
            if len(missing_paths) > 3:
                print(f"      ... 还有 {len(missing_paths) - 3} 个")
            all_checks_passed = False
    except Exception as e:
        print(f"  ❌ 数据集路径检查失败: {e}")
        all_checks_passed = False

    # 5. 检查 base model
    print("\n[5/6] 检查 base model...")
    base_model_path = "/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"
    if os.path.exists(base_model_path):
        print(f"  ✅ Base model 路径存在")
        config_file = os.path.join(base_model_path, "config.json")
        model_files = [f for f in os.listdir(base_model_path) if f.endswith('.safetensors')]
        print(f"  ✅ 发现 {len(model_files)} 个模型文件")
    else:
        print(f"  ❌ Base model 路径不存在: {base_model_path}")
        all_checks_passed = False

    # 6. 检查训练脚本
    print("\n[6/6] 检查训练脚本...")
    train_script = "/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/train_robocasa_atomic_seen.sh"
    if os.path.exists(train_script):
        print(f"  ✅ 训练脚本存在")
        with open(train_script, 'r') as f:
            content = f.read()
            if "/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T" in content:
                print(f"  ✅ 训练脚本路径配置正确")
            else:
                print(f"  ⚠️  警告: 训练脚本路径可能不正确")
    else:
        print(f"  ❌ 训练脚本不存在: {train_script}")
        all_checks_passed = False

    # 总结
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ 所有检查通过! 环境已就绪,可以开始训练")
        print("\n启动训练命令:")
        print("  cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T")
        print("  bash scripts/train_robocasa_atomic_seen.sh")
    else:
        print("❌ 部分检查失败,请修复上述问题后再启动训练")
    print("=" * 70)

    return all_checks_passed

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
