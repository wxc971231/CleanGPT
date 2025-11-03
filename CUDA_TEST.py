"""
测试CUDA和cuDNN安装是否正确
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys

# 新增：检测当前GPU架构是否被当前PyTorch构建支持
def _is_arch_supported():
    if not torch.cuda.is_available():
        return False, "CUDA不可用"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported_arch = torch.cuda.get_arch_list()
        return arch in supported_arch, f"当前GPU架构 {arch}，PyTorch构建支持: {supported_arch}"
    except Exception as e:
        return False, f"架构检测异常: {e}"

def test_cuda_availability():
    """测试CUDA是否可用"""
    print("=== CUDA 可用性测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 新增：打印架构支持信息
        ok, msg = _is_arch_supported()
        print(f"架构支持检测: {'支持' if ok else '不支持'} | {msg}")
        return True
    else:
        print("❌ CUDA不可用")
        return False

def test_cudnn():
    """测试cuDNN是否可用"""
    print("\n=== cuDNN 测试 ===")
    print(f"cuDNN是否可用: {cudnn.is_available()}")
    print(f"cuDNN版本: {cudnn.version()}")
    print(f"cuDNN启用状态: {cudnn.enabled}")
    
    if cudnn.is_available():
        print("✅ cuDNN可用")
        return True
    else:
        print("❌ cuDNN不可用")
        return False

def test_gpu_computation():
    """测试GPU计算功能"""
    print("\n=== GPU 计算测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过GPU计算测试")
        return False
    
    ok, msg = _is_arch_supported()
    if not ok:
        print(f"⚠️  跳过GPU计算测试：当前PyTorch未包含所需GPU架构。{msg}")
        print("提示：升级PyTorch到支持当前架构的构建，或从源码编译并设置TORCH_CUDA_ARCH_LIST。")
        return False
    
    try:
        # 创建测试张量
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # 矩阵乘法测试
        z = torch.mm(x, y)
        print(f"✅ GPU矩阵乘法测试成功，结果形状: {z.shape}")
        
        # 神经网络测试
        model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)
        
        input_tensor = torch.randn(32, 1000, device=device)
        output = model(input_tensor)
        print(f"✅ GPU神经网络测试成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
        return False

def test_memory_allocation():
    """测试GPU内存分配"""
    print("\n=== GPU 内存测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存测试")
        return False
    
    ok, msg = _is_arch_supported()
    if not ok:
        print(f"⚠️  跳过内存测试：当前PyTorch未包含所需GPU架构。{msg}")
        print("提示：升级PyTorch到支持当前架构的构建，或从源码编译并设置TORCH_CUDA_ARCH_LIST。")
        return False
    
    try:
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 获取内存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_before = torch.cuda.memory_allocated()
        cached_before = torch.cuda.memory_reserved()
        
        print(f"总内存: {total_memory / 1024**3:.1f} GB")
        print(f"分配前已用内存: {allocated_before / 1024**2:.1f} MB")
        print(f"分配前缓存内存: {cached_before / 1024**2:.1f} MB")
        
        # 分配大张量
        large_tensor = torch.randn(5000, 5000, device='cuda')
        
        allocated_after = torch.cuda.memory_allocated()
        cached_after = torch.cuda.memory_reserved()
        
        print(f"分配后已用内存: {allocated_after / 1024**2:.1f} MB")
        print(f"分配后缓存内存: {cached_after / 1024**2:.1f} MB")
        print(f"新分配内存: {(allocated_after - allocated_before) / 1024**2:.1f} MB")
        
        # 释放内存
        del large_tensor
        torch.cuda.empty_cache()
        
        print("✅ GPU内存分配测试成功")
        return True
        
    except Exception as e:
        print(f"❌ GPU内存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始CUDA和cuDNN安装验证测试...\n")
    
    results = []
    results.append(test_cuda_availability())
    results.append(test_cudnn())
    results.append(test_gpu_computation())
    results.append(test_memory_allocation())
    
    print("\n=== 测试总结 ===")
    if all(results):
        print("🎉 所有测试通过！CUDA和cuDNN安装成功且可正常使用")
        return 0
    else:
        print("⚠️  部分测试失败，请检查安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())