#!/usr/bin/env python3
"""
BERT模型CUDA Kernel Profiling脚本（增强版）
用于分析Transformer模型中的热点算子
包含详细的kernel信息、内存分析和算子映射
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import json
import os
from collections import defaultdict
import re

# 创建一个简单的Transformer Block用于测试
class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        
        return x


def profile_transformer(batch_size=8, seq_len=128, hidden_size=768, use_real_bert=False):
    """
    Profiling Transformer模型
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        use_real_bert: 是否使用真实的BERT模型（需要transformers库）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if use_real_bert:
        try:
            from transformers import BertModel
            model = BertModel.from_pretrained('bert-base-uncased').to(device)
            print("使用BERT-base模型")
        except:
            print("未安装transformers库，使用简化模型")
            model = SimpleTransformerBlock(hidden_size).to(device)
    else:
        model = SimpleTransformerBlock(hidden_size).to(device)
    
    model.eval()
    
    # 创建输入数据
    if use_real_bert:
        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
        inputs = {'input_ids': input_ids}
    else:
        inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 预热
    print("预热GPU...")
    with torch.no_grad():
        for _ in range(10):
            if use_real_bert:
                _ = model(**inputs)
            else:
                _ = model(inputs)
    torch.cuda.synchronize()
    
    # Profiling
    print(f"\n开始Profiling (batch_size={batch_size}, seq_len={seq_len})...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        with record_function("transformer_forward"):
            with torch.no_grad():
                for _ in range(20):  # 多次运行获得更稳定的统计
                    if use_real_bert:
                        _ = model(**inputs)
                    else:
                        _ = model(inputs)
    
    torch.cuda.synchronize()
    
    # 打印结果
    print("\n" + "="*100)
    print("CUDA Kernel 热点分析 (按CUDA时间排序)")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=80
    ))
    
    print("\n" + "="*100)
    print("CUDA Kernel 调用次数统计")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
        max_name_column_width=80
    ))
    
    # 导出Chrome trace用于可视化
    trace_file = f"bert_trace_bs{batch_size}_seq{seq_len}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace已导出到: {trace_file}")
    print("在Chrome浏览器中打开 chrome://tracing 并加载该文件进行可视化分析")
    
    # ============ 增强的详细分析 ============
    analysis_results = analyze_profiling_results(prof, batch_size, seq_len, hidden_size)
    
    # 打印详细分析
    print_detailed_analysis(analysis_results)
    
    # 保存详细统计数据
    stats_file = f"profiling_stats_bs{batch_size}_seq{seq_len}.json"
    with open(stats_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"\n详细统计数据已保存到: {stats_file}")
    
    # 生成算子调研报告
    report_file = f"kernel_analysis_report_bs{batch_size}_seq{seq_len}.md"
    generate_analysis_report(analysis_results, report_file)
    print(f"算子分析报告已保存到: {report_file}")
    
    return prof


def analyze_profiling_results(prof, batch_size, seq_len, hidden_size):
    """
    详细分析profiling结果
    
    Returns:
        dict: 包含详细分析结果的字典
    """
    all_ops = []
    cuda_kernels = []
    aten_ops = {}
    op_categories = defaultdict(list)
    total_cuda_time = 0
    total_cpu_time = 0
    
    # 遍历所有事件
    for evt in prof.key_averages():
        name = evt.key
        cuda_time_ms = evt.cuda_time_total / 1000.0
        cpu_time_ms = evt.cpu_time_total / 1000.0
        
        total_cuda_time += cuda_time_ms
        total_cpu_time += cpu_time_ms
        
        # 基本信息
        op_info = {
            'name': name,
            'cuda_time_total_ms': cuda_time_ms,
            'cuda_time_avg_ms': evt.cuda_time / 1000.0 if evt.count > 0 else 0,
            'cpu_time_total_ms': cpu_time_ms,
            'cpu_time_avg_ms': evt.cpu_time / 1000.0 if evt.count > 0 else 0,
            'count': evt.count,
            'self_cuda_time_ms': evt.self_cuda_time_total / 1000.0,
            'self_cpu_time_ms': evt.self_cpu_time_total / 1000.0,
        }
        
        # 添加内存信息（如果有）
        if hasattr(evt, 'cuda_memory_usage'):
            op_info['cuda_memory_usage'] = evt.cuda_memory_usage
        
        # 添加shape信息（如果有）
        if hasattr(evt, 'input_shapes') and evt.input_shapes:
            op_info['input_shapes'] = str(evt.input_shapes)
        
        # 添加FLOPs信息（如果有）
        if hasattr(evt, 'flops') and evt.flops > 0:
            op_info['flops'] = evt.flops
        
        all_ops.append(op_info)
        
        # 分类：CUDA kernel vs ATen算子
        if 'void ' in name or '::' in name and 'aten::' not in name:
            cuda_kernels.append(op_info)
        elif 'aten::' in name:
            aten_ops[name] = op_info
            # 按算子类型分类
            category = categorize_operator(name)
            op_categories[category].append(op_info)
    
    # 按时间排序
    all_ops.sort(key=lambda x: x['cuda_time_total_ms'], reverse=True)
    cuda_kernels.sort(key=lambda x: x['cuda_time_total_ms'], reverse=True)
    
    # 统计各类别的占比
    category_stats = {}
    for category, ops in op_categories.items():
        total_time = sum(op['cuda_time_total_ms'] for op in ops)
        category_stats[category] = {
            'total_time_ms': total_time,
            'percentage': (total_time / total_cuda_time * 100) if total_cuda_time > 0 else 0,
            'op_count': len(ops),
            'call_count': sum(op['count'] for op in ops)
        }
    
    # 识别Top算子并映射到native_functions
    top_aten_ops = sorted(aten_ops.values(), key=lambda x: x['cuda_time_total_ms'], reverse=True)[:10]
    top_ops_with_mapping = []
    
    for op in top_aten_ops:
        op_mapping = map_to_native_function(op['name'])
        top_ops_with_mapping.append({
            **op,
            'native_function': op_mapping['function'],
            'potential_cuda_file': op_mapping['cuda_file'],
            'category': categorize_operator(op['name'])
        })
    
    return {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        },
        'summary': {
            'total_cuda_time_ms': total_cuda_time,
            'total_cpu_time_ms': total_cpu_time,
            'total_operators': len(all_ops),
            'aten_operators': len(aten_ops),
            'cuda_kernels': len(cuda_kernels)
        },
        'top_operators': all_ops[:30],
        'top_aten_operators': top_ops_with_mapping,
        'top_cuda_kernels': cuda_kernels[:20],
        'category_statistics': category_stats,
        'detailed_category_breakdown': {
            category: sorted(ops, key=lambda x: x['cuda_time_total_ms'], reverse=True)[:5]
            for category, ops in op_categories.items()
        }
    }


def categorize_operator(op_name):
    """将算子分类"""
    name_lower = op_name.lower()
    
    categories = {
        'Matrix Operations': ['mm', 'matmul', 'bmm', 'addmm', 'baddbmm', 'gemm'],
        'Attention': ['attention', 'scaled_dot_product'],
        'Normalization': ['layer_norm', 'layernorm', 'batch_norm', 'group_norm'],
        'Activation': ['gelu', 'relu', 'silu', 'sigmoid', 'tanh', 'softmax'],
        'Embedding': ['embedding', 'gather'],
        'Elementwise': ['add', 'mul', 'div', 'sub', 'pow'],
        'Reduction': ['sum', 'mean', 'max', 'min'],
        'Memory': ['copy', 'clone', 'contiguous', 'view', 'reshape', 'transpose'],
        'Other': []
    }
    
    for category, keywords in categories.items():
        if any(keyword in name_lower for keyword in keywords):
            return category
    
    return 'Other'


def map_to_native_function(op_name):
    """
    将ATen算子映射到native_functions.yaml中的函数和CUDA实现
    """
    # 提取算子名称（去掉aten::前缀）
    if 'aten::' in op_name:
        func_name = op_name.split('aten::')[1].split('.')[0].split('(')[0]
    else:
        func_name = op_name.split('::')[-1].split('.')[0].split('(')[0]
    
    # 常见算子的CUDA实现文件映射
    cuda_file_mapping = {
        'softmax': 'aten/src/ATen/native/cuda/SoftMax.cu',
        'layer_norm': 'aten/src/ATen/native/cuda/layer_norm_kernel.cu',
        'batch_norm': 'aten/src/ATen/native/cuda/Normalization.cu',
        'addmm': 'aten/src/ATen/native/cuda/Blas.cpp',
        'mm': 'aten/src/ATen/native/cuda/Blas.cpp',
        'bmm': 'aten/src/ATen/native/cuda/Blas.cpp',
        'matmul': 'aten/src/ATen/native/cuda/Blas.cpp',
        'gelu': 'aten/src/ATen/native/cuda/Activation.cu',
        'relu': 'aten/src/ATen/native/cuda/Activation.cu',
        'silu': 'aten/src/ATen/native/cuda/Activation.cu',
        'embedding': 'aten/src/ATen/native/cuda/Embedding.cu',
        'dropout': 'aten/src/ATen/native/cuda/Dropout.cu',
        'linear': 'aten/src/ATen/native/cuda/Linear.cu',
        'add': 'aten/src/ATen/native/cuda/BinaryOps.cu',
        'mul': 'aten/src/ATen/native/cuda/BinaryOps.cu',
    }
    
    cuda_file = cuda_file_mapping.get(func_name.lower(), f'aten/src/ATen/native/cuda/{func_name}.cu')
    
    return {
        'function': func_name,
        'cuda_file': cuda_file,
        'native_function_entry': f'{func_name} in native_functions.yaml'
    }


def print_detailed_analysis(results):
    """打印详细的分析结果"""
    print("\n" + "="*100)
    print("📊 PROFILING 详细分析报告")
    print("="*100)
    
    # 1. 配置信息
    print("\n1️⃣ 配置信息:")
    print("-" * 100)
    config = results['config']
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Sequence Length: {config['seq_len']}")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Device: {config['device']}")
    print(f"  CUDA Version: {config['cuda_version']}")
    print(f"  PyTorch Version: {config['pytorch_version']}")
    
    # 2. 总体统计
    print("\n2️⃣ 总体统计:")
    print("-" * 100)
    summary = results['summary']
    print(f"  总CUDA时间: {summary['total_cuda_time_ms']:.2f} ms")
    print(f"  总CPU时间: {summary['total_cpu_time_ms']:.2f} ms")
    print(f"  ATen算子数量: {summary['aten_operators']}")
    print(f"  CUDA Kernel数量: {summary['cuda_kernels']}")
    print(f"  总算子数量: {summary['total_operators']}")
    
    # 3. 算子类别占比
    print("\n3️⃣ 算子类别占比:")
    print("-" * 100)
    print(f"{'类别':<25} {'总时间(ms)':<15} {'占比(%)':<10} {'算子数':<10} {'调用次数'}")
    print("-" * 100)
    for category, stats in sorted(results['category_statistics'].items(), 
                                   key=lambda x: x[1]['total_time_ms'], reverse=True):
        print(f"{category:<25} {stats['total_time_ms']:<15.2f} {stats['percentage']:<10.1f} "
              f"{stats['op_count']:<10} {stats['call_count']}")
    
    # 4. Top 10 ATen算子（带映射信息）
    print("\n4️⃣ Top 10 ATen算子（含native_functions映射）:")
    print("-" * 100)
    print(f"{'算子名称':<40} {'总时间(ms)':<12} {'调用次数':<10} {'类别':<20}")
    print(f"{'Native函数':<40} {'CUDA文件':<50}")
    print("-" * 100)
    
    for op in results['top_aten_operators']:
        print(f"{op['name']:<40} {op['cuda_time_total_ms']:<12.3f} {op['count']:<10} {op['category']:<20}")
        print(f"  └─ {op['native_function']:<38} {op['potential_cuda_file']:<50}")
        print()
    
    # 5. Top 15 CUDA Kernels
    print("\n5️⃣ Top 15 底层CUDA Kernels:")
    print("-" * 100)
    print(f"{'Kernel名称':<80} {'总时间(ms)':<12} {'调用次数'}")
    print("-" * 100)
    for kernel in results['top_cuda_kernels'][:15]:
        # 截断过长的kernel名称
        kernel_name = kernel['name']
        if len(kernel_name) > 80:
            kernel_name = kernel_name[:77] + "..."
        print(f"{kernel_name:<80} {kernel['cuda_time_total_ms']:<12.3f} {kernel['count']}")
    
    # 6. 各类别Top算子
    print("\n6️⃣ 各类别Top算子详情:")
    print("-" * 100)
    for category, ops in results['detailed_category_breakdown'].items():
        if not ops:
            continue
        print(f"\n【{category}】")
        for i, op in enumerate(ops[:3], 1):
            print(f"  {i}. {op['name']}")
            print(f"     时间: {op['cuda_time_total_ms']:.3f}ms (平均: {op['cuda_time_avg_ms']:.4f}ms)")
            print(f"     调用: {op['count']}次")


def generate_analysis_report(results, output_file):
    """生成Markdown格式的算子分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# BERT模型CUDA算子Profiling分析报告\n\n")
        
        # 配置信息
        f.write("## 1. 实验配置\n\n")
        config = results['config']
        f.write(f"- **Batch Size**: {config['batch_size']}\n")
        f.write(f"- **Sequence Length**: {config['seq_len']}\n")
        f.write(f"- **Hidden Size**: {config['hidden_size']}\n")
        f.write(f"- **GPU设备**: {config['device']}\n")
        f.write(f"- **CUDA版本**: {config['cuda_version']}\n")
        f.write(f"- **PyTorch版本**: {config['pytorch_version']}\n\n")
        
        # 总体统计
        f.write("## 2. 性能总览\n\n")
        summary = results['summary']
        f.write(f"- **总CUDA时间**: {summary['total_cuda_time_ms']:.2f} ms\n")
        f.write(f"- **总CPU时间**: {summary['total_cpu_time_ms']:.2f} ms\n")
        f.write(f"- **ATen算子数量**: {summary['aten_operators']}\n")
        f.write(f"- **CUDA Kernel数量**: {summary['cuda_kernels']}\n\n")
        
        # 算子类别统计
        f.write("## 3. 算子类别分布\n\n")
        f.write("| 类别 | 总时间(ms) | 占比(%) | 算子数 | 调用次数 |\n")
        f.write("|------|-----------|---------|--------|----------|\n")
        for category, stats in sorted(results['category_statistics'].items(),
                                       key=lambda x: x[1]['total_time_ms'], reverse=True):
            f.write(f"| {category} | {stats['total_time_ms']:.2f} | "
                   f"{stats['percentage']:.1f} | {stats['op_count']} | {stats['call_count']} |\n")
        f.write("\n")
        
        # Top算子详细信息
        f.write("## 4. Top 10 关键算子详细分析\n\n")
        for i, op in enumerate(results['top_aten_operators'], 1):
            f.write(f"### 4.{i} {op['name']}\n\n")
            f.write(f"**性能指标:**\n")
            f.write(f"- 总CUDA时间: {op['cuda_time_total_ms']:.3f} ms\n")
            f.write(f"- 平均CUDA时间: {op['cuda_time_avg_ms']:.4f} ms\n")
            f.write(f"- 调用次数: {op['count']}\n")
            f.write(f"- 算子类别: {op['category']}\n\n")
            
            f.write(f"**源码信息:**\n")
            f.write(f"- Native函数: `{op['native_function']}`\n")
            f.write(f"- CUDA实现文件: `{op['potential_cuda_file']}`\n")
            f.write(f"- native_functions.yaml声明: `{op['native_function']}`\n\n")
            
            f.write(f"**调研要点:**\n")
            category = op['category']
            if category == 'Matrix Operations':
                f.write("- 分析cuBLAS库调用\n")
                f.write("- 研究矩阵分块策略\n")
                f.write("- 考察shared memory使用\n")
                f.write("- 并行维度: block/thread tiling\n")
            elif category == 'Activation' and 'softmax' in op['name'].lower():
                f.write("- 分析warp-level reduction\n")
                f.write("- 研究数值稳定性处理（max subtraction）\n")
                f.write("- 考察online softmax算法\n")
                f.write("- 并行维度: 每个warp处理一行\n")
            elif category == 'Normalization':
                f.write("- 分析Welford算法实现\n")
                f.write("- 研究两阶段归约（均值和方差）\n")
                f.write("- 考察数值稳定性\n")
                f.write("- 并行维度: 沿normalization维度并行\n")
            elif 'gelu' in op['name'].lower():
                f.write("- 分析GELU近似方法（tanh vs erf）\n")
                f.write("- 研究向量化实现\n")
                f.write("- 考察memory coalescing\n")
                f.write("- 并行维度: elementwise并行\n")
            
            f.write("\n")
        
        # 优化建议
        f.write("## 5. 性能优化建议\n\n")
        f.write("### 5.1 优先级1 - 高影响算子\n\n")
        
        top3 = results['top_aten_operators'][:3]
        for i, op in enumerate(top3, 1):
            percentage = (op['cuda_time_total_ms'] / results['summary']['total_cuda_time_ms']) * 100
            f.write(f"{i}. **{op['native_function']}** (占总时间{percentage:.1f}%)\n")
            f.write(f"   - 当前耗时: {op['cuda_time_total_ms']:.2f}ms\n")
            f.write(f"   - 优化目标: 减少10-30%执行时间\n\n")
        
        f.write("### 5.2 算子融合机会\n\n")
        f.write("- Fused Attention (QKV projection + Attention)\n")
        f.write("- Fused FFN (Linear + GELU + Linear)\n")
        f.write("- Fused LayerNorm + Linear\n\n")
        
        # 下一步行动
        f.write("## 6. 下一步行动计划\n\n")
        f.write("- [ ] 深入分析Top 3算子的CUDA实现源码\n")
        f.write("- [ ] 使用Nsight Compute进行kernel级别的详细分析\n")
        f.write("- [ ] 实现优化版本的关键算子\n")
        f.write("- [ ] 进行性能对比测试\n")
        f.write("- [ ] 撰写详细的算子分析报告\n\n")
    
    print(f"✅ 分析报告已生成: {output_file}")


def benchmark_kernels():
    """
    单独benchmark关键算子，测试不同输入大小
    """
    print("\n" + "="*100)
    print("🔬 单独Benchmark关键算子")
    print("="*100)
    
    device = torch.device('cuda')
    
    # 测试配置
    configs = [
        {'batch_size': 8, 'seq_len': 128, 'hidden_size': 768},
        {'batch_size': 16, 'seq_len': 128, 'hidden_size': 768},
        {'batch_size': 8, 'seq_len': 256, 'hidden_size': 768},
    ]
    
    all_benchmark_results = []
    
    for config in configs:
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        hidden_size = config['hidden_size']
        
        print(f"\n{'='*80}")
        print(f"配置: Batch={batch_size}, SeqLen={seq_len}, Hidden={hidden_size}")
        print(f"{'='*80}")
        
        benchmark_results = {
            'config': config,
            'kernels': {}
        }
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # 1. Softmax (Attention scores)
        print("\n1️⃣  Softmax (Attention)")
        x = torch.randn(batch_size, 12, seq_len, seq_len, device=device)
        torch.cuda.synchronize()
        
        for _ in range(10):
            _ = torch.softmax(x, dim=-1)
        
        start.record()
        for _ in range(100):
            _ = torch.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        
        softmax_time = start.elapsed_time(end) / 100
        print(f"   平均时间: {softmax_time:.4f} ms")
        print(f"   输入形状: {x.shape}")
        print(f"   内存带宽: {x.numel() * x.element_size() * 2 / (softmax_time / 1000) / 1e9:.2f} GB/s")
        benchmark_results['kernels']['softmax'] = softmax_time
        
        # 2. LayerNorm
        print("\n2️⃣  LayerNorm")
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        layer_norm = torch.nn.LayerNorm(hidden_size).to(device)
        
        for _ in range(10):
            _ = layer_norm(x)
        
        start.record()
        for _ in range(100):
            _ = layer_norm(x)
        end.record()
        torch.cuda.synchronize()
        
        ln_time = start.elapsed_time(end) / 100
        print(f"   平均时间: {ln_time:.4f} ms")
        print(f"   输入形状: {x.shape}")
        print(f"   内存带宽: {x.numel() * x.element_size() * 2 / (ln_time / 1000) / 1e9:.2f} GB/s")
        benchmark_results['kernels']['layernorm'] = ln_time
        
        # 3. MatMul (GEMM) - QKV projection
        print("\n3️⃣  MatMul/GEMM (Linear层)")
        A = torch.randn(batch_size * seq_len, hidden_size, device=device)
        B = torch.randn(hidden_size, hidden_size, device=device)
        
        for _ in range(10):
            _ = torch.matmul(A, B)
        
        start.record()
        for _ in range(100):
            _ = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        
        mm_time = start.elapsed_time(end) / 100
        flops = 2 * A.shape[0] * A.shape[1] * B.shape[1]
        print(f"   平均时间: {mm_time:.4f} ms")
        print(f"   输入形状: A={A.shape}, B={B.shape}")
        print(f"   FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"   吞吐量: {flops / (mm_time / 1000) / 1e12:.2f} TFLOPs/s")
        benchmark_results['kernels']['matmul'] = mm_time
        
        # 4. BMM (Batch MatMul) - Attention QK^T
        print("\n4️⃣  BMM (Attention QK^T)")
        Q = torch.randn(batch_size * 12, seq_len, 64, device=device)
        K = torch.randn(batch_size * 12, seq_len, 64, device=device)
        
        for _ in range(10):
            _ = torch.bmm(Q, K.transpose(1, 2))
        
        start.record()
        for _ in range(100):
            _ = torch.bmm(Q, K.transpose(1, 2))
        end.record()
        torch.cuda.synchronize()
        
        bmm_time = start.elapsed_time(end) / 100
        bmm_flops = 2 * Q.shape[0] * Q.shape[1] * Q.shape[1] * Q.shape[2]
        print(f"   平均时间: {bmm_time:.4f} ms")
        print(f"   输入形状: Q={Q.shape}, K={K.shape}")
        print(f"   吞吐量: {bmm_flops / (bmm_time / 1000) / 1e12:.2f} TFLOPs/s")
        benchmark_results['kernels']['bmm'] = bmm_time
        
        # 5. GELU
        print("\n5️⃣  GELU激活函数")
        x = torch.randn(batch_size, seq_len, hidden_size * 4, device=device)
        gelu = torch.nn.GELU()
        
        for _ in range(10):
            _ = gelu(x)
        
        start.record()
        for _ in range(100):
            _ = gelu(x)
        end.record()
        torch.cuda.synchronize()
        
        gelu_time = start.elapsed_time(end) / 100
        print(f"   平均时间: {gelu_time:.4f} ms")
        print(f"   输入形状: {x.shape}")
        print(f"   内存带宽: {x.numel() * x.element_size() * 2 / (gelu_time / 1000) / 1e9:.2f} GB/s")
        benchmark_results['kernels']['gelu'] = gelu_time
        
        # 6. Dropout
        print("\n6️⃣  Dropout")
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        dropout = torch.nn.Dropout(0.1)
        
        for _ in range(10):
            _ = dropout(x)
        
        start.record()
        for _ in range(100):
            _ = dropout(x)
        end.record()
        torch.cuda.synchronize()
        
        dropout_time = start.elapsed_time(end) / 100
        print(f"   平均时间: {dropout_time:.4f} ms")
        benchmark_results['kernels']['dropout'] = dropout_time
        
        all_benchmark_results.append(benchmark_results)
    
    # 保存benchmark结果
    benchmark_file = 'kernel_benchmark_results.json'
    with open(benchmark_file, 'w') as f:
        json.dump(all_benchmark_results, f, indent=2)
    print(f"\n✅ Benchmark结果已保存到: {benchmark_file}")
    
    return all_benchmark_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BERT模型CUDA Kernel Profiling（增强版）')
    parser.add_argument('--use-real-bert', action='store_true',
                        help='使用真实的BERT模型（需要transformers库）')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 16],
                        help='批次大小列表，默认: 1 8 16')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[128, 256],
                        help='序列长度列表，默认: 128 256')
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='隐藏层大小，默认: 768')
    parser.add_argument('--output-dir', type=str, default='./profiling_results',
                        help='输出目录，默认: ./profiling_results')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='跳过单独的kernel benchmark')
    
    args = parser.parse_args()
    
    print("="*100)
    print("🚀 Transformer模型 CUDA Kernel Profiling (增强版)")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到CUDA设备!")
        return
    
    print(f"\n📊 GPU信息:")
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  计算能力: {torch.cuda.get_device_capability(0)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 输出目录: {args.output_dir}")
    
    # 切换到输出目录
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # 生成所有配置组合
        configs = []
        for bs in args.batch_sizes:
            for seq_len in args.seq_lens:
                configs.append({
                    'batch_size': bs,
                    'seq_len': seq_len,
                    'hidden_size': args.hidden_size,
                    'use_real_bert': args.use_real_bert
                })
        
        print(f"\n📋 将运行 {len(configs)} 个配置的profiling:")
        for i, config in enumerate(configs, 1):
            print(f"  {i}. Batch={config['batch_size']}, SeqLen={config['seq_len']}, "
                  f"Hidden={config['hidden_size']}, RealBERT={config['use_real_bert']}")
        
        # 运行profiling
        all_results = []
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*100}")
            print(f"⏳ [{i}/{len(configs)}] Profiling配置: "
                  f"Batch={config['batch_size']}, SeqLen={config['seq_len']}")
            print(f"{'='*100}")
            
            try:
                prof = profile_transformer(**config)
                all_results.append({
                    'config': config,
                    'success': True
                })
            except Exception as e:
                print(f"❌ 配置 {config} 执行失败: {e}")
                all_results.append({
                    'config': config,
                    'success': False,
                    'error': str(e)
                })
        
        # 单独benchmark（如果需要）
        if not args.skip_benchmark and torch.cuda.is_available():
            benchmark_kernels()
        
        # 生成总结报告
        print("\n" + "="*100)
        print("📈 Profiling执行总结")
        print("="*100)
        
        successful = sum(1 for r in all_results if r['success'])
        print(f"\n成功: {successful}/{len(all_results)} 个配置")
        
        if successful > 0:
            print("\n生成的文件:")
            for r in all_results:
                if r['success']:
                    config = r['config']
                    bs, sl = config['batch_size'], config['seq_len']
                    print(f"  - profiling_stats_bs{bs}_seq{sl}.json (详细统计)")
                    print(f"  - kernel_analysis_report_bs{bs}_seq{sl}.md (分析报告)")
                    print(f"  - bert_trace_bs{bs}_seq{sl}.json (Chrome trace)")
        
        print("\n" + "="*100)
        print("✅ Profiling完成!")
        print("="*100)
        print("\n📝 下一步操作:")
        print("  1. 查看生成的Markdown报告，识别Top 3关键算子")
        print("  2. 在Chrome浏览器中打开 chrome://tracing 查看trace文件")
        print("  3. 根据报告中的源码路径，深入分析CUDA实现")
        print("  4. 开始撰写算子调研文档")
        print(f"\n📂 所有结果已保存到: {os.path.abspath(args.output_dir)}")
        
    finally:
        # 切回原目录
        os.chdir(original_dir)


if __name__ == '__main__':
    main()



