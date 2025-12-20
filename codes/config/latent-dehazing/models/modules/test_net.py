"""
测试ConditionalMambaNet集成的脚本
验证：维度正确性、前向传播、内存使用等
"""

import torch
import torch.nn as nn
import numpy as np
from ConditionalMambaNet import ConditionalMambaNet


def test_basic_forward():
    """测试基本的前向传播"""
    print("=" * 50)
    print("测试1: 基本前向传播")
    print("=" * 50)

    # 创建模型
    model = ConditionalMambaNet(
        img_channel=8,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 4],
        dec_blk_nums=[1, 1, 1, 1],
        d_state=16,
        num_heads=4,
        window_size=8,
        inner_rank=32,
        num_tokens=64
    ).cuda()

    model.eval()

    # 创建测试输入
    B, C, H, W = 2, 8, 128, 128
    inp = torch.randn(B, C, H, W).cuda()
    cond = torch.randn(B, C, H, W).cuda()
    time = torch.randint(0, 100, (B,)).cuda()

    print(f"输入形状: inp={inp.shape}, cond={cond.shape}, time={time.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(inp, cond, time)

    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 验证形状
    assert output.shape == inp.shape, f"输出形状不匹配！期望{inp.shape}, 得到{output.shape}"
    print("✓ 形状检查通过")

    # 验证数值稳定性
    assert not torch.isnan(output).any(), "输出包含NaN！"
    assert not torch.isinf(output).any(), "输出包含Inf！"
    print("✓ 数值稳定性检查通过")

    print()


def test_different_resolutions():
    """测试不同分辨率的输入"""
    print("=" * 50)
    print("测试2: 不同分辨率")
    print("=" * 50)

    model = ConditionalMambaNet(
        img_channel=8,
        width=48,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        window_size=8
    ).cuda()

    model.eval()

    resolutions = [(32, 32), (64, 64), (128, 128), (48, 64)]

    for H, W in resolutions:
        inp = torch.randn(1, 8, H, W).cuda()
        cond = torch.randn(1, 8, H, W).cuda()
        time = torch.tensor([50.0]).cuda()

        try:
            with torch.no_grad():
                output = model(inp, cond, time)
            print(f"✓ 分辨率 {H}x{W}: 成功，输出形状={output.shape}")
        except Exception as e:
            print(f"✗ 分辨率 {H}x{W}: 失败 - {str(e)}")

    print()


def test_time_conditioning():
    """测试时间条件的影响"""
    print("=" * 50)
    print("测试3: 时间条件影响")
    print("=" * 50)

    model = ConditionalMambaNet(
        img_channel=8,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
        window_size=8
    ).cuda()

    model.eval()

    inp = torch.randn(1, 8, 32, 32).cuda()
    cond = torch.randn(1, 8, 32, 32).cuda()

    outputs = {}
    for t in [0, 25, 50, 75, 99]:
        time = torch.tensor([float(t)]).cuda()
        with torch.no_grad():
            output = model(inp, cond, time)
        outputs[t] = output
        print(f"时间步 t={t:2d}: 输出均值={output.mean().item():.6f}, 标准差={output.std().item():.6f}")

    # 检查不同时间步的输出是否不同
    diff_01 = (outputs[0] - outputs[50]).abs().mean()
    diff_02 = (outputs[0] - outputs[99]).abs().mean()
    print(f"\n时间步差异: t=0 vs t=50: {diff_01:.6f}")
    print(f"时间步差异: t=0 vs t=99: {diff_02:.6f}")

    if diff_01 > 1e-5 and diff_02 > 1e-5:
        print("✓ 时间条件正确影响输出")
    else:
        print("✗ 警告：时间条件可能没有正确工作！")

    print()


def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 50)
    print("测试4: 梯度流动")
    print("=" * 50)

    model = ConditionalMambaNet(
        img_channel=8,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
        window_size=8
    ).cuda()

    model.train()

    inp = torch.randn(2, 8, 32, 32, requires_grad=True).cuda()
    cond = torch.randn(2, 8, 32, 32).cuda()
    time = torch.randint(0, 100, (2,)).cuda()

    # 前向传播
    output = model(inp, cond, time)
    loss = output.mean()

    # 反向传播
    loss.backward()

    # 检查梯度
    grad_norms = []
    zero_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm < 1e-7:
                zero_grads.append(name)
        else:
            zero_grads.append(name)

    print(f"总参数数: {len(list(model.parameters()))}")
    print(f"有梯度的参数: {len(grad_norms)}")
    print(f"梯度为零的参数: {len(zero_grads)}")
    print(f"平均梯度范数: {np.mean(grad_norms):.6f}")
    print(f"最大梯度范数: {np.max(grad_norms):.6f}")
    print(f"最小梯度范数: {np.min(grad_norms):.6f}")

    if len(zero_grads) > 0:
        print(f"\n警告: 以下参数梯度为零:")
        for name in zero_grads[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(zero_grads) > 5:
            print(f"  ... 还有 {len(zero_grads) - 5} 个")

    print()


def test_memory_usage():
    """测试内存使用"""
    print("=" * 50)
    print("测试5: 内存使用")
    print("=" * 50)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    configs = [
        {"width": 32, "enc_blk_nums": [1, 1], "name": "Small"},
        {"width": 48, "enc_blk_nums": [1, 1, 1], "name": "Medium"},
        {"width": 64, "enc_blk_nums": [1, 1, 1, 4], "name": "Large"},
    ]

    for config in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = ConditionalMambaNet(
            img_channel=8,
            width=config["width"],
            middle_blk_num=1,
            enc_blk_nums=config["enc_blk_nums"],
            dec_blk_nums=[1] * len(config["enc_blk_nums"]),
            window_size=8
        ).cuda()

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 测试前向传播
        inp = torch.randn(1, 8, 64, 64).cuda()
        cond = torch.randn(1, 8, 64, 64).cuda()
        time = torch.tensor([50.0]).cuda()

        with torch.no_grad():
            output = model(inp, cond, time)

        mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # MB

        print(f"{config['name']} 配置:")
        print(f"  总参数: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  显存占用: {mem_allocated:.2f}MB (分配) / {mem_reserved:.2f}MB (保留)")
        print()

        del model
        torch.cuda.empty_cache()


def test_batch_consistency():
    """测试批次一致性"""
    print("=" * 50)
    print("测试6: 批次一致性")
    print("=" * 50)

    model = ConditionalMambaNet(
        img_channel=8,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
        window_size=8
    ).cuda()

    model.eval()

    # 创建批次输入
    inp_batch = torch.randn(4, 8, 32, 32).cuda()
    cond_batch = torch.randn(4, 8, 32, 32).cuda()
    time_batch = torch.tensor([25.0, 50.0, 75.0, 25.0]).cuda()

    # 批次处理
    with torch.no_grad():
        output_batch = model(inp_batch, cond_batch, time_batch)

    # 单独处理
    outputs_single = []
    for i in range(4):
        with torch.no_grad():
            out = model(
                inp_batch[i:i + 1],
                cond_batch[i:i + 1],
                time_batch[i:i + 1]
            )
        outputs_single.append(out)

    # 比较结果
    for i in range(4):
        diff = (output_batch[i:i + 1] - outputs_single[i]).abs().max()
        print(f"样本 {i}: 批次 vs 单独处理差异 = {diff.item():.6e}")

    max_diff = max((output_batch[i:i + 1] - outputs_single[i]).abs().max().item() for i in range(4))

    if max_diff < 1e-5:
        print("✓ 批次处理一致性检查通过")
    else:
        print(f"✗ 警告：批次处理不一致，最大差异={max_diff:.6e}")

    print()


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("=" * 50)
    print("ConditionalMambaNet 集成测试套件")
    print("=" * 50)
    print("\n")

    try:
        test_basic_forward()
        test_different_resolutions()
        test_time_conditioning()
        test_gradient_flow()
        test_memory_usage()
        test_batch_consistency()

        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()