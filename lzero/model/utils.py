"""
Overview:
    In this file, we provide a set of utility functions for probing network parameters and gradients,
    which can be helpful in analyzing and debugging the inner workings of various models.
"""
from typing import List, Tuple, Union, Dict
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn

###############################
# 1. 计算 average_weight_magnitude
###############################
def compute_average_weight_magnitude(model: nn.Module) -> float:
    """
    计算模型中所有参数的平均绝对值。

    Arguments:
        model: 待评估模型，类型为 nn.Module

    Returns:
        平均权重绝对值（float）
    """
    num_weights = 0
    # 使用模型中第一个参数的设备，保证计算时设备一致
    device = next(model.parameters()).device
    sum_weight_magnitude = torch.tensor(0.0, device=device)

    for p in model.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))
        
    if num_weights == 0:
        return 0.0
    return sum_weight_magnitude.cpu().item() / num_weights

###############################
# 2. 计算 effective_rank
###############################
def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    根据给定的奇异值数组计算 effective rank，公式为：
       effective_rank = exp( - sum_i [p_i * log(p_i)] )
       其中 p_i 是归一化后的奇异值（p_i = s_i / ∑ s_i）

    Arguments:
        singular_values: 奇异值数组，类型为 np.ndarray

    Returns:
        effective rank（float）
    """
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)
    return np.e ** entropy


# 定义一个 Hook 类，用来捕获中间层的输出
class IntermediateOutputHook:
    """
    用于捕获模块输出的 Hook，保存输出张量列表。
    """
    def __init__(self):
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        # 这里使用 detach 防止反向传播干扰，并转移到 CPU 便于后续统计
        self.outputs.append(output.detach().cpu())

def cal_effective_rank(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]], 
    representation_layer_name: str,
) -> float:
    """
    针对模型指定的中间层（representation 层），
    使用 Hook 捕获该层输出，并计算 effective rank。

    Arguments:
        model: 待评估模型，应为 nn.Module 类型。
        inputs: 模型 forward 的输入，可以为 tensor 或 tensor-list。
        representation_layer_name: 模型中表示 representation 层的名称，
                                   该名称必须能够在 model.named_modules() 中找到对应模块。

    Returns:
        effective rank（float）
    """
    # 获取 representation 层模块（若名称不存在将引发 KeyError）
    module_dict = dict(model.named_modules())
    if representation_layer_name not in module_dict:
        raise KeyError(f"Representation layer '{representation_layer_name}' not found in model.named_modules().")
    representation_module = module_dict[representation_layer_name]

    # 注册 hook
    hook = IntermediateOutputHook()
    handle = representation_module.register_forward_hook(hook)
    
    # 执行 forward 推理
    model.eval()
    with torch.no_grad():
        if isinstance(inputs, (list, tuple)):
            _ = model(*inputs)
        else:
            _ = model(inputs)
    
    # 注销 hook，避免内存泄露
    handle.remove()

    if not hook.outputs:
        raise RuntimeError("No outputs captured from the representation layer.")

    # 这里假定有一个或多个 forward（例如在 batch 或多次调用的场景），
    # 将所有输出在 batch 维度上拼接
    if len(hook.outputs) > 1:
        rep_tensor = torch.cat(hook.outputs, dim=0)
    else:
        rep_tensor = hook.outputs[0]

    # 将 representation 展开为二维矩阵： (samples, features)
    rep_tensor = rep_tensor.view(rep_tensor.size(0), -1)

    # 将 tensor 转换为 numpy 数组以使用 numpy.linalg.svd
    rep_np = rep_tensor.cpu().numpy()

    # 计算奇异值
    singular_values = np.linalg.svd(rep_np, full_matrices=False, compute_uv=False)

    # 计算 effective rank
    e_rank = compute_effective_rank(singular_values)

    # 清空 hook 存储（若需要多次调用可以保持清洁状态）
    hook.outputs.clear()
    return e_rank



def compute_dormant_stats(outputs: List[torch.Tensor], threshold: float) -> Tuple[int, int]:
    """
    对给定的一组输出（同一层可能 forward 多次）进行元素级统计。
    
    Arguments:
        outputs: List[torch.Tensor]，每个 tensor 表示一次 forward 的输出
        threshold: 判断 dormant 的阈值，当激活值 <= threshold 时视为 dormant
    
    Returns:
        layer_total: 该层总元素数（累加多个 forward）
        layer_dormant: 该层中满足 dormant 条件的元素数目
    """
    layer_total = 0
    layer_dormant = 0
    for out in outputs:
        flattened = out.view(-1)
        total = flattened.numel()
        dormant = torch.sum(flattened <= threshold).item()
        layer_total += total
        layer_dormant += dormant
    return layer_total, layer_dormant

def cal_dormant_ratio(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    dormant_threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    针对模型中 encoder、transformer backbone 以及 head 三个部分，
    分别统计各部分中所有目标层（例如 nn.Conv2d、nn.Linear、nn.MultiheadAttention 等）的
    dormant ratio（元素级 dormant 百分比），同时返回全局统计指标。
    
    Arguments:
        model: 待评估模型，应包含属性 encoder、transformer（backbone）以及 head（可选）。
        inputs: 模型的输入，支持 tensor 或 tensor-list，要求与模型 forward 调用一致。
        dormant_threshold: 激活值低于该阈值时视为 dormant，默认 1e-2。
    
    Returns:
        results: 包含各部分以及全局 dormant ratio 的字典，单位为百分比（%）。
                 如：{"encoder": 2.5, "transformer": 1.8, "head": 0.5, "global": 1.6}
    """

    # 我们将统计分类为三个部分
    parts = {}
    if hasattr(model, "encoder"):
        parts["encoder"] = model.encoder
    if hasattr(model, "transformer"):
        parts["transformer"] = model.transformer
    
    # 对于 head 部分，查找所有以 "head_" 开头的子模块
    # head_modules = {}
    # for name, module in model.named_children():
    #     if name.startswith("head_"):
    #         head_modules[name] = module
    # if head_modules:
    #     parts["head"] = nn.ModuleDict(head_modules)
    
    if hasattr(model, "head_modules"):
        parts["head"] = model.head_modules

    # if not hasattr(model, "encoder") and not hasattr(model, "transformer") and not hasattr(model, "head"):
    #     parts["model"] = model

    # 定义要捕获的目标模块类型 TODO: 增加更多模块
    target_modules = (nn.Conv2d, nn.Linear)
    
    # 用于存储各部分的 hook（字典：部分名 -> list of (module_name, hook)）
    hooks_dict = {part: [] for part in parts}
    hook_handles = []

    # 为每个部分中的满足类型条件的模块注册 hook
    for part_name, submodule in parts.items():
        for name, module in submodule.named_modules():
            if isinstance(module, target_modules):
                hook = IntermediateOutputHook()
                # 为了避免名称冲突，加上所属部分前缀
                full_name = f"{part_name}/{name}"
                hooks_dict[part_name].append((full_name, hook))
                handle = module.register_forward_hook(hook)
                hook_handles.append(handle)

    # 调用 forward，执行一次推理
    model.eval()
    with torch.no_grad():
        if isinstance(inputs, (list, tuple)):
            _ = model(*inputs)
        else:
            _ = model(inputs)

    # 统计各部分各个模块的 dormant 数量和总数
    results = {}
    total_global = 0
    dormant_global = 0
    for part, hooks in hooks_dict.items():
        part_total = 0
        part_dormant = 0
        for full_name, hook in hooks:
            layer_total, layer_dormant = compute_dormant_stats(hook.outputs, dormant_threshold)
            # 可打印日志，也可记录更详细信息
            # print(f"{full_name}: {layer_dormant}/{layer_total} -> {layer_dormant / layer_total * 100.0 if layer_total > 0 else 0.0}%")
            part_total += layer_total
            part_dormant += layer_dormant
        if part_total > 0:
            ratio = (part_dormant / part_total) * 100.0
        else:
            ratio = 0.0
        results[part] = ratio
        total_global += part_total
        dormant_global += part_dormant

    results["global"] = (dormant_global / total_global) * 100.0 if total_global > 0 else 0.0

    # 清理所有 hook
    for handle in hook_handles:
        handle.remove()
    for hooks in hooks_dict.values():
        for _, hook in hooks:
            hook.outputs.clear()

    return results

def renormalize(inputs: torch.Tensor, first_dim: int = 1) -> torch.Tensor:
    """
    Overview:
        Normalize the input data using the max-min-normalization.
    Arguments:
        - inputs (:obj:`torch.Tensor`): The input data needs to be normalized.
        - first_dim (:obj:`int`): The first dimension of flattening the input data.
    Returns:
        - output (:obj:`torch.Tensor`): The normalized data.
    """
    if first_dim < 0:
        first_dim = len(inputs.shape) + first_dim
    flat_input = inputs.view(*inputs.shape[:first_dim], -1)
    max_val = torch.max(flat_input, first_dim, keepdim=True).values
    min_val = torch.min(flat_input, first_dim, keepdim=True).values
    flat_input = (flat_input - min_val) / (max_val - min_val)

    return flat_input.view(*inputs.shape)


def get_dynamic_mean(model: nn.Module) -> float:
    dynamic_mean = np.abs(model.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

    for block in model.resblocks:
        for name, param in block.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
    dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
    return dynamic_mean


def get_reward_mean(model: nn.Module) -> Tuple[np.ndarray, float]:
    reward_w_dist = model.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

    for name, param in model.fc.named_parameters():
        temp_weights = param.detach().cpu().numpy().reshape(-1)
        reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
    reward_mean = np.abs(reward_w_dist).mean()
    return reward_w_dist, reward_mean


def get_params_mean(model: nn.Module) -> Tuple[np.ndarray, float, float, float]:
    representation_mean = model.representation_network.get_param_mean()
    dynamic_mean = model.dynamics_network.get_dynamic_mean()
    reward_w_dist, reward_mean = model.dynamics_network.get_reward_mean()

    return reward_w_dist, representation_mean, dynamic_mean, reward_mean


def get_gradients(model: nn.Module) -> List[torch.Tensor]:
    grads = []
    for p in model.parameters():
        grad = None if p.grad is None else p.grad.detach()
        grads.append(grad)
    return grads


def set_gradients(model: nn.Module, gradients: List[torch.Tensor]) -> None:
    # TODO due to the drawback of zip operation, we have to check whether gradients match model's parameters
    for g, p in zip(gradients, model.parameters()):
        if g is not None:
            p.grad = g
