import torch
import numpy as np
from typing import List, Tuple

def compute_gradient_conflicts(gradients: List[torch.Tensor]) -> dict:
    """
    计算多个梯度之间的冲突
    
    Args:
        gradients: 梯度列表，每个元素是一个梯度张量
    
    Returns:
        包含各种冲突指标的字典
    """
    results = {}
    n_gradients = len(gradients)
    
    # 确保所有梯度形状相同
    assert all(g.shape == gradients[0].shape for g in gradients), "梯度形状必须相同"
    
    # 1. 余弦相似度矩阵
    cosine_sim_matrix = torch.zeros(n_gradients, n_gradients)
    for i in range(n_gradients):
        for j in range(n_gradients):
            cos_sim = torch.cosine_similarity(
                gradients[i].flatten(), 
                gradients[j].flatten(), 
                dim=0
            )
            cosine_sim_matrix[i, j] = cos_sim
    
    results['cosine_similarity_matrix'] = cosine_sim_matrix
    
    # 2. 梯度冲突得分 (负余弦相似度的平均)
    # 排除对角线元素
    mask = ~torch.eye(n_gradients, dtype=bool)
    conflict_scores = -cosine_sim_matrix[mask]
    results['avg_conflict_score'] = conflict_scores.mean().item()
    results['max_conflict_score'] = conflict_scores.max().item()
    
    # 3. 点积矩阵
    dot_product_matrix = torch.zeros(n_gradients, n_gradients)
    for i in range(n_gradients):
        for j in range(n_gradients):
            dot_prod = torch.dot(gradients[i].flatten(), gradients[j].flatten())
            dot_product_matrix[i, j] = dot_prod
    
    results['dot_product_matrix'] = dot_product_matrix
    
    # 4. 梯度范数
    gradient_norms = [torch.norm(g).item() for g in gradients]
    results['gradient_norms'] = gradient_norms
    
    # 5. 冲突强度 (基于负点积)
    negative_dot_products = []
    for i in range(n_gradients):
        for j in range(i+1, n_gradients):
            dot_prod = torch.dot(gradients[i].flatten(), gradients[j].flatten())
            if dot_prod < 0:  # 负点积表示冲突
                negative_dot_products.append(-dot_prod.item())
    
    results['num_conflicting_pairs'] = len(negative_dot_products)
    results['avg_conflict_intensity'] = np.mean(negative_dot_products) if negative_dot_products else 0
    
    return results

# 使用示例
def example_usage():
    # 生成示例梯度
    torch.manual_seed(42)
    gradients = [
        torch.randn(100),  # 梯度1
        torch.randn(100),  # 梯度2  
        torch.randn(100),  # 梯度3
    ]
    
    # 计算冲突
    conflicts = compute_gradient_conflicts(gradients)
    
    print("梯度冲突分析结果:")
    print(f"平均冲突得分: {conflicts['avg_conflict_score']:.4f}")
    print(f"最大冲突得分: {conflicts['max_conflict_score']:.4f}")
    print(f"冲突梯度对数量: {conflicts['num_conflicting_pairs']}")
    print(f"平均冲突强度: {conflicts['avg_conflict_intensity']:.4f}")
    print(f"梯度范数: {conflicts['gradient_norms']}")
    print("\n余弦相似度矩阵:")
    print(conflicts['cosine_similarity_matrix'])


if __name__ == "__main__":
    example_usage()
