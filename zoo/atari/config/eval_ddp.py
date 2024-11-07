# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# def example(rank, world_size):
#     # 初始化进程组
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
#     # 创建模型
#     model = nn.Linear(10, 10).to(rank)
    
#     # 将模型包装为DDP模型
#     ddp_model = DDP(model, device_ids=[rank])
    
#     # 创建优化器
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # 模拟训练过程
#     for _ in range(5):
#         # 前向传播
#         outputs = ddp_model(torch.randn(20, 10).to(rank))
#         labels = torch.randn(20, 10).to(rank)
#         loss = torch.nn.functional.mse_loss(outputs, labels)
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         print(f'rank {rank}: one step forward')
#         optimizer.step()
    
#     print(f'清理进程组 rank {rank} 开始')
#     # 清理进程组
#     dist.destroy_process_group()
#     print(f'清理进程组 rank {rank} 结束')


# if __name__ == "__main__":
#     world_size = 2
    
#     # 启动多个进程运行example函数
#     torch.multiprocessing.spawn(example, args=(world_size,), nprocs=world_size, join=True)




import torch
import torch.distributed as dist

def all_reduce_example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    tensor = torch.tensor([rank + 1], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: {tensor.item()}")
    dist.destroy_process_group()
    print(f'Rank {rank}:  done')


if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(all_reduce_example, args=(world_size,), nprocs=world_size, join=True)
    print('all done')