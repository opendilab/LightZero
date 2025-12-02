# distutils: language=c++
# cython:language_level=3
from libcpp.vector cimport vector

cdef class MinMaxStatsList:
    """
    最小最大统计列表类
    用于批量管理多个最小最大统计对象，用于 Q 值的归一化
    """
    cdef CMinMaxStatsList *cmin_max_stats_lst  # 指向 C++ 最小最大统计列表的指针

    def __cinit__(self, int num):
        """
        初始化最小最大统计列表

        参数:
            num: 统计对象的数量（对应批量大小）
        """
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        """
        设置价值增量的最大值（用于初始化 min-max 范围）

        参数:
            value_delta_max: 价值变化的最大值
        """
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        """
        析构函数，释放 C++ 对象占用的内存
        """
        del self.cmin_max_stats_lst

cdef class ResultsWrapper:
    """
    搜索结果包装类
    用于包装和管理 C++ 搜索结果对象，包含搜索路径、叶节点等信息
    """
    cdef CSearchResults cresults  # C++ 搜索结果对象

    def __cinit__(self, int num):
        """
        初始化搜索结果包装器

        参数:
            num: 搜索数量（批量大小）
        """
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        """
        获取搜索路径的长度列表

        返回:
            搜索长度列表，即每条搜索路径的深度
        """
        return self.cresults.search_lens

cdef class Roots:
    """
    根节点集合类
    用于批量管理多个 MCTS 搜索树的根节点，是 MuZero 并行搜索的核心数据结构
    """
    cdef int root_num  # 根节点数量，对应批量大小
    cdef CRoots *roots  # 指向 C++ 根节点集合的指针

    def __cinit__(self, int root_num, vector[vector[int]] legal_actions_list):
        """
        初始化根节点集合

        参数:
            root_num: 根节点数量（对应批量大小）
            legal_actions_list: 每个根节点的合法动作列表
        """
        self.root_num = root_num
        self.roots = new CRoots(root_num, legal_actions_list)

    def prepare(self, float root_noise_weight, list noises, list value_prefix_pool, list policy_logits_pool,
                vector[int] & to_play_batch):
        """
        准备根节点（带探索噪声版本，用于训练）

        扩展根节点并添加 Dirichlet 噪声以增强探索，模拟强化学习训练过程

        参数:
            root_noise_weight: 根节点噪声权重，控制探索程度（0.25 是常用值）
            noises: Dirichlet 噪声列表，每个批次元素一个噪声向量
            value_prefix_pool: 价值前缀池（即时奖励 r，MuZero 中的 value_prefix）
            policy_logits_pool: 策略 logits 池（神经网络输出的动作概率原始值）
            to_play_batch: 当前玩家批次（-1 表示单人游戏，1/2 表示双人游戏的玩家编号）
        """
        self.roots[0].prepare(root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, vector[int] & to_play_batch):
        """
        准备根节点（无噪声版本，用于评估）

        扩展根节点但不添加探索噪声，用于模型评估和测试

        参数:
            value_prefix_pool: 价值前缀池
            policy_logits_pool: 策略 logits 池
            to_play_batch: 当前玩家批次
        """
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play_batch)

    def get_trajectories(self):
        """
        获取从每个根节点开始的最佳轨迹

        返回从根节点到叶节点的最佳动作序列（基于访问次数的贪心路径）

        返回:
            轨迹列表，每个轨迹是一个动作序列，例如 [[0, 1, 2], [1, 0, 1], ...]
        """
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        """
        获取每个根节点的子节点访问次数分布

        该分布用于生成 MCTS 改进的策略目标，是强化学习中的重要训练目标

        返回:
            访问次数分布列表，例如 [[1,3,0,2,5], [2,1,4,0,3], ...]
            其中每个内部列表代表一个根节点的子节点访问次数
        """
        return self.roots[0].get_distributions()

    def get_values(self):
        """
        获取每个根节点的平均价值

        价值是通过 value_sum / visit_count 计算得出的

        返回:
            价值列表，每个值为该根节点在搜索中估计的累积价值
        """
        return self.roots[0].get_values()

    def clear(self):
        """
        清空根节点集合，释放子树

        在开始新的搜索前调用，释放上一次搜索的树结构
        """
        self.roots[0].clear()

    def __dealloc__(self):
        """
        析构函数，释放 C++ 对象占用的内存
        """
        del self.roots

    @property
    def num(self):
        """
        获取根节点数量属性

        返回:
            根节点数量（等同于批量大小）
        """
        return self.root_num

    def init_sequential_halving(self, int num_sims, int num_top_acts):
        """
        初始化 Sequential Halving 参数

        参数:
            num_sims: 总模拟次数
            num_top_acts: 初始候选动作数
        """
        self.roots[0].init_sequential_halving(num_sims, num_top_acts)

    def ready_for_next_sh_phase(self):
        """
        检查是否准备好进行下一个 Sequential Halving 阶段

        返回:
            1 如果准备好，0 否则
        """
        return self.roots[0].ready_for_next_sh_phase()

    def apply_next_sh_phase(self, MinMaxStatsList min_max_stats_lst):
        """
        应用下一个 Sequential Halving 阶段

        参数:
            min_max_stats_lst: 最小最大统计列表，用于更新动作评分
        """
        self.roots[0].apply_next_sh_phase(min_max_stats_lst.cmin_max_stats_lst)

    def set_used_visit_num(self, int num):
        """
        设置已使用的访问次数（用于 Sequential Halving 阶段转换检查）

        参数:
            num: 已使用的访问次数
        """
        self.roots[0].set_used_visit_num(num)

cdef class Node:
    """
    单个搜索树节点的 Python 包装类
    对应 MuZero MCTS 中搜索树的一个节点
    """
    cdef CNode cnode  # C++ 节点对象

    def __cinit__(self):
        """
        默认构造函数
        """
        pass

    def __cinit__(self, float prior, vector[int] & legal_actions):
        """
        带参数的构造函数

        参数:
            prior: 先验概率
            legal_actions: 合法动作列表
        """
        pass

    def expand(self, int to_play, int current_latent_state_index, int batch_index, float value_prefix,
               list policy_logits):
        """
        扩展节点，创建所有合法动作的子节点

        在叶节点处调用，基于神经网络的预测扩展子节点，
        并根据策略 logits 初始化每个子节点的先验概率

        参数:
            to_play: 当前玩家（-1 单人游戏，1/2 双人游戏的玩家编号）
            current_latent_state_index: 当前隐状态在搜索路径中的索引
            batch_index: 当前隐状态在批次中的索引
            value_prefix: 价值前缀（即时奖励 r，MuZero 中的概念）
            policy_logits: 策略 logits（神经网络输出的原始动作概率值）
        """
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, current_latent_state_index, batch_index, value_prefix, cpolicy)

def batch_backpropagate(int current_latent_state_index, float discount_factor, list value_prefixs, list values, list policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list to_play_batch):
    """
    批量反向传播（标准版本）

    在叶节点获得神经网络的预测后，沿搜索路径从叶节点向根节点反向传播价值
    更新路径中所有节点的价值和访问次数，这是 MCTS 的关键步骤

    参数:
        current_latent_state_index: 当前隐状态索引
        discount_factor: 折扣因子 γ，用于计算累积奖励（通常 0.99）
        value_prefixs: 价值前缀列表（即时奖励，通过展开获得）
        values: 价值列表（神经网络预测的叶节点价值，V(s)）
        policies: 策略列表（神经网络预测的动作概率分布，π(a|s)）
        min_max_stats_lst: 最小最大统计列表，用于归一化 Q 值到 [0,1] 范围
        results: 搜索结果包装器，包含搜索路径和叶节点信息
        to_play_batch: 玩家批次（-1 单人，1/2 双人游戏）
    """
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_backpropagate(current_latent_state_index, discount_factor, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, to_play_batch)

def batch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, list value_prefixs, list values, list policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list to_play_batch, list no_inference_lst, list reuse_lst, list reuse_value_lst):
    """
    批量反向传播（带价值重用版本）

    优化版本的反向传播，通过重用之前搜索的价值来减少神经网络推理次数
    实现了 ARM (Action Reuse Method) 算法，详见论文: https://arxiv.org/abs/2404.16364

    这个函数处理三类节点：
    1. 无需推理的节点（no_inference_lst）：直接使用重用价值
    2. 需要使用重用价值的节点（reuse_lst）：先通过网络推理获得策略，再用重用价值进行反向传播
    3. 其他节点：正常的网络推理和反向传播

    参数:
        current_latent_state_index: 当前隐状态索引
        discount_factor: 折扣因子 γ
        value_prefixs: 价值前缀列表
        values: 价值列表
        policies: 策略列表
        min_max_stats_lst: 最小最大统计列表
        results: 搜索结果包装器
        to_play_batch: 玩家批次
        no_inference_lst: 无需推理的节点索引列表（完全跳过网络推理）
        reuse_lst: 需要使用重用价值的节点索引列表（进行推理但用重用价值反向传播）
        reuse_value_lst: 重用价值列表（来自上一步的搜索结果）
    """
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies
    cdef vector[float] creuse_value_lst = reuse_value_lst

    cbatch_backpropagate_with_reuse(current_latent_state_index, discount_factor, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, to_play_batch, no_inference_lst, reuse_lst, creuse_value_lst)

def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch):
    """
    批量遍历搜索树（标准 MCTS 选择阶段）

    从根节点开始，使用 UCB（Upper Confidence Bound）公式选择最佳动作，
    不断向下遍历搜索树，直到到达未扩展的叶节点

    这是 MCTS 的核心步骤之一，平衡探索与利用：
    - 利用：选择价值高的节点
    - 探索：选择未充分探索的节点（访问次数少）

    参数:
        roots: 根节点集合
        pb_c_base: UCB 公式中的常数 c2，控制探索强度的基数（通常 19652）
        pb_c_init: UCB 公式中的常数 c1，初始探索系数（通常 1.25）
        discount_factor: 折扣因子 γ
        min_max_stats_lst: 最小最大统计列表，用于归一化价值
        results: 搜索结果包装器，用于存储搜索路径和叶节点信息
        virtual_to_play_batch: 虚拟玩家批次（用于双人游戏模拟对手）

    返回:
        latent_state_index_in_search_path: 叶节点父节点的隐状态在搜索路径中的索引
        latent_state_index_in_batch: 叶节点父节点的隐状态在批次中的索引
        last_actions: 到达叶节点的最后一个动作
        virtual_to_play_batchs: 叶节点处的玩家信息
    """
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst, results.cresults,
                    virtual_to_play_batch)

    return results.cresults.latent_state_index_in_search_path, results.cresults.latent_state_index_in_batch, results.cresults.last_actions, results.cresults.virtual_to_play_batchs

def batch_traverse_with_reuse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch, list true_action, list reuse_value):
    """
    批量遍历搜索树（带价值重用的优化版本）

    在根节点处使用 ARM（Action Reuse Method）评分函数，将重用价值融入到选择过程中
    这使得搜索更倾向于探索真实轨迹的方向，从而在减少计算的同时保持搜索质量

    与 batch_traverse 的关键区别：
    - 在根节点选择动作时，对于真实轨迹中选择的动作使用 ARM 评分
    - ARM 评分用重用价值代替网络预测的价值，更准确地反映该分支的价值
    - 其他节点仍使用标准 UCB 评分

    参数:
        roots: 根节点集合
        pb_c_base: UCB 公式中的常数 c2
        pb_c_init: UCB 公式中的常数 c1
        discount_factor: 折扣因子 γ
        min_max_stats_lst: 最小最大统计列表
        results: 搜索结果包装器
        virtual_to_play_batch: 虚拟玩家批次
        true_action: 真实轨迹中选择的动作（来自上一步搜索）
        reuse_value: 重用价值列表（来自上一步搜索中该动作的结果）

    返回:
        与 batch_traverse 相同
    """
    cbatch_traverse_with_reuse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst, results.cresults,
                    virtual_to_play_batch, true_action, reuse_value)

    return results.cresults.latent_state_index_in_search_path, results.cresults.latent_state_index_in_batch, results.cresults.last_actions, results.cresults.virtual_to_play_batchs
