import math
import random
from easydict import EasyDict
from seller_env import SellerEnv
import time, copy
import math
import random
EPS = 1e-6

class TreeNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        # 将当前节点的环境状态进行深拷贝
        self.history_deepcopy = copy.deepcopy(self.env.history)
        self.round_cnt_deepcopy = copy.deepcopy(self.env.round_cnt)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.eval_episode_return = None
        
    def expand(self, actions):
        if self.env.finished:
            return  # 如果环境已经完成，不进行扩展
        
        # 选择具有最高UCB值的动作进行扩展
        ucb_values = [self.ucb(a) for a in actions]
        best_action_index = ucb_values.index(max(ucb_values))
        action = actions[best_action_index]
        
        _, reward, done, _ = self.env.step([action])
        child_node = TreeNode(self.env, self, action)
        if done:
            child_node.eval_episode_return = reward
        self.children.append(child_node)
        
        # 重置环境到之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)

        # for action in actions:
        #     _, reward, done, _  = self.env.step([action])
        #     child_node = TreeNode(self.env, self, action)
        #     if done:
        #         child_node.eval_episode_return = reward
        #     self.children.append(child_node)

        #     # 重置环境到之前的状态
        #     self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)
    
    def ucb(self, action):
        child = next((c for c in self.children if c.action == action), None)
        if child is None:
            return float('inf')  # 未访问过的节点具有最高优先级
        
        exploitation = child.reward / (child.visits + EPS)
        exploration = math.sqrt(2 * math.log(self.visits) / (child.visits + EPS))
        return exploitation + 1.4 * exploration

    def rollout(self):
        if self.eval_episode_return is not None:
            return self.eval_episode_return
        
        # 重置环境到node之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)
        done = self.env.finished

        while not done:
            legal_actions = list(range(len(self.env.commands))) 
            action = random.choice(legal_actions)
            _, reward, done, _ = self.env.step([action])

        rollout_reward = reward
        # 重置环境到之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)
        return rollout_reward
    
    def backpropagate(self, result):
        self.visits += 1
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)
            
    def select_best_child(self, c_param=1.4):
        choices_weights = [
            (c.reward / (c.visits+EPS)) + c_param * math.sqrt(2 * math.log(self.visits) / (c.visits+EPS)) 
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
class MCTSBot:
    def __init__(self, n_iterations=50):
        self.n_iterations = n_iterations
        
    def search(self, env):
        root = TreeNode(env)
        
        # select_time = 0
        # expand_time = 0
        # rollout_time = 0
        # backprop_time = 0
        
        for _ in range(self.n_iterations):
            node = root
            
            # selection
            # start_time = time.time()
            while node.children:
                node = node.select_best_child()
            # select_time += time.time() - start_time
            
            # expansion
            # start_time = time.time()
            legal_actions = list(range(len(env.commands))) 
            if not node.env.finished and legal_actions:
                node.expand(legal_actions)
                if node.children:  # 确保存在子节点
                    node = random.choice(node.children)
            # expand_time += time.time() - start_time
            
            # rollout  
            # start_time = time.time()
            reward = node.rollout()
            # rollout_time += time.time() - start_time
            
            # backpropagation
            # start_time = time.time() 
            node.backpropagate(reward)
            # backprop_time += time.time() - start_time
        
        # print(f"Selection time: {select_time:.4f}s")
        # print(f"Expansion time: {expand_time:.4f}s") 
        # print(f"Rollout time: {rollout_time:.4f}s")
        # print(f"Backpropagation time: {backprop_time:.4f}s")
        
        return root
    
    def get_action(self, env):
        root = self.search(env)
        
        # 选择访问次数最多的子节点作为最佳动作
        return max(root.children, key=lambda c: c.visits).action

if __name__ == '__main__':
    env_cfg = EasyDict(
    dict(
        agent='deepseek',
        api_key='sk-c4a8fe52693a4aaab64e648c42f40be6',
        # api_key='sk-7866ab6ea8ca408a91971ef18eed4b75',
        # commands=[
        #     '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他',
        #     '询问用户的具体使用情景', '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
        # ],
        commands=[
            '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '和用户共情，从用户的角度解释选择的原因'
        ],
        max_round=5,
        # commands=[
        #     '将你的产品推销给用户'
        # ],
        # max_round=2,
        seed=0,
        lang='zh',
        log_suffix='mcts_0805',
        save_replay=False,
        )
    )

    env = SellerEnv(cfg=env_cfg)

    avg_return = 0
    mcts_bot = MCTSBot(n_iterations=3)
    # for seed in range(1, 6): # TODO
    for seed in [0]: # TODO
        env = SellerEnv(cfg=env_cfg)
        env.seed(seed)
        env.reset()
        while not env.finished:
            action = mcts_bot.get_action(copy.deepcopy(env))
            # env.save_replay = True
            env.save_replay = False
            env_step = env.step([action])
            env.save_replay = False
            print(f'########## Round {env.round_cnt} ##########')
            print(f'MCTS Bot 选择动作: {env.commands[action]}')
            for k in env_step.info:
                print(f'【{k} 的回复】')
                print(env_step.info[k])
            print(f'【reward: {env_step.reward}, done: {env_step.done}】')
        avg_return += env_step.reward
    print(f'对话结束,最终平均收益: {avg_return/6}')
    
