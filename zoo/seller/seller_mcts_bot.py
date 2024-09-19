import math
import random
from easydict import EasyDict
from seller_env import SellerEnv
import time, copy

EPS = 1e-6

class TreeNode:
    def __init__(self, env, parent=None, action=None, root_base_return=None):
        self.env = copy.deepcopy(env)
        self.history_deepcopy = copy.deepcopy(self.env.history)
        self.round_cnt_deepcopy = copy.deepcopy(self.env.round_cnt)
        self.eval_episode_return_deeepcopy = copy.deepcopy(self.env.eval_episode_return)

        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_return = 0.0
        if root_base_return is None:
            self.root_base_return = 0
        else:
            self.root_base_return = root_base_return

        
    def expand(self, actions):
        if self.env.finished:
            return  # 如果环境已经完成，不进行扩展
        
        # 过滤掉已经访问过的动作
        unvisited_actions = [a for a in actions if a not in [c.action for c in self.children]]

        for action in unvisited_actions:

            if self.env.finished:  # 再次检查环境是否完成
                continue
            _, reward, done, _ = self.env.step([action])
            child_node = TreeNode(self.env, self, action, self.root_base_return)
            self.children.append(child_node)

            # 重置环境到之前的状态
            self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy, self.eval_episode_return_deeepcopy)

    def rollout(self):
        # 重置环境到node之前的状态
        done = self.env.finished
        if done:
            return self.env.eval_episode_return - self.root_base_return

        while not done:
            legal_actions = list(range(len(self.env.commands))) 
            action = random.choice(legal_actions) # TODO
            _, reward, done, _ = self.env.step([action])

        rollout_return = self.env.eval_episode_return - self.root_base_return
        # 重置环境到之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy, self.eval_episode_return_deeepcopy)
        return rollout_return

    def backpropagate(self, result):
        self.visits += 1
        self.total_return += result
        if self.parent:
            self.parent.backpropagate(result)

    
    def select_best_child(self):
        ucb_values = [self.ucb(c.action) for c in self.children]
        best_child_index = ucb_values.index(max(ucb_values))
        return self.children[best_child_index]

    def ucb(self, action):
        child = next((c for c in self.children if c.action == action), None)
        if child is None:
            return float('inf')  # 未访问过的节点具有最高优先级
        
        exploitation = child.total_return / (child.visits + 1)
        exploration = math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1))
        # return exploitation + math.sqrt(2) * exploration  # 探索常数为 sqrt(2)
        return exploitation + 1.41 * exploration  # 可以调节探索常数


class MCTSBot:
    def __init__(self, n_iterations=50):
        self.n_iterations = n_iterations
        
    def search(self, env):
        root = TreeNode(env, root_base_return=env.eval_episode_return)
        
        for _ in range(self.n_iterations):
            node = root

            # selection
            while node.children:
                node = node.select_best_child()
            
            # expansion
            if not node.env.finished:
                actions = list(range(len(env.commands)))  # 获取所有可能的动作
                node.expand(actions)
                if node.children:  # 确保存在未访问的子节点
                    # node = random.choice(node.children)  # 改为选择 UCB 值最高的子节点
                    node = node.select_best_child()
            
            # rollout  
            rollout_return = node.rollout()
            
            # backpropagation
            node.backpropagate(rollout_return)
            
        
        return root
    
    def get_action(self, env):
        root = self.search(env)
        print("Visits:", [(c.action, c.visits) for c in root.children])
        # 选择访问次数最多的子节点作为最佳动作
        # return max(root.children, key=lambda c: c.visits).action
        
        # 选择平均奖励最高的子节点作为最佳动作
        best_child = max(root.children, key=lambda c: c.total_return / (c.visits + 1))
        print("Average Reward:", [(c.action, c.total_return / (c.visits + 1)) for c in root.children])
        return best_child.action

if __name__ == '__main__':
    env_cfg = EasyDict(
    dict(
        agent='deepseek',  # or 'lmdeploy'
        api_key=['your deepseek api key'],
        commands=[
            '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因',
            '询问用户的具体使用情景', '询问用户当前还有哪些疑虑'
        ],
        max_round=5,
        seed=0,
        lang='zh',
        log_suffix='mcts_sim10_a9_eps20',
        save_replay=False,
        dynamic_action_space=False,
        )
    )

    env = SellerEnv(cfg=env_cfg)
    avg_return = 0
    eval_episodes = 20
    mcts_bot = MCTSBot(n_iterations=10)

    for seed in range(0, eval_episodes): # TODO
        env.seed(seed=seed, dynamic_seed=False) # NOTE: seed must be before reset
        env.reset(is_eval=True) # NOTE
        while not env.finished:
            print(f'commands: {env.commands}')
            action = mcts_bot.get_action(env)
            env.save_replay = True  # NOTE
            env_step = env.step([action])
            env.save_replay = False  # NOTE: 不存储simulation_env里面的replay
            print(f'============= Round {env.round_cnt} =============')
            print(f'MCTS Bot 选择动作: {env.commands[action]}')
            for k in env_step.info:
                print(f'【{k} 的回复】')
                print(env_step.info[k])
            print(f'【reward: {env_step.reward}, done: {env_step.done}】')
        
        print(f'episode {seed}: evaluation return: {env.eval_episode_return}')
        avg_return += env.eval_episode_return
    print(f'对话结束,最终平均收益: {avg_return}')
    
