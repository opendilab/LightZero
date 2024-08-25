import random
import copy
import math

EPS = 1e-6

class TreeNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
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
        
        # 过滤掉已经访问过的动作
        unvisited_actions = [a for a in actions if a not in [c.action for c in self.children]]
        
        for action in unvisited_actions:
            if self.env.finished:  # 再次检查环境是否完成
                continue
            _, reward, done, _ = self.env.step([action])
            child_node = TreeNode(self.env, self, action)
            if done:
                child_node.eval_episode_return = reward
            self.children.append(child_node)

            # 重置环境到之前的状态
            self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)

    def rollout(self):
        if self.eval_episode_return is not None:
            return self.eval_episode_return
        
        # 重置环境到node之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)
        done = self.env.finished

        while not done:
            legal_actions = list(range(len(self.env.commands))) 
            action = random.choice(legal_actions)
             # TODO: 贪婪策略选择最优动作，而不是随机动作
            # action = max(legal_actions, key=lambda a: self._get_reward_for_action(a))
            _, reward, done, _ = self.env.step([action])

        rollout_reward = reward
        # 重置环境到之前的状态
        self.env.reset_from_history(self.history_deepcopy, self.round_cnt_deepcopy)
        return rollout_reward
    
    # def _get_reward_for_action(self, action):
    #     # 为一个给定的动作返回预计的奖励
    #     rewards = {
    #         0: 1.0,  # action_0: 最优
    #         1: 0.5,  # action_1: 次优
    #         2: 0.2,  # action_2: 一般
    #         3: 0.0   # action_3: 最差
    #     }
    #     return rewards[action]

    def backpropagate(self, result):
        self.visits += 1
        self.reward += result 
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
        
        exploitation = child.reward / (child.visits + EPS)
        exploration = math.sqrt(2 * math.log(self.visits + EPS) / (child.visits + EPS))
        # return exploitation + math.sqrt(2) * exploration  # 调整探索常数为 sqrt(2)
        return exploitation + exploration  # 调整探索常数为 1


class MCTSBot:
    def __init__(self, n_iterations=50):
        self.n_iterations = n_iterations
        
    def search(self, env):
        root = TreeNode(env)
        
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
                    node = random.choice(node.children) # TODO
                    # node = max(node.children, key=lambda c: c.reward / (c.visits + EPS)) # 选择奖励最高的子节点
            
            # rollout  
            reward = node.rollout()
            
            # backpropagation
            node.backpropagate(reward)
            
        
        return root
    
    def get_action(self, env):
        root = self.search(env)
        print("Visits:", [(c.action, c.visits) for c in root.children])
        # 选择访问次数最多的子节点作为最佳动作
        # return max(root.children, key=lambda c: c.visits).action
        
        # 选择平均奖励最高的子节点作为最佳动作
        best_child = max(root.children, key=lambda c: c.reward / (c.visits + EPS))
        print("Average Reward:", [(c.action, c.reward / (c.visits + EPS)) for c in root.children])
        return best_child.action

class ToyEnv:
    def __init__(self, max_round=5):
        self.max_round = max_round
        self.commands = ['action_0', 'action_1', 'action_2', 'action_3']
        self.round_cnt = 0
        self.finished = False
        self.history = []
        self.reset()
        self.last_action = -1

    def reset(self):
        self.round_cnt = 0
        self.finished = False
        self.history = []
        return self.get_observation()

    def step(self, action):
        if self.finished:
            raise ValueError("Cannot step in a finished environment.")

        self.round_cnt += 1
        self.history.append(action)

        # 假设 action_0 总是最优的，action_3 是最差的
        rewards = {
            0: 1.0,  # action_0: 最优
            1: 0.5,  # action_1: 次优
            2: 0.2,  # action_2: 一般
            3: 0.0   # action_3: 最差
        }
        reward = rewards[action[0]]

        # if self.last_action == 0 and action[0] == 0:
        #     reward = 1.0
        # else:
        #     reward = 0.0
        # self.last_action = action[0]

        done = self.round_cnt >= self.max_round
        if done:
            self.finished = True

        info = {}
        return self.get_observation(), reward, done, info

    def reset_from_history(self, history, round_cnt):
        self.history = copy.deepcopy(history)
        self.round_cnt = copy.deepcopy(round_cnt)
        self.finished = self.round_cnt >= self.max_round

    def get_observation(self):
        # 观察可以是当前的回合计数和历史动作记录
        return {
            'round_cnt': self.round_cnt,
            'history': self.history
        }

    def seed(self, seed):
        random.seed(seed)


# 测试 MCTSBot 在 ToyEnv 上的行为
if __name__ == '__main__':
    env = ToyEnv(max_round=5)
    mcts_bot = MCTSBot(n_iterations=50)

    env.seed(0)
    env.reset()

    avg_return = 0
    for _ in range(1):  # 单次测试
        env.reset()
        while not env.finished:
            action = mcts_bot.get_action(copy.deepcopy(env))
            _, reward, done, _ = env.step([action])
            print(f'============== Round {env.round_cnt} ==============')
            print(f'MCTS Bot 选择动作: {env.commands[action]}')
            print(f'【reward: {reward}, done: {done}】')
        avg_return += reward

    print(f'对话结束,最终平均收益: {avg_return/1}')