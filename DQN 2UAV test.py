import numpy as np
import pandas as pd
import numpy as np
import time
import sys
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

UNIT = 5  # pixels   像素
MAZE_H = 100  # grid height
MAZE_W = 100  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r','o']  # 行为
        self.n_actions = 25  # 行为数
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)  # 画一条从(x0,y0)到(x1,y1)的线
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT / 2, UNIT / 2])

        # # hell            #画第一个黑色正方形
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 2.5, hell1_center[1] - 2.5,
        #     hell1_center[0] + 2.5, hell1_center[1] + 2.5,
        #     fill='black')
        # # hell            #画第二个黑色正方形
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 2.5, hell2_center[1] - 2.5,
        #     hell2_center[0] + 2.5, hell2_center[1] + 2.5,
        #     fill='black')

        # create oval     #画黄色的正方形
        x = [74, 83, 74, 70, 82, 81, 76, 88, 73, 88, 79, 82, 72, 72, 89, 83, 75, 83, 79, 82, 80, 77, 72, 70,
             74, 39, 35, 32, 25, 30, 24, 36, 36, 32, 20, 41, 20, 38, 36, 35, 25, 34, 40, 34, 20, 44, 29, 40,
             40, 23, 55, 58, 47, 48, 61, 46, 62, 52, 63, 62, 59, 53, 49, 64, 59, 53, 60, 50, 58, 55, 51, 50,
             59, 63, 62, 63, 58, 57, 49, 52,  9, 18, 10, 21,  9, 9,  1, 0, 16, 23, 85, 62, 86, 83, 87, 45,
             67, 45, 53, 64]
        y = [82, 84, 82, 78, 78, 80, 81, 75, 72, 84, 76, 76, 89, 85, 76, 81, 75, 70, 82, 87, 86, 82, 75, 89,
             78, 32, 24, 28, 24, 43, 37, 40, 30, 34, 20, 28, 38, 37, 33, 31, 36, 24, 32, 25, 30, 28, 40, 40,
             39, 23, 52, 55, 62, 51, 59, 52, 54, 57, 54, 59, 56, 62, 47, 58, 52, 53, 59, 52, 59, 48, 55, 50,
             63, 45, 60, 62, 45, 52, 55, 53, 53, 39, 34, 51, 31, 70, 66, 77, 69, 82, 13, 26, 35, 17, 19, 82,
             73, 67, 85, 83]
        self.res = []
        for i in range(100):
            oval_center = origin + np.array([(x[i]) * UNIT, y[i] * UNIT])
            self.oval = self.canvas.create_oval(
                oval_center[0] - 2.5, oval_center[1] - 2.5,
                oval_center[0] + 2.5, oval_center[1] + 2.5,
                fill='red')
            self.res.append(self.canvas.coords(self.oval))
        origin1 = np.array([UNIT / 2 + UNIT * 50, UNIT / 2 + UNIT * 50])
        self.rect0 = self.canvas.create_oval(
            origin1[0] - 2.5 - 5 * 15, origin1[1] - 2.5 - 5 * 15,  # x0,y0
            origin1[0] + 2.5 + 5 * 15, origin1[1] + 2.5 + 5 * 15,
            fill='')
        origin2 = np.array([UNIT / 2 + UNIT * 50, UNIT / 2 + UNIT * 50])
        self.rect1 = self.canvas.create_oval(
            origin2[0] - 2.5 - 5 * 15, origin2[1] - 2.5 - 5 * 15,  # x0,y0
            origin2[0] + 2.5 + 5 * 15, origin2[1] + 2.5 + 5 * 15,
            fill='')
        # origin3 = np.array([UNIT / 2 + UNIT * 80, UNIT / 2 + UNIT * 80])
        # self.rect2 = self.canvas.create_oval(
        #     origin3[0] - 2.5 - 5 * 15, origin3[1] - 2.5 - 5 * 15,  # x0,y0
        #     origin3[0] + 2.5 + 5 * 15, origin3[1] + 2.5 + 5 * 15,
        #     fill='')
        # oval_center = origin + UNIT * 2
        # self.oval = self.canvas.create_oval(
        #     oval_center[0] - 2.5, oval_center[1] - 2.5,
        #     oval_center[0] + 2.5, oval_center[1] + 2.5,
        #     fill='yellow')

        # create red rect   #画红色的正方形

        # self.rect1 = self.canvas.create_oval(
        #     origin[0] - 2.5-5*30, origin[1] - 2.5-5*30,#x0,y0
        #     origin[0] + 2.5+5*30, origin[1] + 2.5+5*30,
        #     fill='')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.ori_reward = 0
        # self.update()
        # time.sleep(0.0002)
        self.canvas.delete(self.rect0, self.rect1)
        # 删除agent格子

        origin1 = np.array([UNIT / 2 + UNIT * 50, UNIT / 2 + UNIT * 50])
        self.rect0 = self.canvas.create_oval(
            origin1[0] - 2.5 - 5 * 15, origin1[1] - 2.5 - 5 * 15,  # x0,y0
            origin1[0] + 2.5 + 5 * 15, origin1[1] + 2.5 + 5 * 15,
            fill='')
        origin2 = np.array([UNIT / 2 + UNIT * 50, UNIT / 2 + UNIT * 50])
        self.rect1 = self.canvas.create_oval(
            origin2[0] - 2.5 - 5 * 15, origin2[1] - 2.5 - 5 * 15,  # x0,y0
            origin2[0] + 2.5 + 5 * 15, origin2[1] + 2.5 + 5 * 15,
            fill='')
        # origin3 = np.array([UNIT / 2 + UNIT * 80, UNIT / 2 + UNIT * 80])
        # self.rect2 = self.canvas.create_oval(
        #     origin3[0] - 2.5 - 5 * 15, origin3[1] - 2.5 - 5 * 15,  # x0,y0
        #     origin3[0] + 2.5 + 5 * 15, origin3[1] + 2.5 + 5 * 15,
        #     fill='')
        self.count = 0
        # self.rect1 = self.canvas.create_oval(
        #     origin[0] - 2.5-5*30, origin[1] - 2.5-5*30,#x0,y0
        #     origin[0] + 2.5+5*30, origin[1] + 2.5+5*30,
        #     fill='')
        # self.rect2 = self.canvas.create_oval(
        #     origin[0] - 2.5-5*30, origin[1] - 2.5-5*30,#x0,y0
        #     origin[0] + 2.5+5*30, origin[1] + 2.5+5*30,
        #     fill='')
        # return observation
        return self.canvas.coords(self.rect0)+self.canvas.coords(self.rect1)

    def step(self, action):
        # action = [0,1,3]
        self.count += 1

        s0 = self.canvas.coords(self.rect0)
        s1 = self.canvas.coords(self.rect1)
        # s2 = self.canvas.coords(self.rect2)
        s_lis = [s0,s1]
        # s_all = [s0,s1,s2]
        base_action = np.array([[0, 0]for i in range(2)])
        s = action
        for i in range(2):
            t = s % 5
            if t == 0:  # up
                if s_lis[i][1] + 5 * 10 > UNIT:
                    base_action[i][1] -= UNIT * 10  # 设置步长为10
            elif t == 1:  # down
                if s_lis[i][1] + 5 * 10 < (MAZE_H - 1) * UNIT:
                    base_action[i][1] += UNIT * 10  # 加10
            elif t == 2:  # right
                if s_lis[i][0] + 5 * 10 < (MAZE_W - 1) * UNIT:
                    base_action[i][0] += UNIT * 10  # 右移10
            elif t == 3:  # left
                if s_lis[i][0] + 5 * 10 > UNIT:  # 左移10
                    base_action[i][0] -= UNIT * 10
            elif t == 4:
                base_action[i][0] == base_action[i][0]
                base_action[i][1] == base_action[i][1]
            s = int(s//5)
        x0 = base_action[0][0]
        y0 = base_action[0][1]
        x1 = base_action[1][0]
        y1 = base_action[1][1]
        # x2 = base_action[2][0]
        # y2 = base_action[2][1]
        self.canvas.move(self.rect0, x0, y0) # move agent
        self.canvas.move(self.rect1, x1, y1)
        # self.canvas.move(self.rect2, x2, y2)
        # s_2 = self.canvas.coords(self.rect2)
        s_1 = self.canvas.coords(self.rect1)
        s_0 = self.canvas.coords(self.rect0)  # next state
        # reward function
        # sigle = np.zeros(4)
        r = 15
        # (r//3)*UNIT
        new0 = [0,0,0]
        new1 = [0,0,0]
        new0[0] = s_0[0]
        new1[0] = s_0[1]
        new0[1] = s_1[0]
        new1[1] = s_1[1]
        # new0[2] = s_2[0]
        # new1[2] = s_2[1]
        reward_sigle = 0
        dic = {}
        for b in range(2):
            for i in range(2 * r + 1):
                for j in range(2 * r + 1):
                    # reward_sigle = 0
                    sigle = np.zeros((2,4))
                    sigle[b][0] = new0[b] + UNIT * i
                    sigle[b][1] = new1[b] + UNIT * j
                    sigle[b][2] = sigle[b][0] + UNIT
                    sigle[b][3] = sigle[b][1] + UNIT
                    if [sigle[b][0], sigle[b][1], sigle[b][2], sigle[b][3]] in self.res and str([sigle[b][0], sigle[b][1], sigle[b][2], sigle[b][3]]) not in dic :
                        dic[str([sigle[b][0], sigle[b][1], sigle[b][2], sigle[b][3]])] = 1
                        reward_sigle += 1



        #
        delta = reward_sigle - self.ori_reward
        self.ori_reward = reward_sigle

        if self.count >= 200:

            done = True
            s_0 = 'terminal'
            s_1 = ''
            # s_2 = ''
        else:

            done = False
        return s_0+s_1, delta, done

    def render(self):
        time.sleep(0.02)
        self.update()





class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, N_HIDDEN=128):
        super(Net, self).__init__()
        # 一个隐层，一个输出层
        # (x0,y0,x1,y1)
        self.fc1 = nn.Linear(N_STATES, N_HIDDEN)
        # self.fc1.weight.data.normal_(0, 0.2)
        self.fc2 = nn.Linear(N_HIDDEN, 64)
        # self.fc2.weight.data.normal_(0, 0.2)

        self.fc3 = nn.Linear(64, N_ACTIONS)
        # q(s,a)
        # self.out.weight.data.normal_(0, 0.2)

    def forward(self, x):
        # Net的执行逻辑 Linear_fc1 --> relu --> out --> actions_value
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.fc3(x)
        # x = torch.relu(x)
        actions_value = self.fc3(x)

        return actions_value


        return actions_value


class DQN_agent:
    def __init__(
            self,
            actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=128,
            e_greedy=0.9):

        # torch.cuda.set_device(cuda_device)
        self.n_actions = actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        ''
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        's,a,r,s`'

        self.eval_net, self.target_net = Net(self.n_features, self.n_actions, 128), Net(self.n_features, self.n_actions,
                                                                                        128)

        self.target_net.load_state_dict(self.eval_net.state_dict())
        # self.eval_net
        # self.target_net
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.flag = 0  # to indicate whether the buffer can provide a batch of data
        self.memory_counter = 0
        self.epsilon = e_greedy

    def store_transition(self, s, a, r, s_):  # 保存一条replay buffer
        if s_ == 'terminal':
            s_ = s
        # if not hasattr(self, 'memory_counter'):
        #     self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        self.flag = min(self.flag + 1, self.memory_size)

    def learn(self):  # 更新dqn参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 在memory中随机选择batch条buffer
        sample_index = np.random.choice(self.flag, self.batch_size)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.n_features])
        b_a = torch.LongTensor(b_memory[:, self.n_features:self.n_features + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_features + 1:self.n_features + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        # 计算q_target,loss，反向传递
        q_eval = self.eval_net(b_s).gather(1, b_a).squeeze()

        q_next = self.target_net(b_s_).detach()
        '最后一个done == 0非常重要！！！！！！！！！！！！！'
        q_target = b_r.squeeze() + self.gamma * q_next.max(1)[0].view(self.batch_size, 1).squeeze() * (
                    1 - np.array([b_s[i].mean() == b_s_[i].mean() for i in range(len(b_s))]))

        q_target = q_target.float()
        loss = self.loss_func(q_eval, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, observation, e=None):  # 根据epsilon选择最优动作还是随机动作
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if not e:
            e = self.epsilon

        if np.random.uniform() < e:
            actions_value = self.eval_net.forward(observation)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
            action = action
        return action

    def get_optimal_action(self, observation):  # 选择最优动作
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        actions_value = self.eval_net.forward(observation)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]
        return action

    def save(self, filename):
        torch.save(self.eval_net.state_dict(), filename + "saved_critic")

    def load(self, filename):
        print('load model from: ' + filename)
        self.eval_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.eval_net.state_dict())




def update():
    e = 1
    all_reward = []
    for episode in range(10):

        # initial observation
        observation = env.reset()
        episode_reward = 0
        while True:
            # fresh env
            if episode %1 == 0:
                env.render()
            # RL choose action based on observation
            action = dqn_agent.get_action(observation, e)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            episode_reward += reward

            # RL learn from this transition

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
        print('探索：', e, 'episode_reward:', episode_reward,episode)
        all_reward.append(episode_reward)
        # e = min(0.95, e +0.002)


    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    dqn_agent = DQN_agent(actions=25,n_features = 8)
    dqn_agent.load('./DQN e=0.4 lr=0.0001 episode=100')
    # print(RL.q_table)
    # env.after(100, update)
    # # print("hahah")
    # # print(RL.q_table)
    # env.mainloop()
    update()