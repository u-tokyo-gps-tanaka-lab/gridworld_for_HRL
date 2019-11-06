import gym
import chainer
import numpy as np
import random
from chainer import links as L
from chainer import functions as F
from chainerrl.agents import a2c
from chainerrl.agents import a3c
from chainerrl import policy
from chainerrl import v_function
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import policies
from collections import deque
from gym.envs.registration import EnvSpec
xp = np
d4 = ((1,0),(0,1),(0,-1),(-1,0))

class MyHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""
    def __init__(self, trial, n_input_channels=3, 
                 activation=F.relu, bias=0.1,width=None, height=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = trial.suggest_int('n_output_channels', 16, 256)
        self.width = width
        self.height = height
        layer_type = trial.suggest_categorical('layer_type',['MLP', 'CNN'])
        if layer_type == 'CNN':
            n_layers = trial.suggest_int('n_layers',1,3)
            n_channels = trial.suggest_int('n_channels_cnn', 16, 128)
            layers = [
                L.Convolution2D(n_input_channels, n_channels, 3, stride=1, pad=1,
                                initial_bias=bias)]
            for i in range(n_layers):
                layers.append(L.Convolution2D(n_channels, n_channels, 3, stride=1, pad=1, initial_bias=bias))
            layers.append(L.Linear(n_channels * self.width * self.height, self.n_output_channels, initial_bias=bias))        
            
        elif layer_type == 'MLP':
            layers = []
            n_layers = trial.suggest_int('n_layers',1,3)
            n_units = [0] * (n_layers + 1)
            n_units[0] = n_input_channels * self.width * self.height
            n_units[n_layers] = self.n_output_channels
            for i in range(n_layers):
                if i + 1 < n_layers:
                    n_units[i + 1] = trial.suggest_int('n_units_l{}'.format (i), 4, 128)
                    layers.append(L.Linear(n_units[i], n_units[i + 1], initial_bias=bias))
        super(MyHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        if isinstance(self[0], L.Linear):
            h = F.reshape(h,shape=(-1, self.width * self.height * 3))
        elif isinstance(self[0], L.Convolution2D):
            h = h.transpose((0,3,1,2))
        for layer in self:
            if isinstance(layer, L.Linear):
                h = F.reshape(h,shape=(-1,layer.in_size))
            h = layer(h)
            h = self.activation(h)
        return h
   
class MyModel(chainer.ChainList, a2c.A2CModel, a3c.A3CModel):
    def __init__(self, trial, n_actions, width=None, height=None):
        self.head = MyHead(trial, width=width, height=height)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        pi = self.pi(out)
        v = self.v(out)
        return pi, v

    
class MyA3CLSTM(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, trial, width, height, action_size, lstm_size=128):
        obs_size=width*height
        self.head = MyHead(trial, width=width, height=height)
        self.lstm = L.LSTM(self.head.n_output_channels, lstm_size)
        self.pi = policies.FCSoftmaxPolicy(lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.head, self.lstm, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        out = self.lstm(out)
        pi = self.pi(out)
        v = self.v(out)
        return pi,v

def best_move(px, py, tx, ty):
    if px == tx:
        if py > ty:
            return 1
        else:
            return 3
    if px > tx:
        return 2
    return 0


def sample(p, t, width, height, nohole=False):
    ret = xp.zeros(shape=(width, height),dtype=xp.float32)
    while True:
        for x in range(width):
            for y in range(height):
                if not nohole and random.random()<0.1:
                    ret[x][y] = 0
                else:
                    ret[x][y] = 1
        ret[p[0]][p[1]] = 1
        ret[t[0]][t[1]] = 1
        dist = [[-1]*height for x in range(width)]
        q = deque()
        q.append(t)
        dist[t[0]][t[1]] = 0
        while len(q) > 0:
            p1 = q.popleft()
            x, y  = p1
            d = dist[x][y]
            for dx, dy in d4:
                x1, y1 = x + dx, y + dy
                if not (0 <= x1 < width and 0 <= y1 < height):
                    continue
                if ret[x1][y1] == 1 and dist[x1][y1] < 0:
                    q.append((x1, y1))
                    dist[x1][y1] = d + 1
        if dist[p[0]][p[1]] >= 0:
            return ret, dist


def make_obs(f, p, t, width, height, hidden, mode_1):
    r = xp.zeros(shape=(width, height, 3),dtype=xp.float32)
    for x in range(width):
        for y in range(height):
            r[x][y][2] = f[x][y]
    if hidden:
        r[t[0]][t[1]][1] = 0
#    elif mode_1:
#        r[t[0]][t[1]][1] = 0
#        fake_t = t
#        while (not (0 <= fake_t[0] < width)) or (not (0 <= fake_t[1] < height)) or (fake_t == t):
#            fake_t = (t[0] + random.randrange(-1,2), t[1] + random.randrange(-1,2))
#        r[fake_t[0]][fake_t[1]][1] = 1
#        print("fake_t = {}, t = {}".format(fake_t,t))
    else:
        r[t[0]][t[1]][1] = 1
    r[p[0]][p[1]][0] = 1
    return r


class CliffGridEnv(gym.Env):
    def __init__(self, process_idx=0, edge_penalty=0, width=10, height=16, easy=False, nohole=False, hidden_dist=0, mark_env=False, hrl_env=False, debug=False):
        self.process_idx = process_idx
        self.edge_penalty = edge_penalty
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(width, height, 3), dtype=np.float32)
        self.easy = easy
        self.nohole = nohole
        self.actions = []
        self.width = width
        self.height = height
        self.hidden = False
        self.hidden_dist = hidden_dist
        self.reward_range = (float(-self.edge_penalty-1), 100.0)
        self.debug = debug
        self.spec = EnvSpec('Myenv-v0')
        self.mark_env = mark_env
        self.hrl_env = hrl_env
        self.mode_1 = False
        self.goal = (0,0) #true goal 
        self.mark_done = False
        
    def set_lava(self):
        #print(self.p)
        if self.mark_env:
            self.t = (random.randrange(0,self.width-1), random.randrange(0,self.height))
            self.p = (random.randrange(0,self.width), random.randrange(0,self.height))
            self.f, self.dist = sample(self.p, self.t, self.width, self.height, nohole=self.nohole)
            while self.dist[self.p[0]][self.p[1]] <= self.hidden_dist:
                self.p = (random.randrange(0,self.width), random.randrange(0,self.height))
                self.f, self.dist = sample(self.p, self.t, self.width, self.height, nohole=self.nohole)
        else:     
            self.t = (random.randrange(0,self.width), random.randrange(0,self.height))
            self.p = (random.randrange(0,self.width), random.randrange(0,self.height))
            self.f, self.dist = sample(self.p, self.t, self.width, self.height, nohole=self.nohole)
            while self.dist[self.p[0]][self.p[1]] <= self.hidden_dist:
                self.p = (random.randrange(0,self.width), random.randrange(0,self.height))
                self.f, self.dist = sample(self.p, self.t, self.width, self.height, nohole=self.nohole)
        self.goal = self.t
        if self.mode_1:
            fake_t = self.t
            while (not (0 <= fake_t[0] < self.width)) or (not (0 <= fake_t[1] < self.height) or (fake_t == self.t)):
                fake_t = (self.t[0] + random.randrange(-1,2), self.t[1] + random.randrange(-1,2))
            self.t = fake_t
        self.remain = 400
        self.hidden = False
        
    def obs(self):
        return make_obs(self.f, self.p, self.t, self.width, self.height, self.hidden, self.mode_1)

    def reset(self):
        self.set_lava()
        if self.dist[self.p[0]][self.p[1]] <= self.hidden_dist:
            self.hidden = True
        return self.obs()

    def step(self, a):
        self.actions.append(a)
        #print('a={}'.format(a))
        dx, dy = [(1,0),(0,-1),(-1,0),(0,1)][a]
        old_x, old_y = self.p[0], self.p[1]
        x, y = self.p[0] + dx, self.p[1] + dy
        r = -1
        done = False
        self.remain -= 1
        self.p = (x, y)
        if self.remain < 0:
            done = True
        if (not (0 <= x < self.width)) or (not (0 <= y < self.height)) or self.f[self.p[0], self.p[1]] == 0:
            r -= self.edge_penalty
            done = True
        elif self.p == self.goal:
            if self.mark_env and (not self.mark_done):
                self.t = (self.t[0]+1, self.t[1])
                self.goal = self.t
                self.mark_done = True
            elif self.hrl_env and not self.mode_1:
                #done = True
                self.reset()
                self.mode_1 = not self.mode_1
            else:
                r += 100
                done = True
        elif self.hidden and (self.dist[x][y] > self.hidden_dist):
            r -= 20
            done = True
        elif self.easy:
            r = self.dist[old_x][old_y] - self.dist[x][y]
        #print(self.dist[x][y], x, y)
        elif self.dist[x][y] <= self.hidden_dist:
            self.hidden = True
        #print("p = {}, self.t={}, goal = {}, a = {}, r = {}, mode_1={}, hidden={}, done = {}".format(self.p,self.t,self.goal,a,r,self.mode_1,self.hidden,done))
        if done:
            if self.debug:
                print('r={}, actions={}'.format(r, self.actions))
            self.reset()
            self.actions = []
            self.mark_done=False
        return self.obs(), r, done, dict()
