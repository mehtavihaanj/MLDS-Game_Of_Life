import numpy as np
from battle.agents import Agent
from scipy import signal


class priority_agent(Agent):
    def __init__(self,
                 obs_onehot: bool = False,
                 attack_map: bool = False,
                 attack_map_logits: bool = False,
                 ):

        super().__init__(obs_onehot, attack_map, attack_map_logits)
        self.obs_onehot = obs_onehot
        self.attack_map = attack_map
        self.attack_map_logits = attack_map_logits
        self.adj_mat_ind = None
        return

    def get_adj_indices(self, h, w, n=1):
        adj_width = 2 * n + 1
        X, Y = np.indices((h, w))
        x, y = np.indices((adj_width, adj_width))
        x -= n
        y -= n
        X = X.reshape(h, w, 1, 1) + x
        Y = Y.reshape(h, w, 1, 1) + y

        X = X % h
        Y = Y % w

        X = X.reshape(h, w, -1)
        Y = Y.reshape(h, w, -1)
        return X, Y

    def get_pop(self, state_mat):
        abs_map = (state_mat >= 2)  # get locations where there is an agent
        sum_mat = signal.convolve2d(abs_map, np.ones((3, 3)), boundary='wrap', mode='same')

        return sum_mat

    def get_max_pop(self, pop_mat):
        adj_pop = pop_mat[self.X_ind, self.Y_ind]
        max_pop = np.max(adj_pop, axis=2)

        return max_pop

    def smart_adj(self, feat_mat, pop_mat):  # helper function from individual cell perspective

        feat_mat = feat_mat.reshape(-1, 1)

        enemy_list = np.argwhere(feat_mat >= 3)[:, 0]
        empty_list = np.argwhere((feat_mat == 0) * (pop_mat <= 6))[:, 0]

        if len(enemy_list) >= 1:
            # print(enemy_list)
            # select the enemy of lowest population
            # enemy_ind = np.argmax(pop_mat[enemy_list])
            return np.random.choice(enemy_list, 1)[0]
            # return enemy_list[enemy_ind]
        elif len(empty_list) >= 1:
            enemy_ind = np.argmin(pop_mat[empty_list])
            return empty_list[enemy_ind]
        else:
            return 12

    def policy(self, state_mat, action_space=None, obs_space=None):
        attack_mat = np.zeros(state_mat.shape).astype(int)

        # get the indices for adjacent cells
        if self.adj_mat_ind is None:
            h, w = state_mat.shape
            self.X_ind, self.Y_ind = self.get_adj_indices(h, w, n=2)

        X_ind = self.X_ind
        Y_ind = self.Y_ind

        pop_mat = self.get_pop(state_mat)
        max_pop = self.get_max_pop(pop_mat)

        for i in range(attack_mat.shape[0]):  # for each row
            for j in range(attack_mat.shape[1]):  # for each column

                current = state_mat[i, j]
                if current == 2:  # only run if theres an ally cell here.

                    adj_x = X_ind[i, j, :]
                    adj_y = Y_ind[i, j, :]

                    adj_pop = max_pop[adj_x, adj_y]
                    feat_mat = state_mat[adj_x, adj_y]

                    ind = self.smart_adj(feat_mat, adj_pop)  # get cell feats returns adjacent cells
                    attack_mat[i, j] = ind

                    a_x = adj_x[ind]  # x pos being attacked
                    a_y = adj_y[ind]  # y pos being attacked

                    target = state_mat[a_x, a_y]

                    if target == 0:  # go back and edit the population map.
                        state_mat[a_x, a_y] = 2

                        adj_x = X_ind[a_x, a_y, :]
                        adj_y = Y_ind[a_x, a_y, :]

                        pop_mat[adj_x, adj_y] += 1
                        max_pop = self.get_max_pop(pop_mat)

        return attack_mat.astype(int)


class rand_enemy_agent(Agent):
    def rand_adj(self, feat_mat): #helper function from individual cell perspective

        feat_mat = feat_mat.reshape(-1,1)

        enemy_list = np.argwhere(feat_mat >=3)[:,0]
        empty_list = np.argwhere(feat_mat == 0)[:,0]

        if len(enemy_list)>=1:
            #print(enemy_list)
            return np.random.choice(enemy_list,1)[0]
        elif len(empty_list)>=1:
            return np.random.choice(empty_list,1)[0]
        else:
            return 12

    def get_cell_feats(self,state_mat, x,y, n = 2):
        w,h = state_mat.shape
        adj_width = 2*n+1

        xind,yind = np.indices((adj_width,adj_width))
        xind = (xind -n +x)%h
        yind = (yind -n +y)%w

        #gets a matrix of the adjacent cells by 1
        adj_vals = state_mat[xind,yind]
        return adj_vals

    def policy(self, state_mat, action_space = None, obs_space = None):
        attack_mat =np.zeros(state_mat.shape)

        for i in range(attack_mat.shape[0]):
            for j in range(attack_mat.shape[1]):
                attack_mat[i,j] = self.rand_adj(self.get_cell_feats(state_mat, i,j, 2)) #get cell feats returns adjacent cells

        return attack_mat.astype(int)
