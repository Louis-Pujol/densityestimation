''' Tree structure inspired by : https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html '''

import numpy as np

class Tree:
    
    def __init__(self):
        
        self.LEAF = -1
        self.n_nodes = 1
        
        self.left_child = [self.LEAF]
        self.right_child = [self.LEAF]
        self.depth = [0]
        self.parent = [self.LEAF]
        
    
    def split(self, node_id):
        
        if node_id >= self.n_nodes:
            #Error
            return 0
        if self.right_child[node_id] != self.LEAF:
            #Error
            return 0
        
        
        self.left_child[node_id] = self.n_nodes
        self.right_child[node_id] = self.n_nodes+1
        
        self.left_child += [self.LEAF, self.LEAF]
        self.right_child += [self.LEAF, self.LEAF]
        self.depth += [self.depth[node_id] + 1, self.depth[node_id] + 1]
        self.parent += [node_id, node_id]
        self.n_nodes += 2
    
    def go_left(self, node_id):
        
        if node_id >= self.n_nodes:
            #Error
            print("erreur")
            return 0
        if self.left_child[node_id] == self.LEAF:
            #Error
            print("erreur")
            return 0
        
        return self.left_child[node_id]

    def go_right(self, node_id):
        
        if node_id >= self.n_nodes:
            #Error
            print("erreur")
            return 0
        if self.right_child[node_id] == self.LEAF:
            #Error
            print("erreur")
            return 0
        
        return self.right_child[node_id]
    
    def path_to_root(self, node_id):
        
        if node_id >= self.n_nodes:
            #Error
            print("erreur")
            return 0
        
        path = [node_id]
        i = node_id
        
        while self.parent[i] != self.LEAF:
            path.append(self.parent[i])
            i = self.parent[i]
        
        #while( i != 0 ):
            #if i in self.left_child:
                #j = np.where(np.array(self.left_child) == i)[0][0]
            #elif i in self.right_child:
                #j = np.where(np.array(self.right_child) == i)[0][0]
        
            #path += [j]
            #i = j
        
        return path

    def leafs(self):
        
        leafs = []
        for i in range(self.n_nodes):
            if self.right_child[i] == self.LEAF:
                leafs.append(i)
        return leafs
    
    def max_depth(self):
        return np.max(self.depth)
    
    def is_leaf(self, node_id):
        return left_child[node_id] == self.LEAF
            
    
class CartDensityTree(Tree):
    
    def __init__(self):
        super().__init__()
        self.cut_axe = [self.LEAF]
        self.threshold = [self.LEAF]
        self.count = [self.LEAF]
        
    def split(self, node_id, cut_axe, threshold):
        super().split(node_id)
        self.threshold[node_id] = threshold
        self.cut_axe[node_id] = cut_axe
        #TODO maj de count et de volume (nécessite p-e plus d'info)
        
        self.threshold += [self.LEAF, self.LEAF]
        self.cut_axe += [self.LEAF, self.LEAF]
        
    def locate(self, x):
        
        #TODO vérifier la dimension
        i = 0
        while self.left_child[i] != self.LEAF:
            if x[self.cut_axe[i]] <= self.threshold[i]:
                i = left_child[i]
            else:
                i = right_node[i]
                
        return i
        
    def fit(self, X, max_depth=2, method='diadyc'):
        n, d = X.shape
        self.dim = d
        
        indices = [np.array(range(n))]
        self.count[0] = n
        while( self.max_depth() < max_depth ):
            
            for i in self.leafs():
                
                if method == 'diadyc':
                    
                    ''' axe will be the maximizer of ll among dyadic cuts '''
                    c = len(indices[i])
                    axe = 0
                    threshold = np.mean(self.rectangle(i)[axe, :])
                    c_left = len(np.where(X[indices[i], axe] <= threshold)[0])
                    c_right = len(np.where(X[indices[i], axe] > threshold)[0])
                    best_score_cut = c_left * np.log(c_left) + c_right * np.log(c_right)
                    
                    for j in range(1, self.dim):
                        threshold = np.mean(self.rectangle(i)[j, :])
                        c_left = len(np.where(X[indices[i], j] <= threshold)[0])
                        c_right = len(np.where(X[indices[i], j] > threshold)[0])
                        score_cut = c_left * np.log(c_left) + c_right * np.log(c_right)
                        if score_cut > best_score_cut:
                            best_score_cut = score_cut
                            axe = j
                    
                    
                    threshold = np.mean(self.rectangle(i)[axe, :])
                
                
                self.split(i, axe, threshold)
                indices += [np.where(X[indices[i], axe] <= threshold)[0]] 
                indices += [np.where(X[indices[i], axe] > threshold)[0]]
        
        self.count = self.n_nodes * [0]
        for j in range(self.n_nodes):
            self.count[j] = len(indices[j])
            
    
    def rectangle(self, i):
        ''' Return limits of rectangle corresponding to node i '''
        
        rectangle = np.zeros(shape=(self.dim, 2))
        rectangle[:, 1] = np.ones(self.dim)
        
        
        path = self.path_to_root(i)[::-1]
        for j in range(len(path)-1):

            
            if self.left_child[path[j]] == path[j+1]:
                rectangle[self.cut_axe[path[j]], 1] = self.threshold[path[j]]
            elif self.right_child[path[j]] == path[j+1]:
                rectangle[self.cut_axe[path[j]], 0] = self.threshold[path[j]]
                
        return rectangle
    
    def volume(self, i):
        ''' return the volume corresponding to the node i '''
        r = self.rectangle(i)
        return np.prod(r[:, 1] - r[:, 0])
    
    def log_likelihood(self):
        
        ll = 0
        n = self.count[0]
        for i in self.leafs():
            v = self.volume(i)
            c = self.count[i]
            
            if c != 0:
                
                ll += c * np.log( c / (v * n) ) / n
        
        return(ll)
    
    
    

def show_hist2d(tree):
    

    import matplotlib.pyplot as plt
    from matplotlib import cm as cm
    
    n  = tree.count[0]
    
    
    for i in tree.leafs():
        intensity = tree.count[i] / n
        r = tree.rectangle(i)
        plt.plot([r[0,0], r[0,0], r[0,1], r[0,1], r[0, 0]] , [r[1,0], r[1,1], r[1,1], r[1,0], r[1, 0]], c='r')
    
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()
    
        
if __name__ == '__main__':
    
    t = CartDensityTree()
    X = np.random.rand(5000,2)
    t.fit(X, max_depth=4)
    
    print(t.log_likelihood())

    show_hist2d(t)

    #print(t.left_child)
    #print(t.right_child)
    #print(t.cut_axe)
    #print(t.threshold)
    #print(t.leafs())
    
    
    
    

