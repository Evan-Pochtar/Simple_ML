import numpy as np

class Tree_node:
    def __init__(self,):
        self.feature = None
        self.label = None
        self.left_child = None
        self.right_child = None


class Decision_tree:
    def __init__(self,min_entropy, metric='entropy'):
        if metric == 'entropy':
            self.metric = self.entropy
        else:
            self.metric = self.gini_index
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        prediction = []
        for i in range(len(test_x)):
            cur_data = test_x[i]
            cur_node = self.root
            while True:
                if cur_node.label != None:
                    break
                elif cur_node.feature == None:
                    print("You haven't selected the feature yet")
                    exit()
                else:
                    if cur_data[cur_node.feature] == 0:
                        cur_node = cur_node.left_child
                    else:
                        cur_node = cur_node.right_child
            prediction.append(cur_node.label)

        prediction = np.array(prediction)

        return prediction

    def generate_tree(self,data,label):
        cur_node = Tree_node()
        node_entropy = self.metric(label)
        if node_entropy < self.min_entropy:
            cur_node.label = np.argmax(np.bincount(label))
            return cur_node

        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        select_x = data[:, selected_feature]
        
        left_x = data[select_x == 0, :]
        left_y = label[select_x == 0]
        right_x = data[select_x == 1, :]
        right_y = label[select_x == 1]
        if len(left_y) > 0:
            cur_node.left_child = self.generate_tree(left_x, left_y)
        if len(right_y) > 0:
            cur_node.right_child = self.generate_tree(right_x, right_y)

        return cur_node

    def select_feature(self,data,label):
        best_feat = 0
        min_entropy = float('inf')

        for i in range(len(data[0])):
            split_x = data[:,i]
            left_y = label[split_x==0,]
            right_y = label[split_x==1,]

            cur_entropy = self.combined_entropy(left_y,right_y)

            if cur_entropy < min_entropy:
                min_entropy = cur_entropy
                best_feat = i

        return best_feat


    def combined_entropy(self,left_y,right_y):
        result = 0
        wleft = len(left_y) / (len(left_y) + len(right_y))
        wright = len(right_y) / (len(left_y) + len(right_y))
        result = wleft * self.metric(left_y) + wright * self.metric(right_y)

        return result

    def entropy(self,label):
        node_entropy = 0
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / counts.sum()
        for percent in probabilities:
            node_entropy += percent*np.log2(percent+1e-15)
        node_entropy *= -1
        return node_entropy

    def gini_index(self, label):
        gini_index = 0
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / counts.sum()
        gini_index = 1 - sum(probabilities**2)
        return gini_index
