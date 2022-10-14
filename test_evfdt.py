# Very Fast Decision Tree i.e. Hoeffding Tree, described in
# "Mining High-Speed Data Streams" (Domingos & Hulten, 2000)
#
# this program contains 2 classes: Vfdt, VfdtNode
# changed to CART: gini index
#
# Jamie
# 02/06/2018
# ver 0.03


import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.utils import check_array, check_X_y
OSM = True
BOUND_SPLITS = 0
TIE_SPLITS = 0
ATTEMPT_SPLITS = 0
DELAY_SPLITS = 0
def resetGlobal():
    global BOUND_SPLITS, TIE_SPLITS, ATTEMPT_SPLITS , DELAY_SPLITS
    BOUND_SPLITS = 0
    TIE_SPLITS = 0
    ATTEMPT_SPLITS = 0
    DELAY_SPLITS = 0

# VFDT node class
class VfdtNode:
    
    predict_count = 0
    attempt_count = 0
    def __init__(self, possible_split_features):
        """
        nijk: statistics of feature i, value j, class
        possible_split_features: features list
        """
        
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nmax = 0
        self.predict = 0
        self.nijk = {f: {} for f in possible_split_features}
        self.possible_split_features = possible_split_features
        
        
        
    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self
        self.nijk.clear()  # reset stats   也不往下傳了
     
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f
                                for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
    
                new_features = [None if f == split_feature else f
                                for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree
    # to distribute data examples to corresponding leaves
    def sort_example(self, x):   #x是data 各個attribute值
        #print("x是",x)
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            #print(type(self.possible_split_features))
            #print('self.split_feature =',  self.split_feature)
            #print('index =',index)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:
            prediction = max(self.class_frequency,
                             key=self.class_frequency.get)
            #print('預測:',prediction)
            #print('數值:',self.class_frequency[prediction])
            #print('---------------------------------------------')
            #print(self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency    #回傳開leaf最多的class
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):    
        #print('x是',x)
        #print('y是',y)
        feats = self.possible_split_features
        nijk = self.nijk
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]
            if value not in nijk[i]:
                nijk[i][value] = {y: 1}
                #print('結果',nijk[i][value])
            else:
                try:
                    nijk[i][value][y] += 1
                except KeyError:
                    nijk[i][value][y] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        class_frequency = self.class_frequency
        try:
            #print(class_frequency)
            class_frequency[y] += 1
        except KeyError:
            class_frequency[y] = 1
            #self.nmax = np.log(len(self.class_frequency))**2 * np.log(1/delta)/2/(tau**2 )
            
    def check_not_splitting(self):               #原本節點
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            #print('嚼果',j,k)
            X0 -= (k/n)**2
        return X0
   
    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        global BOUND_SPLITS, TIE_SPLITS, ATTEMPT_SPLITS, DELAY_SPLITS, OSM
        class_frequency = self.class_frequency
        if len(class_frequency) == 1:
            return None
        self.nmax = np.log(len(self.class_frequency))**2 * np.log(1/delta)/2/(tau**2)
        D1_class_frequency = {}
        D2_class_frequency = {}
        if(self.total_examples_seen >=self.nmax):#有沒有超過self.nmax，有就強迫他分
            VfdtNode.attempt_count += 1 #tie-split也要算attemp??????
            TIE_SPLITS += 1
            ATTEMPT_SPLITS += 1       ###不確定 #tie-split也要算attemp??????
            nijk = self.nijk
            min = 1
            second_min = 1
            Xa = ''
            split_value = None
            for feature in self.possible_split_features:
                if feature is not None:
                    njk = nijk[feature]
                    gini, value,D1_frequent,D2_frequent = self.gini(njk, class_frequency)
                    if gini < min:
                        min = gini
                        Xa = feature
                        split_value = value
            return [Xa, split_value]  
        if self.new_examples_seen < max(nmin,self.predict):     #跟我們predict出來的值比較
            return None
        
        VfdtNode.attempt_count += 1
        ATTEMPT_SPLITS += 1
        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                gini, value,D1_frequent,D2_frequent = self.gini(njk, class_frequency)
                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                    D1_class_frequency = D1_frequent
                    D2_class_frequency = D2_frequent
                   
                elif min < gini < second_min:
                    second_min = gini
        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            # print(second_min - min, epsilon)
            if second_min - min > epsilon:
                BOUND_SPLITS += 1
                # print('1 node split')
                return [Xa, split_value]
            elif tau != 0 and second_min - min < tau:
                # print('2 node split')
                BOUND_SPLITS += 1
                return [Xa, split_value]
            else:
                #g_d1 = 1
                #g_d2 = 1
                #D1 = sum(D1_class_frequency.values())
                #D2 = sum(D2_class_frequency.values())
                #D = D1+D2
                #for key, v in D1_class_frequency.items():
                    #g_d1 -= (v/D1)**2
                #for key, v in D2_class_frequency.items():
                    #g_d2 -= (v/D2)**2
                #g = g_d1*D1/D + g_d2*D2/D
                #print(second_min-g)
                if(OSM):
                    self.splitTimePredictionOSM(D1_class_frequency, D2_class_frequency,second_min,delta,tau)
                    return None
        else:
            if(OSM):
                self.splitTimePredictionOSM(D1_class_frequency, D2_class_frequency,second_min,delta,tau)
                return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        #print(R)
        #print('n =',np.sqrt(R * R * np.log(1/delta) / (2 * n)))
        return np.sqrt(R * R * np.log(1/delta) / (2 * n))

    def gini(self, njk, class_frequency):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential  該feature下的值然後排序
        return_frequent = {}
        return_frequent2 = {}
        #print(njk.keys())
        if not isinstance(feature_values[0], str):  # numeric  feature values
            sort = np.array(sorted(feature_values))
            #print(sort)
            # vectorized computation, like in R
            split = (sort[0:-1] + sort[1:])/2   #取各值之間值，chapter3 p43
            
            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
            
            for index in range(len(split)):
                #print(index)
                nk = njk[sort[index]]    #nk是該值下的class和數目
                for j in nk:
                    D1_class_frequency[j] += nk[j]     #各個class 的元素個數
                    #print(D1_class_frequency)
                D1 = sum(D1_class_frequency.values())  #剛開始是0
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value-D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value
                
                for key, v in D1_class_frequency.items():
                    g_d1 -= (v/D1)**2
                    #print('D1=',D1)
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v/D2)**2
                    #print('D2=',D2)
                g = g_d1*D1/D + g_d2*D2/D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]     #分裂的數值
                    return_frequent = D1_class_frequency.copy()
                    return_frequent2 = D2_class_frequency.copy()
                    #print(self.total_examples_seen)
                    #print('回傳字典對',return_frequent,'|||||||||||',D2_class_frequency)
                # elif m1 < g < m2:       #算次小的
                    # m2 = g
            return [m1, Xa_value,return_frequent,return_frequent2]

        else:  # discrete feature_values
            length = len(njk)
            if length > 10:  # too many discrete feature values, estimate
                for j, k in njk.items():  #ex:attribute 第j個值 第k個class 有多少個k.values()
                    D1 = sum(k.values())    #
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]  #剛開始全滿
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v/D1)**2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                        return_frequent = D1_class_frequency.copy()
                        return_frequent2 = D2_class_frequency.copy()
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values, get combinations
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v/D1)**2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                        return_frequent = D1_class_frequency.copy()
                        return_frequent2 = D2_class_frequency.copy()
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
                
            return [m1, [Xa_value, right],return_frequent,return_frequent2]

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e/2)
            for i in range(1, end+1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb)/2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e-1)/2)
            for i in range(1, end+1):
                combination.extend(combinations(feature_values, i))

        return combination
         #OSM Function
    def brent(self, a, b, func):  
        fa = func(a)
        fb = func(b)
        if(fa*fb>=0):
            return self.nmax
            #TODO: Fosm = nmax
            return None
        if(abs(fa)<abs(fb)):
            tmp = fa
            fa = fb
            fb = tmp
            tmp = a
            a = b
            b = tmp
        c = a
        fc = fa
        mflag = True
        #TODO: chg oEps and iEps parameter
        while (abs(fb)>0.005 and abs(b-a)>0.5):
            if(fa!=fc and fb!=fc):
                c0 = (a*fb*fc) / ((fa-fb)*(fa-fc))
                c1 = (b*fa*fc) / ((fb-fa)*(fb-fc))
                c2 = (c*fa*fb) / ((fc-fa)*(fc-fb))
                s = c0+c1+c2
            else:
                s = b-fb*(b-a)/(fb-fa)
            if(not (3*a+b)/4<s<b) or (mflag and abs(s-b)>=abs(b-c)/2) or (not mflag and abs(s-b)>=abs(c-d)/2):
                s = (a+b)/2
                mflag = True
            else:
                mflag = False
            fs = func(s)
            d = c
            c = b
            fc = fb
            if(fa*fs<0):
                b = s
            else:
                a = s
            if(abs(fa)<abs(fb)):
                tmp = fa
                fa = fb
                fb = tmp
                tmp = a
                a = b
                b = tmp
        return b
    
    def splitTimePredictionOSM(self,D1_class_frequency,D2_class_frequency,gini2,delta,tau):
        global DELAY_SPLITS
        DELAY_SPLITS += 1
        VfdtNode.predict_count += 1
        D2 = sum(D2_class_frequency.values())
        g_d2 = 1
        for key, v in D2_class_frequency.items():
            g_d2 -= (v/D2)**2
        def predict(x): 
            copy_D1_class_frequency = D1_class_frequency.copy()
            maxclass = max(copy_D1_class_frequency)    #再做E函數的部分
            #print('之前=',copy_D1_class_frequency)
            copy_D1_class_frequency[maxclass] += x     #再做E函數的部分
            #print(copy_D1_class_frequency)
            D1 = sum(copy_D1_class_frequency.values())
            g_d1 = 1
            for key, v in copy_D1_class_frequency.items():
                g_d1 -= (v/D1)**2
            D = D1+D2
            g = g_d1*D1/D + g_d2*D2/D
            #return gini2 - g - np.log(((len(copy_D1_class_frequency))**2) * np.log(1/delta)/(2*D-x))  
            #print(len(copy_D1_class_frequency)**2)
            #print('--------------------------------')
            #print(np.log(1/delta))
            #print('--------------------------------')
            #print((2*D)-x)
            #print(gini2 - g - np.sqrt((len(copy_D1_class_frequency)**2) * np.log(1/delta)/((2*D)-x)))
            return gini2 - g - np.sqrt((len(copy_D1_class_frequency)**2) * np.log(1/delta)/((2*D)-x))
        self.predict = int(self.brent(0, int(self.nmax),predict))
          
                  
         
# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, features, delta=0.01, nmin=100, tau=0.1):
        """
        :features: list of data features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        #self.nmax
        self.root = VfdtNode(features)
        self.n_examples_processed = 0
        print(self.features, self.delta, self.tau,self.n_examples_processed)
        # self.print_tree()
        # print("--- / __init__ ---")

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        X, y = check_X_y(X, y)
        for x, _y in zip(X, y):
            self.__update(x, _y)
        # print("Update!")
        # print("X {}: {}".format(type(X), X))
        # print("y {}: {}".format(type(y), y))
        # self.print_tree()
        # print("---")
        # if isinstance(y, (np.ndarray, list)):
        #     for x, _y in zip(X, y):
        #         self.__update(x, _y)
        # else:
        #     self.__update(X, y)
        # self.print_tree()
        # print("---")
        # print("End update! n_examples_processed={}".format(
        #     self.n_examples_processed))
        # print("--- --- ---")

    # update the tree by adding one training example
    def __update(self, x, _y):
        self.n_examples_processed += 1
        node = self.root.sort_example(x)  #找葉節點
        node.update_stats(x, _y)          #然後更新他

        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:    #這裡重要
            feature = result[0]
            value = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    def node_split(self, node, split_feature, split_value):                  #分樹用
        features = node.possible_split_features
        #print('node_split')
        left = VfdtNode(features)
        right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, X):                          
        X = check_array(X)
        return [self.__predict(x) for x in X]
        # if isinstance(X, (np.ndarray, list)):
        #     return [self.__predict(x) for x in X]
        # else:
        #     leaf = self.__predict(X)

    def __predict(self, x):                         #
        leaf = self.root.sort_example(x)
        return leaf.most_frequent()

    def print_tree(self, node=None):
        if node is None:
            self.print_tree(self.root)
        elif node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)
    
    def __recur_tree(self, n, str_dot='', node=None):
        if node is None:
            return self.__recur_tree(n, node=self.root)
        elif node.is_leaf():
            #print(n)
            node.trace_num = str(n)
            #tmp = str_dot + str(n)+(' [label="%d(Leaf)"] ;\n'%n)
            tmp = str_dot + str(n)+(' [label="sample=%d", fillcolor="#99D9EA"] ;\n'%(node.total_examples_seen))
            if(node.parent is not None):
                if(node.parent.left_child==node):
                    llabel = '<=%.1f'%node.parent.split_value
                else:
                    llabel = '>%.1f'%node.parent.split_value
                label = ' [label="%s"]'%llabel
                tmp += node.parent.trace_num+' -> '+node.trace_num+label+' ;\n'
            return n, tmp
        else:
            #print(n)
            node.trace_num = str(n)
            #tmp = str_dot + str(n)+(' [label="%d"] ;\n'%n)
            tmp = str_dot + str(n)+(' [label="split feature=%d\nsample=%d", fillcolor="#FFFFFF"] ;\n'%(node.split_feature,node.total_examples_seen))
            if(node.parent is not None):
                if(node.parent.left_child==node):
                    llabel = '<=%.1f'%node.parent.split_value
                else:
                    llabel = '>%.1f'%node.parent.split_value
                label = ' [label="%s"]'%llabel
                tmp += node.parent.trace_num+' -> '+node.trace_num+label+' ;\n'
            n, tmp = self.__recur_tree(n+1, str_dot=tmp, node=node.left_child)
            n, tmp = self.__recur_tree(n+1, str_dot=tmp, node=node.right_child)
            return n, tmp
            
    def save_tree(self, fname='VFDT'):
        import os
        from subprocess import call
        n, tmp= self.__recur_tree(0)
        fname = fname+("_%d"%n)+'.dot'
        dot = 'digraph VFDT {\n' + 'node [shape=box , style="filled"];\n' + tmp +'}'
        f = open(fname,'w+')
        f.write(dot)
        f.close()
        call(['dot', '-Tpng', fname, '-o', fname.split('.')[0]+'.png', '-Gdpi=600'], shell=True)
        os.unlink(fname)

def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(
        precision_recall_fscore_support(
            y_test, y_pred, average='weighted',
            labels=np.unique(y_pred)))
    metrics = pd.DataFrame({
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2]}, index=[row_name])
    return metrics
def test_accuracy_n_time(dataset):   #dataset檔名
    import json
    global OSM
    df = pd.read_csv('C:/Users/USER/Desktop/python_module/'+dataset,header=None, sep=',')
    title = list(df.columns.values)
    features = title[:-1]        #最後一項，最後一項是分類
    rows = df.shape[0]            #算總data樹木
    n_training = int(rows)
    array = df.head(n_training).values  #前面數來n個值
    inter = 20000    #間隔
    examples = list()
    
    for i in range(0,n_training,inter):
        examples.append(array[i:i+inter, :] if(i+inter<n_training) else array[i:, :])
        
    # test set is different from training set
    n_test = int(rows*1/4)
    test_set = df.tail(n_test).values   #後面數來n個值
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    print('Training size: ', n_training)
    print('Test set size: ', n_test)
    n = 0
    sample_x = list()
    acc_VFDT = list()
    acc_OSM = list()
    runtime_VFDT = list()
    runtime_OSM = list()
    
    OSM = False     #這裡是普通的樹
    
    tree = Vfdt(features, delta=1e-7, nmin=200, tau=0.05)
    start_time = time.time()
    for training_set in examples:
        n += len(training_set)                 #紀錄總數
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        tree.update(x_train, y_train)
        y_pred = tree.predict(x_test)
        sample_x.append(n)
        acc_VFDT.append(1-accuracy_score(y_test, y_pred))    #計算錯誤率
        runtime_VFDT.append(time.time() - start_time)
        print('Training set:', n, end=', ')
        print('VFDT Error Rate: %.4f' % acc_VFDT[-1])
        
    OSM = True       #這是evfdt
    tree = Vfdt(features, delta=1e-7, nmin=200, tau=0.05)
    start_time = time.time()
    for training_set in examples:                            #每隔20000筆資料跑一次
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        tree.update(x_train, y_train)
        y_pred = tree.predict(x_test)
        acc_OSM.append(1-accuracy_score(y_test, y_pred))
        runtime_OSM.append(time.time() - start_time)
        print('OSM Error Rate: %.4f' % acc_OSM[-1])
    f = open('C:/Users/USER/Desktop/python_module/'+dataset+'_sample_x','w+')
    f.write(json.dumps(sample_x))
    f.close()
    f = open('C:/Users/USER/Desktop/python_module/'+dataset+'_err_VFDT','w+')
    f.write(json.dumps(acc_VFDT))
    f.close()
    f = open('C:/Users/USER/Desktop/python_module/'+dataset+'_time_VFDT','w+')
    f.write(json.dumps(runtime_VFDT))
    f.close()
    f = open('C:/Users/USER/Desktop/python_module/'+dataset+'_err_OSM','w+')
    f.write(json.dumps(acc_OSM))
    f.close()
    f = open('C:/Users/USER/Desktop/python_module/'+dataset+'_time_OSM','w+')
    f.write(json.dumps(runtime_OSM))
    f.close()

def test_split_count(dataset):
    global OSM
    NMIN = [50,200]
    for nmin in NMIN:
        attempt = {'Dataset':[],'VFDT':[],'OSM':[]}
        delay = {'Dataset':[],'VFDT':[],'OSM':[]}
        bound = {'Dataset':[],'VFDT':[],'OSM':[]}
        tie = {'Dataset':[],'VFDT':[],'OSM':[]}
        for dset in dataset:
            attempt['Dataset'].append(dset)
            delay['Dataset'].append(dset)
            bound['Dataset'].append(dset)
            tie['Dataset'].append(dset)
            df = pd.read_csv('C:/Users/USER/Desktop/python_module/'+dset,header=None, sep=',')
            title = list(df.columns.values)
            features = title[:-1]
            rows = df.shape[0]
            n_training = int(rows)
            array = df.head(n_training).values
            n_test = int(rows*1/4)
            test_set = df.tail(n_test).values
            x_test = test_set[:, :-1]
            y_test = test_set[:, -1]
            print('Training size: ', n_training)
            print('Test set size: ', n_test)
            OSM = False
            resetGlobal()
            tree = Vfdt(features, delta=1e-7, nmin=nmin, tau=0.05)
            x_train = array[:, :-1]
            y_train = array[:, -1]
            tree.update(x_train, y_train)
            attempt['VFDT'].append(ATTEMPT_SPLITS)
            delay['VFDT'].append(DELAY_SPLITS)
            bound['VFDT'].append(BOUND_SPLITS)
            tie['VFDT'].append(TIE_SPLITS)
            OSM = True
            resetGlobal()
            tree = Vfdt(features, delta=1e-7, nmin=nmin, tau=0.05)
            x_train = array[:, :-1]
            y_train = array[:, -1]
            tree.update(x_train, y_train)
            attempt['OSM'].append(ATTEMPT_SPLITS)
            delay['OSM'].append(DELAY_SPLITS)
            bound['OSM'].append(BOUND_SPLITS)
            tie['OSM'].append(TIE_SPLITS)
        
        df = pd.DataFrame(attempt)
        df.to_excel("attempts_"+str(nmin)+'.xlsx',index=False,columns=['Dataset','VFDT','OSM'])
        df = pd.DataFrame(delay)
        df.to_excel("delay_"+str(nmin)+'.xlsx',index=False,columns=['Dataset','VFDT','OSM'])
        df = pd.DataFrame(bound)
        df.to_excel("bound_splits_"+str(nmin)+'.xlsx',index=False,columns=['Dataset','VFDT','OSM'])
        df = pd.DataFrame(tie)
        df.to_excel("tie_splits_"+str(nmin)+'.xlsx',index=False,columns=['Dataset','VFDT','OSM'])

def test_tree_visualize(dataset):
    df = pd.read_csv('C:/Users/USER/Desktop/python_module/'+dataset,header=None, sep=',')
    title = list(df.columns.values)
    features = title[:-1]
    rows = df.shape[0]
    n_training = int(rows*1/4)
    array = df.head(n_training).values  #前面數來n個值
    i = int(n_training/3)
    j = int(n_training*2/3)
    set1 = array[:i, :]
    set2 = array[i:j, :]
    set3 = array[j:, :]

    # to simulate continuous training, modify the tree for each training set
    examples = [set1, set2, set3]
    #examples = [set1]
    # test set is different from training set
    n_test = rows - n_training
    test_set = df.tail(n_test).values   #後面數來n個值
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    tree = Vfdt(features, delta=1e-7, nmin=100, tau=0.05)
    print('Total data size: ', rows)
    print('Training size: ', n_training)
    print('Test set size: ', n_test)
    n = 0
    for training_set in examples:
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        tree.update(x_train, y_train)
        y_pred = tree.predict(x_test)  #data train完了

        print('Training set:', n, end=', ')
        print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))
    tree.save_tree(fname="EVFDT_Fin")

def test_run():
    global BOUND_SPLITS, TIE_SPLITS, ATTEMPT_SPLITS , DELAY_SPLITS,OSM
    start_time = time.time()
    df = pd.read_csv('C:/Users/USER/Desktop/python_module/covtypeNorm.csv', header=None, sep=',')
    # df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
    # df = df.drop(df.columns[0], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
    title = list(df.columns.values)                  #第一列當title
    print('標題=',title)
    features = title[:-1]        #最後一項，最後一項是分類
    rows = df.shape[0]            #算總data樹木
    print("Features:", features)

    # change month string to int
    def month_str_to_int(df1):
        import calendar
        d = dict((v.lower(), k) for k, v in enumerate(calendar.month_abbr))
        df1.month = df1.month.map(d)
    # month_str_to_int(df)
    # print(df.head(5)['month'])
    # convert df to data examples
    n_training = int(rows/4)
    array = df.head(n_training).values  #前面數來n個值
    i = int(n_training/3)
    j = int(n_training*2/3)
    set1 = array[:i, :]
    set2 = array[i:j, :]
    set3 = array[j:, :]
    examples = [set1, set2, set3]
    #examples = [set1]
    # test set is different from training set
    n_test = 50000
    test_set = df.tail(n_test).values   #後面數來n個值
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    # Heoffding bound (epsilon) parameter delta: with 1 - delta probability
    # the true mean is at least r_bar - epsilon
    # Efdt parameter nmin: test split if new sample size > nmin
    # feature_values: unique values in every feature
    # tie breaking: when difference is so small,
    # split when diff_g < epsilon < tau
    tree = Vfdt(features, delta=1e-7, nmin=200, tau=0.05)
    print('Total data size: ', rows)
    print('Training size: ', n_training)
    print('Test set size: ', n_test)
    n = 0
    for training_set in examples:
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        tree.update(x_train, y_train)
        y_pred = tree.predict(x_test)  #data train完了

        print('Training set:', n, end=', ')
        print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

    #tree.print_tree(tree.root)
    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))
    print('attemp =', ATTEMPT_SPLITS)
    print('predict =', DELAY_SPLITS)
    tree.save_tree()

if __name__ == "__main__":
    #test_accuracy_n_time('poker-lsn.arff.csv')
    #resetGlobal()
    #test_accuracy_n_time('covtypeNorm.csv')
    #resetGlobal()
    #test_split_count(['poker-lsn.arff.csv','covtypeNorm.csv'])
    #test_run() 
    #test_tree_visualize('poker-lsn.arff.csv')
    #resetGlobal()
    #test_tree_visualize('covtypeNorm.csv')
    test_run()
    OSM = True
    
