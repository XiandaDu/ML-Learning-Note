from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator


def create_data_set():
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return data_set, labels


def create_tree(dataset, labels, feat_labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majority_count(classList)
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    feat_labels.append(best_feat_label)
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    feat_value = [example[best_feat] for example in dataset]
    unique_vals = set(feat_value)
    for value in unique_vals:
        sublabels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(dataset, best_feat, value), sublabels, feat_labels)
    return my_tree


def majority_count(classList):
    class_count = {}
    for vote in classList:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sortedclass_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclass_count[0][0]


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0
        for val in unique_vals:
            sub_data_set = split_data_set(dataset, i, val)
            prob = len(sub_data_set) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def split_data_set(dataset, axis, val):
    ret_data_set = []
    for featVec in dataset:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            ret_data_set.append(reducedFeatVec)
    return ret_data_set


def calc_shannon_ent(dataset):
    num_examples = len(dataset)
    label_counts = {}
    for featVec in dataset:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prop = float(label_counts[key]) / num_examples
        shannon_ent -= prop * log(prop, 2)
    return shannon_ent


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def getTreeDepth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth: max_depth = this_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args, FontProperties=font)


def plot_mid_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plot_tree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = get_num_leafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrPt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plot_tree.totalW = float(get_num_leafs(inTree))  # 获取决策树叶结点数目
    plot_tree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0  # x偏移
    plot_tree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, labels = create_data_set()
    feat_labels = []
    my_tree = create_tree(dataset, labels, feat_labels)
    create_plot(my_tree)
