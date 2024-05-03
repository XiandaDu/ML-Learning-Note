from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator


def create_data_set():
    # Create a sample dataset with features and labels
    dataset = [[0, 0, 0, 0, 'no'],
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
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']  # Features labels
    return dataset, labels


def create_tree(dataset, labels, feat_labels):
    class_list = [example[-1] for example in dataset]
    # If all classes in the dataset are the same, return the class label
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # If there is only one feature left, return the majority class label
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    # Choose the best feature to split the dataset
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    feat_labels.append(best_feat_label)
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    feat_value = [example[best_feat] for example in dataset]
    unique_vals = set(feat_value)
    # Recursively create tree for each unique value of the best feature
    for value in unique_vals:
        sublabels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(dataset, best_feat, value), sublabels, feat_labels)
    return my_tree


def majority_count(class_list):
    # Count the occurrences of each class label
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # Sort the class labels by their counts in descending order
    sortedclass_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclass_count[0][0]  # Return the class label with the highest count


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)  # Calculate the base entropy of the dataset
    best_info_gain = 0
    best_feature = -1
    # Iterate through each feature and calculate its information gain
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
    return best_feature  # Return the index of the feature with the highest information gain


def split_data_set(dataset, axis, val):
    # Split dataset based on a particular feature value
    ret_data_set = []
    for featVec in dataset:
        if featVec[axis] == val:
            reduced_feat_vec = featVec[:axis]
            reduced_feat_vec.extend(featVec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def calc_shannon_ent(dataset):
    # Calculate Shannon entropy of a dataset
    num_examples = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prop = float(label_counts[key]) / num_examples
        shannon_ent -= prop * log(prop, 2)
    return shannon_ent


def get_num_leafs(my_tree):
    # Get the number of leaf nodes in a decision tree
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
    # Get the depth of a decision tree
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # Plot a node with annotation
    arrow_args = dict(arrowstyle="<-")
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntrPt, parentPt, txt_string):
    # Plot text between two points
    x_mid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    y_mid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parentPt, node_txt):
    # Plot the decision tree
    decision_node = dict(boxstyle="sawtooth", fc="0.8")  # Decision node style
    leaf_node = dict(boxstyle="round4", fc="0.8")  # Leaf node style
    num_leafs = get_num_leafs(my_tree)
    firstStr = next(iter(my_tree))
    cntrPt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, node_txt)
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    second_dict = my_tree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntrPt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(inTree):
    # Create a plot for the decision tree
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # Remove x and y axis
    plot_tree.totalW = float(get_num_leafs(inTree))  # Get the total number of leaf nodes
    plot_tree.totalD = float(getTreeDepth(inTree))  # Get the total depth of the tree
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0  # x offset
    plot_tree(inTree, (0.5, 1.0), '')  # Plot the decision tree
    plt.show()


if __name__ == '__main__':
    dataset, labels = create_data_set()
    feat_labels = []
    my_tree = create_tree(dataset, labels, feat_labels)
    create_plot(my_tree)
