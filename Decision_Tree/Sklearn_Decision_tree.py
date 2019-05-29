# Sklearn之使用决策树预测隐形眼睛类型
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd
import graphviz

if __name__ == "__main__":
    with open('lenses.txt','r') as fr:
        lenses = [row.strip().split('\t') for row in fr.readlines()] # .strip()去掉每一行开头和结尾的符号,.split('\t')变成了列表
    lenses_label = []
    for row in lenses:
        lenses_label.append(row[-1])
    print(lenses_label)

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] # 特征标签
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for row in lenses:
            lenses_list.append(row[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []

    lenses_pd = pd.DataFrame(lenses_dict)

    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(),lenses_label)
    dot_data = tree.export_graphviz(clf, out_file = None,                            #绘制决策树
                        feature_names = lenses_pd.keys(),
                        class_names = clf.classes_,
                        filled=True, rounded=True,
                        special_characters=True)
    graph = graphviz.Source(dot_data)
    # graph.render("tree")
    print(clf.predict([[1, 1, 1, 0]]))


