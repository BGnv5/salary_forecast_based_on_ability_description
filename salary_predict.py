import pandas as pd
from collections import defaultdict
import networkx as nx
import random
import matplotlib.pyplot as plt
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# 数据加载
content = pd.read_excel('./jobs_4k.xls')
position_names = content['positionName'].tolist()
# print(len(position_names))
temp = set(position_names)
# print(len(temp))
skill_labels = content['skillLables'].tolist()
# print(len(skill_labels))

# 图可视化
skill_position_grap = defaultdict(list)
for p, s in zip(position_names, skill_labels):
    skill_position_grap[p] += eval(s)
# print(skill_position_grap)
G = nx.Graph(skill_position_grap)

# 30个随机选取的工作岗位为例，进行图可视化
sample_nodes = random.sample(position_names, k=30)
sample_nodes_connections = sample_nodes
for p, skills in skill_position_grap.items():
    if p in sample_nodes:
        sample_nodes_connections += skills

# 抽取G的节点作为子图
sample_graph = G.subgraph(sample_nodes_connections)
f = plt.figure(figsize=(50, 30))
pos = nx.spring_layout(sample_graph, k=1)
nx.draw(sample_graph, pos, with_labels=True, node_size=30, font_size=10)
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
f.savefig('job-connections.png', dpi=180)

# 对核心能力和职位进行排序（按照影响力）
pr = nx.pagerank(G, alpha=0.9)
ranked_position_and_ability = sorted([(name, value) for name, value in pr.items()], key=lambda x:x[1], reverse=True)
# print(ranked_position_and_ability)

# 取出特征X， 不包括salary
X_content = content.drop(['salary'], axis=1)
# 取出target
target = content['salary'].tolist()
# print(target)
# 将X_content内容拼接成字符串
X_content['merged'] = X_content.apply(lambda x:''.join(str(x)), axis=1)
# 转化为list
X_string = X_content['merged'].tolist()
# print(X_string[0])

# 合并到一起
def get_one_row_job_string(x_string_row):
    job_string = ''
    for i, element in enumerate(x_string_row.split('\n')):
        if len(element.split()) == 2:
            _, value = element.split()
            # i=0 是id字段，不要
            if i == 0: continue
            # 只保存value
            job_string += value
    return job_string

def token(string):
    return re.findall('\w+', string)

cutted_X = []
for i, row in enumerate(X_string):
    job_string = get_one_row_job_string(row)
    cutted_X.append(' '.join(list(jieba.cut(''.join(token(job_string))))))
    # print(cutted_X)
    # break

# 使用TFIDF
vectorizer = TfidfVectorizer()
# 原始数据的tfidf特征
X = vectorizer.fit_transform(cutted_X)
# print(X[0])

# 求平均值，比如 薪资10-15k => 12.5k
target_numical = [np.mean(list(map(float, re.findall('\d+',s)))) for s in target]
# print(target_numical[0:5])

# 使用KNN回归
model = KNeighborsRegressor(n_neighbors=2)
y = target_numical
model.fit(X, y)
print(model.score)

def predict_by_label(test_string, model):
    test_words = list(jieba.cut(test_string))
    test_vec = vectorizer.transform(test_words)
    predict_y = model.predict(test_vec)
    return predict_y[0]

test = '测试 北京 3年 专科'
test2 = '测试 北京 4年 专科'
test3 = '算法 北京 4年 本科'
test4 = 'UI 北京 4年 本科'
print(test, predict_by_label(test, model))
print(test2, predict_by_label(test2, model))
print(test3, predict_by_label(test3, model))
print(test4, predict_by_label(test4, model))

sentences= ["广州Java本科3年掌握大数据", "沈阳Java硕士3年掌握大数据", "沈阳Java本科3年掌握大数据", "北京算法硕士3年掌握图像识别"]
for p in sentences:
    print('{}的薪资预测{}'.format(p, predict_by_label(p, model)))
