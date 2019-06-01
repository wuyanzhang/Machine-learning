import os
import jieba
import random

# 函数说明：中文文本处理
def TextProcessing(folder_path,test_size=0.2):
    folder_list = os.listdir(dataset_path)
    data_list = []
    label_list = []
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            data_list.append(word_list)
            label_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list,label_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_data_label = zip(*train_list)
    test_data_list,test_data_label = zip(*test_list)

    all_words_dict = {}
    for wordlist in train_data_list:
        for word in wordlist:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
    all_words_list,all_words_num = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list

# 函数说明：读取文件里的内容，并去重
def makeWordsSet(words_file):
    word_set = set()
    with open(words_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                word_set.add(word)
    return word_set

# 函数说明：文本特征选取
def words_dict(all_words_list,deleteN,stopwords_set):
    feature_words = []
    n = 1
    for i in range(deleteN,len(all_words_list),1):
        if n > 1000:
            break
        if not all_words_list[i].isdigit() and all_words_list[i] not in stopwords_set and 1 < len(all_words_list[i]) < 5:
            feature_words.append(all_words_list[i])
        n += 1
    return feature_words

if __name__ == '__main__':
    dataset_path = './SogouC/Sample'
    all_words_list = TextProcessing(dataset_path,test_size=0.2)

    stopwords_file = './stopwords_cn.txt'
    stopwords_set = makeWordsSet(stopwords_file)
    feature_words = words_dict(all_words_list,100,stopwords_set)
    print(feature_words)