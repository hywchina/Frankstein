'''
20231214 - 原始版本@孙睿晗编写：对文本做embedding然后看tsne
'''
import json
import os
from gpt_caller import Gpt4Caller
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from diskcache import Cache
import itertools

# 设置中文字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

cache  = Cache('./gpt4_cache')
prompt_label_mapping = {
    "信息理解": "info_understanding",
    "逻辑及信息推理": "logical_reasoning",
    "文本创作": "text_creation",
    "代码及表格能力": "coding_form",
    "知识问答": "QA"
}

def line2text_label(line):
    # 机评1450个promtps直接取prompt_type即可
    text = line['conversations'][0]['value']
    label = line['prompt_type']
    return text, label

def line2text_label_jialian(line):
    text = line['conversations'][0]['value']
    # 诗歌的各种tags拼接成为其label
    #label = ','.join(f'{key}: {value}' for key, value in line['tags'].items())
    # 只用体裁作为label
    label = line['tags']['体裁']    
    return text, label

def line2text_label_yiyu(line):
    text = line['prompt']
    #print(type(line['condition']))
    #print(type(line['condition'][0]))
    #print(len(line['condition'][0]))
    label = line['condition'][0]['value']
    return text, label

def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            yield json_object

def get_texts_labels(prompt_file):
    # 处理读取json/jsonl文件
    file_postfix = os.path.splitext(os.path.basename(prompt_file))[1]
    if file_postfix == '.jsonl':
        lines = read_jsonl_file(prompt_file)
    elif file_postfix == '.json':
        lines = json.load(open(prompt_file, 'r', encoding='utf-8'))

    texts = []
    labels = []
    for line in lines:
        text, label = line2text_label(line)
        #print(text, label)
        texts.append(text)
        labels.append(label)

    print(f'There are {len(texts)} lines in {prompt_file}.')
    # 以prompt为key，label为value，构建字典去重
    unique_dict = dict(zip(texts, labels))
    print('There are {} unique prompts.'.format(len(unique_dict)))
    unique_texts = list(unique_dict.keys())
    labels = list(unique_dict.values())
    unique_labels = list(set(labels))
    #print(labels)
    print(f'There are {len(unique_labels)} unique labels.')
    return unique_texts, labels 

def GPT4embedding(texts):
    proxy = Gpt4Caller()

    embeddings = []
    for txt in texts:
        if txt in cache:
            embedding = cache[txt]
        else:
            embedding = proxy.get_embedding(txt)['data'][0]['embedding']
            cache[txt] = embedding
        embeddings.append(embedding)
        #if len(embeddings) % 100 == 0:
        #    print(f'Processed {len(embeddings)} embeddings.')

    # 将嵌入转换为NumPy数组
    embeddings = np.array(embeddings)
    return embeddings

# 使用 PCA
def PCA(embeddings):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title("PCA Visualization")
    plt.show()

# 使用t-SNE进行降维
def TSNE(embeddings, labels):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    reduced_embeddings = tsne.fit_transform(embeddings)

    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['label'] = labels
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep', legend='full')
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Prompt Type')

    print('Saving visualization to tsne_visualization.png...')
    plt.savefig('./visualization/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def UMAP(embeddings, labels, file_name):
    import umap
    from sklearn.preprocessing import LabelEncoder

    # 将字符串标签转换为整数
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)

    # 将降维后的数据和标签合并到 DataFrame
    df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df['label'] = labels

    # 使用 seaborn 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='label', palette='deep', legend='full')
    plt.title(f'UMAP visualization of {file_name} embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Label')

    # 保存图像到本地
    pic_path = './visualization/'
    pic_name = 'umap_' + file_name[:-6] + '.png'
    print(f'Saving visualization to {pic_path+pic_name}...')
    plt.savefig(pic_path + pic_name, dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def dst(a, b):
    return np.dot(a, b)
    #return np.linalg.norm(a - b)

def diversity_calc(embeddings):
    pairs = list(itertools.combinations(embeddings, 2))
    sum = 0
    for pair in pairs:
        a, b = pair
        sum += dst(a, b)
    return sum 

def main():
    file_path = './data/'
    #file_name = 'poetry_instructions.jsonl'
    file_name = 'eval.json'
    texts, labels = get_texts_labels(file_path + file_name)
    embeddings = GPT4embedding(texts)
    np.savez(file_path + 'embeddings_'+ file_name[:-6] + '.npz', embeddings = embeddings, labels = labels)
    print(f'embeddings.shape: {embeddings.shape}')
    div_metric = diversity_calc(embeddings)
    print(f'Diversity metric: {div_metric}')
    #TSNE(embeddings, labels)
    #PCA(embeddings)
    #UMAP(embeddings, labels, file_name)
    pass

if __name__ == '__main__':
    main()