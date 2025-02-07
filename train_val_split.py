import json
import random
from collections import defaultdict
import math
import numpy as np
import torch


def train_val_split2(file_path, val_percentage=0.10, seed=None):
    """
    Track2数据集划分

    从给定的JSON文件中按人员编号分组，选取10%的人员编号数据，同时需确保每个标签的比例和原始数据一致，
    选取的人员编号数据要么全部选中，要么全不选。

    参数:
    - file_path: JSON文件路径
    - val_percentage: 需要选取的数据比例（基于人员编号数量，默认是10%）
    - seed: 随机种子

    返回:
    - train_data: 训练集
    - val_data: 验证集
    - train_category_count: 训练集 tri_category 标签的总数
    - val_category_count: 验证集 tri_category 标签的总数
    """

    if seed is not None:
        random.seed(seed)

    with open(file_path, 'r') as file:
        data = json.load(file)

    grouped_by_person = defaultdict(list)
    for entry in data:
        # 提取人员编号（假设格式为 "人员编号_主题编号.npy"）
        person_id = entry['audio_feature_path'].split('_')[0]
        grouped_by_person[person_id].append(entry)

    # 按标签类别均匀划分人员（young数据集根据tri_category划分）
    tri_category_person = defaultdict(list)
    for person_id, entries in grouped_by_person.items():
        tri_category = entries[0]['tri_category']
        tri_category_person[tri_category].append(person_id)

    total_person_count = len(grouped_by_person)
    num_persons_to_select = round(total_person_count * val_percentage)

    selected_person_ids = set()

    # 计算每个类别的人员数量和需要选取的人员数量
    selected_per_category = defaultdict(int)
    for category, person_ids in tri_category_person.items():
        num_category_persons = len(person_ids)
        num_category_to_select = round(num_category_persons * val_percentage + 0.001)
        selected_per_category[category] = num_category_to_select

    for category, person_ids in tri_category_person.items():
        num_category_to_select = selected_per_category[category]
        selected_person_ids.update(random.sample(person_ids, num_category_to_select))

    # 构建验证集数据
    val_data = []
    for entry in data:
        person_id = entry['audio_feature_path'].split('_')[0]
        if person_id in selected_person_ids:
            val_data.append(entry)

    # 训练集
    train_data = [entry for entry in data if entry not in val_data]

    # 统计 train_data 和 val_data 中 tri_category 标签的总数
    train_category_count = defaultdict(int)
    val_category_count = defaultdict(int)

    for entry in train_data:
        train_category_count[entry['tri_category']] += 1

    for entry in val_data:
        val_category_count[entry['tri_category']] += 1
    # 保存 train_data 和 val_data 到 JSON 文件（如果需要）


    return train_data, val_data, train_category_count, val_category_count

import json
import random
from collections import defaultdict

def train_val_split1(file_path, val_ratio=0.1, random_seed=3407):
    """
    Track1数据集划分

    读取 JSON 文件，并按照指定规则划分训练集和验证集：
      - label=4 的数据按 2:1 划分；
      - label=3 且 id=69 的数据直接放入验证集；
      - 其余数据按照 val_ratio 进行划分。
    
    保证：
      - 同一个 ID 的样本不会同时出现在训练集和验证集中。
      - 返回格式与 train_val_split 保持一致。

    参数:
        file_path (str): JSON 数据文件路径
        val_ratio (float): 验证集占比，默认 0.1
        random_seed (int): 随机种子，默认 3407

    返回:
        tuple: (训练数据列表, 验证数据列表, 训练集类别统计, 验证集类别统计)
    """
    random.seed(random_seed)

    with open(file_path, 'r') as file:
        data = json.load(file)

    train_data, val_data = [], []
    label_to_ids = defaultdict(set)
    id_to_samples = defaultdict(list)

    for item in data:
        label = item["label"]
        id_ = item["id"]
        label_to_ids[label].add(id_)
        id_to_samples[id_].append(item)

    train_ids, val_ids = set(), set()

    for label, ids in label_to_ids.items():
        ids = list(ids)

        # 处理 label=4（2:1 划分）
        if label == 4:
            for id_ in ids:
                samples = id_to_samples[id_]
                if len(samples) >= 3:
                    random.shuffle(samples)
                    train_data.extend(samples[:2])
                    val_data.extend(samples[2:3])
                else:
                    train_data.extend(samples)
            continue

        # 处理 label=3，且 id=69 的情况
        if label == 3:
            for id_ in ids:
                if id_ == "69":  # ID 87 直接入验证集
                    val_data.extend(id_to_samples[id_])
                else:
                    train_data.extend(id_to_samples[id_])
            continue

        # 其他类别按比例随机划分
        random.shuffle(ids)
        split_index = int(len(ids) * (1 - val_ratio))
        train_ids.update(ids[:split_index])
        val_ids.update(ids[split_index:])

    # 根据 ID 划分数据
    for id_ in train_ids:
        train_data.extend(id_to_samples[id_])
    for id_ in val_ids:
        val_data.extend(id_to_samples[id_])

    # 计算类别统计信息
    train_category_count = defaultdict(int)
    val_category_count = defaultdict(int)

    for entry in train_data:
        train_category_count[entry['label']] += 1
    for entry in val_data:
        val_category_count[entry['label']] += 1

    return train_data, val_data, train_category_count, val_category_count
