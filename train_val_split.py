import json
import random
from collections import defaultdict
import math
import numpy as np
import torch


def train_val_split(file_path, val_percentage=0.10, seed=None):
    """
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