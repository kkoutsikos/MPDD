import json
import torch
import numpy as np
from torch.utils.data import Dataset


class AudioVisualDataset(Dataset):
    def __init__(self, json_data, label_count, personalized_feature_file, max_len=10, batch_size=32, audio_path='', video_path=''):
        self.data = json_data
        self.max_len = max_len  # 目标序列长度
        self.batch_size = batch_size  # 记录批次大小

        # Load personalized features
        self.personalized_features = self.load_personalized_features(personalized_feature_file)
        self.audio_path = audio_path
        self.video_path = video_path
        self.label_count = label_count

    def __len__(self):
        return len(self.data)

    def fixed_windows(self, features: torch.Tensor, fixLen=4):
        """
        将二维特征划分为 fixLen 个固定窗口并聚合成固定大小的结果（Tensor 版）。

        参数:
        - features: (timesteps, feature_dim) 的输入特征张量

        返回:
        - (4, feature_dim) 的张量，每一行表示一个窗口的聚合特征
        """
        timesteps, feature_dim = features.shape
        window_size = int(torch.ceil(torch.tensor(timesteps / fixLen)))
        windows = []
        for i in range(fixLen):
            start = i * window_size
            end = min(start + window_size, timesteps)
            window = features[start:end]
            if window.size(0) > 0:
                window_aggregated = torch.mean(window, dim=0)
                windows.append(window_aggregated)
            else:
                windows.append(torch.zeros(feature_dim))

        return torch.stack(windows, dim=0)

    def pad_or_truncate(self, feature, max_len):
        """将输入特征序列进行填充或截断"""
        if feature.shape[0] < max_len:
            padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
            feature = torch.cat((feature, padding), dim=0)
        else:
            feature = feature[:max_len]
        return feature

    def load_personalized_features(self, file_path):
        """
        Load personalized features from the .npy file.
        """

        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            return {entry["id"]: entry["embedding"] for entry in data}
        else:
            raise ValueError("Unexpected data format in the .npy file. Ensure it contains a list of dictionaries.")

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load audio and video features
        audio_feature = np.load(self.audio_path + '/' + entry['audio_feature_path'])
        video_feature = np.load(self.video_path + '/' + entry['video_feature_path'])
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
        video_feature = torch.tensor(video_feature, dtype=torch.float32)

        audio_feature = self.pad_or_truncate(audio_feature, self.max_len)
        video_feature = self.pad_or_truncate(video_feature, self.max_len)

        # Load label
        if self.label_count == 2:
            label = torch.tensor(entry['bin_category'], dtype=torch.long)
        elif self.label_count == 3:
            label = torch.tensor(entry['tri_category'], dtype=torch.long)
        elif self.label_count == 5:
            label = torch.tensor(entry['pen_category'], dtype=torch.long)

        import os

        filepath = entry['audio_feature_path']  # 假设这是包含路径的文件名
        # 提取文件名
        filename = os.path.basename(filepath)
        # 提取人员id并转换为整数
        person_id = int(filename.split('_')[0])
        personalized_id = str(person_id)

        if personalized_id in self.personalized_features:
            personalized_feature = torch.tensor(self.personalized_features[personalized_id], dtype=torch.float32)
        else:
            # If no personalized feature found, use a zero vector
            personalized_feature = torch.zeros(1024, dtype=torch.float32)
            print(f"❗Personalized feature not found for id: {personalized_id}")

        return {
            'A_feat': audio_feature,
            'V_feat': video_feature,
            'emo_label': label,
            'personalized_feat': personalized_feature
        }
