import numpy as np
import csv
import json
import asyncio
from pylsl import StreamInlet, resolve_stream
import torch
import torch.nn as nn
import websockets
import mne
import os
from scipy.signal import resample  # 导入重采样函数

# 定义情绪映射（仅用于推理，不在训练中使用）
emotion_map = [
    ("sleepy", 0.01, -1.00),
    ("tired", -0.01, -1.00),
    ("afraid", -0.12, 0.79),
    ("angry", -0.40, 0.79),
    ("calm", 0.78, -0.67),
    ("relaxed", 0.71, -0.65),
    ("content", 0.81, -0.55),
    ("depressed", -0.81, -0.48),
    ("discontent", -0.68, -0.32),
    ("determined", 0.89, 0.35),
    ("happy", 0.89, 0.17),
    ("anxious", -0.72, -0.80),
    ("good", 0.78, 0.35),
    ("pensive", 0.03, -0.60),
    ("impressed", 0.39, 0.06),
    ("frustrated", -0.60, 0.40),
    ("disappointed", -0.80, -0.03),
    ("bored", -0.35, -0.78),
    ("annoyed", -0.44, 0.76),
    ("enraged", -0.18, 0.83),
    ("excited", 0.70, 0.71),
    ("melancholy", -0.65, -0.65),
    ("satisfied", 0.77, -0.63),
    ("distressed", -0.76, 0.83),
    ("uncomfortable", -0.68, -0.37),
    ("worried", -0.07, 0.32),
    ("amused", 0.55, -0.02),
    ("apathetic", -0.20, -0.12),
    ("peaceful", 0.55, -0.60),
    ("contemplative", 0.58, -0.60),
    ("embarrassed", -0.31, -0.61),
    ("sad", -0.81, -0.40),
    ("hopeful", 0.61, 0.40),
    ("pleased", 0.89, -0.10)
]

# 函数：根据 Valence 和 Arousal 获取最近的情绪标签
def get_emotion_label(valence, arousal):
    closest_emotion = None
    min_distance = float('inf')
    for emotion, e_valence, e_arousal in emotion_map:
        distance = np.sqrt((valence - e_valence) ** 2 + (arousal - e_arousal) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion
    return closest_emotion

# 实时预处理函数
def high_pass_filter_realtime(raw_eeg, l_freq=1.0):
    print(f"Applying high-pass filter: {l_freq} Hz (real-time, IIR)")
    raw_eeg.filter(l_freq=l_freq, h_freq=None, method='iir')  # 使用 IIR 滤波器
    return raw_eeg

def bandpass_filter_realtime(raw_eeg, l_freq=4.0, h_freq=45.0):
    print(f"Applying bandpass filter: {l_freq} - {h_freq} Hz (real-time, IIR)")
    return raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, method='iir')  # 使用 IIR 滤波器

def average_reference_realtime(raw_eeg):
    raw_eeg.set_eeg_reference('average', projection=True)
    raw_eeg.apply_proj()  # 应用参考投影
    return raw_eeg

def preprocess_data_realtime(eeg_data, fs, selected_channels):
    # 将原始 EEG 数据转换为 MNE 的 RawArray 进行处理
    info = mne.create_info(ch_names=selected_channels, sfreq=fs, ch_types='eeg')
    raw_eeg = mne.io.RawArray(eeg_data, info)

    # 应用实时特定的预处理步骤，使用 IIR 滤波器
    raw_eeg = high_pass_filter_realtime(raw_eeg)
    raw_eeg = bandpass_filter_realtime(raw_eeg)
    raw_eeg = average_reference_realtime(raw_eeg)

    # 获取预处理后的数据作为 numpy 数组
    preprocessed_data = raw_eeg.get_data()
    return preprocessed_data

# 计算带功率特征
def compute_bandpower(eeg_data, sfreq, band, relative=False):
    """Compute the average power of the EEG signal in a specific frequency band.

    Parameters
    ----------
    eeg_data : array, shape (n_channels, n_times)
        EEG data.
    sfreq : float
        Sampling frequency.
    band : tuple
        Lower and upper frequencies of the band of interest (e.g., (4, 8) for theta).
    relative : bool
        If True, return the relative power (percentage).

    Returns
    -------
    bp : float or array
        Band power for each channel.
    """
    band = np.asarray(band)
    low, high = band

    # Compute power spectral density (PSD)
    psd, freqs = mne.time_frequency.psd_array_welch(eeg_data, sfreq=sfreq, n_fft=eeg_data.shape[1])

    # Find indices of the band
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Compute average power in the band
    bp = psd[:, idx_band].mean(axis=1)

    if relative:
        total_power = psd.sum(axis=1)
        bp /= total_power

    return bp

# Transformer 模型定义（PyTorch）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, input_dim)
        residual = x
        x = self.layernorm1(x)

        # PyTorch MultiheadAttention 期望输入形状: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)  # 现在 x 的形状为 (seq_len, batch_size, input_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.transpose(0, 1)  # 回到 (batch_size, seq_len, input_dim)
        x = self.dropout1(x)
        x = x + residual

        residual = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, features):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.linear = nn.Linear(patch_size * features, embed_dim)

    def forward(self, x):
        # x 形状: (batch_size, num_patches, patch_size * features)
        x = self.linear(x)  # (batch_size, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, head_size, num_heads, ff_dim, num_transformer_blocks, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer(input_dim=embed_dim, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output_layer = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x 形状: (batch_size, num_patches, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        # x 形状: (batch_size, num_patches, embed_dim)
        # 转换为 (batch_size, embed_dim, num_patches) 以适应 AdaptiveAvgPool1d
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)  # (batch_size, embed_dim, 1)
        x = x.squeeze(-1)  # (batch_size, embed_dim)
        x = self.mlp(x)  # (batch_size, 128)
        x = self.tanh(self.output_layer(x))  # (batch_size, 1)
        return x

# 组合的 Transformer 模型，具有两个独立的编码器
class TransformerModel(nn.Module):
    def __init__(self, patch_size, embed_dim, head_size, num_heads, ff_dim, num_transformer_blocks, features, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, features=features)
        # 用于 Arousal 的 Transformer 编码器
        self.transformer_arousal = TransformerEncoder(
            embed_dim=embed_dim,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout
        )
        # 用于 Valence 的 Transformer 编码器
        self.transformer_valence = TransformerEncoder(
            embed_dim=embed_dim,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout
        )

    def forward(self, x):
        # x 形状: (batch_size, num_patches, patch_size * features)
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)
        arousal_output = self.transformer_arousal(x)  # (batch_size, 1)
        valence_output = self.transformer_valence(x)  # (batch_size, 1)
        return arousal_output, valence_output

# 预测情绪的函数（使用 PyTorch）
def predict_emotion(eeg_data, fs, selected_channels, model, device):
    try:
        # 预处理数据以符合模型输入要求
        preprocessed_data = preprocess_data_realtime(eeg_data, fs, selected_channels)  # Shape: (channels, samples)

        # 计算带功率特征
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        bandpowers = []
        for band in bands.values():
            bp = compute_bandpower(preprocessed_data, fs, band)
            bandpowers.extend(bp.tolist())

        # 将带功率特征转换为 numpy 数组
        bandpowers = np.array(bandpowers)  # Shape: (channels * bands,)

        # 划分为 Patch
        patch_size = 16  # 必须与训练时的 patch_size 一致
        features = len(bands)  # 5 频段
        num_channels = preprocessed_data.shape[0]  # 通道数，例如 2

        # 每个通道有 5 个频段，所以总特征数 = 2 * 5 =10
        # 总 Patch 特征数 = patch_size * features =16 *10=160

        desired_length = patch_size * num_channels * len(bands)  # 160
        if bandpowers.shape[0] < desired_length:
            bandpowers = np.pad(bandpowers, (0, desired_length - bandpowers.shape[0]), 'constant')
        elif bandpowers.shape[0] > desired_length:
            bandpowers = bandpowers[:desired_length]

        # 重塑为 (batch_size=1, num_patches=1, patch_size * features=160)
        patches = bandpowers.reshape(1, 1, desired_length)  # (1,1,160)
        patches = torch.tensor(patches, dtype=torch.float32).to(device)

        # 使用 Transformer 模型进行预测
        model.eval()
        with torch.no_grad():
            output_arousal, output_valence = model(patches)
            valence = output_valence.cpu().numpy()[0, 0]
            arousal = output_arousal.cpu().numpy()[0, 0]

        # 使用预测的 valence 和 arousal 值获取情绪标签
        emotion_label = get_emotion_label(valence, arousal)

        return float(valence), float(arousal), emotion_label
    except Exception as e:
        print(f"Error in predicting emotion: {e}")
        return None, None, None

# WebSocket 处理函数
clients = set()

async def register(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def handler(websocket, path):
    await register(websocket)
    async for message in websocket:
        for client in clients:
            if client != websocket:
                try:
                    await client.send(message)
                except websockets.ConnectionClosed:
                    clients.remove(client)

# EEG 数据处理与预测函数
async def process_eeg_data(model, device, fs, selected_channels):
    # 解析可用的 OpenSignals 流
    print("# Looking for an available OpenSignals stream...")
    os_stream = resolve_stream("name", "OpenSignals")
    if len(os_stream) == 0:
        print("No OpenSignals stream found. Please ensure the EEG device is streaming.")
        return
    else:
        print(f"Found {len(os_stream)} 'OpenSignals' stream(s).")
        for idx, stream in enumerate(os_stream):
            print(f"Stream {idx+1}: {stream.name()}, {stream.type()}, {stream.channel_count()} channels")

    inlet = StreamInlet(os_stream[0])

    # 初始化 EEG 数据缓冲区
    eeg_buffer = []
    eeg_min_length = 100  # 调整为 100 个样本（1 秒）

    # 确保 Results 目录存在
    os.makedirs('./Results', exist_ok=True)

    # 打开 CSV 文件以保存结果
    with open('./Results/EEG_P01.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Valence', 'Arousal', 'Emotion'])

        async with websockets.connect('ws://localhost:8769') as websocket:
            print("WebSocket connection established.")
            while True:
                try:
                    sample, timestamp = inlet.pull_sample()
                    eeg_data = sample[:len(selected_channels)]  # 获取选定通道的数据

                    eeg_buffer.append(eeg_data)

                    if len(eeg_buffer) >= eeg_min_length:
                        eeg_buffer_array = np.array(eeg_buffer[-eeg_min_length:]).T  # 形状: (channels, samples)

                        # 重采样到 128 Hz
                        if fs != 128:
                            num_samples = 128
                            eeg_buffer_array_resampled = resample(eeg_buffer_array, num_samples, axis=1)
                            print(f"Resampled data from {fs} Hz to 128 Hz.")
                        else:
                            eeg_buffer_array_resampled = eeg_buffer_array

                        # 预测情绪
                        valence, arousal, emotion_label = predict_emotion(
                            eeg_buffer_array_resampled, 128, selected_channels, model, device)
                        if valence is not None and arousal is not None:
                            print(f"Timestamp: {timestamp}, Valence: {valence}, Arousal: {arousal}, Emotion: {emotion_label}")

                            # 保存结果到 CSV
                            writer.writerow([timestamp, valence, arousal, emotion_label])

                            # 通过 WebSocket 发送数据给客户端
                            data = {
                                "valence": valence,
                                "arousal": arousal,
                                "emotion": emotion_label,
                                "source": "EEG"
                            }
                            await websocket.send(json.dumps(data))
                        else:
                            print(f"Timestamp: {timestamp}, Prediction Error")
                    else:
                        print("Not enough data, skipping this sample.")
                except Exception as e:
                    print(f"Error in processing EEG data: {e}")

# 主函数启动 WebSocket 服务器和 EEG 数据处理
async def main():
    # 加载训练好的 Transformer 模型（PyTorch）
    model_path = './ModelResults/EmoEst_EEG_Transformer_PP_DEAP_2_v1/transformer_model.pth'
    embed_dim = 128
    patch_size = 16
    head_size = 128
    num_heads = 4
    ff_dim = 256
    num_transformer_blocks = 1
    features = 10  # 2 通道 * 5 频段

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        patch_size=patch_size,
        embed_dim=embed_dim,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        features=features,  # 设为 10
        dropout=0.3
    ).to(device)

    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Transformer model loaded successfully.")
    else:
        print(f"Model file {model_path} not found.")
        return

    # 启动 WebSocket 服务器
    server = websockets.serve(handler, "localhost", 8769)
    await server
    print("WebSocket server started at ws://localhost:8769")

    # 启动 EEG 数据处理
    await process_eeg_data(model, device, fs=100, selected_channels=['Fp1', 'Fp2'])

# 运行集成的 WebSocket 服务器和 EEG 数据处理
if __name__ == "__main__":
    asyncio.run(main())
