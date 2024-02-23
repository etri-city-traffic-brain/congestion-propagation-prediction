import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        sampled_latent = mean + epsilon * std
        return sampled_latent

    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean_layer(encoded)
        logvar = self.logvar_layer(encoded)

        sampled_latent = self.reparameterize(mean, logvar)

        decoded = self.decoder(sampled_latent)
        return decoded, mean, logvar

# 재현성을 위해 랜덤 시드를 설정합니다.
torch.manual_seed(42)
np.random.seed(42)

# 데이터 불러오기 (파일 경로에 맞게 수정하세요)
data = pd.read_hdf('./METRLA/metr-la.h5')
# data_refine = data.iloc[:, 1:-1]  # 첫 열과 마지막 열 제외
data.index.freq = data.index.inferred_freq
column_names = data.columns
index_names = data.index

data = data.replace(0, np.NaN)

# data 누락된 값 저장해둠.(위치)
missing_mask = np.isnan(data)

# 데이터 전처리: 누락된 값을 행의 최솟값으로 대체
min_filled_data = data.apply(lambda col: col.fillna(col.min()), axis=0).values
# data_filled = data_refine.fillna(0).values
if np.all(np.isnan(min_filled_data)):
    min_filled_data = np.zeros_like(min_filled_data)
# print(min_filled_data)

# Min-Max 스케일링
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(min_filled_data)


# 마스킹: 0인 값에 마스크 적용
masked_data = data_scaled.copy()
masked_data[missing_mask] = 0  # 0인 값만 마스킹

# 모델 및 옵티마이저 설정
input_dim = data.shape[1]
hidden_dim = 64
latent_dim = 16
vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 학습 데이터 준비
train_data = masked_data
train_data = torch.FloatTensor(train_data)

# 학습
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    vae.train()
    total_loss = 0
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i + batch_size]
        optimizer.zero_grad()
        reconstructed_batch, mean, logvar = vae(batch_data)

        # Reconstruction loss 계산
        reconstruction_loss = nn.MSELoss()(reconstructed_batch, batch_data)

        # KL Divergence 계산
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # 총 손실
        loss = reconstruction_loss + kl_divergence

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# 예측된 데이터를 하나의 배열로 변환
predicted_data = []
for i in range(0, len(train_data), batch_size):
    batch_data = train_data[i:i + batch_size]
    vae.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        reconstructed_batch, _, _ = vae(batch_data)
    predicted_data.append(reconstructed_batch.numpy())

# 예측된 데이터를 원래 데이터로 복원 (Min-Max 역변환)
predicted_data_stacked = np.vstack(predicted_data)

# 원래 데이터 중에서 마스킹된 부분만 VAE 예측 값으로 대체
restored_data = data_scaled.copy()
restored_data[missing_mask] = predicted_data_stacked[missing_mask]

restored_data = scaler.inverse_transform(restored_data)
restored_data = np.round(restored_data, 6)

# 복원된 데이터를 DataFrame으로 변환
restored_df = pd.DataFrame(restored_data, index=index_names, columns=column_names)

# 첫 열을 저장한 데이터와 합치기
# first_column = data.iloc[:, 0]
# restored_df = pd.concat([first_column, restored_df], axis=1)

# 복원된 데이터 출력
print("Restored Data:")

# 저장할 엑셀 파일 이름 (경로 포함 가능)
restored_h5_filename = './METRLA/metr-la-restored.h5'
restored_df.to_hdf(restored_h5_filename, 'axis0')
print(f'Restored data saved to {restored_h5_filename}')
