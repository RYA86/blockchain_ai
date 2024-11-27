import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class BlockchainAnomalyDetector(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        
        # LSTM Layer untuk sequence detection
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True
        )
        
        # Autoencoder untuk rekonstruksi
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        # Klasifikasi anomali
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM Processing
        lstm_out, _ = self.lstm(x)
        
        # Encoder-Decoder
        encoded = self.encoder(x[:, -1, :])  # Ambil timestamp terakhir
        decoded = self.decoder(encoded)
        
        # Anomaly Detection
        anomaly_prob = self.anomaly_classifier(encoded)
        
        return {
            'anomaly_probability': anomaly_prob,
            'reconstruction_error': torch.mean((x[:, -1, :] - decoded)**2),
            'lstm_features': lstm_out
        }


class BlockchainTransactionDataset(Dataset):
    def __init__(self, transactions):
        # Preprocessing fitur blockchain
        self.scaler = StandardScaler()
        
        # Contoh fitur kompleks
        features = [
            'sender_history_volume',
            'receiver_history_volume', 
            'transaction_frequency',
            'time_between_transactions',
            'wallet_age',
            'transaction_amount',
            'network_congestion',
            'gas_price',
            'previous_transaction_pattern',
            'wallet_diversity_score'
        ]
        
        # Transform dan scale
        self.data = self.scaler.fit_transform(transactions[features])
        self.data = torch.FloatTensor(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_blockchain_anomaly_model(transactions, epochs=50):
    # Setup model, dataset, loss
    dataset = BlockchainTransactionDataset(transactions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = BlockchainAnomalyDetector(input_dim=10)
    optimizer = optim.Adam(model.parameters())
    
    # Loss gabungan
    def combined_loss(outputs):
        anomaly_labels = (outputs['anomaly_probability'] > 0.7).float()
        anomaly_loss = nn.BCELoss()(outputs['anomaly_probability'], anomaly_labels)
        reconstruction_loss = outputs['reconstruction_error']
        return anomaly_loss + reconstruction_loss
    
    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.unsqueeze(1)  # Tambahkan dimensi waktu
            optimizer.zero_grad()
            outputs = model(batch)
            loss = combined_loss(outputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model


def detect_blockchain_anomalies(model, transaction):
    # Prediksi anomali pada transaksi
    with torch.no_grad():
        transaction = transaction.unsqueeze(1)  # Tambahkan dimensi waktu
        result = model(transaction)
        
    return {
        'is_anomaly': result['anomaly_probability'] > 0.7,
        'anomaly_score': result['anomaly_probability'].item(),
        'reconstruction_error': result['reconstruction_error'].item()
    }


# Contoh penggunaan
def main():
    # Simulasi data transaksi
    sample_transactions = pd.DataFrame({
        'sender_history_volume': [1000, 5000, 50000],
        'receiver_history_volume': [500, 2000, 10000],
        'transaction_frequency': [10, 20, 30],
        'time_between_transactions': [5, 15, 25],
        'wallet_age': [1, 2, 3],
        'transaction_amount': [100, 200, 300],
        'network_congestion': [0.5, 0.3, 0.9],
        'gas_price': [20, 15, 10],
        'previous_transaction_pattern': [0.2, 0.5, 0.1],
        'wallet_diversity_score': [0.7, 0.8, 0.9]
    })
    
    # Training model
    blockchain_ai_model = train_blockchain_anomaly_model(sample_transactions)
    
    # Detect anomalies
    test_transaction = torch.randn(1, 10)  # Contoh transaksi
    anomaly_result = detect_blockchain_anomalies(blockchain_ai_model, test_transaction)
    
    print(anomaly_result)


if __name__ == "__main__":
    main()
