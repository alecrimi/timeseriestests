import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

input_window = 10  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 250

# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("your device is: ", device)

# Load your custom dataset with 3 columns
df = pd.read_csv("data/facebook.csv")  # Replace with your actual file path
df['order_date'] = pd.to_datetime(df['order_date'])
df = df.sort_values('order_date')

print("Dataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)
df.head()

# Prepare the data - use both QuantityAccepted and Failed_quantity as features
# but predict only Failed_quantity
quantity_accepted = df["QuantityAccepted"].fillna(method="ffill").values
failed_quantity = df["Failed_quantity"].fillna(method="ffill").values

# Create combined features (both QuantityAccepted and Failed_quantity)
# We'll use both as input features but predict only Failed_quantity
combined_features = np.column_stack((quantity_accepted, failed_quantity))

# Apply logarithmic transformation to stabilize variance
combined_features_log = np.log(combined_features + 1)  # +1 to avoid log(0)
combined_log_return = np.diff(combined_features_log, axis=0)
combined_csum_logreturn = combined_log_return.cumsum(axis=0)

# Plot the data
fig = plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(quantity_accepted, color="blue")
plt.title("Raw Quantity Accepted")
plt.xlabel("Time Steps")
plt.ylabel("Quantity")

plt.subplot(2, 2, 2)
plt.plot(failed_quantity, color="red")
plt.title("Raw Failed Quantity")
plt.xlabel("Time Steps")
plt.ylabel("Failed Quantity")

plt.subplot(2, 2, 3)
plt.plot(combined_csum_logreturn[:, 0], color="green")
plt.title("Normalized Quantity Accepted")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Quantity")

plt.subplot(2, 2, 4)
plt.plot(combined_csum_logreturn[:, 1], color="orange")
plt.title("Normalized Failed Quantity")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Failed Quantity")

plt.tight_layout()
plt.show()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class transformer(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(transformer, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # Output only 1 value (Failed_quantity prediction)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        # Input: both features [tw, 2]
        train_seq = input_data[i:i + tw]  # shape: [tw, 2]
        
        # Target: only Failed_quantity for the next time steps [tw, 1]
        # We take the Failed_quantity (second column) for the next time steps
        train_label = input_data[i + output_window:i + tw + output_window, 1:2]  # shape: [tw, 1]
        
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)

def get_data(data, split):
    series = data

    # Use final 30% for testing (as requested)
    split = round((1 - split) * len(series))  # split = 0.3 means 30% test
    train_data = series[:split]
    test_data = series[split:]

    train_data = train_data.cumsum(axis=0)
    train_data = 2 * train_data  # Data augmentation

    test_data = test_data.cumsum(axis=0)

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window] if output_window > 0 else train_sequence

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window] if output_window > 0 else test_data

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    
    # Input: both features (shape: [input_window, batch_size, 2])
    input_data = torch.stack([item[0] for item in data])  # [batch_size, input_window, 2]
    input_data = input_data.transpose(0, 1)  # [input_window, batch_size, 2]
    
    # Target: only Failed_quantity (shape: [input_window, batch_size, 1])
    target_data = torch.stack([item[1] for item in data])  # [batch_size, input_window, 1]
    target_data = target_data.transpose(0, 1)  # [input_window, batch_size, 1]
    
    return input_data, target_data

def train(train_data):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = max(1, int(len(train_data) / batch_size / 5))
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | "
                  "lr {:02.10f} | {:5.2f} ms | "
                  "loss {:5.7f}".format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

def forecast_seq(model, sequences):
    start_timer = time.time()
    model.eval()
    forecast_seq = torch.Tensor(0)
    actual = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(sequences) - 1):
            data, target = get_batch(sequences, i, 1)
            output = model(data)
            forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
            actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
    timed = time.time() - start_timer
    print(f"{timed:.2f} sec")

    return forecast_seq, actual

def calculate_metrics(y_true, y_pred):
    """Calculate R2, MSE and MAPE metrics"""
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    # Calculate R2 score
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # +1e-8 to avoid division by zero
    
    return r2, mse, mape

# Prepare data with 70% train, 30% test split (final 30% for testing)
train_data, val_data = get_data(combined_log_return, 0.3)  # 70% train, 30% test
 
model = transformer(feature_size=2).to(device)  # 2 input features

criterion = nn.MSELoss()
lr = 0.00005
epochs = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if epoch % 10 == 0 or epoch == epochs:  # Validate every 10 epochs and at the end
        val_loss = evaluate(model, val_data)
        print("-" * 80)
        print("| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.7f}".format(
            epoch, (time.time() - epoch_start_time), val_loss))
        print("-" * 80)
    else:
        print("-" * 80)
        print("| end of epoch {:3d} | time: {:5.2f}s".format(epoch, (time.time() - epoch_start_time)))
        print("-" * 80)

    scheduler.step()

# Final evaluation on test set
test_result, truth = forecast_seq(model, val_data)

# Calculate evaluation metrics
r2, mse, mape = calculate_metrics(truth, test_result)

print("\n" + "="*50)
print("FINAL VALIDATION RESULTS (30% Test Set)")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAPE: {mape:.2f}%")
print("="*50)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(truth, color="red", alpha=0.7, label="Actual")
plt.plot(test_result, color="blue", linestyle="dashed", linewidth=1, label="Forecast")
plt.title(f"Actual vs Forecast Failed Quantity\nR²: {r2:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Failed Quantity (normalized)")
plt.grid(True, alpha=0.3)
plt.show()

# Save model
torch.save(model.state_dict(), "model/failed_quantity_forecasting_transformer.pth")
print("Model saved successfully!")

# Optional: Test on new datasets (if available)
try:
    # Load new dataset for testing
    df_new = pd.read_csv("data/boeing.csv")  # Replace with your test file
    df_new['order_date'] = pd.to_datetime(df_new['order_date'])
    df_new = df_new.sort_values('order_date')
    
    # Prepare test data
    quantity_accepted_new = df_new["QuantityAccepted"].fillna(method="ffill").values
    failed_quantity_new = df_new["Failed_quantity"].fillna(method="ffill").values
    combined_features_new = np.column_stack((quantity_accepted_new, failed_quantity_new))
    combined_features_log_new = np.log(combined_features_new + 1)
    combined_log_return_new = np.diff(combined_features_log_new, axis=0)
    
    # Get test data (using final 30% for testing)
    train_data_new, test_data_new = get_data(combined_log_return_new, 0.3)
    
    # Evaluate on test set
    test_loss = evaluate(model, test_data_new)
    test_result_new, truth_new = forecast_seq(model, test_data_new)
    
    # Calculate metrics for test set
    r2_test, mse_test, mape_test = calculate_metrics(truth_new, test_result_new)
    
    print("\n" + "="*50)
    print("TEST RESULTS ON NEW DATASET")
    print("="*50)
    print(f"Test Loss: {test_loss:.5f}")
    print(f"R² Score: {r2_test:.4f}")
    print(f"MSE: {mse_test:.4f}")
    print(f"MAPE: {mape_test:.2f}%")
    print("="*50)
    
    # Plot test results
    plt.figure(figsize=(12, 6))
    plt.plot(truth_new, color="red", alpha=0.7, label="Actual")
    plt.plot(test_result_new, color="blue", linestyle="dashed", linewidth=1, label="Forecast")
    plt.title(f"Test Set: Actual vs Forecast Failed Quantity\nR²: {r2_test:.4f}, MSE: {mse_test:.4f}, MAPE: {mape_test:.2f}%")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Failed Quantity (normalized)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
except FileNotFoundError:
    print("Test dataset not found. Skipping external validation.")
except Exception as e:
    print(f"Error during external validation: {e}")
