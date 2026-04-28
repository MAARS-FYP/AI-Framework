import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ARVTDNN import ARVTDNN
from rf_chain import RFChain, OperatingPoint, Setting

GENERATE_DATA = True # Set to False to load from 'data/dataset.npz'
DATA_FILE = "data/dataset.npz"

# 1. Synthetic PA Model (Ground Truth)

def apply_pa_distortion(x, gain_db=10.0):
    """
    Simulates the LMH6401 DVGA based on datasheet specifications.
    
    Args:
        x: Input complex signal (Analytic IF)
        gain_db (float): Requested gain. 
    """
    gain_db_clamped = np.clip(np.round(gain_db), -6, 26)
    G = 10**(gain_db_clamped / 20.0)
    v_sat = 4.6 
    smoothness = 4.0
    
    x_amp = np.abs(x)
    scaled_mag = (x_amp * G) / (v_sat + 1e-12)
    denominator = (1 + scaled_mag**(2 * smoothness)) ** (1 / (2 * smoothness))
    y_rapp = (x * G) / (denominator + 1e-12)

    k2 = 0.00022  # 10^(-73/20)
    k3 = 0.00010  # 10^(-80/20)
    
    y_dist = y_rapp + k2 * (y_rapp**2) + k3 * (y_rapp**3)

    pad = np.zeros(2, dtype=np.complex128)
    y_d1 = np.concatenate([pad[:1], y_dist[:-1]])
    y_d2 = np.concatenate([pad[:2], y_dist[:-2]])
    
    y = y_dist + (0.001 * y_d1) + (-0.0005 * y_d2)
    
    return y

# 2. Feature Extraction (Preprocessor)

def prepare_features(input_sig, memory_depth=5, nonlinear_degree=5, aux_param=None):
    """
    Creates the input feature matrix for the ARVTDNN.
    Features: Re/Im of delayed samples AND Magnitude powers of delayed samples.
    """
    n_samples = len(input_sig)
    valid_len = n_samples - memory_depth
    
    features = []
    
    # Create sliding window features
    for m in range(memory_depth):
        start = memory_depth - 1 - m
        end = start + valid_len
        u_delayed = input_sig[start:end]
        
        # 1. Add Linear I/Q parts
        features.append(np.real(u_delayed))
        features.append(np.imag(u_delayed))
        
        # 2. Add Nonlinear Magnitude powers (|u|^2 ... |u|^K)
        mag = np.abs(u_delayed)
        for k in range(2, nonlinear_degree + 1):
            features.append(mag ** k)
            
    # Stack features: Shape (Samples, Num_Features)
    X = np.stack(features, axis=1)
    
    # 3. Append Auxiliary Parameter (Parametric Modeling)
    if aux_param is not None:
        aux_col = np.full((X.shape[0], 1), aux_param, dtype=X.dtype)
        X = np.concatenate([X, aux_col], axis=1)
        
    return X, valid_len

# Main Configuration & Data Prep

FS = 125e6       # 125 MHz Sampling Rate (Updated to match RF Chain)
FC_IF = 25e6     # 25 MHz IF
MEM_DEPTH = 5    # Memory Taps
DEGREE = 5       # Nonlinear Degree

if GENERATE_DATA:
    print("1. Generating Data with Full RF Chain (1, 10, 20 MHz)...")

    rf_chain = RFChain()
    bandwidths = [1e6, 10e6, 20e6]
    # Optimized Sweep
    powers = np.linspace(-60, -20, 12)
    voltages = rf_chain.available_voltages[:2] # Uses 3V and 5V from CSV
    gain_controls = np.linspace(-6, 26, 10) 

    # Synthesizer Amplitude Sweep
    # Range: -13.75 dBm to 20 dBm
    lo_powers = np.linspace(-13.75, 20.0, 6)

    all_inputs = []
    all_targets = []
    all_features = [] 

    total_iterations = len(bandwidths) * len(powers) * len(voltages) * len(gain_controls) * len(lo_powers)
    iteration = 0

    print(f"2. Generating {total_iterations} segments of signal data...")

    for bw in bandwidths:
        # Generate base signal for this bandwidth
        base_sig, _, _, _ = rf_chain.generate_variable_bw_ofdm(bw, FS, num_symbols=30)
        
        for pwr in powers:
            print(f"   Processing BW: {bw/1e6}MHz, Pwr: {pwr:.1f}dBm... ({iteration}/{total_iterations})")
            for v in voltages:
                for lo_pwr in lo_powers:
                    for g_db in gain_controls:
                        iteration += 1
                        # Fixed PA Drive (Hardware does not have a variable driver amp)
                        pa_drive = 0.0
                        
                        op = OperatingPoint(power_pre_lna_dbm=pwr, bandwidth=bw)
                        # Pass LO Power to Setting
                        setting = Setting(lna_voltage=v, pa_drive_db=pa_drive, lo_power_dbm=lo_pwr)
                        
                        # Run chain up to PA input
                        x_pa_in = rf_chain.process_chain_pre_pa(base_sig, op, setting)
                    
                        # Apply Ground Truth PA Model (Parametric)
                        y_pa_out = apply_pa_distortion(x_pa_in, gain_db=g_db)
                        
                        # Normalize gain for NN input (scale -6..26 to 0..1)
                        g_norm = (g_db - (-6)) / (26 - (-6))
                        
                        # Pre-calculate features here to ensure 'g_norm' is matched to this segment
                        feats, valid_len = prepare_features(x_pa_in, MEM_DEPTH, DEGREE, aux_param=g_norm)
                        
                        # Align Target
                        y_aligned = y_pa_out[MEM_DEPTH-1 : MEM_DEPTH-1+valid_len]
                        
                        # OPTIMIZATION: Convert to float32/complex64 immediately to save memory
                        all_features.append(feats.astype(np.float32))
                        all_targets.append(y_aligned.astype(np.complex64))

    print("\n3. Concatenating and Shuffling Data...")
    # Concatenate all segments
    X_full = np.concatenate(all_features, axis=0)
    y_full = np.concatenate(all_targets, axis=0)

    # Global Shuffle before split
    indices = np.random.permutation(len(y_full))
    X_full = X_full[indices]
    y_full = y_full[indices]

    # Save data
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    np.savez(DATA_FILE, X_full=X_full, y_full=y_full)
    print(f"   Data saved to {DATA_FILE}")

else:
    print(f"1. Loading Data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file {DATA_FILE} not found. Please set GENERATE_DATA = True.")
    
    data = np.load(DATA_FILE)
    X_full = data['X_full']
    y_full = data['y_full']
    print("   Data loaded successfully.")

print(f"   Total samples generated: {len(y_full)}")

# Split into Train/Test (80/20)
split_idx = int(len(y_full) * 0.8)
X_train_feat, X_test_feat = X_full[:split_idx], X_full[split_idx:]
y_train, y_test = y_full[:split_idx], y_full[split_idx:]

# Stack Targets [Real, Imag]
Y_train_stack = np.stack([np.real(y_train), np.imag(y_train)], axis=1)
Y_test_stack = np.stack([np.real(y_test), np.imag(y_test)], axis=1)

print(f"   Input Feature Dimension: {X_train_feat.shape[1]}")
print(f"   Training Samples: {X_train_feat.shape[0]}")

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"4. Training on device: {device}")

# Instantiate
model = ARVTDNN(X_train_feat.shape[1]).to(device)

# Checkpointing
os.makedirs("models", exist_ok=True)
best_val_loss = float('inf')
best_ckpt_path = "models/best_arvtdnn_pa_model_sw.pth"

# Prepare Dataloaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train_feat), 
    torch.from_numpy(Y_train_stack)
)
val_dataset = TensorDataset(
    torch.from_numpy(X_test_feat), 
    torch.from_numpy(Y_test_stack)
)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) # Reduced LR slightly
criterion = nn.MSELoss()

# Training Loop
epochs = 15
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss_acc = 0.0
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss_acc += criterion(out, yb).item() * xb.size(0)
    val_loss = val_loss_acc / len(val_loader.dataset)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"   Checkpoint saved (val improved from {best_val_loss:.8e} to {val_loss:.8e})")
        best_val_loss = val_loss
    else:
        print(f"   No checkpoint saved (val {val_loss:.8e} did not improve over {best_val_loss:.8e})")

    print(f"   Epoch {epoch}/{epochs} - Train Loss: {train_loss:.8e} - Val Loss: {val_loss:.8e}")

# Evaluation with

print("5. Evaluating & Generating Passband Plots...")

model.eval()
with torch.inference_mode():
    X_eval = torch.from_numpy(X_test_feat.astype(np.float32)).to(device)
    y_pred_tensor = model(X_eval).cpu().numpy()

# 1. Reconstruct Analytic Prediction
y_pred_complex = y_pred_tensor[:, 0] + 1j * y_pred_tensor[:, 1]

# 2. Calculate NMSE
# y_test is already the aligned target chunk
error = y_test - y_pred_complex
nmse_db = 10 * np.log10(np.sum(np.abs(error)**2) / np.sum(np.abs(y_test)**2))
print(f"   NMSE on Test Set: {nmse_db:.2f} dB")


# Visualization (Handling IF signals)
# Since inputs/outputs are Analytic IF (centered at 25 MHz), taking Real part gives the physical signal.

y_test_physical = np.real(y_test)
y_pred_physical = np.real(y_pred_complex)

# Reconstruct approximate input physical signal from feature matrix
# Column 0 is Real(u[n]), which is effectively the physical signal for Analytic IF at baseband equivalent view
x_test_physical = X_test_feat[:, 0]

plt.figure(figsize=(10, 6))

plt.psd(y_test_physical, NFFT=1024, Fs=FS, label='Actual PA Output (Physical)', color='blue')
plt.psd(y_pred_physical, NFFT=1024, Fs=FS, label='NN Predicted Output (Physical)', linestyle='--', color='orange')
plt.psd(x_test_physical, NFFT=1024, Fs=FS, label='Input Signal (Physical)', alpha=0.5, linestyle=':', color='green')
plt.title(f'Physical Spectrum (Analytic IF Data)\nNMSE: {nmse_db:.2f} dB')
plt.legend()
plt.xlim(0, FS/2)
plt.grid(True, which='both')
plt.show()

plt.figure(figsize=(10, 6))
# AM/AM Check: Magnitude of Analytic Signal represents envelope
# We need to approximate the input envelope from the features (Real + Imag)
# Feature 0 is Real, Feature 1 is Imag
input_envelope = np.sqrt(X_test_feat[:, 0]**2 + X_test_feat[:, 1]**2)
output_envelope = np.abs(y_test)

plt.scatter(input_envelope, output_envelope, s=1, label='Actual PA Physics', alpha=0.8)
plt.scatter(input_envelope, np.abs(y_pred_complex), s=1, label='Neural Network Model', color='orange', alpha=0.5)
plt.xlabel('Input Amplitude |u|')
plt.ylabel('Output Amplitude |y|')
plt.title('AM/AM Characteristic (Envelope)')
plt.legend()
plt.grid(True)
plt.show()
