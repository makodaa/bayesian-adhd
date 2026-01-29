import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax

import numpy as np

class EEGCNNLSTM(nn.Module):
    def __init__(self, num_band_features=7, num_classes=4,
                 cnn_kernels_1=32,
                 cnn_kernel_size_1=3,
                 cnn_kernels_2=32,
                 cnn_kernel_size_2=3,
                 cnn_dropout=0.3,
                 cnn_dense=16,
                 lstm_hidden_size=32,
                 lstm_layers=4,
                 lstm_dense=64,
                 dropout=0.3):
        super().__init__()

        pad1 = cnn_kernel_size_1 // 2
        self.conv1   = nn.Conv2d(1, int(cnn_kernels_1), kernel_size=cnn_kernel_size_1, padding=pad1)
        self.pool1 = nn.AvgPool2d(2)

        pad2 = cnn_kernel_size_2 // 2
        self.conv2 = nn.Conv2d(int(cnn_kernels_1), int(cnn_kernels_2), kernel_size=cnn_kernel_size_2, padding=pad2)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 77, 19)
            out = self._forward_cnn(dummy)   # [B, C, H, W]
            b, c, h, w = out.shape
            self.seq_len = h                      # sequence length (rows)
            self.cnn_feat_dim = c * w             # CNN features per timestep

        # Dense layer BEFORE LSTM
        self.cnn_dense = nn.Linear(self.cnn_feat_dim, int(cnn_dense))

        # Two stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=int(cnn_dense),
            hidden_size=int(lstm_hidden_size),
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        # self.lstm_dense = nn.Linear(int(lstm_hidden_size), int(lstm_dense))
        self.band_fc = nn.Sequential(
            nn.Linear(num_band_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # final classifier (match your original final style: dropout + linear)
        self.classifier = nn.Sequential(
            nn.Linear(int(lstm_hidden_size) + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def _forward_cnn(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.cnn_dropout(x)
        return x

    def forward(self, x_eeg, x_band):
        # 1️⃣ CNN feature extraction
        x = self._forward_cnn(x_eeg)             # [B, C, H, W]

        # 2️⃣ Prepare sequence for LSTM
        x = x.permute(0, 2, 1, 3)                 # [B, H, C, W]
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # [B, H, C*W]

        # 3️⃣ Dense layer for each timestep
        x = F.relu(self.cnn_dense(x))     # [B, H, dense_size]

        # 4️⃣ Two-layer LSTM
        lstm_out, _ = self.lstm(x)                # [B, H, hidden_size]

        # 5️⃣ Use last time step (or mean/attention if preferred)
        eeg_feat = lstm_out[:, -1, :]
        # eeg_feat = lstm_out.mean(dim=1)                    # [B, hidden_size]
        # eeg_feat = self.lstm_dense(eeg_feat)

        # --- Band features ---
        band_feat = self.band_fc(x_band)        # [B, 32]

        # --- Fusion ---
        fused = torch.cat([eeg_feat, band_feat], dim=1)

        # 6️⃣ Fully connected head
        x = self.classifier(fused)

        return x

    def fit(self, train_loader, test_loader, epochs, criterion, optimizer, device, patience=100, is_quiet=False):
        best_val_loss = float('inf')
        no_improve = 0

        train_losses, train_accs = [], []
        val_losses, val_accs     = [], []

        best_state = None
        for epoch in range(epochs):
            # --- Train ---
            self.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for xb_eeg, xb_band, yb in train_loader:
                xb_eeg = xb_eeg.to(device)
                xb_band = xb_band.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = self(xb_eeg, xb_band)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb_eeg.size(0)
                train_correct += (out.argmax(1) == yb).sum().item()
                train_total += yb.size(0)

            train_loss /= train_total
            train_acc  = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # --- Validate ---
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for xb_eeg, xb_band, yb in test_loader:
                    xb_eeg = xb_eeg.to(device)
                    xb_band = xb_band.to(device)
                    yb = yb.to(device)
                    out = self(xb_eeg, xb_band)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb_eeg.size(0)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_total += yb.size(0)

            val_loss /= val_total
            val_acc  = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if not is_quiet:
                print(f"Epoch {epoch+1:03d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # if val_loss - train_loss > 0.2:
            #     print("Overfitting detected.")
            #     break

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return {
            "train_accs": np.array(train_accs),
            "train_losses": np.array(train_losses),
            "val_accs":   np.array(val_accs),
            "val_losses": np.array(val_losses)
        }