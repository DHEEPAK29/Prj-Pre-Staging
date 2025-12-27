import mlflow
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import prometheus_client
from prometheus_client import Counter, Histogram
import json
from datetime import datetime
import os
 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXPERIMENT TRACKING (MLFLOW & WANDB)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name) 
    if not mlflow.active_run():
        mlflow.start_run()
    print("✓ MLflow started")

def log_mlflow_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_mlflow_metrics(metrics, step):
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)

def setup_wandb(project_name, experiment_name, config):
    wandb.init(project=project_name, name=experiment_name, config=config)
    print("✓ W&B initialized")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# VISUALIZATION & ARCHITECTURE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def export_to_netron(model, input_shape, onnx_path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
                      input_names=['input'], output_names=['output'])
    print(f"✓ Model exported to {onnx_path} (visualize at netron.app)")

def setup_tensorboard(log_dir="./runs"):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    print("✓ TensorBoard initialized")
    return writer

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL DEFINITION  
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=200):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image data
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TRAINING ENGINE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train_model(model, train_loader, val_loader, optimizer, criterion, 
                epochs=5, device='cpu', mlflow_log=True, wandb_log=True, tb_writer=None):
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Logging
        metrics = {"train_loss": avg_train_loss, "epoch": epoch}
        if mlflow_log: log_mlflow_metrics(metrics, epoch)
        if wandb_log: wandb.log(metrics)
        if tb_writer: 
            for k, v in metrics.items(): tb_writer.add_scalar(k, v, epoch)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MAIN EXECUTION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Tracking
    setup_mlflow("Track")
    setup_wandb("sequence", "baseline-run", {"num_classes": 100})
 
    model = SimpleModel(input_size=784, hidden_size=128, output_size=10).to(device)
    export_to_netron(model, (1, 784))
 
    dummy_x = torch.randn(100, 784)
    dummy_y = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
 
    train_model(model, train_loader, train_loader, optimizer, criterion, device=device)
 
    mlflow.end_run()
    wandb.finish()
    print("✓ Pipeline Complete!")

if __name__ == "__main__":
    main()
