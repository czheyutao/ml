import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, scaler_y, epochs=10, lr=0.01, device='cuda', warmup_epochs=2, modelname='lstm'):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return max(0.1, 1.0 / (1.0 + 0.01 * (epoch - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.MSELoss()
    best_mae = float('inf')
    mse_history = []
    mae_history = []
    std_history = []
    train_mse_history = []
    train_mae_history = []
    train_std_history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_mse, train_mae, train_std = evaluate(model, train_loader, scaler_y, device, mode='Train')
        train_mse_history.append(train_mse)
        train_mae_history.append(train_mae)
        train_std_history.append(train_std)

        print(f"Epoch {epoch+1} Train Loss: {np.mean(train_losses):.4f}")
        
        mse, mae, std = evaluate(model, val_loader, scaler_y, device, mode='Val')
        mse_history.append(mse)
        mae_history.append(mae)
        std_history.append(std)

        if mae < best_mae:
            torch.save(model.state_dict(), f'./ml/best_model_{modelname}.pt')
            print(f"Saved model with MAE: {mae:.4f}")
            best_mae = mae
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Learning Rate: {current_lr:.6f}")

    print("Finished training")
    print(f"Best MAE: {best_mae:.4f}")

    plt.figure(figsize=(10, 15))
    
    plt.subplot(3, 1, 1)
    plt.plot(range(1, epochs + 1), train_mse_history, label='Train MSE', color='blue')
    plt.plot(range(1, epochs + 1), mse_history, label='Val MSE', color='red')
    plt.title('MSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(1, epochs + 1), train_mae_history, label='Train MAE', color='blue')
    plt.plot(range(1, epochs + 1), mae_history, label='Val MAE', color='red')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(range(1, epochs + 1), train_std_history, label='Train STD', color='blue')
    plt.plot(range(1, epochs + 1), std_history, label='Val STD', color='red')
    plt.title('STD over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('STD')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./ml/assets/validation_metrics_{modelname}.png')
    plt.close()

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb[:1])  # use only the first sample in the batch
            pred = pred.squeeze(0).cpu().numpy()
            true = yb[:1].squeeze(0).cpu().numpy()
            # Inverse transform to original scale
            pred = scaler_y.inverse_transform(pred)
            true = scaler_y.inverse_transform(true)
            break  # Only visualize one sample

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(true)), true, label='Ground Truth', color='blue')
    plt.plot(range(len(pred)), pred, label='Prediction', color='orange')
    plt.title('Pred vs Ground Truth')
    plt.xlabel('Day')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./ml/assets/prediction_vs_truth_{modelname}.png')
    plt.close()

def evaluate(model, loader, scaler_y, device='cuda', mode='Val'):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yb = yb.numpy()
            pred = scaler_y.inverse_transform(pred.squeeze(-1))
            yb = scaler_y.inverse_transform(yb.squeeze(-1))
            preds.append(pred)
            trues.append(yb)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    std = np.std(preds - trues)
    print(f"[{mode}][Eval] MSE: {mse:.4f} | MAE: {mae:.4f} | STD: {std:.4f}")
    return mse, mae, std