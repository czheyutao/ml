import argparse
from power_dataset import load_and_preprocess, build_dataloader
from train_eval import train_model
from models.lstm import LSTMForecast
from models.new_transformer import EnhancedTransformerForecast

def main(model_type="new_transformer", long_term=False):
    input_days = 90
    pred_days = 365 if long_term else 90

    features_train, targets_train, sx, sy = load_and_preprocess("/data/hyt/ml/data/train.csv")
    features_test, targets_test, _, _ = load_and_preprocess("/data/hyt/ml/data/test.csv")

    train_loader = build_dataloader(features_train, targets_train, input_days, pred_days, batch_size=32, shuffle=True)
    test_loader = build_dataloader(features_test, targets_test, input_days, pred_days, batch_size=32, shuffle=False)

    input_dim = features_train.shape[1]

    if model_type == "lstm":
        model = LSTMForecast(input_dim=input_dim, hidden_dim=64, num_layers=2, pred_len=pred_days)
    elif model_type == "transformer":
        model = EnhancedTransformerForecast(
            input_dim=input_dim, 
            d_model=64, 
            nhead=4, 
            num_layers=2, 
            pred_len=pred_days,
            dropout=0.1,
            model_variant="original"
        )
    elif model_type == "new_transformer":
        model = EnhancedTransformerForecast(
            input_dim=input_dim, 
            d_model=64, 
            nhead=4, 
            num_layers=2, 
            pred_len=pred_days,
            dropout=0.1,
            model_variant="enhanced"
        )
    else:
        raise ValueError("Unsupported model type")

    train_model(
        model, 
        train_loader, 
        test_loader, 
        scaler_y=sy, 
        epochs=300, 
        lr=0.01, 
        device='cuda:0',
        warmup_epochs=2,
        modelname=model_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a forecasting model")
    parser.add_argument('--model_type', type=str, default='new_transformer', choices=['lstm', 'transformer', 'new_transformer'])
    parser.add_argument('--long_term', action='store_true', help='Use long-term prediction (365 days)')
    args = parser.parse_args()
    main(model_type=args.model_type, long_term=args.long_term)