import pandas as pd
import numpy as np
from pathlib import Path
from models.tea_harvest_model import TeaHarvestModel
import argparse

def load_data(data_path: str) -> pd.DataFrame:
    """
    学習データの読み込み
    
    Args:
        data_path (str): データファイルのパス
        
    Returns:
        pd.DataFrame: 読み込まれたデータ
    """
    df = pd.read_csv(data_path)
    return df

def main():
    parser = argparse.ArgumentParser(description='茶葉収穫時期予測モデルの学習')
    parser.add_argument('--data_path', type=str, required=True, help='学習データのパス')
    parser.add_argument('--model_path', type=str, default='models/saved', help='モデルの保存先')
    parser.add_argument('--epochs', type=int, default=100, help='学習エポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    
    args = parser.parse_args()
    
    # データの読み込み
    print('データを読み込んでいます...')
    df = load_data(args.data_path)
    
    # モデルの初期化
    print('モデルを初期化しています...')
    model = TeaHarvestModel()
    
    # データの前処理
    print('データの前処理を実行しています...')
    X, y = model.preprocess_data(df)
    
    # モデルの学習
    print('モデルの学習を開始します...')
    history = model.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
    
    # モデルの保存
    print('モデルを保存しています...')
    model.save(args.model_path)
    
    print('学習が完了しました。')
    print(f'最終的なMAE: {history["mae"][-1]:.2f}')
    print(f'最終的なval_MAE: {history["val_mae"][-1]:.2f}')

if __name__ == '__main__':
    main() 