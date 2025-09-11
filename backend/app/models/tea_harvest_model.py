import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

class TeaHarvestModel:
    """
    茶葉収穫時期予測のためのLSTMモデル
    
    特徴量:
    - 気温（平均、最高、最低）
    - 湿度
    - 降水量
    - 日照時間
    - 地域（エンコード済み）
    """
    
    def __init__(self, sequence_length: int = 30):
        """
        モデルの初期化
        
        Args:
            sequence_length (int): 時系列データのシーケンス長（デフォルト: 30日）
        """
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self.scaler = None
        self.region_encoder = None
        self.imputer = None
        self.feature_ranges = None
        self.logger = logging.getLogger(__name__)
        
    def _build_model(self) -> Sequential:
        """
        LSTMモデルの構築
        
        Returns:
            Sequential: 構築されたモデル
        """
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 6), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # 収穫日までの日数を予測
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        データの検証
        
        Args:
            df (pd.DataFrame): 入力データ
            
        Raises:
            ValueError: データが不正な場合
        """
        # 必要な列の確認
        required_columns = ['date', 'temperature', 'humidity', 'precipitation', 'sunshine', 'region']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"必要な列が不足しています: {required_columns}")
            
        # データ型の確認
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                raise ValueError(f"日付の変換に失敗しました: {e}")
                
        # 数値列の確認
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'sunshine']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"{col}は数値型である必要があります")
                
        # 欠損値の確認
        missing_values = df[numeric_columns].isnull().sum()
        if missing_values.any():
            self.logger.warning(f"欠損値が検出されました: {missing_values[missing_values > 0]}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値の処理
        
        Args:
            df (pd.DataFrame): 入力データ
            
        Returns:
            pd.DataFrame: 欠損値処理済みデータ
        """
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'sunshine']
        
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(df[numeric_columns])
            
        df[numeric_columns] = self.imputer.transform(df[numeric_columns])
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量の正規化
        
        Args:
            df (pd.DataFrame): 入力データ
            
        Returns:
            pd.DataFrame: 正規化済みデータ
        """
        features = ['temperature', 'humidity', 'precipitation', 'sunshine']
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df[features])
            
        df[features] = self.scaler.transform(df[features])
        return df
    
    def _encode_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        地域のエンコーディング
        
        Args:
            df (pd.DataFrame): 入力データ
            
        Returns:
            pd.DataFrame: エンコード済みデータ
        """
        if self.region_encoder is None:
            self.region_encoder = {region: idx for idx, region in enumerate(df['region'].unique())}
            
        df['region_encoded'] = df['region'].map(self.region_encoder)
        return df
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        派生特徴量の計算
        
        Args:
            df (pd.DataFrame): 入力データ
            
        Returns:
            pd.DataFrame: 派生特徴量追加済みデータ
        """
        # 移動平均
        df['temperature_ma7'] = df['temperature'].rolling(window=7, min_periods=1).mean()
        df['humidity_ma7'] = df['humidity'].rolling(window=7, min_periods=1).mean()
        
        # 季節性の特徴量
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        データの前処理
        
        Args:
            df (pd.DataFrame): 生データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 前処理済みの特徴量とターゲット
        """
        self.logger.info("データの前処理を開始します")
        
        # データの検証
        self._validate_data(df)
        
        # 欠損値の処理
        df = self._handle_missing_values(df)
        
        # 派生特徴量の計算
        df = self._calculate_derived_features(df)
        
        # 特徴量の正規化
        df = self._normalize_features(df)
        
        # 地域のエンコーディング
        df = self._encode_regions(df)
        
        # シーケンスデータの作成
        X, y = self._create_sequences(df)
        
        self.logger.info("データの前処理が完了しました")
        return X, y
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列シーケンスデータの作成
        
        Args:
            df (pd.DataFrame): 前処理済みデータ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: シーケンスデータとターゲット
        """
        features = [
            'temperature', 'humidity', 'precipitation', 'sunshine',
            'region_encoded', 'temperature_ma7', 'humidity_ma7',
            'month', 'day_of_year'
        ]
        
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length):
            X.append(df[features].values[i:i+self.sequence_length])
            y.append((df['harvest_date'].iloc[i+self.sequence_length] - df['date'].iloc[i]).days)
            
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> dict:
        """
        モデルの学習
        
        Args:
            X (np.ndarray): 特徴量
            y (np.ndarray): ターゲット
            epochs (int): エポック数
            batch_size (int): バッチサイズ
            
        Returns:
            dict: 学習履歴
        """
        self.logger.info(f"モデルの学習を開始します（エポック数: {epochs}, バッチサイズ: {batch_size}）")
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        self.logger.info("モデルの学習が完了しました")
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測の実行
        
        Args:
            X (np.ndarray): 入力データ
            
        Returns:
            np.ndarray: 予測結果（収穫日までの日数）
        """
        return self.model.predict(X)
    
    def save(self, model_path: str):
        """
        モデルの保存
        
        Args:
            model_path (str): 保存先のパス
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # モデルの保存
        self.model.save(str(model_path / 'model.keras'))
        
        # 前処理用オブジェクトの保存
        joblib.dump({
            'scaler': self.scaler,
            'region_encoder': self.region_encoder,
            'imputer': self.imputer,
            'feature_ranges': self.feature_ranges
        }, str(model_path / 'preprocessors.pkl'))
        
        self.logger.info(f"モデルを保存しました: {model_path}")
    
    @classmethod
    def load(cls, model_path: str) -> 'TeaHarvestModel':
        """
        モデルの読み込み
        
        Args:
            model_path (str): モデルのパス
            
        Returns:
            TeaHarvestModel: 読み込まれたモデル
        """
        model_path = Path(model_path)
        
        # モデルのインスタンス化
        model = cls()
        
        # モデルの読み込み
        model.model = tf.keras.models.load_model(str(model_path / 'model.keras'))
        
        # 前処理用オブジェクトの読み込み
        preprocessors = joblib.load(str(model_path / 'preprocessors.pkl'))
        model.scaler = preprocessors['scaler']
        model.region_encoder = preprocessors['region_encoder']
        model.imputer = preprocessors['imputer']
        model.feature_ranges = preprocessors['feature_ranges']
        
        model.logger.info(f"モデルを読み込みました: {model_path}")
        return model 