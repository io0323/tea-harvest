from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from models.tea_harvest_model import TeaHarvestModel
import io
import logging
from .exceptions import (
    TeaHarvestException,
    DataValidationError,
    ModelNotFoundError,
    PredictionError,
    PreprocessingError,
    InvalidFileFormatError
)
from .error_handlers import (
    tea_harvest_exception_handler,
    data_validation_error_handler,
    model_not_found_error_handler,
    prediction_error_handler,
    preprocessing_error_handler,
    invalid_file_format_error_handler
)

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TeaHarvest API")

# エラーハンドラーの登録
app.add_exception_handler(TeaHarvestException, tea_harvest_exception_handler)
app.add_exception_handler(DataValidationError, data_validation_error_handler)
app.add_exception_handler(ModelNotFoundError, model_not_found_error_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(PreprocessingError, preprocessing_error_handler)
app.add_exception_handler(InvalidFileFormatError, invalid_file_format_error_handler)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルの読み込み
MODEL_PATH = Path("models/saved")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = TeaHarvestModel.load(MODEL_PATH)
        logger.info("モデルの読み込みが完了しました")
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗しました: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "TeaHarvest API"}

@app.post("/predict")
async def predict_harvest_date(
    file: UploadFile = File(...),
    region: str = "静岡",
    year: int = 2024
):
    """
    気象データから茶葉の収穫時期を予測するAPIエンドポイント
    
    Args:
        file (UploadFile): 気象データのCSVファイル
        region (str): 地域名
        year (int): 予測対象年
        
    Returns:
        dict: 予測結果（収穫日と信頼度スコア）
        
    Raises:
        ModelNotFoundError: モデルが読み込まれていない場合
        InvalidFileFormatError: ファイルフォーマットが不正な場合
        DataValidationError: データの検証に失敗した場合
        PreprocessingError: データの前処理に失敗した場合
        PredictionError: 予測の実行に失敗した場合
    """
    if model is None:
        raise ModelNotFoundError("予測モデルが読み込まれていません")
    
    try:
        # ファイルの検証
        if not file.filename.endswith('.csv'):
            raise InvalidFileFormatError(
                "CSVファイルのみ対応しています",
                file_info={"filename": file.filename, "content_type": file.content_type}
            )
        
        # ファイルの読み込み
        try:
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise InvalidFileFormatError(
                f"ファイルの読み込みに失敗しました: {str(e)}",
                file_info={"filename": file.filename}
            )
        
        # データの検証
        try:
            required_columns = ['date', 'temperature', 'humidity', 'precipitation', 'sunshine']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataValidationError(
                    "必要な列が不足しています",
                    details={"missing_columns": missing_columns}
                )
        except Exception as e:
            raise DataValidationError(str(e))
        
        # 地域の追加
        df['region'] = region
        
        # データの前処理
        try:
            X, _ = model.preprocess_data(df)
        except Exception as e:
            raise PreprocessingError(
                f"データの前処理に失敗しました: {str(e)}",
                preprocessing_step="preprocess_data"
            )
        
        # 予測の実行
        try:
            days_to_harvest = model.predict(X[-1:])[0][0]
        except Exception as e:
            raise PredictionError(
                f"予測の実行に失敗しました: {str(e)}",
                model_state={"input_shape": X.shape}
            )
        
        # 予測日付の計算
        last_date = pd.to_datetime(df['date'].iloc[-1])
        predicted_date = last_date + timedelta(days=int(days_to_harvest))
        
        # 信頼度スコアの計算（簡易的な実装）
        confidence_score = max(0.0, min(1.0, 1.0 - abs(days_to_harvest - 30) / 30))
        
        logger.info(f"予測が完了しました - 地域: {region}, 予測日: {predicted_date.strftime('%Y-%m-%d')}")
        
        return {
            "predicted_date": predicted_date.strftime("%Y-%m-%d"),
            "confidence_score": float(confidence_score),
            "region": region,
            "year": year
        }
        
    except TeaHarvestException:
        raise
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {str(e)}")
        raise TeaHarvestException(f"予期せぬエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 