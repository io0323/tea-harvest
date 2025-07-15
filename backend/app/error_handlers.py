from fastapi import Request
from fastapi.responses import JSONResponse
from .exceptions import (
    TeaHarvestException,
    DataValidationError,
    ModelNotFoundError,
    PredictionError,
    PreprocessingError,
    InvalidFileFormatError
)
import logging

logger = logging.getLogger(__name__)

async def tea_harvest_exception_handler(request: Request, exc: TeaHarvestException):
    """
    茶葉収穫予測システムの基本例外ハンドラー
    """
    logger.error(f"エラーが発生しました: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "システムエラーが発生しました", "detail": str(exc)}
    )

async def data_validation_error_handler(request: Request, exc: DataValidationError):
    """
    データ検証エラーのハンドラー
    """
    logger.warning(f"データ検証エラー: {str(exc)}, 詳細: {exc.details}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "データ検証エラー",
            "detail": str(exc),
            "validation_details": exc.details
        }
    )

async def model_not_found_error_handler(request: Request, exc: ModelNotFoundError):
    """
    モデル未発見エラーのハンドラー
    """
    logger.error(f"モデル未発見エラー: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "予測モデルが利用できません",
            "detail": str(exc)
        }
    )

async def prediction_error_handler(request: Request, exc: PredictionError):
    """
    予測エラーのハンドラー
    """
    logger.error(f"予測エラー: {str(exc)}, モデル状態: {exc.model_state}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "予測の実行中にエラーが発生しました",
            "detail": str(exc)
        }
    )

async def preprocessing_error_handler(request: Request, exc: PreprocessingError):
    """
    前処理エラーのハンドラー
    """
    logger.error(f"前処理エラー: {str(exc)}, ステップ: {exc.preprocessing_step}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "データの前処理中にエラーが発生しました",
            "detail": str(exc),
            "preprocessing_step": exc.preprocessing_step
        }
    )

async def invalid_file_format_error_handler(request: Request, exc: InvalidFileFormatError):
    """
    ファイルフォーマットエラーのハンドラー
    """
    logger.warning(f"ファイルフォーマットエラー: {str(exc)}, ファイル情報: {exc.file_info}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "ファイルフォーマットが不正です",
            "detail": str(exc),
            "file_info": exc.file_info
        }
    ) 