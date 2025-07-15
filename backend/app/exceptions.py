class TeaHarvestException(Exception):
    """茶葉収穫予測システムの基本例外クラス"""
    pass

class DataValidationError(TeaHarvestException):
    """データ検証エラー"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class ModelNotFoundError(TeaHarvestException):
    """モデルが見つからないエラー"""
    pass

class PredictionError(TeaHarvestException):
    """予測実行時のエラー"""
    def __init__(self, message: str, model_state: dict = None):
        super().__init__(message)
        self.model_state = model_state or {}

class PreprocessingError(TeaHarvestException):
    """データ前処理時のエラー"""
    def __init__(self, message: str, preprocessing_step: str = None):
        super().__init__(message)
        self.preprocessing_step = preprocessing_step

class InvalidFileFormatError(TeaHarvestException):
    """ファイルフォーマットエラー"""
    def __init__(self, message: str, file_info: dict = None):
        super().__init__(message)
        self.file_info = file_info or {} 