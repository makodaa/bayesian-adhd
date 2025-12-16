import io
import pandas as pd
from werkzeug.datastructures import FileStorage
from ..config import ALLOWED_EXTENSIONS
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)


class FileService:
    """Handle file upload, validation, and parsing."""
    @staticmethod
    def is_allowed_file(filename:str) -> bool:
        """Check if file extension is allowed"""
        allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        logger.debug(f"File extension check for '{filename}': {allowed}")
        return allowed
    
    @staticmethod
    def read_csv(file: FileStorage) -> pd.DataFrame:
        """Read and parse CSV file into DataFrame"""
        logger.info(f"Reading CSV file: {file.filename}")
        binary = file.stream.read()
        try:
            csv_string = binary.decode('utf-8').replace('\r','')
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode file '{file.filename}' as UTF-8", exc_info=True)
            raise ValueError("Could not decode file content as UTF-8. It might be a binary file.") from e
        
        csv_io = io.StringIO(csv_string)
        df = pd.read_csv(csv_io)
        csv_io.close()
        
        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    @staticmethod
    def validate_eeg_data(df: pd.DataFrame) -> bool:
        """Validate that DataFrame contains valid EEG data."""
        logger.debug("Validating EEG data")
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            logger.error("Validation failed: No numeric columns found in file")
            raise ValueError("No numeric columns found in file.")
        logger.info(f"EEG data validation passed: {len(numeric_cols)} numeric columns found")
        return True