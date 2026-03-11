import io
import pandas as pd
from werkzeug.datastructures import FileStorage
from ..config import ALLOWED_EXTENSIONS, ELECTRODE_CHANNELS
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
        binary = file.stream.read()
        return FileService.read_csv_bytes(binary, file.filename)

    @staticmethod
    def read_csv_bytes(binary: bytes, filename: str | None = None) -> pd.DataFrame:
        """Read and parse CSV bytes into DataFrame."""
        file_label = filename or "<in-memory>"
        logger.info(f"Reading CSV content: {file_label}")

        try:
            csv_string = binary.decode('utf-8').replace('\r','')
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode file '{file_label}' as UTF-8", exc_info=True)
            raise ValueError("Could not decode file content as UTF-8. It might be a binary file.") from e

        csv_io = io.StringIO(csv_string)
        df = pd.read_csv(csv_io)
        csv_io.close()

        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    @staticmethod
    def validate_eeg_data(df: pd.DataFrame) -> bool:
        """Validate that DataFrame contains valid EEG data.

        Checks performed:
        1. The file is not empty (at least one data row).
        2. At least one numeric column is present.
        3. All 19 required EEG channels from the 10-20 system are present.
           The check handles both named-column CSVs and zero/one-indexed numeric
           header CSVs (which are implicitly treated as the full channel set).

        Raises ``ValueError`` with a user-facing message on any failure.
        """
        logger.debug("Validating EEG data")

        # 1. Non-empty rows
        if len(df) == 0:
            logger.error("Validation failed: EEG file contains no data rows")
            raise ValueError("The uploaded EEG file is empty (no data rows found).")

        # 2. Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            logger.error("Validation failed: No numeric columns found in file")
            raise ValueError("No numeric columns found in file.")

        # 3. Required channel coverage
        # If the CSV uses purely numeric headers (0-indexed or 1-indexed for 19
        # channels) we treat it as a full-channel recording — consistent with the
        # frontend logic in inspectCsvFile().
        col_names = list(df.columns)
        try:
            header_numbers = [int(c) for c in col_names]
            is_0_to_18 = (
                len(header_numbers) == 19
                and all(header_numbers[i] == i for i in range(19))
            )
            is_1_to_19 = (
                len(header_numbers) == 19
                and all(header_numbers[i] == i + 1 for i in range(19))
            )
        except (ValueError, TypeError):
            is_0_to_18 = False
            is_1_to_19 = False

        if not (is_0_to_18 or is_1_to_19):
            # Named-column CSV — check for the required electrode channels
            col_names_lower = {c.lower() for c in col_names}
            missing = [
                ch for ch in ELECTRODE_CHANNELS
                if ch.lower() not in col_names_lower
            ]
            if missing:
                logger.error(
                    f"Validation failed: Missing EEG channels: {missing}"
                )
                raise ValueError(
                    f"The EEG file is missing {len(missing)} required channel(s): "
                    f"{', '.join(missing)}. "
                    "Upload a complete 19-channel recording using the 10-20 system."
                )

        logger.info(
            f"EEG data validation passed: {len(df)} rows, "
            f"{len(numeric_cols)} numeric columns"
        )
        return True
