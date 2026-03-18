from typing import Optional
from werkzeug.datastructures import FileStorage
from .file_service import FileService
from .eeg_service import EEGService
from .band_analysis_service import BandAnalysisService
from ..db.repositories.recordings import RecordingsRepository
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

class RecordingService:
    """Manage EEG recording upload, analysis, and storage"""

    def __init__(self,
                 recordings_repo: RecordingsRepository,
                 file_service: FileService,
                 eeg_service: EEGService,
                 band_analysis_service: BandAnalysisService
                ):
        self.recordings_repo = recordings_repo
        self.file_service = file_service
        self.eeg_service = eeg_service
        self.band_analysis_service = band_analysis_service
    
    def get_recordings_by_subject(self, subject_id: int) -> list:
        """Get all recordings for a subject."""
        logger.debug(f"RecordingService: fetching recordings for subject {subject_id}")
        return self.recordings_repo.get_by_subject(subject_id)
    
    def get_recording_by_id(self, recording_id: int) -> Optional[dict]:
        """Get recording by ID."""
        logger.debug(f"RecordingService: fetching recording {recording_id}")
        return self.recordings_repo.get_by_id(recording_id)
    
    def create_recording(self, subject_id: int, file_name: str, **kwargs) -> int:
        """Create a new recording entry."""
        logger.info(f"RecordingService: creating recording for subject {subject_id}: {file_name}")
        return self.recordings_repo.create_recording(
            subject_id=subject_id,
            file_name=file_name,
            **kwargs
        )

    def process_and_store(
        self,
        subject_id: int,
        file: FileStorage,
        referral_name=None,
        referral_institution=None,
        technician_name=None,
        sleep_hours=None,
        coffee_hours_ago=None,
        drugs_hours_ago=None,
        meal_hours_ago=None,
        medication=None,
        recorded_minutes=None,
        duration_minutes=None,
        artifacts_noted=None,
        notes=None,
    ) -> dict:
        """Process uploaded EEG file and store results."""
        logger.info(f"Processing and storing recording for subject {subject_id}: {file.filename}")
        
        df = self.file_service.read_csv(file)
        self.file_service.validate_eeg_data(df)

        # Create recording entry
        logger.debug("Creating recording entry in database")
        recording_id = self.recordings_repo.create_recording(
            subject_id=subject_id,
            file_name=file.filename,
            referral_name=referral_name,
            referral_institution=referral_institution,
            technician_name=technician_name,
            sleep_hours=sleep_hours,
            coffee_hours_ago=coffee_hours_ago,
            drugs_hours_ago=drugs_hours_ago,
            meal_hours_ago=meal_hours_ago,
            medication=medication,
            recorded_minutes=recorded_minutes,
            duration_minutes=duration_minutes,
            artifacts_noted=artifacts_noted,
            notes=notes
        )

        # Classify and save results
        logger.info(f"Starting classification for recording {recording_id}")
        classification_result = self.eeg_service.classify_and_save(recording_id,df)

        # Compute and save band powers
        logger.info(f"Computing band powers for result {classification_result['result_id']}")
        band_powers = self.band_analysis_service.compute_and_save(classification_result['result_id'], df)

        logger.info(f"Recording processing complete for recording {recording_id}")
        return {
            'recording_id': recording_id,
            'classification': classification_result,
            'band_powers' : band_powers
        }
