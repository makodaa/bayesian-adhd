from typing import Optional
from ..db.repositories.results import ResultsRepository
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

class ResultsService:
    """Manage analysis results and related operations."""
    
    def __init__(self, results_repo: ResultsRepository):
        self.results_repo = results_repo
    
    def get_result_by_id(self, result_id: int) -> Optional[dict]:
        """Get result by ID."""
        logger.debug(f"ResultsService: fetching result {result_id}")
        return self.results_repo.get_by_id(result_id)
    
    def create_result(self, recording_id: int, classification: str, confidence_score: float, clinician_id: int | None = None) -> int:
        """Create a new result entry."""
        logger.info(f"ResultsService: creating result for recording {recording_id}")
        return self.results_repo.create_result(
            recording_id=recording_id,
            classification=classification,
            confidence_score=confidence_score,
            clinician_id=clinician_id
        )
    
    def get_all_results_with_details(self):
        """Get all results with subject and recording details."""
        logger.debug("ResultsService: fetching all results with details")
        query = """
        SELECT 
            r.id as result_id,
            r.predicted_class,
            r.confidence_score,
            r.inferenced_at,
            rec.id as recording_id,
            rec.file_name,
            s.id as subject_id,
            s.subject_code,
            s.age,
            s.gender,
            c.first_name,
            c.last_name
        FROM results r
        JOIN recordings rec ON r.recording_id = rec.id
        JOIN subjects s ON rec.subject_id = s.id
        LEFT JOIN clinicians c ON r.clinician_id = c.id
        ORDER BY r.inferenced_at DESC;
        """
        try:
            with self.results_repo.get_connection() as conn:
                cursor = self.results_repo.get_dict_cursor(conn)
                cursor.execute(query)
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch results with details: {e}", exc_info=True)
            raise
    
    def get_result_with_full_details(self, result_id: int):
        """Get detailed result including band powers and ratios."""
        logger.debug(f"ResultsService: fetching full details for result {result_id}")
        query = """
        SELECT 
            r.id as result_id,
            r.predicted_class,
            r.confidence_score,
            r.inferenced_at,
            rec.id as recording_id,
            rec.file_name,
            rec.sleep_hours,
            rec.food_intake,
            rec.caffeinated,
            rec.medicated,
            rec.medication_intake,
            rec.notes,
            s.id as subject_id,
            s.subject_code,
            s.age,
            s.gender,
            c.first_name as clinician_first_name,
            c.last_name as clinician_last_name,
            c.middle_name as clinician_middle_name,
            c.occupation as clinician_occupation
        FROM results r
        JOIN recordings rec ON r.recording_id = rec.id
        JOIN subjects s ON rec.subject_id = s.id
        LEFT JOIN clinicians c ON r.clinician_id = c.id
        WHERE r.id = %s;
        """
        try:
            with self.results_repo.get_connection() as conn:
                cursor = self.results_repo.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"Result {result_id} not found")
                    return None
                
                # Get band powers
                cursor.execute(
                    "SELECT * FROM band_powers WHERE result_id = %s ORDER BY electrode, frequency_band;",
                    (result_id,)
                )
                result['band_powers'] = cursor.fetchall()
                
                # Get ratios
                cursor.execute(
                    "SELECT * FROM ratios WHERE result_id = %s;",
                    (result_id,)
                )
                result['ratios'] = cursor.fetchall()
                
                logger.info(f"Retrieved full details for result {result_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch full details for result {result_id}: {e}", exc_info=True)
            raise
