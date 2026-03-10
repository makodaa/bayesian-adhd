import json

from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class ResultsRepository(BaseRepository):
    def create_result(
        self,
        recording_id,
        classification,
        confidence_score,
        clinician_id=None,
        vanderbilt_scale_type=None,
        vanderbilt_inattentive_count=None,
        vanderbilt_hyperactive_impulsive_count=None,
        vanderbilt_performance_impairment_count=None,
        vanderbilt_adhd_inattentive_met=None,
        vanderbilt_adhd_hyperactive_impulsive_met=None,
        vanderbilt_adhd_combined_met=None,
        vanderbilt_interpretation=None,
        vanderbilt_domain_scores=None,
        vanderbilt_symptom_scores=None,
        vanderbilt_performance_scores=None,
    ):
        """Create a new result and return its ID."""
        logger.info(f"Creating result for recording {recording_id}: classification={classification}, confidence={confidence_score*100:.2f}%")
        query = """
        INSERT INTO results(
            recording_id,
            clinician_id,
            predicted_class,
            confidence_score,
            vanderbilt_scale_type,
            vanderbilt_inattentive_count,
            vanderbilt_hyperactive_impulsive_count,
            vanderbilt_performance_impairment_count,
            vanderbilt_adhd_inattentive_met,
            vanderbilt_adhd_hyperactive_impulsive_met,
            vanderbilt_adhd_combined_met,
            vanderbilt_interpretation,
            vanderbilt_domain_scores,
            vanderbilt_symptom_scores,
            vanderbilt_performance_scores
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (
                        recording_id,
                        clinician_id,
                        classification,
                        confidence_score,
                        vanderbilt_scale_type,
                        vanderbilt_inattentive_count,
                        vanderbilt_hyperactive_impulsive_count,
                        vanderbilt_performance_impairment_count,
                        vanderbilt_adhd_inattentive_met,
                        vanderbilt_adhd_hyperactive_impulsive_met,
                        vanderbilt_adhd_combined_met,
                        vanderbilt_interpretation,
                        json.dumps(vanderbilt_domain_scores)
                        if vanderbilt_domain_scores is not None
                        else None,
                        json.dumps(vanderbilt_symptom_scores)
                        if vanderbilt_symptom_scores is not None
                        else None,
                        json.dumps(vanderbilt_performance_scores)
                        if vanderbilt_performance_scores is not None
                        else None,
                    ),
                )
                result_id = cursor.fetchone()[0]
                logger.info(f"Result created successfully with ID: {result_id}")
                return result_id
        except Exception as e:
            logger.error(f"Failed to create result for recording {recording_id}: {e}", exc_info=True)
            raise
    
    def get_by_id(self, result_id):
        """Get result by ID."""
        logger.debug(f"Fetching result by ID: {result_id}")
        query = "SELECT * FROM results WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Result found: {result_id}")
                else:
                    logger.warning(f"Result not found: {result_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch result {result_id}: {e}", exc_info=True)
            raise
    
    def get_by_recording(self, recording_id):
        """Get all results for a recording."""
        logger.debug(f"Fetching all results for recording: {recording_id}")
        query = "SELECT * FROM results WHERE recording_id = %s ORDER BY created_at DESC;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (recording_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} results for recording {recording_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch results for recording {recording_id}: {e}", exc_info=True)
            raise

    def count_results_for_subject(self, subject_id):
        """Count all saved results for a subject across recordings."""
        query = """
        SELECT COUNT(r.id)
        FROM results r
        JOIN recordings rec ON rec.id = r.recording_id
        WHERE rec.subject_id = %s;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (subject_id,))
                row = cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.error(
                f"Failed to count results for subject {subject_id}: {e}",
                exc_info=True,
            )
            raise
