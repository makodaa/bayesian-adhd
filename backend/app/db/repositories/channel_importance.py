"""Repository for managing channel importance data."""

from typing import Optional, Dict, Any
from psycopg2.extras import Json
from ..connection import get_db_connection
from ...core.logging_config import get_app_logger

logger = get_app_logger(__name__)


class ChannelImportanceRepository:
    """Repository for channel_importance table operations."""

    @staticmethod
    def save(result_id: int, channel_importance_data: Dict[str, Any]) -> int:
        """
        Save channel importance analysis for a result.
        
        Args:
            result_id: The ID of the result this analysis belongs to
            channel_importance_data: Dictionary containing:
                - topographic_map: Base64 encoded image
                - bar_chart: Base64 encoded image
                - regional_chart: Base64 encoded image
                - importance_scores: Dict of channel: raw score
                - normalized_importance: Dict of channel: normalized score
                - predicted_class: Classification result
                - confidence_score: Model confidence
        
        Returns:
            int: ID of the created channel_importance record
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO channel_importance 
                (result_id, topographic_map, bar_chart, regional_chart, 
                 importance_scores, normalized_scores, predicted_class, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (result_id) 
                DO UPDATE SET
                    topographic_map = EXCLUDED.topographic_map,
                    bar_chart = EXCLUDED.bar_chart,
                    regional_chart = EXCLUDED.regional_chart,
                    importance_scores = EXCLUDED.importance_scores,
                    normalized_scores = EXCLUDED.normalized_scores,
                    predicted_class = EXCLUDED.predicted_class,
                    confidence_score = EXCLUDED.confidence_score,
                    created_at = NOW()
                RETURNING id
                """,
                (
                    result_id,
                    channel_importance_data['topographic_map'],
                    channel_importance_data['bar_chart'],
                    channel_importance_data['regional_chart'],
                    Json(channel_importance_data['importance_scores']),
                    Json(channel_importance_data['normalized_importance']),
                    channel_importance_data['predicted_class'],
                    channel_importance_data['confidence_score']
                )
            )
            
            channel_importance_id = cursor.fetchone()[0]
            conn.commit()
            
            logger.info(f"Saved channel importance for result {result_id}, ID: {channel_importance_id}")
            return channel_importance_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving channel importance: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_by_result_id(result_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve channel importance analysis for a result.
        
        Args:
            result_id: The ID of the result
        
        Returns:
            Dict containing channel importance data or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, result_id, topographic_map, bar_chart, regional_chart,
                       importance_scores, normalized_scores, predicted_class, 
                       confidence_score, created_at
                FROM channel_importance
                WHERE result_id = %s
                """,
                (result_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row[0],
                'result_id': row[1],
                'topographic_map': row[2],
                'bar_chart': row[3],
                'regional_chart': row[4],
                'importance_scores': row[5],
                'normalized_importance': row[6],
                'predicted_class': row[7],
                'confidence_score': row[8],
                'created_at': row[9]
            }
            
        except Exception as e:
            logger.error(f"Error retrieving channel importance for result {result_id}: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def delete_by_result_id(result_id: int) -> bool:
        """
        Delete channel importance analysis for a result.
        
        Args:
            result_id: The ID of the result
        
        Returns:
            bool: True if deleted, False if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "DELETE FROM channel_importance WHERE result_id = %s RETURNING id",
                (result_id,)
            )
            
            deleted = cursor.fetchone() is not None
            conn.commit()
            
            if deleted:
                logger.info(f"Deleted channel importance for result {result_id}")
            
            return deleted
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting channel importance for result {result_id}: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
