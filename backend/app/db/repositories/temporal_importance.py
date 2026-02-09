"""
Temporal Importance Repository

Handles database operations for temporal importance analysis results.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime

from ..connection import get_db_connection
from ...core.logging_config import get_app_logger

logger = get_app_logger(__name__)


class TemporalImportanceRepository:
    """Repository for temporal importance data"""

    @staticmethod
    def save(
        result_id: int,
        predicted_class: str,
        confidence_score: float,
        time_points: List[float],
        importance_scores: List[float],
        window_size_ms: int,
        stride_ms: int,
        time_curve_plot: str,
        heatmap_plot: str,
        statistics_plot: str
    ) -> int:
        """
        Save temporal importance analysis to database.
        
        Args:
            result_id: ID of the associated result
            predicted_class: Predicted ADHD class
            confidence_score: Confidence of prediction
            time_points: List of time points (seconds)
            importance_scores: List of importance scores
            window_size_ms: Size of occlusion window in milliseconds
            stride_ms: Stride for sliding window in milliseconds
            time_curve_plot: Base64 encoded time-importance curve
            heatmap_plot: Base64 encoded heatmap visualization
            statistics_plot: Base64 encoded statistics plot
            
        Returns:
            ID of the saved temporal importance record
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO temporal_importance (
                    result_id,
                    predicted_class,
                    confidence_score,
                    time_points,
                    importance_scores,
                    window_size_ms,
                    stride_ms,
                    time_curve_plot,
                    heatmap_plot,
                    statistics_plot,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (
                    result_id,
                    predicted_class,
                    confidence_score,
                    json.dumps(time_points),
                    json.dumps(importance_scores),
                    window_size_ms,
                    stride_ms,
                    time_curve_plot,
                    heatmap_plot,
                    statistics_plot,
                ),
            )

            temporal_importance_id = cursor.fetchone()[0]
            conn.commit()

            logger.info(
                f"Saved temporal importance (ID: {temporal_importance_id}) for result {result_id}"
            )
            return temporal_importance_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving temporal importance: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_by_result_id(result_id: int) -> Optional[Dict]:
        """
        Get temporal importance data by result ID.
        
        Args:
            result_id: ID of the result
            
        Returns:
            Dictionary with temporal importance data or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT
                    id,
                    result_id,
                    predicted_class,
                    confidence_score,
                    time_points,
                    importance_scores,
                    window_size_ms,
                    stride_ms,
                    time_curve_plot,
                    heatmap_plot,
                    statistics_plot,
                    created_at
                FROM temporal_importance
                WHERE result_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (result_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "id": row[0],
                "result_id": row[1],
                "predicted_class": row[2],
                "confidence_score": row[3],
                "time_points": json.loads(row[4]) if isinstance(row[4], str) else row[4],
                "importance_scores": json.loads(row[5]) if isinstance(row[5], str) else row[5],
                "window_size_ms": row[6],
                "stride_ms": row[7],
                "time_curve_plot": row[8],
                "heatmap_plot": row[9],
                "statistics_plot": row[10],
                "created_at": row[11],
            }

        except Exception as e:
            logger.error(
                f"Error getting temporal importance for result {result_id}: {e}",
                exc_info=True,
            )
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_by_id(temporal_importance_id: int) -> Optional[Dict]:
        """
        Get temporal importance data by ID.
        
        Args:
            temporal_importance_id: ID of the temporal importance record
            
        Returns:
            Dictionary with temporal importance data or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT
                    id,
                    result_id,
                    predicted_class,
                    confidence_score,
                    time_points,
                    importance_scores,
                    window_size_ms,
                    stride_ms,
                    time_curve_plot,
                    heatmap_plot,
                    statistics_plot,
                    created_at
                FROM temporal_importance
                WHERE id = %s
                """,
                (temporal_importance_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "id": row[0],
                "result_id": row[1],
                "predicted_class": row[2],
                "confidence_score": row[3],
                "time_points": json.loads(row[4]) if isinstance(row[4], str) else row[4],
                "importance_scores": json.loads(row[5]) if isinstance(row[5], str) else row[5],
                "window_size_ms": row[6],
                "stride_ms": row[7],
                "time_curve_plot": row[8],
                "heatmap_plot": row[9],
                "statistics_plot": row[10],
                "created_at": row[11],
            }

        except Exception as e:
            logger.error(
                f"Error getting temporal importance {temporal_importance_id}: {e}",
                exc_info=True,
            )
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def delete_by_result_id(result_id: int) -> bool:
        """
        Delete temporal importance data by result ID.
        
        Args:
            result_id: ID of the result
            
        Returns:
            True if deleted, False if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM temporal_importance WHERE result_id = %s",
                (result_id,)
            )
            
            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted temporal importance for result {result_id}")
            
            return deleted

        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting temporal importance: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def list_all(limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List all temporal importance records.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of temporal importance records (without plot data)
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT
                    id,
                    result_id,
                    predicted_class,
                    confidence_score,
                    window_size_ms,
                    stride_ms,
                    created_at
                FROM temporal_importance
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )

            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "result_id": row[1],
                    "predicted_class": row[2],
                    "confidence_score": row[3],
                    "window_size_ms": row[4],
                    "stride_ms": row[5],
                    "created_at": row[6],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error listing temporal importance: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
