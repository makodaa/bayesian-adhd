from ..db.repositories.clinicians import CliniciansRepository
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)


class ClinicianAuthService:
    """Service for clinician authentication and login management."""
    
    def __init__(self, clinicians_repo: CliniciansRepository):
        """Initialize the auth service with clinicians repository."""
        self.clinicians_repo = clinicians_repo
    
    def authenticate(self, username, password):
        """
        Authenticate a clinician with username and password.
        
        Args:
            username: Clinician's username (first_name + space + last_name)
            password: Clinician's password
        
        Returns:
            dict: Clinician data if authentication successful, None otherwise
        """
        logger.info(f"Attempting to authenticate clinician: {username}")
        try:
            clinician = self.clinicians_repo.get_by_username(username)
            
            if not clinician:
                logger.warning(f"Clinician not found: {username}")
                return None
            
            if not clinician.get('password_hash'):
                logger.warning(f"Clinician has no password set: {username}")
                return None
            
            # Verify password
            if self.clinicians_repo.verify_password(clinician['id'], password):
                logger.info(f"Clinician authenticated successfully: {username}")
                return clinician
            else:
                logger.warning(f"Invalid password for clinician: {username}")
                return None
        except Exception as e:
            logger.error(f"Error during authentication: {e}", exc_info=True)
            return None
    
    def get_clinician_display_name(self, clinician):
        """Get display name for a clinician."""
        if clinician:
            return f"{clinician.get('first_name', '')} {clinician.get('last_name', '')}".strip()
        return None
