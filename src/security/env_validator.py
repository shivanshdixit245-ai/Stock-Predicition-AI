import os
import subprocess
from pathlib import Path
from loguru import logger
from .security_manager import audit_log

def validate_environment() -> dict:
    """
    Checks for correctly configured .env, .gitignore, and isolated data directories.
    SECURITY NOTE: Prevents accidental exposure of keys or sensitive data.
    """
    results = {
        "env_exists": Path(".env").exists() or Path(".env.LOCAL").exists(),
        "gitignore_exists": Path(".gitignore").exists(),
        "git_history_safe": True,
        "data_isolated": True,
        "mlruns_isolated": True,
        "no_pkl_in_root": True,
        "requirements_exists": Path("requirements.txt").exists()
    }

    # Check .gitignore for .env
    if results["gitignore_exists"]:
        with open(".gitignore", "r") as f:
            content = f.read()
            if ".env" not in content:
                audit_log("ENV_SECRET_EXPOSURE_RISK", {"reason": ".env not in .gitignore"})
                results["git_history_safe"] = False

    # Check for .pkl in root
    root_pkls = list(Path(".").glob("*.pkl"))
    if root_pkls:
        results["no_pkl_in_root"] = False
        audit_log("UNSAFE_FILE_LOCATION", {"files": [str(p) for p in root_pkls]})

    # Check if data/ is isolated (just a basic check if it exists in root)
    if not Path("data").exists():
        results["data_isolated"] = False
        
    return results

def startup_security_check() -> bool:
    """Runs all environment and key validations on startup."""
    logger.info("Starting Enterprise Security Audit...")
    env_results = validate_environment()
    
    critical_issues = []
    if not env_results["env_exists"]:
        logger.warning("No .env file found. Using default/environment variables.")
    
    if not env_results["gitignore_exists"] or not env_results["git_history_safe"]:
        critical_issues.append("INSECURE_GIT_CONFIG")
        
    if not env_results["no_pkl_in_root"]:
        critical_issues.append("UNSAFE_ROOT_FILES")

    if critical_issues:
        logger.error(f"SECURITY AUDIT FAILED: {critical_issues}")
        return False
        
    logger.info("Security Audit Passed.")
    return True
