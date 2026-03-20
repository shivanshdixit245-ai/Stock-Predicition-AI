import os
import re
import json
import time
import hashlib
import functools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from config import settings

# ============================================================================
# SECTION 1 — API KEY SECURITY
# ============================================================================

class SecretsManager:
    """
    Singleton manager for API key security, validation, and rotation detection.
    SECURITY NOTE: Prevents hardcoding or insecure storage of sensitive keys.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecretsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._key_hashes = self._load_key_hashes()
        self._initialized = True

    def _load_key_hashes(self) -> dict:
        hash_file = Path("data/security/key_metadata.json")
        if hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_key_hashes(self):
        hash_file = Path("data/security/key_metadata.json")
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w") as f:
            json.dump(self._key_hashes, f)

    def validate_key(self, name: str, value: str) -> bool:
        """Validates key format before usage."""
        if not value or len(value.strip()) == 0:
            return False
            
        # Common placeholder patterns
        if any(x in value.lower() for x in ["your_", "xxx", "test", "demo"]):
            return False

        if name == "NEWS_API_KEY":
            # NewsAPI: 32 alphanumeric
            return bool(re.match(r"^[a-z0-9]{32}$", value.lower()))
        
        if name == "ANTHROPIC_API_KEY":
            # Anthropic: starts with sk-ant-
            return value.startswith("sk-ant-")
            
        if name == "GEMINI_API_KEY":
            # Gemini: rough check
            return len(value) > 30

        return True

    def get_secret(self, key_name: str) -> str:
        """Reads from environment ONLY and validates."""
        val = os.getenv(key_name, "")
        if not self.validate_key(key_name, val):
            logger.warning(f"Key {key_name} is missing, invalid, or placeholder. Masked: {self.mask_secret(val)}")
            return ""
            
        # Check for rotation (90 days)
        current_hash = hashlib.sha256(val.encode()).hexdigest()
        last_meta = self._key_hashes.get(key_name, {})
        
        if last_meta.get("hash") != current_hash:
            self._key_hashes[key_name] = {
                "hash": current_hash,
                "updated_at": datetime.now().isoformat()
            }
            self._save_key_hashes()
        else:
            updated_at = datetime.fromisoformat(last_meta.get("updated_at"))
            if datetime.now() - updated_at > timedelta(days=90):
                logger.warning(f"SECURITY ADVISORY: Key {key_name} has not been rotated in >90 days.")
                
        return val

    def mask_secret(self, value: str) -> str:
        """Masks first 4 + **** + last 4."""
        if not value: return "[MISSING]"
        if len(value) <= 8: return "****"
        return f"{value[:4]}****{value[-4:]}"

    def is_feature_available(self, feature: str) -> bool:
        """Gracefully disables features if key missing."""
        if feature == "news":
            return bool(self.get_secret("NEWS_API_KEY"))
        if feature == "ai_assistant":
            return bool(self.get_secret("ANTHROPIC_API_KEY") or self.get_secret("GEMINI_API_KEY"))
        return True

_secrets_manager = SecretsManager()

def get_secret(key_name: str) -> str:
    return _secrets_manager.get_secret(key_name)

def mask_secret(value: str) -> str:
    return _secrets_manager.mask_secret(value)


# ============================================================================
# SECTION 2 — RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Token bucket algorithm for sophisticated API rate limiting.
    SECURITY NOTE: Prevents API abuse and potential DDoS/Quota exhaustion.
    """
    def __init__(self, name: str, rate: float, capacity: int):
        self.name = name
        self.rate = rate  # Tokens per second
        self.capacity = capacity
        self.bucket_file = Path(f"data/security/rate_{name}.json")
        self._tokens, self._last_update = self._load_state()

    def _load_state(self) -> Tuple[float, float]:
        if self.bucket_file.exists():
            try:
                with open(self.bucket_file, "r") as f:
                    data = json.load(f)
                    return data["tokens"], data["last_update"]
            except: pass
        return self.capacity, time.time()

    def _save_state(self):
        self.bucket_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bucket_file, "w") as f:
            json.dump({"tokens": self._tokens, "last_update": self._last_update}, f)

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        # Add new tokens
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now
        
        if self._tokens >= tokens:
            self._tokens -= tokens
            self._save_state()
            return True
        
        logger.warning(f"RATE LIMIT HIT: {self.name}. Retry in {self.get_wait_time():.1f}s")
        return False

    def get_wait_time(self) -> float:
        now = time.time()
        elapsed = now - self._last_update
        tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        if tokens >= 1: return 0
        return (1 - tokens) / self.rate

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "remaining": int(self._tokens),
            "reset_in": self.get_wait_time()
        }

# Global Limiters
news_limiter = RateLimiter("newsapi", 100/(24*3600), 100) # 100/day
yf_limiter = RateLimiter("yfinance", 2000/3600, 2000)      # 2000/hour
ai_limiter = RateLimiter("anthropic", 50/60, 50)          # 50/min
ui_limiter = RateLimiter("streamlit", 100/60, 100)        # 100 reruns/min

class CircuitBreaker:
    """
    Prevents cascade failures when external APIs are unresponsive.
    SECURITY NOTE: Implements regional stability and graceful degradation.
    """
    def __init__(self, name: str, threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED" # CLOSED, OPEN, HALF_OPEN
        self.failures = 0
        self.last_failure_time = 0

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"CIRCUIT BREAKER: {self.name} entered HALF_OPEN state.")
                else:
                    logger.error(f"CIRCUIT BREAKER: {self.name} is OPEN. Blocking request.")
                    return {"error": "CIRCUIT_BREAKER_OPEN", "api": self.name}

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    logger.info(f"CIRCUIT BREAKER: {self.name} RECOVERED.")
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                logger.warning(f"FAILURE DETECTED: {self.name} ({self.failures}/{self.threshold})")
                
                if self.failures >= self.threshold:
                    self.state = "OPEN"
                    logger.error(f"CIRCUIT BREAKER: {self.name} OPENED due to repeated failures.")
                raise e
        return wrapper

# Decorator factory
def circuit_breaker(api_name: str):
    return CircuitBreaker(api_name)


# ============================================================================
# SECTION 3 — INPUT VALIDATION & SANITIZATION
# ============================================================================

class InputValidator:
    """
    Strict validation and sanitization for all user-provided inputs.
    SECURITY NOTE: Prevents XSS, SQLi, and Prompt Injection.
    """
    @staticmethod
    def validate_ticker(ticker: str) -> str:
        if not ticker: raise ValueError("Ticker cannot be empty.")
        t = ticker.upper().strip()
        
        # SQL keywords - check FIRST
        if any(x in t for x in ["SELECT", "DROP", "INSERT", "UPDATE"]):
            audit_log("SUSPICIOUS_INPUT", {"input": t, "type": "SQLI_KEYWORDS"})
            raise ValueError("Invalid characters detected.")
            
        if len(t) > 6: raise ValueError("Ticker too long (max 6 chars).")
        # Regex: allowed A-Z and optional dash
        if not re.match(r"^[A-Z]{1,5}(-[A-Z])?$", t):
            raise ValueError("Invalid ticker format.")
            
        return t

    @staticmethod
    def validate_date_range(start: str, end: str) -> Tuple[str, str]:
        try:
            s_dt = datetime.fromisoformat(start)
            e_dt = datetime.fromisoformat(end)
        except: raise ValueError("Dates must be in ISO format (YYYY-MM-DD).")
        
        if s_dt > e_dt: raise ValueError("Start date must be before end date.")
        if s_dt < datetime(2000, 1, 1): raise ValueError("Data before 2000 is not supported.")
        if e_dt > datetime.now(): raise ValueError("End date cannot be in the future.")
        if (e_dt - s_dt).days > 365 * 10: raise ValueError("Max date range is 10 years.")
        return start, end

    @staticmethod
    def validate_threshold(value: float, name: str) -> float:
        if not isinstance(value, (int, float)): raise ValueError(f"{name} must be numeric.")
        if np.isnan(value) or np.isinf(value): raise ValueError(f"{name} contains invalid math value.")
        
        if name == "buy_threshold":
            if not (0.51 <= value <= 0.99): raise ValueError("Buy threshold must be 0.51-0.99.")
        if name == "sell_threshold":
            if not (0.01 <= value <= 0.49): raise ValueError("Sell threshold must be 0.01-0.49.")
        return float(value)

    @staticmethod
    def sanitize_chat_input(text: str) -> str:
        if not text: return ""
        # 1. Truncate
        sanitized = text[:500]
        # 2. Strip HTML
        sanitized = re.sub(r'<[^>]*?>', '', sanitized)
        # 3. Unicode normalize
        import unicodedata
        sanitized = unicodedata.normalize('NFKC', sanitized)
        # 4. Remove null bytes
        sanitized = sanitized.replace('\0', '')
        
        # 5. Prompt Injection Defense
        injections = [
            "ignore previous", "ignore all", "system prompt", "you are now",
            "act as", "forget everything", "jailbreak", "DAN", "developer mode",
            "disable restrictions", "pretend you are", "new persona", "bypass",
            "override instructions", "reveal your prompt", "print your instructions"
        ]
        if any(x in sanitized.lower() for x in injections):
            audit_log("INJECTION_ATTEMPT", {"raw_input": sanitized})
            return "Please ask a question about the stock analysis."
            
        return sanitized.strip()


# ============================================================================
# SECTION 4 — DATA SECURITY
# ============================================================================

class DataSecurityManager:
    """
    Handles safe file loading and data cleaning.
    SECURITY NOTE: Prevents path traversal and unsafe serialization (pickle).
    """
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        bad_cols = [c for c in clean_df.columns if any(x in str(c).lower() for x in ["password", "key", "secret", "token"])]
        if bad_cols:
            logger.warning(f"DROPPING SENSITIVE COLUMNS: {bad_cols}")
            clean_df.drop(columns=bad_cols, inplace=True)
        
        # Numeric cleanup
        clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return clean_df

    @staticmethod
    def safe_parquet_load(path: Union[str, Path]) -> Optional[pd.DataFrame]:
        p = Path(path)
        if not p.exists() or p.suffix != ".parquet": return None
        if p.stat().st_size > 500 * 1024 * 1024:
            logger.error(f"FILE TOO LARGE (500MB+): {p}")
            return None
        if ".." in str(p):
            audit_log("INVALID_FILE_ACCESS", {"path": str(p), "reason": "PATH_TRAVERSAL"})
            return None
        try:
            return pd.read_parquet(p)
        except Exception as e:
            logger.error(f"Parquet load failed for {p}: {e}")
            return None

    @staticmethod
    def safe_pickle_load(path: Union[str, Path]) -> Any:
        p = Path(path)
        # SECURITY RULE: Only allow pickle from models dir
        if "data" not in str(p.parent) or "models" not in str(p.parent):
            audit_log("INVALID_FILE_ACCESS", {"path": str(p), "reason": "PICKLE_OUTSIDE_MODELS"})
            logger.error(f"UNSAFE PICKLE LOAD REJECTED: {p}")
            return None
            
        if not p.exists() or p.suffix != ".pkl": return None
        if p.stat().st_size > 200 * 1024 * 1024: return None
        
        try:
            import joblib
            return joblib.load(p)
        except Exception as e:
            logger.error(f"Pickle load failed for {p}: {e}")
            return None

    @staticmethod
    def safe_file_write(path: Union[str, Path], data: Any) -> bool:
        p = Path(path)
        allowed_dirs = ["data", "reports", "mlruns", "logs"]
        if not any(x in str(p.absolute()) for x in allowed_dirs):
            logger.error(f"NON-ALLOWED DIRECTORY WRITE ATTEMPT: {p}")
            return False
        
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return True
        except: return False


# ============================================================================
# SECTION 5 — LOGGING & AUDIT
# ============================================================================

class SecureLogger:
    """Redacts secrets from logs and maintains security audit trail."""
    def __init__(self):
        self.audit_file = "logs/security_audit.log"
        Path("logs").mkdir(exist_ok=True)
        
        # Configure loguru for rotation
        logger.add("logs/app.log", rotation="10MB", retention=5, level="INFO")
        logger.add(self.audit_file, rotation="5MB", level="WARNING", filter=lambda r: "AUDIT" in r["message"])

    def redact(self, message: str) -> str:
        # Patterms for secrets
        patterns = [
            r"sk-ant-[a-zA-Z0-9-]{10,}",
            r"bearer [a-zA-Z0-9\._-]{10,}",
            r"api-key[:=]\s*[a-zA-Z0-9]{10,}"
        ]
        redacted = message
        for p in patterns:
            redacted = re.sub(p, "[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted

    def log_event(self, level: str, msg: str, audit: bool = False):
        redacted_msg = self.redact(msg)
        if audit:
            full_msg = f"[SECURITY_AUDIT] {redacted_msg}"
            logger.log("WARNING", full_msg) # Log as warning to ensure it hits audit file
        else:
            logger.log(level, redacted_msg)

_secure_logger = SecureLogger()

def audit_log(event_type: str, details: dict) -> None:
    _secure_logger.log_event("WARNING", f"EVENT: {event_type} | DETAILS: {json.dumps(details)}", audit=True)
