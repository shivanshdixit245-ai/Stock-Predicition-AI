import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes import router
from src.security.security_manager import audit_log, secure_logger

# --- Security: Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Market Signal AI — Secure API",
    description="Enterprise-grade stock prediction backend",
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Security: Middleware ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.herokuapp.com"])

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # Block suspiciously large bodies (> 1MB)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1 * 1024 * 1024:
        audit_log("SUSPICIOUS_REQUEST", {"reason": "BODY_TOO_LARGE", "ip": request.client.host})
        return JSONResponse(status_code=413, content={"error": "Request body too large."})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Log request (masked IP)
    ip_parts = request.client.host.split(".")
    masked_ip = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx" if len(ip_parts) == 4 else "xxx.xxx.xxx.xxx"
    logger.info(f"API Request: {request.method} {request.url.path} | IP: {masked_ip} | Status: {response.status_code}")
    
    return response

# CORS Middleware (Restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted to dashboard domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err_id = str(uuid.uuid4())
    logger.error(f"UNHANDLED API ERROR [{err_id}]: {exc}")
    audit_log("API_SERVER_ERROR", {"request_id": err_id, "path": request.url.path})
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred. Our team has been notified.",
            "request_id": err_id
        }
    )

@app.on_event("startup")
async def startup_event():
    from src.security.env_validator import startup_security_check
    startup_security_check()
    logger.info("Market Signal Secure API initialized.")
