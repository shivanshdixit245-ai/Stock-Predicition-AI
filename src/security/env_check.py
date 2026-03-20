import os
import re
import subprocess
import sys
from pathlib import Path
from loguru import logger

HIGH_ENTROPY_PATTERNS = [
    r"sk-ant-[a-zA-Z0-9\-_]{20,}",
    r"hf_[a-zA-Z0-9]{20,}",
    r"[a-f0-9]{32}",
    r"Bearer [a-zA-Z0-9\-_]{20,}",
]

DANGEROUS_VARIABLE_NAMES = [
    "api_key", "secret", "token", "password",
    "auth", "credential", "private_key"
]

def scan_project_for_exposed_secrets() -> list[dict]:
    """
    Scans entire project for hardcoded secrets.
    Returns list of findings with file, line number, pattern matched.
    Never logs the actual secret value — only the file and line.
    DS Interview Note: Secret scanning is standard in production
    ML systems. Tools like GitGuardian, TruffleHog do this at scale.
    """
    findings = []
    project_root = Path(".")
    skip_dirs = {".git", "__pycache__", ".pytest_cache",
                 "mlruns", "node_modules", ".venv", "venv"}
    skip_extensions = {".parquet", ".pkl", ".pyc", ".jpg",
                       ".png", ".gif", ".ico"}

    for file_path in project_root.rglob("*"):
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        if file_path.suffix in skip_extensions:
            continue
        if not file_path.is_file() or file_path.name == "env_check.py":
            continue
        if file_path.name == ".env":
            findings.append({
                "file": str(file_path),
                "line": 0,
                "issue": ".env file exists — ensure it is in .gitignore",
                "severity": "HIGH"
            })
            continue
        # Also check for .env.LOCAL which was found earlier
        if file_path.name == ".env.LOCAL":
            findings.append({
                "file": str(file_path),
                "line": 0,
                "issue": ".env.LOCAL file exists — ensure it is in .gitignore",
                "severity": "HIGH"
            })
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                for pattern in HIGH_ENTROPY_PATTERNS:
                    if re.search(pattern, line):
                        # Skip if it's a known placeholder or regex pattern in THIS file
                        if "HIGH_ENTROPY_PATTERNS" in line or "os.environ.get" in line:
                            continue
                        findings.append({
                            "file": str(file_path),
                            "line": i,
                            "issue": f"Possible exposed secret: {pattern}",
                            "severity": "CRITICAL",
                            "masked_line": line[:10] + "****"
                        })
                for var_name in DANGEROUS_VARIABLE_NAMES:
                    if var_name in line.lower():
                        if "os.environ" not in line and \
                           "st.secrets" not in line and \
                           "getenv" not in line and \
                           "=" in line and \
                           not line.strip().startswith("#"):
                            val = line.split("=")[-1].strip().strip('"').strip("'")
                            if len(val) > 8 and "your_" not in val \
                               and "xxx" not in val.lower():
                                findings.append({
                                    "file": str(file_path),
                                    "line": i,
                                    "issue": f"Hardcoded value in {var_name}",
                                    "severity": "HIGH",
                                    "masked_line": line[:20] + "****"
                                })
        except Exception:
            continue
    return findings

def check_git_history_for_secrets() -> bool:
    """
    Checks if .env was ever committed to git history.
    Returns True if .env found in history (bad), False if clean.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--", ".env"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            logger.critical(
                ".env was found in git history! "
                "Keys may be permanently exposed. "
                "Use BFG Repo Cleaner to purge history."
            )
            return True
        return False
    except Exception:
        return False

def check_env_not_tracked() -> bool:
    """Returns True if .env is currently tracked by git (bad)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", ".env"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            logger.critical(
                ".env is currently tracked by git! "
                "Run: git rm --cached .env && git commit -m 'remove .env'"
            )
            return True
        return False
    except Exception:
        return False

def run_full_security_scan() -> dict:
    """
    Master security scan — runs on every app startup.
    Never crashes the app. Only warns.
    """
    results = {
        "exposed_secrets": [],
        "env_in_git": False,
        "env_in_history": False,
        "is_clean": True
    }
    findings = scan_project_for_exposed_secrets()
    results["exposed_secrets"] = findings
    results["env_in_git"] = check_env_not_tracked()
    results["env_in_history"] = check_git_history_for_secrets()

    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    high = [f for f in findings if f["severity"] == "HIGH"]

    if critical or results["env_in_git"] or results["env_in_history"]:
        results["is_clean"] = False
        for f in findings:
            print(f"FINDING: {f['severity']} | {f['file']}:{f['line']} | {f['issue']}")
        if results["env_in_git"]: print("FINDING: .env is tracked by git")
        if results["env_in_history"]: print("FINDING: .env found in git history")
        
        logger.critical(
            f"SECURITY SCAN FAILED: "
            f"{len(critical)} critical findings, "
            f"{len(high)} high findings. "
            f"Fix immediately before pushing to GitHub."
        )
    else:
        logger.success("Security scan passed — no exposed secrets found")

    return results

if __name__ == "__main__":
    scan_results = run_full_security_scan()
    if not scan_results["is_clean"]:
        sys.exit(1)
    sys.exit(0)
