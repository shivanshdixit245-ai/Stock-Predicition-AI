import pkg_resources
from loguru import logger

def check_dependencies() -> dict:
    """
    Checks installed packages against known vulnerable versions.
    SECURITY NOTE: Mitigates supply chain attacks and known CVEs.
    """
    vulnerabilities = {
        "requests": "2.28.0",
        "pillow": "9.3.0",
        "cryptography": "41.0.0",
        "numpy": "1.22.0",
        "pandas": "1.4.0"
    }
    
    report = []
    for pkg, min_version in vulnerabilities.items():
        try:
            installed = pkg_resources.get_distribution(pkg).version
            if pkg_resources.parse_version(installed) < pkg_resources.parse_version(min_version):
                logger.warning(f"VULNERABLE PACKAGE DETECTED: {pkg} {installed} < {min_version}")
                report.append({
                    "package": pkg,
                    "installed": installed,
                    "required": min_version,
                    "is_vulnerable": True
                })
        except pkg_resources.DistributionNotFound:
            pass
            
    return {"vulnerable_packages": report, "checked_at": str(pkg_resources.time.time())}
