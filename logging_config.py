"""
Production-ready logging configuration for AI Agent Verification System
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_production_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Configure production-grade logging with rotation and structured output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    # Get log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Application Log File Handler (rotating)
    app_log_file = log_path / "application.log"
    app_file_handler = logging.handlers.RotatingFileHandler(
        filename=app_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    app_file_handler.setLevel(numeric_level)
    app_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(app_file_handler)
    
    # Error Log File Handler (errors only)
    error_log_file = log_path / "errors.log"
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Verification Log File Handler (track all verifications)
    verification_log_file = log_path / "verifications.log"
    verification_file_handler = logging.handlers.RotatingFileHandler(
        filename=verification_log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=20,
        encoding='utf-8'
    )
    verification_file_handler.setLevel(logging.INFO)
    verification_file_handler.setFormatter(detailed_formatter)
    
    # Create verification logger
    verification_logger = logging.getLogger('verification')
    verification_logger.addHandler(verification_file_handler)
    verification_logger.setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('multipart').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('insightface').setLevel(logging.WARNING)
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("AI Agent Verification System - Production Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    logger.info(f"Application Log: {app_log_file}")
    logger.info(f"Error Log: {error_log_file}")
    logger.info(f"Verification Log: {verification_log_file}")
    logger.info("=" * 80)
    
    return root_logger


class VerificationLogger:
    """Specialized logger for tracking verification requests"""
    
    def __init__(self):
        self.logger = logging.getLogger('verification')
    
    def log_request(self, user_id: str, has_dob: bool, has_gender: bool):
        """Log incoming verification request"""
        self.logger.info(
            f"REQUEST | user_id={user_id} | dob={'provided' if has_dob else 'missing'} | "
            f"gender={'provided' if has_gender else 'missing'}"
        )
    
    def log_result(self, user_id: str, decision: str, score: float, 
                   face_score: float, processing_time: float):
        """Log verification result"""
        self.logger.info(
            f"RESULT | user_id={user_id} | decision={decision} | score={score:.2f} | "
            f"face_score={face_score:.2f} | time={processing_time:.2f}s"
        )
    
    def log_error(self, user_id: str, error_type: str, error_msg: str):
        """Log verification error"""
        self.logger.error(
            f"ERROR | user_id={user_id} | error_type={error_type} | message={error_msg}"
        )
    
    def log_cache_hit(self, user_id: str):
        """Log cache hit"""
        self.logger.info(f"CACHE_HIT | user_id={user_id}")
    
    def log_cache_miss(self, user_id: str):
        """Log cache miss"""
        self.logger.debug(f"CACHE_MISS | user_id={user_id}")


# Singleton instance
_verification_logger = None

def get_verification_logger() -> VerificationLogger:
    """Get singleton verification logger instance"""
    global _verification_logger
    if _verification_logger is None:
        _verification_logger = VerificationLogger()
    return _verification_logger
