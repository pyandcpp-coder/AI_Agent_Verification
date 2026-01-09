"""
Production Monitoring and Metrics for AI Agent Verification System
"""

import time
import psutil
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class VerificationMetrics:
    """Store verification metrics"""
    total_requests: int = 0
    approved: int = 0
    rejected: int = 0
    in_review: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Response times (in seconds)
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Component performance
    face_agent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    entity_agent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    gender_pipeline_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_request(self, decision: str, response_time: float):
        """Record a verification request"""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if decision == "APPROVED":
            self.approved += 1
        elif decision == "REJECTED":
            self.rejected += 1
        elif decision in ["REVIEW", "IN_REVIEW"]:
            self.in_review += 1
    
    def record_error(self, error_type: str):
        """Record an error"""
        self.errors += 1
        self.error_types[error_type] += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time"""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100
    
    def get_approval_rate(self) -> float:
        """Get approval rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.approved / self.total_requests) * 100
    
    def get_error_rate(self) -> float:
        """Get error rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.errors / self.total_requests) * 100


class SystemMonitor:
    """Monitor system resources and application metrics"""
    
    def __init__(self):
        self.metrics = VerificationMetrics()
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            stats = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": disk.percent,
                    "free_gb": round(disk.free / (1024**3), 2)
                }
            }
            
            # GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    stats["gpu"] = {
                        "available": True,
                        "name": torch.cuda.get_device_name(0),
                        "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                        "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                    }
                else:
                    stats["gpu"] = {"available": False}
            except:
                stats["gpu"] = {"available": False}
            
            return stats
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application metrics"""
        with self._lock:
            uptime = datetime.now() - self.start_time
            
            return {
                "uptime": {
                    "seconds": int(uptime.total_seconds()),
                    "formatted": str(uptime).split('.')[0]
                },
                "requests": {
                    "total": self.metrics.total_requests,
                    "approved": self.metrics.approved,
                    "rejected": self.metrics.rejected,
                    "in_review": self.metrics.in_review,
                    "errors": self.metrics.errors
                },
                "rates": {
                    "approval_rate": round(self.metrics.get_approval_rate(), 2),
                    "error_rate": round(self.metrics.get_error_rate(), 2),
                    "cache_hit_rate": round(self.metrics.get_cache_hit_rate(), 2)
                },
                "performance": {
                    "avg_response_time": round(self.metrics.get_avg_response_time(), 2),
                    "p95_response_time": round(self.metrics.get_p95_response_time(), 2),
                    "requests_per_minute": self._calculate_rpm()
                },
                "cache": {
                    "hits": self.metrics.cache_hits,
                    "misses": self.metrics.cache_misses
                },
                "errors": dict(self.metrics.error_types)
            }
    
    def _calculate_rpm(self) -> float:
        """Calculate requests per minute"""
        uptime = datetime.now() - self.start_time
        if uptime.total_seconds() < 60:
            return 0.0
        return (self.metrics.total_requests / uptime.total_seconds()) * 60
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        system_stats = self.get_system_stats()
        app_metrics = self.get_application_metrics()
        
        # Determine health
        is_healthy = True
        warnings = []
        
        # Check CPU
        cpu_percent = system_stats.get("cpu", {}).get("percent", 0)
        if cpu_percent > 90:
            is_healthy = False
            warnings.append(f"High CPU usage: {cpu_percent}%")
        elif cpu_percent > 75:
            warnings.append(f"Elevated CPU usage: {cpu_percent}%")
        
        # Check Memory
        mem_percent = system_stats.get("memory", {}).get("percent", 0)
        if mem_percent > 90:
            is_healthy = False
            warnings.append(f"High memory usage: {mem_percent}%")
        elif mem_percent > 75:
            warnings.append(f"Elevated memory usage: {mem_percent}%")
        
        # Check Error Rate
        error_rate = app_metrics.get("rates", {}).get("error_rate", 0)
        if error_rate > 10:
            is_healthy = False
            warnings.append(f"High error rate: {error_rate}%")
        elif error_rate > 5:
            warnings.append(f"Elevated error rate: {error_rate}%")
        
        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }
    
    def record_verification(self, decision: str, response_time: float):
        """Record a verification"""
        with self._lock:
            self.metrics.record_request(decision, response_time)
    
    def record_error(self, error_type: str):
        """Record an error"""
        with self._lock:
            self.metrics.record_error(error_type)
    
    def record_cache_hit(self):
        """Record cache hit"""
        with self._lock:
            self.metrics.record_cache_hit()
    
    def record_cache_miss(self):
        """Record cache miss"""
        with self._lock:
            self.metrics.record_cache_miss()


# Singleton instance
_monitor = None

def get_monitor() -> SystemMonitor:
    """Get singleton monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor
