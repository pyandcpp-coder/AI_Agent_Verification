"""
Redis Cache Manager for KYC Verification Results

This module manages a Redis cache to store verification results and prevent
reprocessing of users that have already been verified.

Features:
- Store verification results with TTL (time-to-live)
- Check if user has already been verified
- Retrieve cached verification results
- Clean up old entries automatically
"""

import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisVerificationCache:
    """Manages Redis cache for KYC verification results."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_hours: int = 24,
        key_prefix: str = 'kyc_verification'
    ):
        """
        Initialize Redis cache manager.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            ttl_hours: Time-to-live for cache entries in hours (default: 24 hours)
            key_prefix: Prefix for all Redis keys
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl_seconds = ttl_hours * 3600
        self.key_prefix = key_prefix
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,  # Automatically decode responses to strings
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
            self.enabled = True
        except redis.ConnectionError as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.warning("⚠️  Cache will be disabled. System will continue without caching.")
            self.redis_client = None
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ Redis initialization error: {e}")
            self.redis_client = None
            self.enabled = False
    
    def _make_key(self, user_id: str) -> str:
        """Generate Redis key for a user ID."""
        return f"{self.key_prefix}:{user_id}"
    
    def is_user_verified(self, user_id: str) -> bool:
        """
        Check if a user has already been verified.
        
        Args:
            user_id: User ID to check
            
        Returns:
            True if user is in cache, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(user_id)
            exists = self.redis_client.exists(key)
            
            if exists:
                logger.info(f"🔍 Cache HIT: User {user_id} already verified")
                return True
            else:
                logger.debug(f"🔍 Cache MISS: User {user_id} not in cache")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking cache for user {user_id}: {e}")
            return False
    
    def get_verification_result(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached verification result for a user.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            Cached verification result dict or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            key = self._make_key(user_id)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                result = json.loads(cached_data)
                logger.info(f"📦 Retrieved cached result for user {user_id}")
                return result
            else:
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decoding cached data for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving cache for user {user_id}: {e}")
            return None
    
    def store_verification_result(
        self,
        user_id: str,
        verification_result: Dict[str, Any],
        api_response: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store verification result in cache.
        
        Args:
            user_id: User ID
            verification_result: Verification result from AI processing
            api_response: Response from the POST API (optional)
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(user_id)
            
            # Prepare cache entry
            cache_entry = {
                'user_id': user_id,
                'verification_result': verification_result,
                'api_response': api_response,
                'cached_at': datetime.now().isoformat(),
                'ttl_hours': self.ttl_seconds / 3600
            }
            
            # Store in Redis with TTL
            self.redis_client.setex(
                key,
                self.ttl_seconds,
                json.dumps(cache_entry)
            )
            
            logger.info(f"💾 Stored verification result for user {user_id} (TTL: {self.ttl_seconds/3600}h)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing cache for user {user_id}: {e}")
            return False
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from cache.
        
        Args:
            user_id: User ID to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(user_id)
            result = self.redis_client.delete(key)
            
            if result:
                logger.info(f"🗑️  Removed user {user_id} from cache")
                return True
            else:
                logger.warning(f"⚠️  User {user_id} not found in cache")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error removing user {user_id} from cache: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all verification cache entries.
        
        WARNING: This will delete all cached verification results!
        
        Returns:
            True if cleared successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🗑️  Cleared {deleted} cache entries")
                return True
            else:
                logger.info("ℹ️  Cache is already empty")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {
                'enabled': False,
                'status': 'disabled',
                'message': 'Redis cache is not available'
            }
        
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.redis_client.keys(pattern)
            total_entries = len(keys)
            
            # Get memory usage
            info = self.redis_client.info('memory')
            used_memory_human = info.get('used_memory_human', 'N/A')
            
            # Get some sample entries
            sample_keys = keys[:5] if len(keys) >= 5 else keys
            samples = []
            for key in sample_keys:
                ttl = self.redis_client.ttl(key)
                samples.append({
                    'key': key,
                    'ttl_seconds': ttl,
                    'ttl_hours': round(ttl / 3600, 2) if ttl > 0 else 0
                })
            
            return {
                'enabled': True,
                'status': 'connected',
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'total_entries': total_entries,
                'memory_used': used_memory_human,
                'key_prefix': self.key_prefix,
                'default_ttl_hours': self.ttl_seconds / 3600,
                'sample_entries': samples
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {
                'enabled': False,
                'status': 'error',
                'error': str(e)
            }
    
    def bulk_check_users(self, user_ids: list) -> Dict[str, bool]:
        """
        Check multiple users at once (more efficient than individual checks).
        
        Args:
            user_ids: List of user IDs to check
            
        Returns:
            Dictionary mapping user_id -> bool (True if verified, False if not)
        """
        if not self.enabled:
            return {str(uid): False for uid in user_ids}
        
        try:
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            keys = [self._make_key(str(uid)) for uid in user_ids]
            
            for key in keys:
                pipe.exists(key)
            
            results = pipe.execute()
            
            # Map results back to user IDs
            verification_status = {}
            for user_id, exists in zip(user_ids, results):
                verification_status[str(user_id)] = bool(exists)
            
            verified_count = sum(verification_status.values())
            logger.info(f"🔍 Bulk check: {verified_count}/{len(user_ids)} users already verified")
            
            return verification_status
            
        except Exception as e:
            logger.error(f"❌ Error in bulk user check: {e}")
            return {str(uid): False for uid in user_ids}
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("🔌 Redis connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing Redis connection: {e}")


def load_cache_config_from_env() -> Dict[str, Any]:
    """
    Load Redis cache configuration from environment variables.
    
    Environment Variables:
        REDIS_ENABLED: Enable/disable Redis cache (default: true)
        REDIS_HOST: Redis server host (default: localhost)
        REDIS_PORT: Redis server port (default: 6379)
        REDIS_DB: Redis database number (default: 0)
        REDIS_PASSWORD: Redis password (optional)
        REDIS_TTL_HOURS: Cache TTL in hours (default: 24)
        REDIS_KEY_PREFIX: Key prefix (default: kyc_verification)
    
    Returns:
        Configuration dictionary
    """
    return {
        'enabled': os.getenv('REDIS_ENABLED', 'true').lower() == 'true',
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', '6379')),
        'db': int(os.getenv('REDIS_DB', '0')),
        'password': os.getenv('REDIS_PASSWORD'),
        'ttl_hours': int(os.getenv('REDIS_TTL_HOURS', '24')),
        'key_prefix': os.getenv('REDIS_KEY_PREFIX', 'kyc_verification')
    }


# Global cache instance (singleton pattern)
_cache_instance: Optional[RedisVerificationCache] = None


def get_cache() -> RedisVerificationCache:
    """
    Get or create global cache instance.
    
    Returns:
        Global RedisVerificationCache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        config = load_cache_config_from_env()
        
        if config['enabled']:
            _cache_instance = RedisVerificationCache(
                host=config['host'],
                port=config['port'],
                db=config['db'],
                password=config['password'],
                ttl_hours=config['ttl_hours'],
                key_prefix=config['key_prefix']
            )
        else:
            logger.info("ℹ️  Redis cache is disabled by configuration")
            # Return a disabled cache instance
            _cache_instance = RedisVerificationCache(host='disabled')
    
    return _cache_instance


if __name__ == "__main__":
    """Test the Redis cache functionality."""
    
    print("=" * 70)
    print("REDIS CACHE TEST")
    print("=" * 70)
    
    # Initialize cache
    cache = get_cache()
    
    # Print stats
    print("\n📊 Cache Statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test storing a verification result
    print("\n" + "-" * 70)
    print("Testing cache operations...")
    
    test_user_id = "159062"
    test_result = {
        "user_id": 159062,
        "agent_id": "79",
        "final_decision": "APPROVED",
        "status_code": 2,
        "extracted_data": {
            "aadhaar": "877694160807",
            "dob": "16-01-2002",
            "gender": "Male"
        },
        "rejection_reasons": ["Verification Document is not proper"]
    }
    
    test_api_response = {
        "message": "Its not a pending status KYC",
        "success": False
    }
    
    # Store result
    print(f"\n1. Storing verification result for user {test_user_id}...")
    success = cache.store_verification_result(test_user_id, test_result, test_api_response)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Check if user is verified
    print(f"\n2. Checking if user {test_user_id} is verified...")
    is_verified = cache.is_user_verified(test_user_id)
    print(f"   Result: {'✅ Found in cache' if is_verified else '❌ Not in cache'}")
    
    # Retrieve result
    print(f"\n3. Retrieving cached result for user {test_user_id}...")
    cached_result = cache.get_verification_result(test_user_id)
    if cached_result:
        print(f"   ✅ Retrieved result:")
        print(f"      Decision: {cached_result['verification_result']['final_decision']}")
        print(f"      Cached at: {cached_result['cached_at']}")
    else:
        print(f"   ❌ No cached result found")
    
    # Bulk check
    print(f"\n4. Testing bulk check...")
    test_user_ids = ["159062", "159063", "159064"]
    bulk_results = cache.bulk_check_users(test_user_ids)
    for uid, verified in bulk_results.items():
        status = "✅ Verified" if verified else "❌ Not verified"
        print(f"   User {uid}: {status}")
    
    # Final stats
    print("\n" + "-" * 70)
    print("📊 Final Cache Statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        if key != 'sample_entries':
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
