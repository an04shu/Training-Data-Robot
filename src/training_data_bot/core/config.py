"""
Simple configuration management for Training Data Bot.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class SimpleConfig:
    """Simple configuration class."""
    
    def __init__(self):
        # Application info
        self.app_name = "Training Data Curation Bot"
        self.app_version = "0.1.0"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Decodo API
        self.decodo = self._create_decodo_config()
        
        # Processing
        self.processing = self._create_processing_config()
        
        # Storage
        self.storage = self._create_storage_config()
        
        # Quality
        self.quality = self._create_quality_config()
        
        # Dashboard
        self.dashboard = self._create_dashboard_config()
        
        # Security
        self.security = self._create_security_config()
        
        # Monitoring
        self.monitoring = self._create_monitoring_config()

    def _create_decodo_config(self):
        """Create Decodo configuration."""
        class DecodoConfig:
            api_key = os.getenv("DECODO_API_KEY", "mock_api_key")
            base_url = os.getenv("DECODO_BASE_URL", "https://api.decodo.com")
            user_id = os.getenv("DECODO_USER_ID")
            basic_auth = os.getenv("DECODO_BASIC_AUTH")
            timeout = int(os.getenv("DECODO_TIMEOUT", "60"))
            max_retries = int(os.getenv("DECODO_MAX_RETRIES", "3"))
            rate_limit = int(os.getenv("DECODO_RATE_LIMIT", "10"))
            backoff_factor = 2.0
            retry_status_codes = [429, 502, 503, 504]
        
        return DecodoConfig()

    def _create_processing_config(self):
        """Create processing configuration."""
        class ProcessingConfig:
            max_workers = int(os.getenv("MAX_WORKERS", "4"))
            chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
            batch_size = int(os.getenv("BATCH_SIZE", "10"))
            async_batch_size = 5
            connection_pool_size = 10
        
        return ProcessingConfig()

    def _create_storage_config(self):
        """Create storage configuration."""
        class StorageConfig:
            data_dir = Path(os.getenv("DATA_DIR", "./data"))
            output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
            cache_dir = Path(os.getenv("CACHE_DIR", "./cache"))
            temp_dir = Path(os.getenv("TEMP_DIR", "./temp"))
            database_url = os.getenv("DATABASE_URL", "sqlite:///./data/training_data_bot.db")
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            def __post_init__(self):
                # Create directories
                for dir_path in [self.data_dir, self.output_dir, self.cache_dir, self.temp_dir]:
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        config = StorageConfig()
        config.__post_init__()
        return config

    def _create_quality_config(self):
        """Create quality configuration."""
        class QualityConfig:
            toxicity_threshold = float(os.getenv("TOXICITY_THRESHOLD", "0.8"))
            bias_threshold = float(os.getenv("BIAS_THRESHOLD", "0.7"))
            min_text_length = int(os.getenv("MIN_TEXT_LENGTH", "10"))
            max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
            similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
            diversity_threshold = 0.6
        
        return QualityConfig()

    def _create_dashboard_config(self):
        """Create dashboard configuration."""
        class DashboardConfig:
            host = os.getenv("DASHBOARD_HOST", "localhost")
            port = int(os.getenv("DASHBOARD_PORT", "8501"))
            debug = os.getenv("DASHBOARD_DEBUG", "false").lower() == "true"
            title = "Training Data Curation Dashboard"
        
        return DashboardConfig()

    def _create_security_config(self):
        """Create security configuration."""
        class SecurityConfig:
            secret_key = os.getenv("SECRET_KEY", "dev_secret_key_change_in_production")
            encrypt_credentials = os.getenv("ENCRYPT_CREDENTIALS", "true").lower() == "true"
            enable_rate_limiting = True
            audit_logging = True
        
        return SecurityConfig()

    def _create_monitoring_config(self):
        """Create monitoring configuration."""
        class MonitoringConfig:
            enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
            metrics_port = int(os.getenv("METRICS_PORT", "8080"))
            enable_health_check = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"
            track_token_usage = True
            track_processing_time = True
        
        return MonitoringConfig()

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> SimpleConfig:
    """Get cached settings instance."""
    return SimpleConfig()


# Global settings instance
settings = get_settings()


def reload_settings() -> SimpleConfig:
    """Reload settings (useful for testing)."""
    get_settings.cache_clear()
    global settings
    settings = get_settings()
    return settings 