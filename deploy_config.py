"""
AWE V10 Surgery - Deploy Configuration & Infrastructure Fixes.

Fixes handled here:
FIX #14: nvidia-smi power limit clamp (350W cap on L40S)
FIX #20: S3 upload multipart threshold (parquet > 5GB)
FIX #21: Docker image size split (multi-stage build hints)
FIX #26: CloudFront signed URL TTL (12h default)
FIX #35: nvidia-persistenced daemon enable
FIX #36: CUDA MPS activation for multi-tenant
FIX #37: ECC memory mode check
FIX #44: Container health-check endpoint
FIX #47: Graceful shutdown SIGTERM handler
"""

import os
import signal
import sys
import time
import logging
import json
from typing import Optional, Dict

logger = logging.getLogger("DeployConfig")


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

class GPUConfig:
    """FIX #14, #35, #36, #37: GPU infrastructure settings."""
    
    def __init__(self):
        self.power_limit_watts = 350  # FIX #14: L40S max = 350W
        self.persistence_mode = True  # FIX #35
        self.mps_enabled = False      # FIX #36
        self.ecc_required = True      # FIX #37
    
    def apply_gpu_settings(self):
        """Apply all GPU configuration at container boot."""
        import subprocess
        
        # FIX #35: Enable nvidia-persistenced
        if self.persistence_mode:
            try:
                subprocess.run(
                    ["nvidia-smi", "-pm", "1"], 
                    check=True, capture_output=True, timeout=10
                )
                logger.info("[FIX #35] nvidia-persistenced ENABLED. Cold-start latency reduced.")
            except Exception as e:
                logger.warning(f"[FIX #35] nvidia-persistenced failed: {e}")
        
        # FIX #14: Power limit clamp
        try:
            subprocess.run(
                ["nvidia-smi", "-pl", str(self.power_limit_watts)],
                check=True, capture_output=True, timeout=10
            )
            logger.info(f"[FIX #14] GPU power limit set to {self.power_limit_watts}W.")
        except Exception as e:
            logger.warning(f"[FIX #14] Power limit set failed: {e}")
        
        # FIX #37: Check ECC memory
        if self.ecc_required:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=ecc.mode.current", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                ecc_status = result.stdout.strip()
                if "Enabled" in ecc_status:
                    logger.info("[FIX #37] ECC memory: ENABLED (safe for long runs).")
                else:
                    logger.warning(f"[FIX #37] ECC memory: {ecc_status}! "
                                   f"Bit-flips in 48GB VRAM may corrupt weights silently.")
            except Exception as e:
                logger.warning(f"[FIX #37] ECC check failed: {e}")
        
        # FIX #36: CUDA MPS for multi-tenant
        if self.mps_enabled:
            try:
                os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
                os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps-log"
                subprocess.run(
                    ["nvidia-cuda-mps-control", "-d"],
                    check=True, capture_output=True, timeout=10
                )
                logger.info("[FIX #36] CUDA MPS ENABLED for multi-tenant serving.")
            except Exception as e:
                logger.warning(f"[FIX #36] MPS activation failed: {e}")
    
    def get_status(self) -> dict:
        return {
            "power_limit_watts": self.power_limit_watts,
            "persistence_mode": self.persistence_mode,
            "mps_enabled": self.mps_enabled,
            "ecc_required": self.ecc_required,
        }


# =============================================================================
# S3 UPLOAD CONFIGURATION  
# =============================================================================

class S3Config:
    """FIX #20: S3 multipart upload for large GLB files."""
    
    def __init__(self):
        # FIX #20: Multipart threshold — default 5GB, but GLBs can be 1-3GB
        # Setting to 100MB so multipart kicks in earlier for reliability
        self.multipart_threshold_bytes = 100 * 1024 * 1024  # 100MB
        self.multipart_chunksize_bytes = 64 * 1024 * 1024   # 64MB per part
        self.max_concurrency = 10
        self.use_accelerate = True
    
    def get_transfer_config(self) -> dict:
        """Returns boto3 TransferConfig-compatible dict."""
        return {
            "multipart_threshold": self.multipart_threshold_bytes,
            "multipart_chunksize": self.multipart_chunksize_bytes,
            "max_concurrency": self.max_concurrency,
            "use_threads": True,
        }
    
    def upload_glb(self, local_path: str, s3_bucket: str, s3_key: str):
        """Upload GLB with multipart + progress logging."""
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            
            config = TransferConfig(
                multipart_threshold=self.multipart_threshold_bytes,
                multipart_chunksize=self.multipart_chunksize_bytes,
                max_concurrency=self.max_concurrency,
                use_threads=True,
            )
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"[FIX #20] Uploading {local_path} ({file_size_mb:.1f}MB) to s3://{s3_bucket}/{s3_key}")
            
            s3_client = boto3.client('s3')
            
            # Use transfer accelerate if available
            if self.use_accelerate:
                s3_client = boto3.client('s3', config=boto3.session.Config(
                    s3={'use_accelerate_endpoint': True}
                ))
            
            s3_client.upload_file(local_path, s3_bucket, s3_key, Config=config)
            logger.info(f"[FIX #20] Upload complete: s3://{s3_bucket}/{s3_key}")
            
        except ImportError:
            logger.warning("[FIX #20] boto3 not installed. S3 upload skipped.")
        except Exception as e:
            logger.error(f"[FIX #20] S3 upload failed: {e}")


# =============================================================================
# CLOUDFRONT CDN CONFIGURATION
# =============================================================================

class CDNConfig:
    """FIX #26: CloudFront signed URL with proper TTL."""
    
    def __init__(self):
        # FIX #26: TTL 12 horas ao invés de 7 dias (reduz exposição de IP)
        self.signed_url_ttl_seconds = 12 * 3600  # 12 hours
        self.distribution_domain = os.getenv("CF_DISTRIBUTION_DOMAIN", "")
        self.key_pair_id = os.getenv("CF_KEY_PAIR_ID", "")
        self.private_key_path = os.getenv("CF_PRIVATE_KEY_PATH", "")
    
    def generate_signed_url(self, resource_path: str) -> Optional[str]:
        """Generate CloudFront signed URL with 12h TTL."""
        if not self.distribution_domain or not self.key_pair_id:
            logger.warning("[FIX #26] CloudFront not configured. Returning raw URL.")
            return f"https://{self.distribution_domain}/{resource_path}"
        
        try:
            from botocore.signers import CloudFrontSigner
            import datetime
            import rsa
            
            with open(self.private_key_path, 'rb') as f:
                private_key = rsa.PrivateKey.load_pkcs1(f.read())
            
            def rsa_signer(message):
                return rsa.sign(message, private_key, 'SHA-1')
            
            cf_signer = CloudFrontSigner(self.key_pair_id, rsa_signer)
            
            expiry = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=self.signed_url_ttl_seconds
            )
            
            url = cf_signer.generate_presigned_url(
                f"https://{self.distribution_domain}/{resource_path}",
                date_less_than=expiry
            )
            
            logger.info(f"[FIX #26] Signed URL generated (TTL={self.signed_url_ttl_seconds}s)")
            return url
            
        except Exception as e:
            logger.warning(f"[FIX #26] Signed URL generation failed: {e}")
            return f"https://{self.distribution_domain}/{resource_path}"


# =============================================================================
# DOCKER/CONTAINER CONFIGURATION
# =============================================================================

class ContainerConfig:
    """FIX #21, #44, #47: Container lifecycle management."""
    
    def __init__(self):
        # FIX #21: Docker multi-stage build recommendations
        self.max_image_size_gb = 15.0  # Target: < 15GB (was 22GB)
        self.health_check_port = 8080  # FIX #44
        self._shutdown_handlers = []   # FIX #47
        self._is_healthy = False
    
    def get_dockerfile_hints(self) -> dict:
        """
        FIX #21: Recommendations for multi-stage Docker build to reduce image from 22GB to ~14GB.
        """
        return {
            "stage_1_builder": {
                "base": "nvidia/cuda:12.4-devel-ubuntu22.04",
                "purpose": "Compile CUDA kernels + pybind11",
                "install": ["gcc", "g++", "python3-dev", "pybind11"],
                "compile": "python setup.py build_ext --inplace",
            },
            "stage_2_runtime": {
                "base": "nvidia/cuda:12.4-runtime-ubuntu22.04",
                "purpose": "Runtime only — no compiler, no dev headers",
                "copy_from_builder": ["socket_engine_cuda*.so", "*.pth"],
                "pip_install": "pip install --no-cache-dir torch torchvision diffusers transformers",
                "savings": "~8GB smaller (no devel packages)",
            },
            "tips": [
                "Use .dockerignore to exclude checkpoints/, output/, __pycache__/",
                "Pin exact versions in requirements.txt",
                "Use --mount=type=cache,target=/root/.cache for pip cache",
            ]
        }
    
    # FIX #44: Health check endpoint
    def start_health_check_server(self, port: int = None):
        """Start a minimal HTTP health-check server on a background thread."""
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        port = port or self.health_check_port
        config = self
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    status = 200 if config._is_healthy else 503
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    body = json.dumps({
                        "status": "healthy" if config._is_healthy else "starting",
                        "timestamp": time.time(),
                    })
                    self.wfile.write(body.encode())
                elif self.path == "/ready":
                    self.send_response(200 if config._is_healthy else 503)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Silent logging
        
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"[FIX #44] Health check server started on port {port}")
        return server
    
    def set_healthy(self, healthy: bool = True):
        """Mark container as healthy (called after model warmup)."""
        self._is_healthy = healthy
        logger.info(f"[FIX #44] Health status: {'HEALTHY' if healthy else 'UNHEALTHY'}")
    
    # FIX #47: Graceful shutdown handler
    def register_shutdown_handler(self, handler):
        """Register a callback to run on SIGTERM/SIGINT."""
        self._shutdown_handlers.append(handler)
    
    def install_signal_handlers(self):
        """
        FIX #47: Instala SIGTERM/SIGINT handlers para graceful shutdown.
        Sem isso, kill -15 mata o container mid-inference e corrompe checkpoints.
        """
        def _shutdown_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"[FIX #47] Recebido {sig_name}. Iniciando graceful shutdown...")
            
            for handler in self._shutdown_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"[FIX #47] Shutdown handler falhou: {e}")
            
            logger.info("[FIX #47] Graceful shutdown completo.")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, _shutdown_signal)
        signal.signal(signal.SIGINT, _shutdown_signal)
        logger.info("[FIX #47] SIGTERM/SIGINT handlers instalados.")


# =============================================================================
# TEXTURE ENTROPY AUDIT (usado pelo titan_master)
# =============================================================================

def audit_texture_entropy(image_rgb, threshold: float = 50.0) -> dict:
    """Simple entropy check for the orchestrator."""
    import cv2
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    passed = laplacian_var >= threshold
    return {"passed": passed, "score": float(laplacian_var)}


# =============================================================================
# AGGREGATED DEPLOY CONFIG
# =============================================================================

class DeployConfig:
    """Master deployment configuration — aggregates all infra fixes."""
    
    def __init__(self):
        self.gpu = GPUConfig()
        self.s3 = S3Config()
        self.cdn = CDNConfig()
        self.container = ContainerConfig()
    
    def boot(self):
        """Full boot sequence for cloud deployment."""
        logger.info("=" * 60)
        logger.info("[DeployConfig V10] Cloud Infrastructure Boot")
        logger.info("=" * 60)
        
        # FIX #47: Signal handlers first
        self.container.install_signal_handlers()
        
        # FIX #44: Health check (starts unhealthy)
        self.container.start_health_check_server()
        
        # FIX #14, #35, #36, #37: GPU settings
        self.gpu.apply_gpu_settings()
        
        logger.info("[DeployConfig] Infrastructure boot complete!")
    
    def mark_ready(self):
        """Called after model warmup to mark container as ready."""
        self.container.set_healthy(True)
    
    def get_full_config(self) -> dict:
        return {
            "gpu": self.gpu.get_status(),
            "s3": self.s3.get_transfer_config(),
            "cdn": {
                "ttl_seconds": self.cdn.signed_url_ttl_seconds,
                "domain": self.cdn.distribution_domain,
            },
            "container": {
                "image_target_gb": self.container.max_image_size_gb,
                "health_port": self.container.health_check_port,
            }
        }
