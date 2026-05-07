"""
Django settings for aii_server.

Non-secret config lives in aii_config/server/server.yaml.
Secrets (API keys, passwords, OAuth) stay in .env files.
Env vars override yaml values where both exist.
"""

import os
from pathlib import Path

from aii_lib.server_url import DEFAULT_SERVER_PORT
from aii_lib.utils.run_mode import pod_id
from dotenv import load_dotenv

# ---- Paths ----
# aii_server/config/settings.py → BASE_DIR = aii_server/
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent  # aii_server/ → project root

# Load .env files — secrets only (does NOT override existing env vars)
load_dotenv(REPO_ROOT / ".env")  # root .env (API keys)
load_dotenv(BASE_DIR / ".env")  # aii_server/.env (OAuth secrets, DB password)

# Load server config yaml — non-secret settings (private overrides
# from server.private.yaml deep-merged on top).
_SERVER_CONFIG_PATH = REPO_ROOT / "aii_config" / "server" / "server.yaml"
from aii_lib.utils.config_overrides import load_config_with_overrides

_cfg: dict = load_config_with_overrides(_SERVER_CONFIG_PATH)

_server = _cfg.get("server", {})
_cors_cfg = _cfg.get("cors", {})
_db_cfg = _cfg.get("database", {})
_data_cfg = _cfg.get("data", {})
_email_cfg = _cfg.get("email", {})
_security_cfg = _cfg.get("security", {})
_auth_cfg = _cfg.get("auth", {})
_pod_cfg = _cfg.get("pod_lifecycle", {})
_logging_cfg = _cfg.get("logging", {})

# ---- Request logging — per-field value truncation ----
# Consumed by ``agent_abilities.middleware._summarize_dict`` so big
# fields (e.g. base64 image_data) don't dump multi-MB into the log.
AII_LOG_TRUNCATE_CHARS = int(_logging_cfg.get("truncate_chars", 200))

# ---- v26 pod lifecycle tunables (consumed by services.runpod_orchestrator
# and services.run_jobs) ----
AII_POD_LIVENESS_CACHE_SECONDS = int(_pod_cfg.get("liveness_cache_seconds", 10))
AII_POD_HEARTBEAT_INTERVAL_SECONDS = int(_pod_cfg.get("heartbeat_interval_seconds", 60))

# ---- v26 SSE stream tunables (consumed by aii_pipeline.run.ipc.sse) ----
# Heartbeat: idle ``ping`` interval. A connection that sees no real
# events for this many seconds gets ``event: ping`` so the FE watchdog
# (3× this value) doesn't kick stale-but-healthy connections.
AII_SSE_HEARTBEAT_SECONDS = float(_pod_cfg.get("sse_heartbeat_seconds", 15.0))

# ---- v26 orchestrator launch params (consumed by
# services.runpod_provision.launch_orchestrator_pod). RUNPOD_API_KEY itself
# is read from .env, never from server.yaml. ----
_pod_orch_cfg = _pod_cfg.get("orchestrator", {}) or {}
AII_POD_ORCH_TEMPLATE_ID = _pod_orch_cfg.get("template_id", "") or ""
AII_POD_ORCH_IMAGE = _pod_orch_cfg.get("image", "") or ""
AII_POD_ORCH_ABILITY_URL = _pod_orch_cfg.get("ability_url", "") or ""
AII_POD_ORCH_VOLUME_ID = _pod_orch_cfg.get("network_volume_id", "") or ""
AII_POD_ORCH_DATA_CENTER = _pod_orch_cfg.get("data_center_id", "") or ""

# Dynamic ability_url fallback: each redeploy gets a new RunPod pod_id, so
# baking a static URL into server.yaml means every deploy needs a yaml edit.
# When the configured value is empty/placeholder AND we're running on RunPod,
# derive it from RUNPOD_POD_ID (matches aii_runpod.deploy.remote pattern).
if not AII_POD_ORCH_ABILITY_URL or AII_POD_ORCH_ABILITY_URL.startswith("REPLACE"):
    _pod = pod_id()
    if _pod:
        from aii_runpod import pod_proxy_url

        AII_POD_ORCH_ABILITY_URL = pod_proxy_url(_pod, 8020)

# ---- Web App Mode ----
# False = abilities-only server (open-source/local use, no dashboard/auth/postgres)
# True  = full dashboard + abilities (private deployment)
WEB_APP_MODE = _server.get("web_app_mode", False)

AII_PIPELINE_DIR = REPO_ROOT / "aii_pipeline"
AII_CONFIG_DIR = REPO_ROOT / "aii_config"

# Data root — local: aii_data/, RunPod: /ai-inventor/aii_data/ (network volume)
# Env var overrides yaml (for RunPod volume mounts)
_data_root = os.environ.get("AII_DATA_DIR") or _data_cfg.get("root_dir") or ""
AII_DATA_DIR = Path(_data_root) if _data_root else REPO_ROOT / "aii_data"
AII_DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DATA_DIR = AII_DATA_DIR / "users"
USERS_DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / ".cache"
DISP_MESSAGES_DIR = CACHE_DIR / "messages"

# ---- Pipeline IPC ----
# Per-run mutations (stop, send_message) forward to the pipeline IPC
# service. The pipeline boots an HTTP server (see aii_pipeline.run.ipc.server)
# and exposes /runs/{id}/{send_user_msg, stop}. Server-side handlers in
# dashboard/api/run_*.py translate the new server endpoint names to the
# pipeline's internal IPC paths. Override with AII_PIPELINE_URL when running
# the pipeline IPC behind a proxy or on a non-default port.
PIPELINE_URL = os.environ.get("AII_PIPELINE_URL", "http://127.0.0.1:8021")

# Ensure cache dirs exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISP_MESSAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---- Security ----
DEBUG = _server.get("debug", True)
if DEBUG:
    SECRET_KEY = os.environ.get(
        "DJANGO_SECRET_KEY", "django-insecure-dev-key-do-not-use-in-production"
    )
    ALLOWED_HOSTS = _server.get("allowed_hosts", ["*"])
else:
    SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]  # required in production
    ALLOWED_HOSTS = _server.get("allowed_hosts", ["localhost", "127.0.0.1"])

# ---- CORS ----
# security.frontend_url is a shortcut that sets both CORS and CSRF origins.
_frontend_url = _security_cfg.get("frontend_url", "")
_cors_origins = _cors_cfg.get("origins", [])
if _frontend_url and _frontend_url not in _cors_origins:
    _cors_origins = [_frontend_url, *_cors_origins]

if _cors_cfg.get("allow_all", DEBUG):
    CORS_ALLOW_ALL_ORIGINS = True
elif _cors_origins:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = _cors_origins
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = []
CORS_ALLOW_CREDENTIALS = True

# ---- CSRF ----
_csrf_origins = _security_cfg.get("csrf_trusted_origins", [])
if _frontend_url and _frontend_url not in _csrf_origins:
    _csrf_origins = [_frontend_url, *_csrf_origins]
# In dev, auto-trust localhost origins
if DEBUG:
    _dev_origins = [
        "http://localhost:3000",
        f"http://localhost:{DEFAULT_SERVER_PORT}",
        f"http://127.0.0.1:{DEFAULT_SERVER_PORT}",
    ]
    for _dev_origin in _dev_origins:
        if _dev_origin not in _csrf_origins:
            _csrf_origins.append(_dev_origin)
if _csrf_origins:
    CSRF_TRUSTED_ORIGINS = _csrf_origins

# ---- Apps ----
_BASE_APPS = [
    "django.contrib.contenttypes",
    "corsheaders",
    "agent_abilities",
]

_DASHBOARD_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    # allauth
    "allauth",
    "allauth.account",
    "allauth.headless",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.github",
    # project
    "dashboard",
]

INSTALLED_APPS = _DASHBOARD_APPS + _BASE_APPS if WEB_APP_MODE else _BASE_APPS

if WEB_APP_MODE:
    SITE_ID = 1
    # Used by dashboard.apps.DashboardConfig.ready() to overwrite the default
    # `example.com` Site row on startup so allauth email subjects /
    # greetings reflect this deployment instead of the upstream placeholder.
    AII_SITE_NAME = _cfg.get("site", {}).get("name", "AI Inventor")
    _site_domain = _cfg.get("site", {}).get("domain", "")
    if not _site_domain:
        # Derive from frontend_url (strip scheme + path) so local dev gets
        # a sensible default ("localhost:3000") without requiring extra config.
        from urllib.parse import urlparse

        _site_domain = (
            urlparse(_frontend_url).netloc
            if _frontend_url
            else f"localhost:{_server.get('port', DEFAULT_SERVER_PORT)}"
        )
    AII_SITE_DOMAIN = _site_domain

# ---- Middleware ----
_BASE_MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "agent_abilities.middleware.RequestLogger",
]

_DASHBOARD_MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # AII_DEV_AUTOLOGIN=1 → silently sign in as ``admin`` on every anonymous
    # request. Must come after AuthenticationMiddleware (so request.user is
    # set) and before AccountMiddleware (so allauth sees the logged-in
    # session). No-op when the env var is unset, so prod is unaffected.
    "dashboard.middleware.DevAutologinMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "agent_abilities.middleware.RequestLogger",
]

MIDDLEWARE = _DASHBOARD_MIDDLEWARE if WEB_APP_MODE else _BASE_MIDDLEWARE

# Django's CommonMiddleware appends a trailing slash to un-slashed URLs
# via 308 redirect when ``APPEND_SLASH=True`` (default). The Next.js
# proxy in front of us has its own ``trailingSlash`` rule, and the FE
# clients (orval-generated + hand-written) emit no-trailing-slash API
# paths per the OpenAPI schema convention. Both ends agree on
# no-trailing-slash for ``/api/*`` — turning this off prevents every
# poll from eating a 308 round-trip.
APPEND_SLASH = False

ROOT_URLCONF = "config.urls"

if WEB_APP_MODE:
    TEMPLATES = [
        {
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [BASE_DIR / "templates"],
            "APP_DIRS": True,
            "OPTIONS": {
                "context_processors": [
                    "django.template.context_processors.request",
                    "django.contrib.auth.context_processors.auth",
                    "django.contrib.messages.context_processors.messages",
                ],
            },
        },
    ]
else:
    TEMPLATES = [
        {
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [],
            "APP_DIRS": False,
            "OPTIONS": {"context_processors": []},
        },
    ]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---- Database ----
if WEB_APP_MODE:
    # Postgres — connection from server.yaml; password from .env (secret).
    # In DEBUG, fall back to "aii_dev" so local dev works without a .env. In
    # production, require POSTGRES_PASSWORD explicitly (KeyError on missing).
    if DEBUG:
        _postgres_password = os.environ.get("POSTGRES_PASSWORD", "aii_dev")
    else:
        _postgres_password = os.environ["POSTGRES_PASSWORD"]
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": _db_cfg.get("name", "aii_inventor"),
            "USER": _db_cfg.get("user") or "aii",
            "PASSWORD": _postgres_password,
            "HOST": _db_cfg.get("host", "localhost"),
            "PORT": str(_db_cfg.get("port", 5432)),
            "CONN_HEALTH_CHECKS": True,
            "OPTIONS": {
                # psycopg3 native pool — bounded per-process pool of warm
                # connections, borrowed/returned per request instead of one-
                # per-request open/close. Caps PG connection count at
                # max_size×workers regardless of request concurrency.
                "pool": {"min_size": 8, "max_size": 40, "timeout": 10},
            },
        }
    }
else:
    # SQLite — no external DB needed for abilities-only mode.
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": AII_DATA_DIR / "db" / "aii_server.sqlite3",
        }
    }

# ---- Rate limiting (Ninja throttling) ----
# Uses Django's default cache (LocMemCache). For multi-process prod, switch to Redis.
_rate_limits = _auth_cfg.get("rate_limits", {})
NINJA_DEFAULT_THROTTLE_RATES = {
    "anon": _rate_limits.get("anon", "20/min"),
    "user": _rate_limits.get("user", "120/min"),
    "auth": _rate_limits.get("auth", "5/min"),
    "llm": _rate_limits.get("llm", "200/min"),
    "run_start": _rate_limits.get("run_start", "5/min"),
}

# ---- Auth ----
# Shared bearer-token auth for every internal aii HTTP surface:
# ``/agent_abilities/*`` (Django) plus the run-bus AppSink and run sources
# (send_message, stop) that the orchestrator pod exposes for cross-pod reads
# from aii_server. Without this, anyone reaching any of those ports can
# drain compute/LLM credits, leak Claude credentials, or stop runs.
#
# The token is auto-generated at boot and persisted to
# ``<AII_DATA_DIR>/.internal_key`` (chmod 600) — both the server pod and
# the orchestrator pod see the same file via the shared RunPod network
# volume. Override via ``AII_INTERNAL_KEY`` env var in shared deployments
# where the data dir isn't writable by clients (e.g. cross-machine).
import secrets as _secrets

_INTERNAL_KEY_FILE = AII_DATA_DIR / ".internal_key"
_internal_key_env = os.environ.get("AII_INTERNAL_KEY", "").strip()
if _internal_key_env:
    AII_INTERNAL_KEY = _internal_key_env
elif _INTERNAL_KEY_FILE.is_file():
    AII_INTERNAL_KEY = _INTERNAL_KEY_FILE.read_text(encoding="utf-8").strip()
else:
    AII_INTERNAL_KEY = _secrets.token_hex(32)
    _INTERNAL_KEY_FILE.write_text(AII_INTERNAL_KEY, encoding="utf-8")
    try:
        _INTERNAL_KEY_FILE.chmod(0o600)
    except OSError:
        pass
# Propagate to subprocesses (aii_launcher, pipeline, etc.).
os.environ["AII_INTERNAL_KEY"] = AII_INTERNAL_KEY


def _internal_bearer(request):
    """Constant-time bearer-token check for /agent_abilities/*."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None  # → 401
    if _secrets.compare_digest(auth[7:], AII_INTERNAL_KEY):
        return True
    return None


INTERNAL_AUTH = _internal_bearer

if WEB_APP_MODE:
    _min_pw_len = _auth_cfg.get("min_password_length", 5)
    AUTH_PASSWORD_VALIDATORS = [
        {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
        {
            "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
            "OPTIONS": {"min_length": _min_pw_len},
        },
        {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
        {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
    ]

    # ---- Allauth ----
    AUTHENTICATION_BACKENDS = [
        "django.contrib.auth.backends.ModelBackend",
        "allauth.account.auth_backends.AuthenticationBackend",
    ]

    # Account settings
    ACCOUNT_LOGIN_BY_CODE_ENABLED = False
    ACCOUNT_LOGIN_METHODS = {"email", "username"}
    ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
    ACCOUNT_EMAIL_VERIFICATION = _auth_cfg.get("email_verification", "mandatory")

    # Don't persist OAuth access/refresh tokens — we only do "log in with X",
    # we never call provider APIs on the user's behalf.
    SOCIALACCOUNT_STORE_TOKENS = False

    # OAuth credentials live in env vars only — `SOCIALACCOUNT_PROVIDERS["<provider>"]["APP"]`
    # ships them to allauth without needing rows in `socialaccount_socialapp`.
    SOCIALACCOUNT_PROVIDERS = {
        "github": {
            "SCOPE": ["user:email"],
            "APP": {
                "client_id": os.environ.get("GITHUB_CLIENT_ID", ""),
                "secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
            },
        },
    }

    # Redirects — send users to the frontend after login/logout
    _redirect_base = _frontend_url.rstrip("/") if _frontend_url else ""
    LOGIN_REDIRECT_URL = f"{_redirect_base}/runs" if _redirect_base else "/"
    LOGOUT_REDIRECT_URL = f"{_redirect_base}/login" if _redirect_base else "/"
    # Django's @login_required redirects unauth requests here. Headless
    # mode means there's no Django login page; point at the SPA login.
    LOGIN_URL = f"{_redirect_base}/login" if _redirect_base else "/login"
    SOCIALACCOUNT_LOGIN_ON_GET = True  # Skip "are you sure" intermediate page
    SOCIALACCOUNT_CONNECT_REDIRECT_URL = "/"  # After connecting account, go home

    # ---- Headless allauth ----
    # All auth flows are JSON via /_allauth/browser/v1/...; the HTML
    # signup/login/etc. views inside `accounts/` are disabled. The
    # `accounts/` mount is kept ONLY for OAuth provider callbacks.
    HEADLESS_ONLY = True
    HEADLESS_FRONTEND_URLS = {
        "account_confirm_email": f"{_redirect_base}/verify-email/{{key}}",
        "account_reset_password": f"{_redirect_base}/password/reset",
        "account_reset_password_from_key": f"{_redirect_base}/password/reset/key/{{key}}",
        "account_signup": f"{_redirect_base}/signup",
        "socialaccount_login_error": f"{_redirect_base}/login/error",
    }

# ---- i18n ----
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

# ---- URL prefix (when mounted under /app/) ----
_app_prefix = _server.get("url_prefix", "")
if _app_prefix:
    FORCE_SCRIPT_NAME = _app_prefix
    if WEB_APP_MODE:
        LOGIN_REDIRECT_URL = f"{_app_prefix}/"
        LOGOUT_REDIRECT_URL = f"{_app_prefix}/"
        LOGIN_URL = f"{_app_prefix}/login"
        SOCIALACCOUNT_CONNECT_REDIRECT_URL = f"{_app_prefix}/"

# ---- Static files ----
STATIC_URL = f"{_app_prefix}/static/" if _app_prefix else "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- Sessions ----
# cached_db: read-through cache (LocMemCache by default) in front of the DB
# session store. Auth-only endpoints (FE polling) hit the cache, not PG, so
# session lookups stop competing for connections under load.
SESSION_ENGINE = "django.contrib.sessions.backends.cached_db"

# ---- Request size limits ----
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # 5MB max request body

# ---- Production security ----
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    CSRF_COOKIE_SECURE = True
    CSRF_COOKIE_SAMESITE = "Lax"
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# ---- Email ----
if WEB_APP_MODE:
    # Dev: prints to console. Prod: set email.host in server.yaml + credentials in .env.
    _email_host = _email_cfg.get("host", "")
    DEFAULT_FROM_EMAIL = _email_cfg.get("from_address", "AI Inventor <noreply@example.com>")
    SERVER_EMAIL = DEFAULT_FROM_EMAIL
    if _email_host:
        EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
        EMAIL_HOST = _email_host
        EMAIL_PORT = int(_email_cfg.get("port", 587))
        EMAIL_USE_TLS = _email_cfg.get("use_tls", True)
        EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
        EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
    else:
        EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

# ---- Logging (loguru) ----
# Two sinks: colored stderr for humans + rotating JSONL for replay/grep.
# Logs live on the persistent volume so they survive pod restarts on RunPod.
import logging
import sys

from loguru import logger

LOG_DIR = AII_DATA_DIR / "logs" / "server"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()  # drop loguru's default stderr handler

logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    ),
)

logger.add(
    LOG_DIR / "aii_server.jsonl",
    level="DEBUG",
    rotation="100 MB",
    retention=5,
    serialize=True,  # JSON-per-line — message + extras + caller info
)


# Django/uvicorn logging: forward WARNING+ stdlib records to loguru.
# Request logging is handled by the RequestLogger middleware. This
# handler only catches unhandled exceptions with full tracebacks.
class _DjangoHandler(logging.Handler):
    """Forward WARNING+ stdlib records to loguru, tagged ``source=django``."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Service Unavailable" in msg:
            return
        full_msg = msg
        if record.exc_info and record.exc_info[1]:
            import traceback

            full_msg += "\n" + "".join(traceback.format_exception(*record.exc_info))
        bound = logger.bind(source="django")
        if record.levelno >= logging.ERROR:
            bound.bind(type="server_error").error(full_msg)
        else:
            bound.bind(type="server_event").warning(full_msg)


LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
    "handlers": {
        "loguru": {"class": "config.settings._DjangoHandler"},
        "null": {"class": "logging.NullHandler"},
    },
    "root": {"handlers": ["null"], "level": "WARNING"},
    "loggers": {
        "django": {"handlers": ["null"], "level": "WARNING", "propagate": False},
        "django.server": {"handlers": ["null"], "level": "WARNING", "propagate": False},
        "django.request": {
            "handlers": ["loguru"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.security": {
            "handlers": ["loguru"],
            "level": "WARNING",
            "propagate": False,
        },
        "uvicorn": {"handlers": ["null"], "level": "WARNING", "propagate": False},
        "uvicorn.access": {
            "handlers": ["null"],
            "level": "WARNING",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["loguru"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
