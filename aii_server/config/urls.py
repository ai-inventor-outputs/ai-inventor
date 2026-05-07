"""
URL configuration for aii_server.

Django serves the JSON/REST surface only — abilities, /api, /_allauth,
/admin. The SPA frontend lives behind ``next start`` on a separate port
and rewrites Django paths to here. Unknown URLs return JSON 404.

When WEB_APP_MODE is False, only /agent_abilities/ is registered.
"""

from agent_abilities.api import abilities_api
from django.conf import settings
from django.urls import path, re_path

# ---- Core URL patterns ----
# Abilities API is always available.
urlpatterns = [
    path("agent_abilities/", abilities_api.urls),
]


# ---- Dashboard routes (WEB_APP_MODE only) ----
if settings.WEB_APP_MODE:
    from dashboard.api import api
    from django.contrib import admin
    from django.urls import include

    urlpatterns += [
        path("admin/", admin.site.urls),
        # /_allauth/oauth/ hosts the OAuth provider handshake URLs that
        # Google/GitHub redirect back to (`/_allauth/oauth/<provider>/
        # login/callback/`). HTML signup/login/etc. inside this URL pack
        # are disabled by HEADLESS_ONLY=True; only provider login +
        # callback views remain functional.
        path("_allauth/oauth/", include("allauth.urls")),
        path("_allauth/", include("allauth.headless.urls")),
        path("api/", api.urls),
    ]

    def _api_404(request, path=""):
        """Return JSON 404 for unmatched /api/ routes."""
        from django.http import JsonResponse

        return JsonResponse({"detail": f"Not found: /api/{path}"}, status=404)

    def _allauth_404(request, path=""):
        """Return JSON 404 for unmatched /_allauth/ routes."""
        from django.http import JsonResponse

        return JsonResponse({"detail": f"Not found: /_allauth/{path}"}, status=404)

    def _json_405(request, *args, **kwargs):
        from django.http import JsonResponse

        return JsonResponse({"detail": "Method not allowed"}, status=405)

    handler405 = _json_405

    urlpatterns += [
        re_path(r"^api/(?P<path>.*)$", _api_404),
        re_path(r"^_allauth/(?P<path>.*)$", _allauth_404),
    ]
