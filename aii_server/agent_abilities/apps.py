"""AppConfig for the agent_abilities app — registers signal handlers on AppConfig.ready."""

from django.apps import AppConfig
from loguru import logger


class AbilitiesConfig(AppConfig):
    name = "agent_abilities"
    verbose_name = "AI Abilities"

    def ready(self):
        try:
            from .api import register_ability_routes
            from .discovery import discover_abilities

            discover_abilities()
            register_ability_routes()
        except Exception:
            logger.exception("Ability discovery failed — abilities will not be available")
