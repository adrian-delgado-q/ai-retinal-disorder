import reflex as rx


config = rx.Config(
    app_name="reflex_frontend",
    state_auto_setters=False,
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
