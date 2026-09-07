from pathlib import Path


def load_theme(app):

    theme = Path(__file__).parent / "theme.qss"

    if theme.exists():
        app.setStyleSheet(theme.read_text())