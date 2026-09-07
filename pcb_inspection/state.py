from .config import CLASSES

frame = None

stats = {
    cls: 0
    for cls in CLASSES
}

fps = 0.0

running = True