from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
)


class InfoCard(QWidget):

    def __init__(self, title: str, value: str = "--"):
        super().__init__()

        self.title = QLabel(title)
        self.value = QLabel(value)

        self.title.setObjectName("cardTitle")
        self.value.setObjectName("cardValue")

        layout = QHBoxLayout(self)

        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(self.title)
        layout.addStretch()
        layout.addWidget(self.value)

    def set_value(self, value):

        self.value.setText(str(value))