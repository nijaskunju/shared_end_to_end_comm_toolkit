"""Shared Qt stylesheet fragments for the toolkit GUIs."""

from pathlib import Path

_CHECKMARK_PATH = (Path(__file__).resolve().parent / "assets" / "checkmark.svg").as_posix()

# Styling ::indicator replaces Fusion's native drawing, so the checkmark must come from an image.
TOGGLE_STYLESHEET = """
QCheckBox, QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
}
QCheckBox:disabled, QRadioButton:disabled {
    color: #777777;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #90a4ae;
    background-color: #2b2b2b;
}
QCheckBox::indicator {
    border-radius: 4px;
}
QRadioButton::indicator {
    border-radius: 10px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #64b5f6;
    background-color: #263238;
}
QCheckBox::indicator:checked {
    border-color: #64b5f6;
    background-color: #1e88e5;
    image: url("%CHECKMARK%");
}
QRadioButton::indicator:checked {
    border-color: #64b5f6;
    background-color: qradialgradient(
        cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
        stop: 0 #42a5f5, stop: 0.5 #42a5f5, stop: 0.56 #2b2b2b, stop: 1 #2b2b2b
    );
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #555555;
    background-color: #232323;
}
QCheckBox::indicator:checked:disabled {
    background-color: #455a64;
}
""".replace("%CHECKMARK%", _CHECKMARK_PATH)
