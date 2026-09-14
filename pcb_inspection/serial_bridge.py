import time
import serial
import serial.tools.list_ports
import threading
import logging
from typing import Callable, Optional

from .config import (
    SERIAL_PORT,
    SERIAL_BAUD,
    ENABLE_SERIAL_BRIDGE,
)

logger = logging.getLogger("SerialBridge")


class ArduinoBridge:
    """
    Serial Bridge handler connecting Jetson Orin Nano with Arduino Due.
    Communicates over UART/USB Serial (115200 baud).
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baudrate: int = SERIAL_BAUD,
        trigger_callback: Optional[Callable[[], None]] = None,
        feedback_callback: Optional[Callable[[str], None]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.trigger_callback = trigger_callback
        self.feedback_callback = feedback_callback

        self.ser: Optional[serial.Serial] = None
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def find_arduino_port(self) -> Optional[str]:
        """Auto-detect Arduino Due USB serial port if default port does not exist."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino" in p.description or "ttyACM" in p.device or "ttyUSB" in p.device:
                return p.device
        return self.port if ports else None

    def connect(self) -> bool:
        """Establish serial connection with Arduino Due."""
        if not ENABLE_SERIAL_BRIDGE:
            logger.info("Serial bridge is disabled in config.")
            return False

        port_to_open = self.find_arduino_port() or self.port
        try:
            self.ser = serial.Serial(
                port=port_to_open,
                baudrate=self.baudrate,
                timeout=0.5,
                write_timeout=0.5,
            )
            time.sleep(2.0)  # Wait for Arduino Due bootloader reset
            self.running = True
            self.read_thread = threading.Thread(
                target=self._read_loop,
                name="Arduino-Serial-Reader",
                daemon=True,
            )
            self.read_thread.start()
            logger.info(f"Connected to Arduino Due on {port_to_open} at {self.baudrate} baud.")
            return True

        except Exception as e:
            logger.warning(f"Failed to open serial port '{port_to_open}': {e}")
            self.ser = None
            return False

    def send_command(self, cmd: str) -> bool:
        """Send formatted command string to Arduino Due."""
        with self._lock:
            if self.ser is None or not self.ser.is_open:
                logger.debug(f"Serial port not connected. Command '{cmd}' dropped.")
                return False

            try:
                msg = f"{cmd.strip()}\n"
                self.ser.write(msg.encode("utf-8"))
                self.ser.flush()
                logger.info(f"Sent command to Arduino: '{cmd}'")
                return True
            except Exception as e:
                logger.error(f"Error sending command '{cmd}': {e}")
                return False

    def send_ok(self) -> bool:
        """Send OK result signal to Arduino Due."""
        return self.send_command("OK")

    def send_ng(self) -> bool:
        """Send NG result signal to Arduino Due."""
        return self.send_command("NG")

    def send_start(self) -> bool:
        """Send START conveyor command to Arduino Due."""
        return self.send_command("START")

    def send_stop(self) -> bool:
        """Send STOP conveyor command to Arduino Due."""
        return self.send_command("STOP")

    def send_reset(self) -> bool:
        """Send RESET command to Arduino Due."""
        return self.send_command("RESET")

    def _read_loop(self):
        """Background thread reading incoming serial lines from Arduino Due."""
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                logger.info(f"Received from Arduino: '{line}'")

                # Handle trigger from PCB arrival sensor
                if line == "TRIGGER" or "TRIGGER" in line:
                    if self.trigger_callback:
                        self.trigger_callback()

                # Dispatch feedback callback
                if self.feedback_callback:
                    self.feedback_callback(line)

            except Exception as e:
                if self.running:
                    logger.error(f"Serial read error: {e}")
                break

    def disconnect(self):
        """Close serial connection cleanly."""
        self.running = False
        with self._lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None
        logger.info("Arduino Serial Bridge disconnected.")
