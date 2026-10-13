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
    Includes industrial Fail-Safe Heartbeat sender thread.
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baudrate: int = SERIAL_BAUD,
        feedback_callback: Optional[Callable[[str], None]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.feedback_callback = feedback_callback

        self.ser: Optional[serial.Serial] = None
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready_ack = threading.Event()
        self._centering_ack = threading.Event()
        self._resume_ack = threading.Event()
        self._result_ack = {"OK": threading.Event(), "NG": threading.Event()}

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

            # Start reading thread
            self.read_thread = threading.Thread(
                target=self._read_loop,
                name="Arduino-Serial-Reader",
                daemon=True,
            )
            self.read_thread.start()

            # Start industrial heartbeat thread (1.5s interval)
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="Arduino-Heartbeat-Sender",
                daemon=True,
            )
            self.heartbeat_thread.start()

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
                if cmd != "PING":
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

    def send_result(self, result: str, timeout: float = 1.5) -> bool:
        """Send an inspection result and wait for Arduino to accept it."""
        result = result.upper()
        if result not in self._result_ack:
            return False
        ack_event = self._result_ack[result]
        ack_event.clear()
        if not self.send_command(result):
            return False
        return ack_event.wait(timeout)

    def send_start(self) -> bool:
        """Send START conveyor command to Arduino Due."""
        return self.send_command("START")

    def send_stop(self) -> bool:
        """Send STOP conveyor command to Arduino Due."""
        return self.send_command("STOP")

    def send_resume(self, timeout: float = 1.5) -> bool:
        """Resume after NG removal and wait for Arduino confirmation."""
        self._resume_ack.clear()
        if not self.send_command("RESUME"):
            return False
        if self._resume_ack.wait(timeout):
            return True
        logger.error("Arduino did not acknowledge RESUME; issuing STOP.")
        self.send_stop()
        return False

    def begin_centering(self, duration_ms: int) -> bool:
        """Start a bounded centering move; the reader thread receives its stop acknowledgement."""
        duration_ms = max(0, min(8000, int(duration_ms)))
        self._centering_ack.clear()
        return self.send_command(f"CENTER:{duration_ms}")

    def centering_complete(self) -> bool:
        return self._centering_ack.is_set()

    def send_ready(self, timeout: float = 1.5) -> bool:
        """Stop the conveyor and wait for Arduino's stopped-state acknowledgement."""
        self._ready_ack.clear()
        if not self.send_command("PCB_READY"):
            return False
        if self._ready_ack.wait(timeout):
            return True

        # A missing acknowledgement means the stop state is unconfirmed.
        # Issue STOP as a fail-safe and let the caller hold inspection.
        logger.error("Arduino did not acknowledge PCB_READY; issuing STOP.")
        self.send_stop()
        return False

    
    def send_reset(self) -> bool:
        """Send RESET command to Arduino Due."""
        return self.send_command("RESET")

    def _heartbeat_loop(self):
        """Industrial Fail-Safe Heartbeat sender: Send PING to Arduino Due every 1.5s."""
        while self.running and self.ser and self.ser.is_open:
            time.sleep(1.5)
            self.send_command("PING")

    def _read_loop(self):
        """Background thread reading incoming serial lines from Arduino Due."""
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line != "PONG":
                    logger.info(f"Received from Arduino: '{line}'")

                if line == "ACK:PCB_READY_STOPPED":
                    self._ready_ack.set()
                elif line == "ACK:CENTERING_STOPPED":
                    self._centering_ack.set()
                elif line == "ACK:RESUME_RUNNING":
                    self._resume_ack.set()
                elif line == "ACK:OK":
                    self._result_ack["OK"].set()
                elif line == "ACK:NG":
                    self._result_ack["NG"].set()

                # Dispatch feedback callback
                if self.feedback_callback and line != "PONG":
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
