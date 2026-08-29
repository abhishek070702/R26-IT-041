"""
PN532 NFC V3 card reader for Raspberry Pi.

Tries I2C first, then UART (factory default on this board).
Set RFID_ALLOW_KEYBOARD=1 to type an id (laptop tests only).
"""

import os
import platform
import subprocess
import time
from typing import List, Optional


PN532_I2C_ADDRESS = 0x24
PREAMBLE = 0x00
STARTCODE1 = 0x00
STARTCODE2 = 0xFF
POSTAMBLE = 0x00
HOST_TO_PN532 = 0xD4
PN532_TO_HOST = 0xD5
ACK_FRAME = bytes([0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00])
COMMAND_GET_FIRMWARE = 0x02
COMMAND_SAM_CONFIGURATION = 0x14
COMMAND_INLIST_PASSIVE_TARGET = 0x4A


def _uid_to_id(uid) -> str:
    return "".join(f"{byte:02X}" for byte in uid)


class PN532Error(RuntimeError):
    pass


class PN532I2C:
    def __init__(self, bus_id: int = 1, address: int = PN532_I2C_ADDRESS):
        from smbus2 import SMBus, i2c_msg

        self._i2c_msg = i2c_msg
        self.address = address
        self.bus = SMBus(bus_id)
        time.sleep(0.1)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def _write(self, data: bytes):
        msg = self._i2c_msg.write(self.address, data)
        self.bus.i2c_rdwr(msg)

    def _read(self, count: int) -> bytes:
        msg = self._i2c_msg.read(self.address, count)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                status = self._read(1)
            except OSError:
                time.sleep(0.02)
                continue
            if status and status[0] == 0x01:
                return True
            time.sleep(0.02)
        return False

    def read_data(self, count: int) -> bytes:
        raw = self._read(count + 1)
        if not raw or raw[0] != 0x01:
            raise PN532Error("PN532 I2C device was not ready.")
        return raw[1:]

    def write_data(self, data: bytes):
        self._write(data)


class PN532UART:
    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200):
        import serial

        if not os.path.exists(port):
            raise PN532Error(
                f"Serial port {port} was not found. Enable UART hardware in raspi-config."
            )

        self.uart = serial.Serial(port, baudrate=baudrate, timeout=1)
        self.uart.reset_input_buffer()
        self.uart.reset_output_buffer()
        self.uart.write(b"\x55\x55\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        time.sleep(0.5)
        leftover = self.uart.read(self.uart.in_waiting or 0)
        if leftover:
            print(f"UART wakeup leftover on {port} @{baudrate}: {leftover.hex()}")

    def close(self):
        try:
            self.uart.close()
        except Exception:
            pass

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.uart.in_waiting:
                return True
            time.sleep(0.02)
        return False

    def read_data(self, count: int) -> bytes:
        data = self.uart.read(count)
        if len(data) < count:
            raise PN532Error("PN532 UART read timed out.")
        return data

    def write_data(self, data: bytes):
        self.uart.write(data)
        self.uart.flush()

    def dump_rx(self) -> bytes:
        waiting = self.uart.in_waiting
        data = self.uart.read(waiting) if waiting else b""
        print(f"UART RX bytes ({len(data)}): {data.hex() or 'none'}")
        return data


class PN532:
    def __init__(self, transport):
        self.transport = transport

    def close(self):
        self.transport.close()

    def _write_frame(self, payload: List[int]):
        length = len(payload)
        checksum = (-sum(payload)) & 0xFF
        frame = bytes(
            [
                PREAMBLE,
                STARTCODE1,
                STARTCODE2,
                length & 0xFF,
                (-length) & 0xFF,
                *payload,
                checksum,
                POSTAMBLE,
            ]
        )
        self.transport.write_data(frame)

    def _read_ack(self) -> bool:
        return self.transport.read_data(6) == ACK_FRAME

    def _read_frame(self, length: int) -> bytes:
        response = self.transport.read_data(length + 8)
        offset = 0
        while offset < len(response) and response[offset] == 0x00:
            offset += 1
        if offset >= len(response) or response[offset] != 0xFF:
            raise PN532Error("PN532 response start code was missing.")
        offset += 1
        if offset + 1 >= len(response):
            raise PN532Error("PN532 response was truncated.")
        frame_len = response[offset]
        length_checksum = response[offset + 1]
        if ((frame_len + length_checksum) & 0xFF) != 0:
            raise PN532Error("PN532 response length checksum failed.")
        offset += 2
        data = response[offset : offset + frame_len]
        data_checksum = response[offset + frame_len]
        if len(data) != frame_len or ((sum(data) + data_checksum) & 0xFF) != 0:
            raise PN532Error("PN532 response checksum failed.")
        if data[0] != PN532_TO_HOST:
            raise PN532Error("PN532 response type was invalid.")
        return bytes(data[1:])

    def call_function(
        self,
        command: int,
        params: Optional[List[int]] = None,
        response_length: int = 0,
        timeout: float = 1.0,
    ) -> Optional[bytes]:
        payload = [HOST_TO_PN532, command, *(params or [])]
        self._write_frame(payload)
        if not self.transport.wait_ready(timeout):
            return None
        if not self._read_ack():
            raise PN532Error("PN532 did not acknowledge the command.")
        if not self.transport.wait_ready(timeout):
            return None
        response = self._read_frame(response_length + 2)
        if not response or response[0] != command + 1:
            raise PN532Error("PN532 returned an unexpected command.")
        return response[1:]

    def get_firmware_version(self) -> bytes:
        response = self.call_function(COMMAND_GET_FIRMWARE, response_length=4, timeout=3.0)
        if not response or len(response) < 4:
            rx = b""
            dump = getattr(self.transport, "dump_rx", None)
            if callable(dump):
                rx = dump()
            raise PN532Error(
                "PN532 firmware version was not available. "
                f"UART received {len(rx)} byte(s). "
                "Connect the pins labeled TX and RX (not SDA/SCL): "
                "PN532 TX -> Pi pin 10, PN532 RX -> Pi pin 8, both switches OFF. "
                "If still no reply, swap TX and RX."
            )
        return response[:4]

    def sam_configuration(self):
        response = self.call_function(
            COMMAND_SAM_CONFIGURATION,
            params=[0x01, 0x14, 0x01],
            response_length=0,
            timeout=1.0,
        )
        if response is None:
            raise PN532Error("PN532 SAM configuration failed.")

    def read_passive_target(self, timeout: float = 0.5) -> Optional[bytes]:
        response = self.call_function(
            COMMAND_INLIST_PASSIVE_TARGET,
            params=[0x01, 0x00],
            response_length=19,
            timeout=timeout,
        )
        if not response or response[0] < 1:
            return None
        uid_length = response[5]
        uid = response[6 : 6 + uid_length]
        if not uid or len(uid) != uid_length:
            return None
        return bytes(uid)


def _scan_i2c(bus_id: int) -> List[int]:
    from smbus2 import SMBus

    found: List[int] = []
    bus = SMBus(bus_id)
    try:
        for address in range(0x03, 0x78):
            try:
                bus.write_quick(address)
                found.append(address)
            except OSError:
                continue
    finally:
        bus.close()
    return found


def _print_i2c_help(bus_id: int, found: List[int]):
    if found:
        shown = ", ".join(f"0x{addr:02X}" for addr in found)
        print(f"I2C bus {bus_id} devices: {shown}")
    else:
        print(f"I2C bus {bus_id} has no devices.")


def _print_firmware(interface: str, version: bytes):
    print(
        f"PN532 connected over {interface}. "
        f"Firmware {version[1]}.{version[2]} (IC 0x{version[0]:02X})."
    )


def _uart_ports() -> List[str]:
    configured = os.getenv("PN532_UART_PORT", "").strip()
    if configured:
        return [configured]

    serial0 = "/dev/serial0"
    serial0_target = ""
    if os.path.exists(serial0):
        serial0_target = os.path.realpath(serial0)
        print("serial0 ->", serial0_target)

    # Pi 5: serial0 is the debug header (ttyAMA10), not GPIO pins 8/10.
    skip_debug = {"ttyAMA10", "/dev/ttyAMA10"}
    ports = []
    for name in (
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/ttyACM0",
        "/dev/ttyAMA0",
        "/dev/ttyS0",
        serial0,
    ):
        if not os.path.exists(name):
            continue
        target = os.path.realpath(name)
        if os.path.basename(target) in skip_debug or target in skip_debug:
            print(
                f"Skipping {name} ({target}). That is the Pi 5 debug UART, "
                "not header pins 8 and 10."
            )
            continue
        if name not in ports:
            ports.append(name)

    if not ports:
        raise PN532Error(
            "GPIO UART is not enabled on this Raspberry Pi 5. "
            "Add this line to /boot/firmware/config.txt then reboot:\n"
            "  dtoverlay=uart0-pi5\n"
            "After reboot, /dev/ttyAMA0 must exist. "
            "PN532 TX -> pin 10, PN532 RX -> pin 8."
        )
    return ports


def _open_i2c() -> PN532:
    bus_ids = [int(os.getenv("PN532_I2C_BUS", "1"))]
    if 0 not in bus_ids:
        bus_ids.append(0)

    last_error = None
    for bus_id in bus_ids:
        if not os.path.exists(f"/dev/i2c-{bus_id}"):
            continue
        try:
            found = _scan_i2c(bus_id)
        except Exception as error:
            last_error = error
            print(f"I2C bus {bus_id} scan failed: {error}")
            continue
        _print_i2c_help(bus_id, found)
        if PN532_I2C_ADDRESS not in found:
            continue
        reader = PN532(PN532I2C(bus_id=bus_id))
        time.sleep(0.2)
        version = reader.get_firmware_version()
        reader.sam_configuration()
        _print_firmware(f"i2c-{bus_id}", version)
        return reader

    raise PN532Error(
        str(last_error)
        if last_error
        else "No PN532 at I2C 0x24."
    )


def _print_uart_status():
    serial0 = "/dev/serial0"
    if os.path.exists(serial0):
        try:
            print("serial0 ->", os.path.realpath(serial0))
        except OSError:
            print("serial0 exists")
    for cmd in (
        ["pinctrl", "get", "14", "15"],
        ["raspi-gpio", "get", "14,15"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip())
            break

    print(
        "UART wiring (switches BOTH OFF):\n"
        "  Use the pins printed TX and RX on the PN532, not SDA/SCL.\n"
        "  VCC -> Pi 5V pin 2, GND -> pin 6\n"
        "  PN532 TX -> Pi RX pin 10\n"
        "  PN532 RX -> Pi TX pin 8"
    )


def _open_uart() -> PN532:
    try:
        import serial  # noqa: F401
    except ImportError as error:
        raise PN532Error(
            "UART mode needs pyserial. Install with: python3 -m pip install pyserial"
        ) from error

    _print_uart_status()
    last_error = None
    bauds = [int(os.getenv("PN532_UART_BAUD", "115200")), 9600]
    for port in _uart_ports():
        for baud in bauds:
            print(f"Trying PN532 UART on {port} at {baud} baud...")
            reader = None
            try:
                reader = PN532(PN532UART(port=port, baudrate=baud))
                version = reader.get_firmware_version()
                reader.sam_configuration()
                _print_firmware(f"uart {port} @{baud}", version)
                return reader
            except Exception as error:
                last_error = error
                print(f"PN532 uart {port} @{baud} failed: {error}")
                if reader is not None:
                    reader.close()

    raise PN532Error(
        str(last_error)
        if last_error
        else "PN532 UART was not found."
    )


def _open_pn532() -> PN532:
    requested = os.getenv("PN532_INTERFACE", "auto").strip().lower() or "auto"

    if requested == "spi":
        raise PN532Error("SPI is not used. Use I2C or UART.")

    if requested == "auto":
        attempts = ["i2c", "uart"]
    elif requested in {"i2c", "uart"}:
        attempts = [requested]
    else:
        raise PN532Error(f"Unknown PN532_INTERFACE={requested}. Use auto, i2c, or uart.")

    last_error = None
    for name in attempts:
        try:
            if name == "i2c":
                return _open_i2c()
            return _open_uart()
        except Exception as error:
            last_error = error
            print(f"PN532 {name} failed: {error}")

    raise PN532Error(str(last_error) if last_error else "PN532 was not found.")


def _read_uid_from_pn532(timeout_seconds: float) -> str:
    reader = _open_pn532()
    deadline = None
    if timeout_seconds and timeout_seconds > 0:
        deadline = time.time() + timeout_seconds

    print("Waiting for RFID / NFC card tap...")
    last_reminder = time.time()

    try:
        while True:
            uid = reader.read_passive_target(timeout=0.5)
            if uid:
                rfid_id = _uid_to_id(uid)
                print("RFID card UID:", rfid_id)
                return rfid_id

            now = time.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError("No RFID card was tapped in time.")

            if now - last_reminder >= 15:
                print("Still waiting for a card tap...")
                last_reminder = now
    finally:
        reader.close()


def _read_uid_from_keyboard() -> str:
    rfid_id = input("Enter RFID card ID: ").strip()
    if not rfid_id:
        rfid_id = "TEST_RFID_001"
    print("RFID card ID:", rfid_id)
    return rfid_id


def keyboard_fallback_allowed() -> bool:
    flag = os.getenv("RFID_ALLOW_KEYBOARD", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    return "windows" in platform.system().lower()


def read_rfid_uid(timeout_seconds: Optional[float] = None) -> str:
    """
    Return a stable card id from a PN532 tap.
    On Windows, or with RFID_ALLOW_KEYBOARD=1, falls back to typed input.
    """
    if timeout_seconds is None:
        timeout_seconds = float(os.getenv("PN532_TIMEOUT", "0"))

    if keyboard_fallback_allowed() and os.getenv("PN532_INTERFACE", "").strip().lower() == "keyboard":
        return _read_uid_from_keyboard()

    try:
        return _read_uid_from_pn532(timeout_seconds)
    except Exception as error:
        print("PN532 read failed:", error)
        if keyboard_fallback_allowed():
            print("Falling back to keyboard RFID input.")
            return _read_uid_from_keyboard()
        raise


if __name__ == "__main__":
    try:
        print(read_rfid_uid())
    except Exception as error:
        print("RFID test failed:", error)
        raise SystemExit(1) from error
