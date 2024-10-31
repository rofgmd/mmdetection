# SerialCommunicate.py
# Implements serial communication functionalities for ORB_SLAM2 in Python.
# Translated from C++ by hyhh on 2021/10/27.

import serial

# Serial port setup
serial_port = serial.Serial(
    port='/dev/ttyUSB0',  # Ubuntu上的串口设备路径
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

# Communication constants and variables
header = bytes([0xAA, 0xBB])
ender = bytes([0xEE, 0xFF])
header1 = bytes([0xAA, 0xBB, 0x1E, 0xEE, 0xFF])
header2 = bytes([0xAA, 0xBB, 0x1E, 0x10, 0x10, 0xEE, 0xFF])

# Union-like functionality using int to bytes conversion
def to_bytes(value):
    """Convert a short integer to a 2-byte representation."""
    return value.to_bytes(2, byteorder='little', signed=True)

def from_bytes(b):
    """Convert a 2-byte array back to a short integer."""
    return int.from_bytes(b, byteorder='little', signed=True)

# Initialize serial port (already done in serial.Serial instantiation above)
def serial_init():
    if not serial_port.is_open:
        serial_port.open()

# Send control speeds to the robot's wheels
def write_speed(angle, left_v, right_v, speed):
    buf = bytearray(8)
    
    # Setting message header
    buf[0:2] = header
    buf[2] = int(angle) & 0xFF
    buf[3] = int(left_v) & 0xFF
    buf[4] = int(right_v) & 0xFF
    buf[5] = int(speed) & 0xFF
    
    # Setting message tail
    buf[6:8] = ender
    
    # Send data through serial port
    serial_port.write(buf)

# Compute an 8-bit cyclic redundancy check (CRC)
def get_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc
