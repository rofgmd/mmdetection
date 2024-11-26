# ConsoleApplication3.py
# Python translation of the main function for sending serial commands

import time
from mmdet.serial_util.SerialCommunicate import serial_init, write_speed

def send_command(a, b, c, d):
    """Send command to microcontroller."""
    for _ in range(10):  # Send command multiple times to ensure it's received
        write_speed(a, b, c, d)
        time.sleep(1)  # 1 s delay

def main():
    serial_init()  # Initialize and open serial port
    time.sleep(3)  # Wait for 3 seconds
    
    send_command(30, 0, 0, 0)  # Initial command
    time.sleep(1)
    
    for a in range(5):
        if a == 2 or a == 3:
            send_command(30, 150, 150, 150)
        else:
            send_command(90, 150, 150, 150)
        time.sleep(1)

    send_command(0, 200, 200, 200)  # Final command
    time.sleep(1)

    send_command(0, 0, 0, 0)  # Final command
    time.sleep(1)

if __name__ == "__main__":
    main()
