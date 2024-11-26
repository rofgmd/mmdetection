import time
from mmdet.serial_util.control_car import send_command
from mmdet.serial_util.SerialCommunicate import serial_init

def main():
    serial_init()
    time.sleep(1)

    for a in range(5):
        send_command(0,200,200, 200)

    send_command(0,0,0,0)
    time.sleep(1)

if __name__ == "__main__":
    main()

