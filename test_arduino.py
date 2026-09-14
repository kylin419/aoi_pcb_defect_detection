#!/usr/bin/env python3
"""
Arduino Due Interactive Serial Test Tool for Jetson Orin Nano
-------------------------------------------------------------
用來在 Jetson 終端機手動測試與 Arduino Due 通訊的互動式工具
"""

import sys
import time
import serial
import serial.tools.list_ports

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "ttyACM" in p.device or "ttyUSB" in p.device:
            return p.device
    return "/dev/ttyACM0"

def main():
    port = find_arduino_port()
    baud = 115200

    print("=========================================================")
    print("   Arduino Due Serial Interactive Test Tool (Jetson)    ")
    print("=========================================================")
    print(f"嘗試連線至串口: {port} (Baud Rate: {baud})")

    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)
        time.sleep(2.0)
        print("✅ 成功連接 Arduino Due！")
    except Exception as e:
        print(f"❌ 錯誤: 無法開啟序列埠 {port}: {e}")
        print("提示: 請確定已插上 USB 線，或嘗試執行 sudo usermod -a -G dialout $USER")
        sys.exit(1)

    print("\n可用的測試指令:")
    print("  START    - 啟動輸送帶運轉 (綠燈)")
    print("  STOP     - 停止輸送帶 (黃燈)")
    print("  OK       - 傳送良品 OK 訊號 (輸送帶繼續送出)")
    print("  NG       - 傳送瑕疵 NG 訊號 (擋板偏轉/紅燈警報)")
    print("  RESET    - 重置復位機構與警報")
    print("  READ_IR  - 讀取當前紅外線感測器狀態")
    print("  PING     - 測試通訊連線 (回應 PONG)")
    print("  exit     - 離開測試程式")
    print("---------------------------------------------------------\n")

    try:
        while True:
            cmd = input("輸入指令 > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                break

            # 傳送指令至 Arduino
            ser.write(f"{cmd.upper()}\n".encode("utf-8"))
            ser.flush()

            # 讀取 Arduino 回覆
            time.sleep(0.1)
            while ser.in_waiting:
                response = ser.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    print(f"  [Arduino 回覆]: {response}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n序列埠連線已關閉。")

if __name__ == "__main__":
    main()
