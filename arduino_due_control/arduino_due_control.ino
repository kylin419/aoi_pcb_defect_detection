/*
 * AOI PCB Defect Detection System - Arduino Due Industrial Controller
 * -------------------------------------------------------------------
 * 工業級邊緣運算 AOI PCB 瑕疵檢測系統 - Arduino Due 硬體控制器 (正式產線規律版)
 * 
 * 硬體平台: Arduino Due (Atmel SAM3X8E ARM Cortex-M3, 3.3V 邏輯準位)
 * 通訊介面: 雙序列埠支援 (Programming Port & Native USB Port, 115200 bps)
 * 
 * 工業級防護特色:
 * 1. A4988 步進馬達 (Stepper Motor) 脈衝與方向控制
 * 2. 雙向心跳包斷線保護 (Fail-Safe Heartbeat Protection - Jetson 當機/斷線自動防護停機)
 * 3. 預設安全判斷 (Default-to-NG Failsafe)
 * 4. PCB 鏡頭正中央定位與單張遲滯防重覆觸發機制 (Hysteresis Trigger)
 * 5. 分流機構 (OK/NG 伺服馬達 / 氣壓電磁閥)
 * 6. 三色警示燈 (Tower Light) 與蜂鳴器 (Buzzer) 控制
 */

#include <Servo.h>

// ==========================================
// 硬體腳位定義 (Pin Definitions for Arduino Due)
// 註: Arduino Due 邏輯準位為 3.3V！
// ==========================================

// 感測器與按鈕腳位
const int PIN_IR_SENSOR        = 2;  // PCB 到位紅外線感測器 (遮擋觸發 LOW / NPN Input)
const int PIN_SENSOR_TRIGGER   = PIN_IR_SENSOR; // 相容性別名
const int PIN_EMERGENCY_STOP   = 3;  // 緊急停止按鈕 (內部上拉, Press -> LOW)

// 輸送帶步進馬達驅動器腳位 (A4988 驅動模組)
const int PIN_STEPPER_STEP     = 4;  // A4988 STEP 脈衝腳位
const int PIN_STEPPER_DIR      = 5;  // A4988 DIR 方向腳位
const int PIN_STEPPER_ENABLE   = 6;  // A4988 ENABLE (EN) 使能腳位 (LOW = Enable, HIGH = Disable)

// 分流機構腳位 (OK/NG 分流)
const int PIN_SERVO_FLAP       = 7;  // OK/NG 擋板伺服馬達 PWM 腳位
const int PIN_NG_CYLINDER      = 8;  // NG 氣壓缸/電磁閥繼電器腳位

// 三色警示燈與蜂鳴器腳位
const int PIN_LED_GREEN        = 9;  // 綠燈 (系統正常 / OK)
const int PIN_LED_YELLOW       = 10; // 黃燈 (待機 / 辨識中)
const int PIN_LED_RED          = 11; // 紅燈 (瑕疵 / NG / 異常)
const int PIN_BUZZER           = 12; // 蜂鳴器

// ==========================================
// 系統參數配置 (System Configuration Parameters)
// ==========================================
const long SERIAL_BAUD_RATE    = 115200; // 序列埠傳輸速率

// 伺服馬達角度 (Servo Angles for OK / NG paths)
const int SERVO_ANGLE_OK       = 0;   // OK 直行角度 (度)
const int SERVO_ANGLE_NG       = 60;  // NG 偏轉/推送角度 (度)

// 步進馬達速度與脈衝設定 (Stepper Motor Settings)
const int STEPPER_PULSE_DELAY_US = 1000; // 脈衝週期 (微秒, NEMA17 建議設定 800~1500us 避免失步抖動)

// 紅外線感測器觸發邏輯位準設定 (預設遮擋為 LOW；若感測器相反，可改為 HIGH)
const int IR_TRIGGER_STATE     = LOW; 

// 動作計時器 (Timers in milliseconds)
const unsigned long TIMEOUT_INSPECTION_MS   = 15000; // 等待 Jetson 辨識逾時 (15秒)
const unsigned long DURATION_SORT_OK_MS     = 1500;  // OK 板通過時間
const unsigned long DURATION_SORT_NG_MS     = 2000;  // NG 板推離時間
const unsigned long DURATION_BUZZER_MS      = 300;   // NG 蜂鳴器響起時間

// 🛡️【工業級 Fail-Safe 心跳包斷線保護】
// 若在運轉中超過此時間未收到 Jetson 發送的心跳包，將自動強制停機防護 (預設 5000ms = 5秒)
const unsigned long HEARTBEAT_TIMEOUT_MS    = 5000;  

// ==========================================
// 系統狀態機枚舉 (State Machine Enums)
// ==========================================
enum SystemState {
  STATE_IDLE,             // 待機狀態 (Standby)
  STATE_CONVEYOR_RUNNING, // 輸送帶運轉中 (Scanning for PCB)
  STATE_WAIT_INSPECTION,  // PCB 到位停機，等待 Jetson 辨識結果 (Waiting for Jetson OK/NG)
  STATE_SORT_OK,          // 處置 OK 板 (OK Pass Through)
  STATE_SORT_NG,          // 處置 NG 板 (NG Divert / Eject)
  STATE_EMERGENCY_STOP    // 緊急停止狀態 (Emergency Stop / Communication Fail-Safe)
};

// 全域變數
SystemState currentState = STATE_IDLE;
Servo sorterServo;

bool systemEnabled = true;          // 系統啟用開關
unsigned long stateTimer = 0;        // 狀態計時器
unsigned long lastStepTime = 0;      // 馬達脈衝計時器
unsigned long buzzerTimer = 0;      // 蜂鳴器計時器
unsigned long lastHeartbeatTime = 0; // 最後一次收到心跳包的時間點
bool buzzerActive = false;

// 序列埠指令緩衝區
String inputBuffer = "";

// ==========================================
// 函式宣告 (Function Prototypes)
// ==========================================
void setupPins();
void handleSerialInput();
void parseCommand(String cmd);
void runStateMachine();
void stepMotorNonBlocking();
void triggerBuzzer(unsigned long durationMs);
void updateBuzzer();
void setTowerLight(bool green, bool yellow, bool red);
void emergencyStop();
void checkHeartbeatSafety();

// 雙序列埠輸出輔助函式 (同時支援 Programming Port & Native USB Port)
void sendSerialPrintln(String msg) {
  Serial.println(msg);
  SerialUSB.println(msg);
}

void sendSerialPrint(String msg) {
  Serial.print(msg);
  SerialUSB.print(msg);
}

// ==========================================
// Arduino Setup 初始設定
// ==========================================
void setup() {
  // 同時初始化 Programming Port (Serial) 與 Native USB Port (SerialUSB)
  Serial.begin(SERIAL_BAUD_RATE);
  SerialUSB.begin(SERIAL_BAUD_RATE);
  inputBuffer.reserve(64);

  // 初始化硬體腳位
  setupPins();

  // 初始化伺服馬達
  sorterServo.attach(PIN_SERVO_FLAP);
  sorterServo.write(SERVO_ANGLE_OK);

  // 初始燈光狀態：黃燈亮 (待機中)
  setTowerLight(false, true, false);

  lastHeartbeatTime = millis();

  // 傳送開機完成訊號至雙序列埠
  sendSerialPrintln("STATUS:READY");
  sendSerialPrintln("INFO:Arduino Due AOI Controller Initialized (Fail-Safe Active)");
}

// ==========================================
// Arduino Main Loop 主迴圈
// ==========================================
void loop() {
  // 1. 檢查緊急停止按鈕
  if (digitalRead(PIN_EMERGENCY_STOP) == LOW) {
    emergencyStop();
  }

  // 2. 處理序列埠命令 (同時檢查兩個 USB 孔)
  handleSerialInput();

  // 3. 更新蜂鳴器狀態
  updateBuzzer();

  // 4. 工業級心跳包斷線檢查
  checkHeartbeatSafety();

  // 5. 執行主控制狀態機
  if (systemEnabled && currentState != STATE_EMERGENCY_STOP) {
    runStateMachine();
  }
}

// ==========================================
// 硬體腳位初始化
// ==========================================
void setupPins() {
  pinMode(PIN_IR_SENSOR, INPUT_PULLUP);
  pinMode(PIN_EMERGENCY_STOP, INPUT_PULLUP);

  pinMode(PIN_STEPPER_STEP, OUTPUT);
  pinMode(PIN_STEPPER_DIR, OUTPUT);
  pinMode(PIN_STEPPER_ENABLE, OUTPUT);

  pinMode(PIN_NG_CYLINDER, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // 預設使能馬達 (LOW 電位使能)
  digitalWrite(PIN_STEPPER_ENABLE, LOW);
  digitalWrite(PIN_STEPPER_DIR, HIGH); // 正轉進料方向
  digitalWrite(PIN_NG_CYLINDER, LOW);  // 氣壓缸收回
  digitalWrite(PIN_BUZZER, LOW);
}

// ==========================================
// 序列埠資料讀取與解析 (同時監聽 Serial 與 SerialUSB)
// ==========================================
void handleSerialInput() {
  // 1. 讀取 Programming Port (Serial)
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        inputBuffer.trim();
        parseCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += inChar;
    }
  }

  // 2. 讀取 Native USB Port (SerialUSB)
  while (SerialUSB.available()) {
    char inChar = (char)SerialUSB.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        inputBuffer.trim();
        parseCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += inChar;
    }
  }
}

void parseCommand(String cmd) {
  cmd.toUpperCase();
  
  // 收到任何指令時，更新心跳時間戳記
  lastHeartbeatTime = millis();

  if (cmd == "OK" || cmd == "CMD:OK" || cmd == "RESULT:OK") {
    if (currentState == STATE_WAIT_INSPECTION) {
      sendSerialPrintln("ACK:OK");
      currentState = STATE_SORT_OK;
      stateTimer = millis();
    }
  } 
  else if (cmd == "NG" || cmd == "CMD:NG" || cmd == "RESULT:NG") {
    if (currentState == STATE_WAIT_INSPECTION) {
      sendSerialPrintln("ACK:NG");
      currentState = STATE_SORT_NG;
      stateTimer = millis();
      triggerBuzzer(DURATION_BUZZER_MS);
    }
  }
  else if (cmd == "START" || cmd == "CMD:START") {
    systemEnabled = true;
    currentState = STATE_CONVEYOR_RUNNING;
    setTowerLight(true, false, false);
    sendSerialPrintln("STATUS:RUNNING");
  }
  else if (cmd == "STOP" || cmd == "CMD:STOP") {
    currentState = STATE_IDLE;
    setTowerLight(false, true, false);
    sendSerialPrintln("STATUS:IDLE");
  }
  else if (cmd == "RESET" || cmd == "CMD:RESET") {
    currentState = STATE_IDLE;
    digitalWrite(PIN_NG_CYLINDER, LOW);
    sorterServo.write(SERVO_ANGLE_OK);
    setTowerLight(false, true, false);
    sendSerialPrintln("STATUS:RESET_COMPLETE");
  }
  else if (cmd == "PING" || cmd == "HEARTBEAT") {
    sendSerialPrintln("PONG");
  }
  else if (cmd == "READ_IR" || cmd == "IR_STATUS" || cmd == "CMD:IR") {
    int irVal = digitalRead(PIN_IR_SENSOR);
    if (irVal == IR_TRIGGER_STATE) {
      sendSerialPrintln("INFO:IR_SENSOR_STATE:TRIGGERED (PCB Detected)");
    } else {
      sendSerialPrintln("INFO:IR_SENSOR_STATE:CLEAR (No PCB)");
    }
  }
  else {
    sendSerialPrint("ERR:UNKNOWN_COMMAND:");
    sendSerialPrintln(cmd);
  }
}

// ==========================================
// 工業級心跳包斷線檢查 (Fail-Safe Safety Protection)
// ==========================================
void checkHeartbeatSafety() {
  // 僅在運轉中進行斷線心跳包檢測
  if (currentState == STATE_CONVEYOR_RUNNING || currentState == STATE_WAIT_INSPECTION) {
    if (millis() - lastHeartbeatTime > HEARTBEAT_TIMEOUT_MS) {
      sendSerialPrintln("ALARM:JETSON_COMMUNICATION_LOST_FAILSAFE_STOP");
      emergencyStop();
    }
  }
}

// ==========================================
// 系統狀態機核心邏輯 (System State Machine)
// ==========================================
void runStateMachine() {
  switch (currentState) {

    case STATE_IDLE:
      // 待機中，馬達不發出脈衝
      setTowerLight(false, true, false);
      break;

    case STATE_CONVEYOR_RUNNING:
      // 1. 運轉輸送帶 (綠燈亮)
      stepMotorNonBlocking();
      setTowerLight(true, false, false);

      // 2. 只要紅外線一遮擋，馬達立刻 100% 停止並切換黃燈！
      if (digitalRead(PIN_IR_SENSOR) == IR_TRIGGER_STATE) {
        currentState = STATE_WAIT_INSPECTION;
        stateTimer = millis();
        setTowerLight(false, true, false); // 切換黃燈 (辨識中)

        sendSerialPrintln("INFO:IR_SENSOR_TRIGGERED:PCB_DETECTED");
        sendSerialPrintln("INFO:STEPPER_MOTOR_STOPPED");
        sendSerialPrintln("TRIGGER");
        sendSerialPrintln("STATUS:WAIT_INSPECTION");
      }
      break;

    case STATE_WAIT_INSPECTION:
      // 【步進馬達 100% 完全停止，黃燈亮】
      setTowerLight(false, true, false);

      // 等待 Jetson 鏡頭與 YOLOv12 辨識完成傳回結果 ("OK" 或 "NG")
      // 預設 Fail-Safe 安全原則：若超過 TIMEOUT 時間未收到結果，自動判斷為 NG 進行防護處理
      if (millis() - stateTimer > TIMEOUT_INSPECTION_MS) {
        sendSerialPrintln("WARN:INSPECTION_TIMEOUT_FAILSAFE_NG");
        currentState = STATE_SORT_NG;
        stateTimer = millis();
        triggerBuzzer(DURATION_BUZZER_MS * 2);
      }
      break;

    case STATE_SORT_OK:
      // 處置良品 (OK): 復位擋板，運轉輸送帶將板子送出 (綠燈亮)
      sorterServo.write(SERVO_ANGLE_OK);
      digitalWrite(PIN_NG_CYLINDER, LOW);
      setTowerLight(true, false, false);

      stepMotorNonBlocking();

      if (millis() - stateTimer > DURATION_SORT_OK_MS) {
        sendSerialPrintln("STATUS:SORT_OK_DONE");
        currentState = STATE_CONVEYOR_RUNNING;
      }
      break;

    case STATE_SORT_NG:
      // 處置瑕疵品 (NG): 轉動擋板 / 啟動氣壓缸推離，紅燈亮起，馬達 100% 保持停止
      sorterServo.write(SERVO_ANGLE_NG);
      digitalWrite(PIN_NG_CYLINDER, HIGH); // 推送 NG 板
      setTowerLight(false, false, true);   // 紅燈亮
      // （注意：此處不呼叫 stepMotorNonBlocking()，馬達 100% 保持停止保護）
      break;

    case STATE_EMERGENCY_STOP:
      // 急停/斷線鎖死狀態，需傳送 RESET 指令復歸
      setTowerLight(false, false, true);
      digitalWrite(PIN_STEPPER_ENABLE, HIGH); // 關閉馬達驅動器
      break;
  }
}

// ==========================================
// 步進馬達非阻塞脈衝產生器 (Non-blocking Stepper Control)
// ==========================================
void stepMotorNonBlocking() {
  unsigned long currentMicros = micros();
  if (currentMicros - lastStepTime >= STEPPER_PULSE_DELAY_US) {
    lastStepTime = currentMicros;
    digitalWrite(PIN_STEPPER_STEP, HIGH);
    delayMicroseconds(2); // 脈衝最小寬度 2us
    digitalWrite(PIN_STEPPER_STEP, LOW);
  }
}

// ==========================================
// 警示燈控制 (Tower Light Control)
// ==========================================
void setTowerLight(bool green, bool yellow, bool red) {
  digitalWrite(PIN_LED_GREEN, green ? HIGH : LOW);
  digitalWrite(PIN_LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(PIN_LED_RED, red ? HIGH : LOW);
}

// ==========================================
// 蜂鳴器控制 (Buzzer Alarm)
// ==========================================
void triggerBuzzer(unsigned long durationMs) {
  digitalWrite(PIN_BUZZER, HIGH);
  buzzerActive = true;
  buzzerTimer = millis() + durationMs;
}

void updateBuzzer() {
  if (buzzerActive && millis() >= buzzerTimer) {
    digitalWrite(PIN_BUZZER, LOW);
    buzzerActive = false;
  }
}

// ==========================================
// 緊急停止處理 (Emergency Stop Handler)
// ==========================================
void emergencyStop() {
  currentState = STATE_EMERGENCY_STOP;
  digitalWrite(PIN_STEPPER_ENABLE, HIGH); // 關閉馬達驅動器
  digitalWrite(PIN_NG_CYLINDER, LOW);
  digitalWrite(PIN_BUZZER, HIGH);
  setTowerLight(false, false, true);
  sendSerialPrintln("ALARM:EMERGENCY_STOP_ACTIVATED");
}
