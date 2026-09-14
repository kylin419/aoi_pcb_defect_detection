/*
 * AOI PCB Defect Detection System - Arduino Due Controller
 * --------------------------------------------------------
 * 邊緣運算 AOI PCB 瑕疵檢測系統 - Arduino Due 硬體控制器
 * 
 * 硬體平台: Arduino Due (Atmel SAM3X8E ARM Cortex-M3, 3.3V 邏輯準位)
 * 通訊介面: UART Serial (115200 bps)
 * 
 * 功能描述:
 * 1. 輸送帶步進馬達 (Stepper Motor) 脈衝與方向控制
 * 2. PCB 到位光電感測器 (Photoelectric Sensor) 觸發
 * 3. 分流機構 controls (OK/NG 伺服馬達 / 氣壓電磁閥)
 * 4. 三色警示燈 (Tower Light) 與蜂鳴器 (Buzzer) 控制
 * 5. 非阻塞 (Non-blocking) 狀態機與 millis() 時間調度
 */

#include <Servo.h>

// ==========================================
// 硬體腳位定義 (Pin Definitions for Arduino Due)
// 註: Arduino Due 邏輯準位為 3.3V！
// ==========================================

// 感測器與按鈕腳位
const int PIN_IR_SENSOR = 2;                   // PCB 到位紅外線感測器 (遮擋觸發 LOW / NPN Input)
const int PIN_SENSOR_TRIGGER = PIN_IR_SENSOR;  // 相容性別名
const int PIN_EMERGENCY_STOP = 3;              // 緊急停止按鈕 (內部上拉, Press -> LOW)

// 輸送帶步進馬達驅動器腳位 (如 TB6600 / DM542)
const int PIN_STEPPER_STEP = 4;    // PUL+ / STEP 脈衝腳位
const int PIN_STEPPER_DIR = 5;     // DIR+ / DIR 方向腳位
const int PIN_STEPPER_ENABLE = 6;  // ENA+ / ENABLE 使能腳位 (LOW = Enable)

// 分流機構腳位 (OK/NG 分流)
const int PIN_SERVO_FLAP = 7;   // OK/NG 擋板伺服馬達 PWM 腳位
const int PIN_NG_CYLINDER = 8;  // NG 氣壓缸/電磁閥繼電器腳位

// 三色警示燈與蜂鳴器腳位
const int PIN_LED_GREEN = 9;    // 綠燈 (系統正常 / OK)
const int PIN_LED_YELLOW = 10;  // 黃燈 (待機 / 辨識中)
const int PIN_LED_RED = 11;     // 紅燈 (瑕疵 / NG / 異常)
const int PIN_BUZZER = 12;      // 蜂鳴器

// ==========================================
// 系統參數配置 (System Configuration Parameters)
// ==========================================
const long SERIAL_BAUD_RATE = 115200;  // 序列埠傳輸速率

// 伺服馬達角度 (Servo Angles for OK / NG paths)
const int SERVO_ANGLE_OK = 0;   // OK 直行角度 (度)
const int SERVO_ANGLE_NG = 60;  // NG 偏轉/推送角度 (度)

// 步進馬達速度與脈衝設定 (Stepper Motor Settings)
const int STEPPER_PULSE_DELAY_US = 400;  // 脈衝週期 (微秒), 越小速度越快

// 動作計時器 (Timers in milliseconds)
const unsigned long TIMEOUT_INSPECTION_MS = 5000;  // 等待 Jetson 辨識逾時 (5秒)
const unsigned long DURATION_SORT_OK_MS = 1500;    // OK 板通過時間
const unsigned long DURATION_SORT_NG_MS = 2000;    // NG 板推離時間
const unsigned long DURATION_BUZZER_MS = 300;      // NG 蜂鳴器響起時間

// ==========================================
// 系統狀態機枚舉 (State Machine Enums)
// ==========================================
enum SystemState {
  STATE_IDLE,              // 待機狀態 (Standby)
  STATE_CONVEYOR_RUNNING,  // 輸送帶運轉中 (Scanning for PCB)
  STATE_WAIT_INSPECTION,   // PCB 到位，等待 Jetson 辨識結果 (Waiting for Jetson OK/NG)
  STATE_SORT_OK,           // 處置 OK 板 (OK Pass Through)
  STATE_SORT_NG,           // 處置 NG 板 (NG Divert / Eject)
  STATE_EMERGENCY_STOP     // 緊急停止狀態 (Emergency Stop)
};

// 全域變數
SystemState currentState = STATE_IDLE;
Servo sorterServo;

bool systemEnabled = true;       // 系統啟用開關
unsigned long stateTimer = 0;    // 狀態計時器
unsigned long lastStepTime = 0;  // 馬達脈衝計時器
unsigned long buzzerTimer = 0;   // 蜂鳴器計時器
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

// ==========================================
// Arduino Setup 初始設定
// ==========================================
void setup() {
  // 初始化序列埠 (與 Jetson Orin Nano 連線)
  Serial.begin(SERIAL_BAUD_RATE);
  inputBuffer.reserve(64);

  // 初始化硬體腳位
  setupPins();

  // 初始化伺服馬達
  sorterServo.attach(PIN_SERVO_FLAP);
  sorterServo.write(SERVO_ANGLE_OK);

  // 初始燈光狀態：黃燈亮 (待機中)
  setTowerLight(false, true, false);

  // 傳送開機完成訊號給 Jetson
  Serial.println("STATUS:READY");
  Serial.println("INFO:Arduino Due AOI Controller Initialized");
}

// ==========================================
// Arduino Main Loop 主迴圈
// ==========================================
void loop() {
  // 1. 檢查緊急停止按鈕
  if (digitalRead(PIN_EMERGENCY_STOP) == LOW) {
    emergencyStop();
  }

  // 2. 處理序列埠命令
  handleSerialInput();

  // 3. 更新蜂鳴器狀態
  updateBuzzer();

  // 4. 執行主控制狀態機
  if (systemEnabled && currentState != STATE_EMERGENCY_STOP) {
    runStateMachine();
  }
}

// ==========================================
// 硬體腳位初始化
// ==========================================
void setupPins() {
  pinMode(PIN_SENSOR_TRIGGER, INPUT_PULLUP);
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
  digitalWrite(PIN_STEPPER_DIR, HIGH);  // 正轉進料方向
  digitalWrite(PIN_NG_CYLINDER, LOW);   // 氣壓缸收回
  digitalWrite(PIN_BUZZER, LOW);
}

// ==========================================
// 序列埠資料讀取與解析 (Serial Communication)
// ==========================================
void handleSerialInput() {
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
}

void parseCommand(String cmd) {
  cmd.toUpperCase();

  if (cmd == "OK" || cmd == "CMD:OK" || cmd == "RESULT:OK") {
    if (currentState == STATE_WAIT_INSPECTION) {
      Serial.println("ACK:OK");
      currentState = STATE_SORT_OK;
      stateTimer = millis();
    }
  } else if (cmd == "NG" || cmd == "CMD:NG" || cmd == "RESULT:NG") {
    if (currentState == STATE_WAIT_INSPECTION) {
      Serial.println("ACK:NG");
      currentState = STATE_SORT_NG;
      stateTimer = millis();
      triggerBuzzer(DURATION_BUZZER_MS);
    }
  } else if (cmd == "START" || cmd == "CMD:START") {
    systemEnabled = true;
    currentState = STATE_CONVEYOR_RUNNING;
    setTowerLight(true, false, false);
    Serial.println("STATUS:RUNNING");
  } else if (cmd == "STOP" || cmd == "CMD:STOP") {
    currentState = STATE_IDLE;
    setTowerLight(false, true, false);
    Serial.println("STATUS:IDLE");
  } else if (cmd == "RESET" || cmd == "CMD:RESET") {
    currentState = STATE_IDLE;
    digitalWrite(PIN_NG_CYLINDER, LOW);
    sorterServo.write(SERVO_ANGLE_OK);
    setTowerLight(false, true, false);
    Serial.println("STATUS:RESET_COMPLETE");
  } else if (cmd == "PING") {
    Serial.println("PONG");
  } else {
    Serial.print("ERR:UNKNOWN_COMMAND:");
    Serial.println(cmd);
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
      // 1. 運轉輸送帶
      stepMotorNonBlocking();
      setTowerLight(true, false, false);

      // 2. 偵測是否有 PCB 板到達檢測點
      if (digitalRead(PIN_SENSOR_TRIGGER) == LOW) {
        // PCB 到位，向 Jetson 發送觸發訊號
        Serial.println("TRIGGER");
        Serial.println("STATUS:WAIT_INSPECTION");

        currentState = STATE_WAIT_INSPECTION;
        stateTimer = millis();
        setTowerLight(false, true, false);  // 黃燈 (辨識中)
      }
      break;

    case STATE_WAIT_INSPECTION:
      // 【步進馬達停止】此狀態下不呼叫 stepMotorNonBlocking()，馬達保持靜止
      // 等待 Jetson 鏡頭與 YOLOv12 辨識完成傳回結果 ("OK" 或 "NG")
      // 若超過 TIMEOUT 時間未收到結果，自動判斷為 NG 進行保護保護處理
      if (millis() - stateTimer > TIMEOUT_INSPECTION_MS) {
        Serial.println("WARN:INSPECTION_TIMEOUT");
        currentState = STATE_SORT_NG;
        stateTimer = millis();
        triggerBuzzer(DURATION_BUZZER_MS * 2);
      }
      break;

    case STATE_SORT_OK:
      // 處置良品 (OK): 復位擋板，運轉輸送帶將板子送出
      sorterServo.write(SERVO_ANGLE_OK);
      digitalWrite(PIN_NG_CYLINDER, LOW);
      setTowerLight(true, false, false);

      stepMotorNonBlocking();

      if (millis() - stateTimer > DURATION_SORT_OK_MS) {
        Serial.println("STATUS:SORT_OK_DONE");
        currentState = STATE_CONVEYOR_RUNNING;
      }
      break;

    case STATE_SORT_NG:
      // 處置瑕疵品 (NG): 轉動擋板 / 啟動氣壓缸推離
      sorterServo.write(SERVO_ANGLE_NG);
      digitalWrite(PIN_NG_CYLINDER, HIGH);  // 推送 NG 板
      setTowerLight(false, false, true);    // 紅燈亮

      stepMotorNonBlocking();

      if (millis() - stateTimer > DURATION_SORT_NG_MS) {
        // 處置完畢，復位機構
        digitalWrite(PIN_NG_CYLINDER, LOW);
        sorterServo.write(SERVO_ANGLE_OK);
        Serial.println("STATUS:SORT_NG_DONE");
        currentState = STATE_CONVEYOR_RUNNING;
      }
      break;

    case STATE_EMERGENCY_STOP:
      // 急停鎖死狀態，需傳送 RESET 指令復歸
      setTowerLight(false, false, true);
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
    delayMicroseconds(2);  // 脈衝最小寬度 2us
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
  digitalWrite(PIN_STEPPER_ENABLE, HIGH);  // 關閉馬達驅動器
  digitalWrite(PIN_NG_CYLINDER, LOW);
  digitalWrite(PIN_BUZZER, HIGH);
  setTowerLight(false, false, true);
  Serial.println("ALARM:EMERGENCY_STOP_ACTIVATED");
}
