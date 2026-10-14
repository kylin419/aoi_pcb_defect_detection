import sys
def rep(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == count, f"{path}: expected {count} got {s.count(old)} for {old[:60]!r}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new))

# ---------------- Arduino ----------------
A = "arduino_due_control/arduino_due_control.ino"
rep(A, "// 緊急停止按鈕 (內部上拉, Press -> LOW)", "// 緊急停止按鈕 (常閉NC接GND, 內部上拉; 按下/斷線 -> HIGH)")
rep(A, "void emergencyStop();", "void emergencyStop(const char* reason);\nvoid resetFromAlarm();\nbool estopPressed();\nunsigned long lastAlarmReport = 0;")
rep(A, """  if (digitalRead(PIN_EMERGENCY_STOP) == HIGH) {
    emergencyStop();
  }""", """  if (estopPressed()) {
    emergencyStop("EMERGENCY_STOP_ACTIVATED");
  }""")
rep(A, """  if (systemEnabled && currentState != STATE_EMERGENCY_STOP) {
    runStateMachine();
  }""", """  if (systemEnabled && currentState != STATE_EMERGENCY_STOP) {
    runStateMachine();
  }

  // 7. 警報期間每2秒回報一次, 讓 GUI 保持警報顯示
  if (currentState == STATE_EMERGENCY_STOP && millis() - lastAlarmReport >= 2000) {
    lastAlarmReport = millis();
    sendSerialPrintln("ALARM:STILL_ACTIVE");
  }""")
rep(A, """  lastHeartbeatTime = millis();

  if (cmd == "OK" """, """  lastHeartbeatTime = millis();

  // 警報狀態: 只接受 RESET / PING, 其餘拒絕
  if (currentState == STATE_EMERGENCY_STOP) {
    if (cmd == "RESET" || cmd == "CMD:RESET") {
      resetFromAlarm();
    } else if (cmd == "PING" || cmd == "HEARTBEAT") {
      sendSerialPrintln("PONG");
    } else if (cmd == "STOP" || cmd == "CMD:STOP") {
      // 已經停止, 忽略
    } else {
      sendSerialPrintln("ERR:IN_ALARM_SEND_RESET");
    }
    return;
  }

  if (cmd == "OK" """)
rep(A, """    systemEnabled = true;
    digitalWrite(PIN_NG_CYLINDER, LOW);""", """    systemEnabled = true;
    digitalWrite(PIN_STEPPER_ENABLE, LOW);   // 確保驅動器已重新使能
    digitalWrite(PIN_NG_CYLINDER, LOW);""")
rep(A, """      sendSerialPrintln("ALARM:JETSON_COMMUNICATION_LOST_FAILSAFE_STOP");
      emergencyStop();""", """      emergencyStop("JETSON_COMMUNICATION_LOST_FAILSAFE_STOP");""")
rep(A, """void emergencyStop() {
  currentState = STATE_EMERGENCY_STOP;
  digitalWrite(PIN_STEPPER_ENABLE, HIGH); // 關閉馬達驅動器
  digitalWrite(PIN_NG_CYLINDER, LOW);
  digitalWrite(PIN_BUZZER, HIGH);
  setTowerLight(false, false, true);
  sendSerialPrintln("ALARM:EMERGENCY_STOP_ACTIVATED");
}""", """// 常閉(NC)接線: 按下或斷線 -> HIGH
bool estopPressed() {
  return digitalRead(PIN_EMERGENCY_STOP) == HIGH;
}

void emergencyStop(const char* reason) {
  if (currentState == STATE_EMERGENCY_STOP) return;   // 只觸發一次
  currentState = STATE_EMERGENCY_STOP;
  digitalWrite(PIN_STEPPER_ENABLE, HIGH); // 關閉馬達驅動器
  digitalWrite(PIN_NG_CYLINDER, LOW);
  digitalWrite(PIN_BUZZER, HIGH);
  buzzerActive = false;                   // 警報蜂鳴常響, 不被計時器關掉
  setTowerLight(false, false, true);
  lastAlarmReport = millis();
  sendSerialPrint("ALARM:");
  sendSerialPrintln(String(reason));
}

// 警報復歸: 緊急按鈕必須已放開才允許
void resetFromAlarm() {
  if (estopPressed()) {
    sendSerialPrintln("ERR:ESTOP_STILL_PRESSED");
    return;
  }
  digitalWrite(PIN_BUZZER, LOW);
  buzzerActive = false;
  digitalWrite(PIN_NG_CYLINDER, LOW);
  sorterServo.write(SERVO_ANGLE_OK);
  lastHeartbeatTime = millis();
  currentState = STATE_IDLE;
  setTowerLight(false, true, false);
  sendSerialPrintln("STATUS:RESET_COMPLETE");
}""")

# ---------------- Worker ----------------
W = "pcb_inspection/workers/inference_worker.py"
rep(W, "    ros_command_received = pyqtSignal(str)\n", "    ros_command_received = pyqtSignal(str)\n    alarm_changed = pyqtSignal(bool, str)\n    arduino_line = pyqtSignal(str)\n")
rep(W, "        self.running = False\n        \n        self.initialized = False", "        self.running = False\n        self.alarm_active = False\n        \n        self.initialized = False")
rep(W, """                feedback_callback=lambda msg: self.log.emit(f"Arduino: {msg}"),
            )""", """                feedback_callback=self.arduino_line.emit,   # reader thread -> queued to worker thread
            )
            self.arduino_line.connect(
                self._handle_arduino_line, Qt.ConnectionType.QueuedConnection
            )""")
rep(W, "    def _on_ros_command(self, cmd: str):", '''    # ------------------------------------------------------------------
    # Emergency-stop alarm handling
    # ------------------------------------------------------------------

    def _handle_arduino_line(self, line: str):
        if line == "ALARM:STILL_ACTIVE":
            if not self.alarm_active:
                self._enter_alarm("EMERGENCY_STOP_ACTIVATED")
            return
        self.log.emit(f"Arduino: {line}")
        if line.startswith("ALARM:"):
            self._enter_alarm(line[6:])
        elif line in ("STATUS:RESET_COMPLETE", "STATUS:READY") and self.alarm_active:
            self.alarm_active = False
            self.alarm_changed.emit(False, "")
            self.inspection_state.emit("IDLE")
            self.log.emit("Alarm cleared. System is IDLE; press Start to resume.")
        elif line == "ERR:ESTOP_STILL_PRESSED":
            self.error.emit("Emergency stop is still pressed / E-stop circuit open. Release it first.")

    def _enter_alarm(self, reason: str):
        was_active = self.alarm_active
        self.alarm_active = True
        self.running = False
        self.inspect_waiting = False
        self._centering_active = False
        self._centering_deadline = None
        self._defect_detection_deadline = None
        self.pcb_present = False
        self._ng_hold = False
        if not was_active:
            self.error.emit(f"ALARM: {reason} - conveyor stopped. Release E-stop, then press Reset Alarm.")
            self.stopped.emit()
        self.alarm_changed.emit(True, reason)

    @pyqtSlot()
    def reset_alarm(self):
        """Ask Arduino to clear the alarm; GUI clears only after Arduino confirms."""
        if not self.serial_bridge:
            self.alarm_active = False
            self.alarm_changed.emit(False, "")
            return
        self.log.emit("Sending RESET to Arduino (clear alarm)...")
        self.serial_bridge.send_reset()

    def _on_ros_command(self, cmd: str):''')
rep(W, """        if self.running:
            return

        self.running = True
        # Reset PCB""", """        if self.alarm_active:
            self.error.emit("Cannot start: emergency-stop alarm is active. Reset the alarm first.")
            return

        if self.running:
            return

        self.running = True
        # Reset PCB""")
rep(W, """        if self.serial_bridge:
            self.serial_bridge.send_reset()
        self.log.emit("Production yield statistics reset.")""", """        self.log.emit("Production yield statistics reset.")""")

# ---------------- Toolbar ----------------
T = "pcb_inspection/gui/widgets/toolbar.py"
rep(T, "    exit_clicked = pyqtSignal()\n", "    exit_clicked = pyqtSignal()\n    reset_alarm_clicked = pyqtSignal()\n    fullscreen_clicked = pyqtSignal()\n")
rep(T, "        self.exit_btn = self.create_button(", '''        self.alarm_btn = self.create_button(
            "🚨 Reset Alarm",
            "Clear emergency-stop alarm (release the E-stop button first)",
        )
        self.alarm_btn.setStyleSheet(
            "QPushButton{background:#C0392B;color:white;font-weight:bold;}"
            "QPushButton:hover{background:#E74C3C;}"
        )
        self.alarm_btn.setVisible(False)

        self.fullscreen_btn = self.create_button(
            "⛶ Fullscreen",
            "Toggle fullscreen (F11, Esc to exit)",
        )

        self.exit_btn = self.create_button(''')
rep(T, "        layout.addStretch()\n", "        layout.addWidget(self.alarm_btn)\n        layout.addStretch()\n")
rep(T, "        layout.addWidget(self.reload_btn)\n", "        layout.addWidget(self.fullscreen_btn)\n        layout.addWidget(self.reload_btn)\n")
rep(T, "        self.exit_btn.clicked.connect(self.exit_clicked.emit)\n", "        self.exit_btn.clicked.connect(self.exit_clicked.emit)\n        self.alarm_btn.clicked.connect(self.reset_alarm_clicked.emit)\n        self.fullscreen_btn.clicked.connect(self.fullscreen_clicked.emit)\n")
rep(T, "        button.setMinimumWidth(100)", "        button.setMinimumWidth(80)")
s = open(T, encoding="utf-8").read()
s += '''
    def set_alarm(self, active: bool):
        """Show Reset Alarm button and lock run controls while E-stop alarm is active."""
        self.alarm_btn.setVisible(active)
        if active:
            self.start_btn.setEnabled(False)
            self.trigger_btn.setEnabled(False)
        else:
            self.set_running(False)
'''
open(T, "w", encoding="utf-8").write(s)

# ---------------- Statusbar ----------------
S = "pcb_inspection/gui/widgets/statusbar.py"
rep(S, '        self.inspection = StatusItem("Inspection")\n', '        self.inspection = StatusItem("Inspection")\n        self.safety = StatusItem("Safety")\n')
rep(S, "        self.addPermanentWidget(self.inspection)\n", "        self.addPermanentWidget(self.safety)\n        self.addPermanentWidget(self.inspection)\n")
rep(S, '        self.set_inspection_status("IDLE")\n\n    def set_inspection_status', '        self.set_inspection_status("IDLE")\n        self.set_safety(False)\n\n    def set_safety(self, alarm: bool, reason: str = ""):\n        if alarm:\n            self.safety.set_state(f"ALARM {reason}".strip(), "#ff5555")\n        else:\n            self.safety.set_state("Normal", "#00ff7f")\n\n    def set_inspection_status')

# ---------------- Video / Log ----------------
rep("pcb_inspection/gui/widgets/video_widget.py", "        self.setMinimumSize(960, 640)\n", "        self.setMinimumSize(320, 240)\n        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)\n")
rep("pcb_inspection/gui/widgets/log_widget.py", "        self.setMaximumHeight(180)\n", "        self.setMinimumHeight(70)\n        self.setMaximumHeight(260)\n")

# ---------------- Window ----------------
M = "pcb_inspection/gui/window.py"
rep(M, "from PyQt6.QtCore import Qt, QThread, QMetaObject\n", "from PyQt6.QtCore import Qt, QThread, QMetaObject\nfrom PyQt6.QtGui import QShortcut, QKeySequence\n")
rep(M, "    QMessageBox,\n)", "    QMessageBox,\n    QLabel,\n    QScrollArea,\n    QFrame,\n)")
rep(M, "        self.resize(1600, 920)\n", "        self.resize(1280, 720)\n        self.setMinimumSize(800, 480)\n")
rep(M, """        top.addWidget(self.video, 3)
        top.addWidget(self.dashboard, 1)
        root.addLayout(top, 4)""", """        # Dashboard scrolls when the window is small
        self.dash_scroll = QScrollArea()
        self.dash_scroll.setWidgetResizable(True)
        self.dash_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.dash_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dash_scroll.setStyleSheet("QScrollArea{background:transparent;}")
        self.dash_scroll.setWidget(self.dashboard)
        self.dash_scroll.setMinimumWidth(self.dashboard.minimumSizeHint().width() + 24)

        top.addWidget(self.video, 3)
        top.addWidget(self.dash_scroll, 1)
        root.addLayout(top, 4)""")
rep(M, """        root.setSpacing(10)
""", """        root.setSpacing(10)

        # Emergency-stop alarm banner (hidden unless alarm)
        self.alarm_banner = QLabel("")
        self.alarm_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alarm_banner.setStyleSheet(
            "background:#C0392B;color:white;font-size:16px;font-weight:bold;padding:8px;border-radius:6px;"
        )
        self.alarm_banner.setVisible(False)
        root.addWidget(self.alarm_banner)
""")
rep(M, "        self.toolbar.exit_clicked.connect(self.close)\n", """        self.toolbar.exit_clicked.connect(self.close)
        self.toolbar.reset_alarm_clicked.connect(self.on_alarm_reset)
        self.toolbar.fullscreen_clicked.connect(self.toggle_fullscreen)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self.exit_fullscreen)
        self.worker.alarm_changed.connect(self.on_alarm_changed)
""")
rep(M, "    def on_ros_command(self, cmd: str):", '''    def on_alarm_changed(self, active: bool, reason: str):
        self.toolbar.set_alarm(active)
        self.status.set_safety(active, reason)
        if active:
            self.alarm_banner.setText(
                f"🚨 EMERGENCY STOP / ALARM: {reason}  —  放開緊急按鈕後按「Reset Alarm」復歸"
            )
            self.alarm_banner.setVisible(True)
            self.dashboard.set_inspection_state("IDLE")
        else:
            self.alarm_banner.setVisible(False)
            self.log.success("Alarm cleared. System is IDLE.")

    def on_alarm_reset(self):
        reply = QMessageBox.question(
            self,
            "Reset Alarm",
            "請確認緊急停止按鈕已放開、輸送帶上無異物。\\n確定要解除警報嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            QMetaObject.invokeMethod(self.worker, "reset_alarm", Qt.ConnectionType.QueuedConnection)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def on_ros_command(self, cmd: str):''')

# ---------------- app ----------------
P = "app.py"
rep(P, "    window = MainWindow()\n    window.show()\n", '    window = MainWindow()\n    if "--fullscreen" in sys.argv:\n        window.showFullScreen()\n    else:\n        window.showMaximized()\n')
print("ALL OK")
