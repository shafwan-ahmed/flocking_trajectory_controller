/*
  Dual 28BYJ-48 Stepper Bot — ESP8266 (NodeMCU) Differential-Drive (UDP)
  ------------------------------------------------------------------------
  Receives per-wheel speed commands from navigate_csv_trajectory.py:

      "L<left_speed> R<right_speed>"   e.g. "L200 R160"   (range -255..255)
      "X"                              stop both motors immediately

  Sign = direction, magnitude = step rate. Each motor steps independently
  and non-blocking, so both wheels are always in motion at differential
  rates — smooth arcs and spins instead of stop-one-wheel pivots.

  A watchdog auto-stops the motors if no packet arrives for 300 ms.

  PIN ASSIGNMENTS (NodeMCU silkscreen labels -> GPIO):
    Left motor  ULN2003 IN1-IN4 -> D0, D1, D2, D4  (GPIO16, GPIO5, GPIO4, GPIO2)
    Right motor ULN2003 IN1-IN4 -> D5, D6, D7, D8  (GPIO14, GPIO12, GPIO13, GPIO15)

  BOOT-PIN NOTES (only matters at power-on/reset/flash, not while running):
    D0 (GPIO16) has no interrupt/PWM support but works fine as a plain output.
    D4 (GPIO2)  is pulled HIGH at boot and shares the onboard LED — it'll
                blink briefly on reset, harmless for a stepper coil.
    D8 (GPIO15) is pulled LOW at boot — avoid anything that forces it HIGH
                externally during power-on/flashing.

  If a direction is mirrored on your build (e.g. +speed goes backward, or
  the two motors arc the wrong way), flip LEFT_REVERSED / RIGHT_REVERSED
  below rather than rewiring anything.
*/

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// ---------- WiFi credentials ----------
const char* WIFI_SSID = "ASUS_1E_NIRO_2.4G";
const char* WIFI_PASS = "niro@2026";

// ---------- UDP ----------
const unsigned int UDP_PORT = 4210;
WiFiUDP udp;

// ---------- Motor pin assignments (NodeMCU GPIO numbers) ----------
const int leftPins[4]  = {16, 5, 4, 2};    // D0, D1, D2, D4 -> IN1, IN2, IN3, IN4
const int rightPins[4] = {14, 12, 13, 15}; // D5, D6, D7, D8 -> IN1, IN2, IN3, IN4

const bool LEFT_REVERSED = false;
const bool RIGHT_REVERSED = false;

// ---------- Stepper timing ----------
// |speed| = 255 steps at MIN_STEP_US; |speed| -> 1 approaches MAX_STEP_US.
const unsigned long MIN_STEP_US = 800;        // fastest half-step interval
const unsigned long MAX_STEP_US = 5000;       // slowest half-step interval
const unsigned long COMMAND_TIMEOUT_MS = 300; // auto-stop if no command received

const int halfStepSeq[8][4] = {
  {1,0,0,0},
  {1,1,0,0},
  {0,1,0,0},
  {0,1,1,0},
  {0,0,1,0},
  {0,0,1,1},
  {0,0,0,1},
  {1,0,0,1}
};

int leftSeqIndex = 0;
int rightSeqIndex = 0;

int leftSpeed = 0;      // -255..255, 0 = stopped
int rightSpeed = 0;

unsigned long lastCommandMillis = 0;
unsigned long lastLeftStepMicros = 0;
unsigned long lastRightStepMicros = 0;
bool coilsOff = true;

void setPins(const int pins[4], const int state[4]) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(pins[i], state[i]);
  }
}

void stepOnce(const int pins[4], int &seqIndex, int dir) {
  seqIndex = (seqIndex + dir + 8) % 8;
  setPins(pins, halfStepSeq[seqIndex]);
}

void motorsOff() {
  const int off[4] = {0,0,0,0};
  setPins(leftPins, off);
  setPins(rightPins, off);
  coilsOff = true;
}

// Map |speed| (1..255) to a half-step interval (MAX..MIN microseconds).
unsigned long stepIntervalUs(int speed) {
  int mag = abs(speed);
  if (mag == 0) return 0;  // never step
  return MAX_STEP_US - (unsigned long)(MAX_STEP_US - MIN_STEP_US) * (mag - 1) / 254;
}

// Non-blocking per-motor tick: step a motor if its own interval elapsed.
void tickMotor(const int pins[4], int &seqIndex, int speed,
               unsigned long &lastStepMicros, bool reversed) {
  if (reversed) speed = -speed;
  unsigned long interval = stepIntervalUs(speed);
  if (interval == 0) return;
  if (micros() - lastStepMicros >= interval) {
    lastStepMicros = micros();
    stepOnce(pins, seqIndex, speed > 0 ? +1 : -1);
    coilsOff = false;
  }
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
  Serial.print("Listening for UDP commands on port ");
  Serial.println(UDP_PORT);

  for (int i = 0; i < 4; i++) {
    pinMode(leftPins[i], OUTPUT);
    pinMode(rightPins[i], OUTPUT);
  }
  motorsOff();
  lastCommandMillis = millis();
}

void loop() {
  // Check for an incoming UDP packet
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char buf[32] = {0};
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len > 0) {
      buf[len] = '\0';
      lastCommandMillis = millis();

      if (buf[0] == 'X') {
        leftSpeed = 0;
        rightSpeed = 0;
      } else {
        int l = 0, r = 0;
        // Expect "L<int> R<int>"
        if (sscanf(buf, "L%d R%d", &l, &r) == 2) {
          leftSpeed = constrain(l, -255, 255);
          rightSpeed = constrain(r, -255, 255);
        } else {
          leftSpeed = 0;   // unparsable packet: fail safe
          rightSpeed = 0;
        }
      }
    }
  }

  // Watchdog: stop if no command received recently
  if (millis() - lastCommandMillis > COMMAND_TIMEOUT_MS) {
    leftSpeed = 0;
    rightSpeed = 0;
  }

  if (leftSpeed == 0 && rightSpeed == 0) {
    if (!coilsOff) motorsOff();
    return;
  }

  tickMotor(leftPins, leftSeqIndex, leftSpeed, lastLeftStepMicros, LEFT_REVERSED);
  tickMotor(rightPins, rightSeqIndex, rightSpeed, lastRightStepMicros, RIGHT_REVERSED);
}
