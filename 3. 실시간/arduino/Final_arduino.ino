// 졸업 프로젝트 : 류서진 & 박소연

#define speedPin 4
#define Rotation 5

// 1. 모터
int MotorSpeed = 128;
int myDelay = 1000;

// 2. LED
int LED = 12;

void setup() {
  Serial.begin(9600);

  pinMode(speedPin, OUTPUT);  
  pinMode(Rotation, OUTPUT);
  pinMode(LED, OUTPUT);
  
}

void loop() {
  while (Serial.available() > 0)
  { char c = Serial.read();

    if (c == '1'){
      MotorUP();             // 레크 상승
      delay(5000);

      MotorStop();           // 정지
      delay(5000);
    }
    else if (c == '2'){
      LedOn();
      delay(1000);

      LedOff();
      delay(5000);
    }
  }
}

// 1. 노트북 거치대 Low : DCMotor
void MotorUP() {
  analogWrite(speedPin, MotorSpeed);
  digitalWrite(Rotation, HIGH);
}
void MotorStop() {
  analogWrite(speedPin, 0);
  digitalWrite(Rotation, LOW);
}

// 2. 노트북 거치대 High : LED
void LedOn() {
  digitalWrite(LED, HIGH);
}
void LedOff() {
  digitalWrite(LED, LOW);
}
