# ESP32 Edge AI Keyword Inference

Real-time keyword recognition on ESP32 using TFLite Micro.

## Hardware
- ESP32-WROOM-32E DevKit
- SPH0645LM4H-B I2S MEMS Microphone

## Features
- 4-word keyword recognition: up, down, left, right
- 98.8% accuracy on ESP32-collected features
- AUTOSAR-layered software architecture
- WiFi + MQTT telemetry to IoT dashboard
- TFLite Micro CNN inference (~113KB model)

## Architecture
SPH0645 Mic -> I2S Driver -> MFCC Feature Extraction -> TFLite Micro CNN -> MQTT Publish

## ML Pipeline
- 100 samples per word collected directly from ESP32
- Features: 32 frames x 36 coefficients (12 MFCC + delta + delta2)
- Model: 3x Conv2D + GlobalAveragePooling + Dense

## IoT Dashboard
Detections published to MQTT topic `esp32/keyword` as JSON:
```json
{"word": "up", "confidence": 0.92}
```
Dashboard repo: https://github.com/Vish88git/esp32-iot-dashboard

## Build
```bash
idf.py build
idf.py -p COM3 flash monitor
```
Requires ESP-IDF v5.5.3