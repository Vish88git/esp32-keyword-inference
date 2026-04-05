#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2s.h"
#include "esp_heap_caps.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "mqtt_client.h"
#include "driver/gpio.h"

/* TFLite Micro */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Model */
#include "keyword_model.h"

static const char *TAG = "KW";

/* ── WiFi + MQTT config ── */
#define WIFI_SSID       "Airtel_Airel_9480663585"
#define WIFI_PASS       "air53927"
#define MQTT_BROKER     "mqtt://192.168.1.9:1883"
#define MQTT_TOPIC      "esp32/keyword"

/* ── Audio config ── */
#define I2S_WS          25
#define I2S_SCK         26
#define I2S_SD          22
#define SAMPLE_RATE     16000
#define RECORD_SAMPLES  16000

/* ── Model config ── */
#define N_MFCC          12
#define N_FRAMES        32
#define N_FEATURES      36
#define NORM_MEAN       (0.558983f)
#define NORM_STD        (3.523827f)

/* ── Labels ── */
static const char *LABELS[] = {"up", "down", "left", "right"};
#define NUM_LABELS 4

/* ── TFLite globals ── */
static const tflite::Model        *model       = nullptr;
static tflite::MicroInterpreter   *interpreter = nullptr;
static TfLiteTensor               *input       = nullptr;
static TfLiteTensor               *output      = nullptr;

/* ── WiFi + MQTT globals ── */
static esp_mqtt_client_handle_t mqtt_client  = NULL;
static bool wifi_connected = false;
static bool mqtt_connected = false;

/* ── Tensor arena ── */
static uint8_t tensor_arena[100 * 1024];

/* ── Audio buffer ── */
static int16_t *audio_buf = nullptr;

/* ── FFT buffers ── */
#define N_FFT   512
#define HOP_LEN 256
#define N_MELS  40

static float fft_real[N_FFT];
static float fft_imag[N_FFT];
static float features[N_FRAMES][N_FEATURES];
static float mfcc_buf[N_FRAMES][N_MFCC];

/* ── Forward declarations ── */
static void publish_keyword(const char *word, float confidence);

/* ─────────────────────────────────────────────
 * I2S setup
 * ───────────────────────────────────────────── */
static void i2s_init(void)
{
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = 256,
        .use_apll             = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK,
        .ws_io_num    = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD,
    };
    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

/* ─────────────────────────────────────────────
 * Record audio
 * ───────────────────────────────────────────── */
static void record_audio(void)
{
    int32_t raw[256];
    size_t  bytes_read = 0;
    int     recorded   = 0;

    int discard = 2000;
    while (discard > 0) {
        int n = (discard < 256) ? discard : 256;
        i2s_read(I2S_NUM_0, raw, n * 4, &bytes_read, portMAX_DELAY);
        discard -= bytes_read / 4;
    }

    while (recorded < RECORD_SAMPLES) {
        int n = RECORD_SAMPLES - recorded;
        if (n > 256) n = 256;
        i2s_read(I2S_NUM_0, raw, n * 4, &bytes_read, portMAX_DELAY);
        int got = bytes_read / 4;
        for (int i = 0; i < got && recorded < RECORD_SAMPLES; i++) {
            audio_buf[recorded++] = (int16_t)(raw[i] >> 9);
        }
    }
}

/* ─────────────────────────────────────────────
 * Find speech segment
 * ───────────────────────────────────────────── */
static int find_speech_start(void)
{
    const int frame_len   = 512;
    const int hop         = 256;
    float     peak_energy = 0.0f;
    int       peak_frame  = 0;

    int num_frames = (RECORD_SAMPLES - frame_len) / hop;
    for (int f = 0; f < num_frames; f++) {
        float energy = 0.0f;
        int   start  = f * hop;
        for (int i = start; i < start + frame_len; i++) {
            float s = (float)audio_buf[i];
            energy += s * s;
        }
        if (energy > peak_energy) {
            peak_energy = energy;
            peak_frame  = f;
        }
    }

    int center   = peak_frame * hop;
    int half_win = SAMPLE_RATE / 4;
    int start    = center - half_win;
    if (start < 0) start = 0;
    if (start + SAMPLE_RATE / 2 > RECORD_SAMPLES)
        start = RECORD_SAMPLES - SAMPLE_RATE / 2;
    return start;
}

/* ─────────────────────────────────────────────
 * FFT
 * ───────────────────────────────────────────── */
static void fft(float *re, float *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float ur = re[i+j];
                float ui = im[i+j];
                float vr = re[i+j+len/2]*cr - im[i+j+len/2]*ci;
                float vi = re[i+j+len/2]*ci + im[i+j+len/2]*cr;
                re[i+j]       = ur + vr;
                im[i+j]       = ui + vi;
                re[i+j+len/2] = ur - vr;
                im[i+j+len/2] = ui - vi;
                float ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr;
                cr = ncr;
            }
        }
    }
}

/* ─────────────────────────────────────────────
 * Extract features
 * ───────────────────────────────────────────── */
static void extract_features(int seg_start)
{
    float mel_low  = 0.0f;
    float mel_high = 2595.0f * log10f(1.0f + (SAMPLE_RATE / 2.0f) / 700.0f);

    float mel_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        float mel = mel_low + i * (mel_high - mel_low) / (N_MELS + 1);
        mel_points[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }

    int bins[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        bins[i] = (int)floorf((N_FFT + 1) * mel_points[i] / SAMPLE_RATE);
        if (bins[i] >= N_FFT / 2) bins[i] = N_FFT / 2 - 1;
    }

    for (int f = 0; f < N_FRAMES; f++) {
        int frame_start = seg_start + f * HOP_LEN;

        for (int i = 0; i < N_FFT; i++) {
            int idx = frame_start + i;
            float sample = (idx < RECORD_SAMPLES) ?
                           (float)audio_buf[idx] / 32768.0f : 0.0f;
            float window = 0.5f * (1.0f -
                           cosf(2.0f * (float)M_PI * i / (N_FFT - 1)));
            fft_real[i] = sample * window;
            fft_imag[i] = 0.0f;
        }

        fft(fft_real, fft_imag, N_FFT);

        float power[N_FFT / 2 + 1];
        for (int i = 0; i <= N_FFT / 2; i++) {
            power[i] = fft_real[i]*fft_real[i] + fft_imag[i]*fft_imag[i];
        }

        float mel_energy[N_MELS];
        for (int m = 0; m < N_MELS; m++) {
            mel_energy[m] = 0.0f;
            for (int k = bins[m]; k < bins[m+1]; k++) {
                float w = (float)(k - bins[m]) / (bins[m+1] - bins[m]);
                mel_energy[m] += w * power[k];
            }
            for (int k = bins[m+1]; k < bins[m+2]; k++) {
                float w = (float)(bins[m+2] - k) /
                          (bins[m+2] - bins[m+1]);
                mel_energy[m] += w * power[k];
            }
            mel_energy[m] = logf(mel_energy[m] + 1e-6f);
        }

        for (int c = 0; c < N_MFCC; c++) {
            float sum = 0.0f;
            for (int m = 0; m < N_MELS; m++) {
                sum += mel_energy[m] *
                       cosf((float)M_PI * (c + 1) *
                            (m + 0.5f) / N_MELS);
            }
            mfcc_buf[f][c] = sum;
        }
    }

    for (int f = 0; f < N_FRAMES; f++) {
        for (int c = 0; c < N_MFCC; c++) {
            int f1  = (f > 0)          ? f - 1 : 0;
            int f2  = (f < N_FRAMES-1) ? f + 1 : N_FRAMES - 1;
            int f11 = (f > 1)          ? f - 2 : 0;
            int f22 = (f < N_FRAMES-2) ? f + 2 : N_FRAMES - 1;

            float delta  = (mfcc_buf[f2][c] - mfcc_buf[f1][c]) / 2.0f;
            float delta2 = (mfcc_buf[f22][c] - 2.0f*mfcc_buf[f][c]
                           + mfcc_buf[f11][c]) / 4.0f;

            features[f][c]            = mfcc_buf[f][c];
            features[f][c + N_MFCC]   = delta;
            features[f][c + 2*N_MFCC] = delta2;
        }
    }

    for (int f = 0; f < N_FRAMES; f++) {
        for (int c = 0; c < N_FEATURES; c++) {
            features[f][c] = (features[f][c] - NORM_MEAN) / NORM_STD;
        }
    }
}

/* ─────────────────────────────────────────────
 * Run inference
 * ───────────────────────────────────────────── */
static int run_inference(void)
{
    float *in_data = input->data.f;
    for (int f = 0; f < N_FRAMES; f++) {
        for (int c = 0; c < N_FEATURES; c++) {
            in_data[f * N_FEATURES + c] = features[f][c];
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return -1;
    }

    /* Find best label */
    float *out_data  = output->data.f;
    int    best      = 0;
    float  best_score = out_data[0];
    for (int i = 1; i < NUM_LABELS; i++) {
        if (out_data[i] > best_score) {
            best_score = out_data[i];
            best       = i;
        }
    }

    ESP_LOGI(TAG, "Scores: up=%.2f down=%.2f left=%.2f right=%.2f",
             out_data[0], out_data[1], out_data[2], out_data[3]);
    ESP_LOGI(TAG, "Result: %s (%.1f%%)", LABELS[best], best_score * 100.0f);
    publish_keyword(LABELS[best], best_score);

    return best;
}

/* ─────────────────────────────────────────────
 * WiFi event handler
 * ───────────────────────────────────────────── */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT &&
               event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_connected = false;
        ESP_LOGI(TAG, "WiFi disconnected, retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "WiFi connected. IP: " IPSTR,
                 IP2STR(&event->ip_info.ip));
        wifi_connected = true;
    }
}

/* ─────────────────────────────────────────────
 * WiFi init
 * ───────────────────────────────────────────── */
static void wifi_init(void)
{
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                &wifi_event_handler, NULL);

    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid,
            WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password,
            WIFI_PASS, sizeof(wifi_config.sta.password));

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    ESP_LOGI(TAG, "Connecting to WiFi: %s", WIFI_SSID);

    int retries = 0;
    while (!wifi_connected && retries < 20) {
        vTaskDelay(pdMS_TO_TICKS(500));
        retries++;
    }

    if (!wifi_connected) {
        ESP_LOGE(TAG, "WiFi connection failed");
    }
}

/* ─────────────────────────────────────────────
 * MQTT event handler
 * ───────────────────────────────────────────── */
static void mqtt_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            mqtt_connected = true;
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected");
            mqtt_connected = false;
            break;
        default:
            break;
    }
}

/* ─────────────────────────────────────────────
 * MQTT init
 * ───────────────────────────────────────────── */
static void mqtt_init(void)
{
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = MQTT_BROKER;

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY,
                                    mqtt_event_handler, NULL);
    esp_mqtt_client_start(mqtt_client);

    int retries = 0;
    while (!mqtt_connected && retries < 10) {
        vTaskDelay(pdMS_TO_TICKS(500));
        retries++;
    }

    if (!mqtt_connected) {
        ESP_LOGE(TAG, "MQTT connection failed");
    }
}

/* ─────────────────────────────────────────────
 * Publish keyword detection to MQTT
 * ───────────────────────────────────────────── */
static void publish_keyword(const char *word, float confidence)
{
    if (!mqtt_connected) {
        ESP_LOGW(TAG, "MQTT not connected, skipping publish");
        return;
    }

    char payload[64];
    snprintf(payload, sizeof(payload),
             "{\"word\":\"%s\",\"confidence\":%.2f}", word, confidence);

    esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC, payload, 0, 1, 0);
    ESP_LOGI(TAG, "Published: %s", payload);
}

/* ─────────────────────────────────────────────
 * app_main
 * ───────────────────────────────────────────── */
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Keyword Inference starting...");

    i2s_init();

    audio_buf = (int16_t*)heap_caps_malloc(
                    RECORD_SAMPLES * sizeof(int16_t),
                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (audio_buf == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate audio buffer");
        return;
    }

    wifi_init();
    mqtt_init();

    model = tflite::GetModel(keyword_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch");
        return;
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Model loaded. Input shape: [%d,%d,%d,%d]",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Press BOOT button to record a word");

    gpio_set_direction(GPIO_NUM_0, GPIO_MODE_INPUT);
    gpio_set_pull_mode(GPIO_NUM_0, GPIO_PULLUP_ONLY);

    while (1) {
        if (gpio_get_level(GPIO_NUM_0) == 0) {
            vTaskDelay(pdMS_TO_TICKS(50));
            if (gpio_get_level(GPIO_NUM_0) == 0) {
                ESP_LOGI(TAG, "Recording...");
                record_audio();
                ESP_LOGI(TAG, "Extracting features...");
                int seg = find_speech_start();
                extract_features(seg);
                ESP_LOGI(TAG, "Running inference...");
                run_inference();
                while (gpio_get_level(GPIO_NUM_0) == 0) {}
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}