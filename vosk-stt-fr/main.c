#include "vosk-api/src/vosk_api.h"
#include <portaudio.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "command_parser.h"
#include "llm_client.h"

// Configuration standard pour micro USB
#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 4096 

// Couleurs
#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_CYAN "\033[0;36m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RED   "\033[1;31m"

VoskModel *model = NULL;
VoskRecognizer *recognizer = NULL;
PaStream *stream = NULL;
volatile sig_atomic_t stop_requested = 0;

void cleanup() {
    if (stream) {
        Pa_AbortStream(stream);
        Pa_CloseStream(stream);
    }
    Pa_Terminate();
    if (recognizer) vosk_recognizer_free(recognizer);
    if (model) vosk_model_free(model);
    printf("\n%s[INFO] Ressources nettoyées.%s\n", COLOR_CYAN, COLOR_RESET);
}

void sigint_handler(int sig) { 
    stop_requested = 1; 
    printf("\nArrêt demandé...\n");
}

void process_result(const char *json_result) {
    const char *key = "\"text\"";
    const char *text_start = strstr(json_result, key);
    if (!text_start) return;

    text_start = strchr(text_start, ':');
    if (!text_start) return;

    text_start = strchr(text_start, '\"');
    if (!text_start) return;
    text_start++;

    const char *text_end = strchr(text_start, '\"');
    if (!text_end) return;

    int len = text_end - text_start;
    if (len <= 0) return;

    char *spoken_text = malloc(len + 1);
    strncpy(spoken_text, text_start, len);
    spoken_text[len] = '\0';

    if (strlen(spoken_text) == 0) {
        free(spoken_text);
        return;
    }

    printf("Transcription : %s\n", spoken_text);

    CommandType cmd = parse_command(spoken_text);
    if (cmd != CMD_UNKNOWN) {
        printf("%s>>> COMMANDE DÉTECTÉE : [%s]%s\n", COLOR_GREEN, get_command_action(cmd), COLOR_RESET);
    } else {
        printf("%s>>> [RELAY LLM] Envoi à l'IA...%s\n", COLOR_YELLOW, COLOR_RESET);
        send_to_llm(spoken_text);
    }
    free(spoken_text);
}

int main() {
    PaError err;
    signal(SIGINT, sigint_handler);

    // 1. Chargement du modèle
    printf("Chargement du modèle Vosk...\n");
    model = vosk_model_new("vosk-model-small-fr-0.22");
    if (model == NULL) {
        fprintf(stderr, "%s[ERREUR] Impossible de charger le modèle.%s\n", COLOR_RED, COLOR_RESET);
        return -1;
    }

    recognizer = vosk_recognizer_new(model, (float)SAMPLE_RATE);

    // 2. Initialisation Audio
    printf("Initialisation PortAudio...\n");
    err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "Erreur Pa_Initialize: %s\n", Pa_GetErrorText(err));
        cleanup();
        return -1;
    }

    // 3. RECHERCHE AUTOMATIQUE DU MICRO USB
    int numDevices = Pa_GetDeviceCount();
    int usbDeviceIndex = -1;

    printf("\n--- Recherche des micros ---\n");
    for(int i = 0; i < numDevices; i++) {
        const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(i);
        // On affiche tout pour le debug
        printf("ID %d : %s (Entrées: %d)\n", i, deviceInfo->name, deviceInfo->maxInputChannels);
        
        // Si c'est un micro (inputs > 0) et qu'il a "USB" dans le nom
        if (deviceInfo->maxInputChannels > 0 && strstr(deviceInfo->name, "USB") != NULL) {
            usbDeviceIndex = i;
        }
    }
    printf("----------------------------\n");

    PaStreamParameters inputParameters;
    
    if (usbDeviceIndex != -1) {
        printf("%s[SUCCÈS] Micro USB trouvé à l'ID PortAudio : %d%s\n", COLOR_GREEN, usbDeviceIndex, COLOR_RESET);
        inputParameters.device = usbDeviceIndex;
    } else {
        printf("%s[ATTENTION] Pas de 'USB' dans le nom. Essai avec le périphérique par défaut...%s\n", COLOR_YELLOW, COLOR_RESET);
        inputParameters.device = Pa_GetDefaultInputDevice();
    }

    // Vérification finale
    if (inputParameters.device == paNoDevice) {
        fprintf(stderr, "%s[ERREUR] Aucun micro valide trouvé.%s\n", COLOR_RED, COLOR_RESET);
        cleanup();
        return -1;
    }

    const PaDeviceInfo *devInfo = Pa_GetDeviceInfo(inputParameters.device);
    inputParameters.channelCount = 1;
    inputParameters.sampleFormat = paInt16;
    inputParameters.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    // 4. Ouverture du flux
    printf("Ouverture du flux sur '%s' @ %.0f Hz...\n", devInfo->name, (float)SAMPLE_RATE);
    err = Pa_OpenStream(&stream, &inputParameters, NULL, SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, NULL, NULL);
    
    if (err != paNoError) {
        fprintf(stderr, "%s[ERREUR] Pa_OpenStream: %s%s\n", COLOR_RED, Pa_GetErrorText(err), COLOR_RESET);
        printf("Essai de changer le SAMPLE_RATE dans le code si 44100 ne marche pas.\n");
        cleanup();
        return -1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "Erreur StartStream: %s\n", Pa_GetErrorText(err));
        cleanup();
        return -1;
    }

    printf("%s>>> PRÊT. Je vous écoute...%s\n", COLOR_GREEN, COLOR_RESET);

    // 5. Boucle principale
    int16_t buffer[FRAMES_PER_BUFFER];
    while (!stop_requested) {
        err = Pa_ReadStream(stream, buffer, FRAMES_PER_BUFFER);
        if (err != paNoError && err != paInputOverflowed) {
             fprintf(stderr, "Erreur lecture: %s\n", Pa_GetErrorText(err));
        }
        
        if (vosk_recognizer_accept_waveform_s(recognizer, buffer, FRAMES_PER_BUFFER)) {
            process_result(vosk_recognizer_result(recognizer));
        }
    }

    process_result(vosk_recognizer_final_result(recognizer));
    cleanup();
    return 0;
}
