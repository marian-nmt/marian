#pragma once
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

DLLEXPORT void* init(char* path, int device_num);
DLLEXPORT char* translate(void* marian, char* sent);
DLLEXPORT void free_result(char* to_free);
