#ifndef HIGHRESTIMER_CLASS_H
#define HIGHRESTIMER_CLASS_H

#include "windows.h"

class HighResTimer
{

    LARGE_INTEGER start;
    LARGE_INTEGER stop;
    double frequency;

public:

    HighResTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        frequency = (double)freq.QuadPart;
    }
    void StartTimer()
    {
        QueryPerformanceCounter(&start);
    }
    double StopTimer()
    {
        QueryPerformanceCounter(&stop);
        return ((stop.QuadPart - start.QuadPart)/frequency);
    }

};

#endif // HIGHRESTIMER_CLASS_H
