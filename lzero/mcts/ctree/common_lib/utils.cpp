// C++11

#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

void get_time_and_set_rand_seed()
{
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER uli;
  uli.LowPart = ft.dwLowDateTime;
  uli.HighPart = ft.dwHighDateTime;
  uint64_t timestamp = (uli.QuadPart - 116444736000000000ULL) / 10000000ULL;
  srand(timestamp % RAND_MAX);
#else
    timeval tv;
    gettimeofday(&tv, nullptr);
    srand(tv.tv_usec);
#endif
}