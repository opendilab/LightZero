// C++11

#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

uint64_t get_time()
{
#ifdef _WIN32
    FILETIME ft;
    LARGE_INTEGER li;

    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    uint64_t t = li.QuadPart;  // 100-nanosecond intervals since January 1, 1601 UTC
    t -= 116444736000000000ULL;  // 100-nanosecond intervals since January 1, 1970 UTC
    t /= 10;  // microseconds since January 1, 1970 UTC
    return t;
#else
    timeval tv;
    gettimeofday(&tv, nullptr);
    srand(tv.tv_usec);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
}


#ifdef _WIN32
constexpr int get_action_num(const std::vector<float> &policy_logits) {
    return policy_logits.size();
}
#else
#endif
