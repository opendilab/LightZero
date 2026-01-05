// C++11

#include "cminimax.h"
#include <algorithm>
#include <cmath>

namespace tools {

    // ========== CMinMaxStats Implementation ==========
    CMinMaxStats::CMinMaxStats() {
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
        this->value_delta_max = 0.01;
        this->c_visit = 1;
        this->c_scale = 1.0;
    }

    CMinMaxStats::~CMinMaxStats() {}

    void CMinMaxStats::set_delta(float value_delta_max) {
        this->value_delta_max = value_delta_max;
    }

    void CMinMaxStats::set_static_val(float value_delta_max, int c_visit, float c_scale) {
        this->value_delta_max = value_delta_max;
        this->c_visit = c_visit;
        this->c_scale = c_scale;
    }

    void CMinMaxStats::update(float value) {
        this->maximum = std::max(this->maximum, value);
        this->minimum = std::min(this->minimum, value);
    }

    void CMinMaxStats::clear() {
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
    }

    float CMinMaxStats::normalize(float value) {
        float delta = this->maximum - this->minimum;
        delta = std::max(delta, this->value_delta_max);
        return (value - this->minimum) / delta;
    }

    // ========== CMinMaxStatsList Implementation ==========
    CMinMaxStatsList::CMinMaxStatsList() {
        this->num = 0;
    }

    CMinMaxStatsList::CMinMaxStatsList(int num) {
        this->num = num;
        for (int i = 0; i < num; ++i) {
            this->stats_lst.push_back(CMinMaxStats());
        }
    }

    CMinMaxStatsList::~CMinMaxStatsList() {}

    void CMinMaxStatsList::set_delta(float value_delta_max) {
        for (int i = 0; i < this->num; ++i) {
            this->stats_lst[i].set_delta(value_delta_max);
        }
    }

    void CMinMaxStatsList::set_static_val(float value_delta_max, int c_visit, float c_scale) {
        for (int i = 0; i < this->num; ++i) {
            this->stats_lst[i].set_static_val(value_delta_max, c_visit, c_scale);
        }
    }
}
