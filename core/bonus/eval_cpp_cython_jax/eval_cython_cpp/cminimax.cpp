#include "cminimax.h"

namespace tools{

    CMinMaxStats::CMinMaxStats(){
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
        this->value_delta_max = 0.;
    }

    CMinMaxStats::~CMinMaxStats(){}

    void CMinMaxStats::set_delta(float value_delta_max){
        this->value_delta_max = value_delta_max;
    }

    void CMinMaxStats::update(float value){
        if(value > this->maximum){
            this->maximum = value;
        }
        if(value < this->minimum){
            this->minimum = value;
        }
    }

    void CMinMaxStats::clear(){
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
    }

    float CMinMaxStats::normalize(float value){
        float norm_value = value;
        float delta = this->maximum - this->minimum;
        norm_value = (norm_value - this->minimum) / delta;
        return norm_value;
    }

    //*********************************************************

}