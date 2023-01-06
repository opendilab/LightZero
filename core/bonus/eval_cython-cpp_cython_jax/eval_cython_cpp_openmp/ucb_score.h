#ifndef UCB_SCORE_H
#define UCB_SCORE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>
#include <map>


namespace tree {

    float cpp_ucb_score(std::vector<float> &child_visit_count, std::vector<float> &child_prior, std::vector<float> &child_reward, std::vector<float> &child_value, float maximum, float minimum, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount);
    
}

#endif