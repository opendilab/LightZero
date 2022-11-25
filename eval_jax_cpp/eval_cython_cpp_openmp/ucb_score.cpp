#include <iostream>
#include "cminimax.h"
#include "ucb_score.h"
#include <algorithm>
#include <map>

namespace tree {


    float cpp_ucb_score(std::vector<float> &child_visit_count, std::vector<float> &child_prior, std::vector<float> &child_reward, std::vector<float> &child_value, float maximum, float minimum, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount){
        #pragma omp simd
        float ucb_value=0.0;
        for(int i = 0;i < total_children_visit_counts;i++){
            float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
            pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
            pb_c *= (sqrt(total_children_visit_counts) / (child_visit_count[i] + 1));

            prior_score = pb_c * child_prior[i];

            value_score = child_reward[i] +discount * child_value[i];
            value_score = (value_score - minimum) / (maximum - minimum);

            if (value_score < 0) value_score = 0;

            ucb_value = prior_score + value_score;
        }
        return ucb_value;
    }
}
