# distutils: language=c++
from libcpp.vector cimport vector

cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

cdef extern from "ucb_score.cpp":
    pass

cdef extern from "ucb_score.h" namespace "tree":
    cdef float cpp_ucb_score(vector[float] child_visit_count, vector[float] child_prior, vector[float] child_reward, vector[float] child_value, float maximum, float minimum, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount)