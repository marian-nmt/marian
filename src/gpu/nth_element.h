#pragma once

#include <thrust/partition.h>
#include <thrust/find.h>
#include <thrust/fill.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>

#include "thrust_bind/bind.hpp"

namespace thrust {

template<class Iterator, class Compare>
inline void nth_element(Iterator first,
                        Iterator nth,
                        Iterator last,
                        Compare compare)
{
    if(nth == last) return;

    using namespace thrust::placeholders;

    typedef typename thrust::iterator_traits<Iterator>::value_type value_type;

    while(1)
    {
        value_type value = *nth;

        Iterator new_nth = thrust::partition(
            first, last, thrust::experimental::bind(compare, _1, value)
        );

        Iterator old_nth = thrust::find(new_nth, last, value);

        value_type new_value = *new_nth;

        thrust::fill_n(new_nth, 1, value);
        thrust::fill_n(old_nth, 1, new_value);

        new_value = *nth;

        if(value == new_value) break;

        if(thrust::distance(first, nth) < thrust::distance(first, new_nth)) {
            last = new_nth;
        }
        else {
            first = new_nth;
        }
    }
}

}
