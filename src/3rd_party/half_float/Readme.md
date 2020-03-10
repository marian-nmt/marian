half_float
========

#### 16 bit floating-point data type for C++ ####

Implements a `HalfFloat` class that implements all the common arithmetic operations for a 16 bit 
floating-point type (10 bits mantissa, 5 bits exponent and one sign bit) and can thus be used (almost)
interchangeably with regular `float`s. Not all operations have efficent implementations (some just convert to `float`, 
compute the result and convert back again) - if in doubt, check out the source code.

The implementation tries to adhere to IEEE 754 in that it supports NaN and Infinity, but fails in other points:

 - no difference between qnan and snan
 - no traps
 - no well-defined rounding mode


We also supply a specialization for `std::numeric_limits<half>` that `half` be usable in template code
dependent on type traits.


#### Usage ####

     // get some halfs (half is a typedef for HalfFloat)
     half a = 1.0f;
     half b = 0.5f;
     
     // and have some FUN
     half c = (a+b) / (a-b);
     ++c;
     
     // now that we have a result in loosy precision,
     // convert it back to double precision.
     // if anybody asks, it's for the lulz.
     double result = c;


Credits to _Chris Maiwald_ for the conversion code to `double` and extensive testing.


#### License ####

3-clause BSD license: use it for anything, but give credit, don't blame us if your rocket crashes and don't advertise with it (who would).