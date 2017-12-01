#include <stdint.h>
#include "thrust_functions.h"


__fp16 uint16_as_fp16 (uint16_t a)
{
    __fp16 res;
#if defined (__cplusplus)
    memcpy (&res, &a, sizeof (res));
#else /* __cplusplus */
    volatile union {
        __fp16 f;
        uint16_t i;
    } cvt;
    cvt.i = a;
    res = cvt.f;
#endif /* __cplusplus */
    return res;
}

uint32_t fp32_as_uint32 (float a)
{
    uint32_t res;
#if defined (__cplusplus)
    memcpy (&res, &a, sizeof (res));
#else /* __cplusplus */
    volatile union {
        float f;
        uint32_t i;
    } cvt;
    cvt.f = a;
    res = cvt.i;
#endif /* __cplusplus */
    return res;
}

/* host version of device function __float2half_rn() */
__fp16 float2half_rn (float a)
{
    uint32_t ia = fp32_as_uint32 (a);
    uint16_t ir;

    ir = (ia >> 16) & 0x8000;
    if ((ia & 0x7f800000) == 0x7f800000) {
        if ((ia & 0x7fffffff) == 0x7f800000) {
            ir |= 0x7c00; /* infinity */
        } else {
            ir = 0x7fff; /* canonical NaN */
        }
    } else if ((ia & 0x7f800000) >= 0x33000000) {
        int shift = (int)((ia >> 23) & 0xff) - 127;
        if (shift > 15) {
            ir |= 0x7c00; /* infinity */
        } else {
            ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
            if (shift < -14) { /* denormal */
                ir |= ia >> (-1 - shift);
                ia = ia << (32 - (-1 - shift));
            } else { /* normal */
                ir |= ia >> (24 - 11);
                ia = ia << (32 - (24 - 11));
                ir = ir + ((14 + shift) << 10);
            }
            /* IEEE-754 round to nearest of even */
            if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
                ir++;
            }
        }
    }
    return uint16_as_fp16 (ir);
}
