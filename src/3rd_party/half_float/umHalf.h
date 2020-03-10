
///////////////////////////////////////////////////////////////////////////////////
/*
Copyright (c) 2006-2008,
Chris "Krishty" Maiwald, Alexander "Aramis" Gessler

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the class, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the Development Team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
///////////////////////////////////////////////////////////////////////////////////

#ifndef UM_HALF_H_INCLUDED
#define UM_HALF_H_INCLUDED

#include <limits>
#include <algorithm>

#include <stdint.h>

#undef min
#undef max

///////////////////////////////////////////////////////////////////////////////////
/** 1. Represents a half-precision floating point value (16 bits) that behaves
 *  nearly conformant to the IEE 754 standard for floating-point computations.
 *
 *  Not all operators have special implementations, most perform time-consuming
 *  conversions from half to float and back again.
 *  Differences to IEEE 754:
 *  - no difference between qnan and snan
 *  - no traps
 *  - no well-defined rounding mode
 */
///////////////////////////////////////////////////////////////////////////////////
class HalfFloat
{
	friend HalfFloat operator+ (HalfFloat, HalfFloat);
	friend HalfFloat operator- (HalfFloat, HalfFloat);
	friend HalfFloat operator* (HalfFloat, HalfFloat);
	friend HalfFloat operator/ (HalfFloat, HalfFloat);

public:

	enum { BITS_MANTISSA = 10 };
	enum { BITS_EXPONENT = 5 };

	enum { MAX_EXPONENT_VALUE = 31 };
	enum { BIAS = MAX_EXPONENT_VALUE/2 };

	enum { MAX_EXPONENT = BIAS };
	enum { MIN_EXPONENT = -BIAS };

	enum { MAX_EXPONENT10 = 9 };
	enum { MIN_EXPONENT10 = -9 };

public:

	/** Default constructor. Unitialized by default.
	 */
	inline HalfFloat() {}

	/** Construction from an existing half
	 */
	inline HalfFloat(const HalfFloat& other)
		: bits(other.GetBits())
	{}

	/** Construction from existing values for mantissa, sign
	 *  and exponent. No validation is performed.
	 *  @note The exponent is unsigned and biased by #BIAS
	 */
	inline HalfFloat(uint16_t _m,uint16_t _e,uint16_t _s);


	/** Construction from a single-precision float
	 */
	inline HalfFloat(float other);

	/** Conversion operator to convert from half to float
	 */
	inline operator float() const;

	/** Assignment operator to assign another half to
	 *  *this* object.
	 */
	inline HalfFloat& operator= (HalfFloat other);
	inline HalfFloat& operator= (float other);

	/** Comparison operators
	 */
	inline bool operator== (HalfFloat other) const;
	inline bool operator!= (HalfFloat other) const;


	/** Relational comparison operators
	 */
	inline bool operator<  (HalfFloat other) const;
	inline bool operator>  (HalfFloat other) const;
	inline bool operator<= (HalfFloat other) const;
	inline bool operator>= (HalfFloat other) const;

	inline bool operator<  (float other) const;
	inline bool operator>  (float other) const;
	inline bool operator<= (float other) const;
	inline bool operator>= (float other) const;


	/** Combined assignment operators
	 */
	inline HalfFloat& operator += (HalfFloat other);
	inline HalfFloat& operator -= (HalfFloat other);
	inline HalfFloat& operator *= (HalfFloat other);
	inline HalfFloat& operator /= (HalfFloat other);

	inline HalfFloat& operator += (float other);
	inline HalfFloat& operator -= (float other);
	inline HalfFloat& operator *= (float other);
	inline HalfFloat& operator /= (float other);

	/** Post and prefix increment operators
	 */
	inline HalfFloat& operator++();
	inline HalfFloat operator++(int);

	/** Post and prefix decrement operators
	 */
	inline HalfFloat& operator--();
	inline HalfFloat operator--(int);

	/** Unary minus operator
	 */
	inline HalfFloat operator-() const;


	/** Provides direct access to the bits of a half float
	 */
	inline uint16_t GetBits() const;
	inline uint16_t& GetBits();


	/** Classification of floating-point types
	 */
	inline bool IsNaN() const;
	inline bool IsInfinity() const;
	inline bool IsDenorm() const;

	/** Returns the sign of the floating-point value -
	 *  true stands for positive.
	 */
	inline bool GetSign() const;

public:

	union
	{
		uint16_t bits;			// All bits
		struct
		{
			uint16_t Frac : 10;	// mantissa
			uint16_t Exp  : 5;		// exponent
			uint16_t Sign : 1;		// sign
		} IEEE;
	};


	union IEEESingle
	{
		float Float;
		struct
		{
			uint32_t Frac : 23;
			uint32_t Exp  : 8;
			uint32_t Sign : 1;
		} IEEE;
	};
};

/** 2. Binary operations
 */
inline HalfFloat operator+ (HalfFloat one, HalfFloat two);
inline HalfFloat operator- (HalfFloat one, HalfFloat two);
inline HalfFloat operator* (HalfFloat one, HalfFloat two);
inline HalfFloat operator/ (HalfFloat one, HalfFloat two);

inline float operator+ (HalfFloat one, float two);
inline float operator- (HalfFloat one, float two);
inline float operator* (HalfFloat one, float two);
inline float operator/ (HalfFloat one, float two);

inline float operator+ (float one, HalfFloat two);
inline float operator- (float one, HalfFloat two);
inline float operator* (float one, HalfFloat two);
inline float operator/ (float one, HalfFloat two);



///////////////////////////////////////////////////////////////////////////////////
/** 3. Specialization of std::numeric_limits for type half.
 */
///////////////////////////////////////////////////////////////////////////////////
namespace std {
template <>
class numeric_limits<HalfFloat> {

 public:

	// General -- meaningful for all specializations.

    static const bool is_specialized = true;
    static HalfFloat min ()
		{return HalfFloat(0,1,0);}
    static HalfFloat max ()
		{return HalfFloat((uint16_t)~0,HalfFloat::MAX_EXPONENT_VALUE-1,0);}
    static const int radix = 2;
    static const int digits = 10;   // conservative assumption
    static const int digits10 = 2;  // conservative assumption
	static const bool is_signed		= true;
    static const bool is_integer	= true;
    static const bool is_exact		= false;
    static const bool traps			= false;
    static const bool is_modulo		= false;
    static const bool is_bounded	= true;

	static const HalfFloat lowest() {
	  return HalfFloat((uint16_t)~0,HalfFloat::MAX_EXPONENT_VALUE-1,(uint16_t)~0);
	}

	// Floating point specific.

    static HalfFloat epsilon ()
		{return HalfFloat(0.00097656f);} // from OpenEXR, needs to be confirmed
    static HalfFloat round_error ()
		{return HalfFloat(0.00097656f/2);}
    static const int min_exponent10 = HalfFloat::MIN_EXPONENT10;
    static const int max_exponent10 = HalfFloat::MAX_EXPONENT10;
    static const int min_exponent   = HalfFloat::MIN_EXPONENT;
    static const int max_exponent   = HalfFloat::MAX_EXPONENT;

    static const bool has_infinity			= true;
    static const bool has_quiet_NaN			= true;
    static const bool has_signaling_NaN		= true;
    static const bool is_iec559				= false;
    static const bool has_denorm			= denorm_present;
    static const bool tinyness_before		= false;
    static const float_round_style round_style = round_to_nearest;

    static HalfFloat denorm_min ()
		{return HalfFloat(1,0,1);}
    static HalfFloat infinity ()
		{return HalfFloat(0,HalfFloat::MAX_EXPONENT_VALUE,0);}
    static HalfFloat quiet_NaN ()
		{return HalfFloat(1,HalfFloat::MAX_EXPONENT_VALUE,0);}
    static HalfFloat signaling_NaN ()
		{return HalfFloat(1,HalfFloat::MAX_EXPONENT_VALUE,0);}
 };
} // end namespace std


#include "umHalf.inl"

#ifndef UM_HALF_NO_TYPEDEFS
	typedef HalfFloat float16;
#endif

#endif // !! UM_HALF_H_INCLUDED
