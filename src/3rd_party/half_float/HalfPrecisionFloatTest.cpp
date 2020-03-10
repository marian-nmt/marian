


#include "umHalf.h"
#include <iostream>
#include <assert.h>

#define VALIDATE(x) if (!(x)){std::cout << "Failed: " <<  #x << std::endl;assert((x));}

int main(int, char*)
{
   half h = 1.f, h2 = 2.f;
   --h2;
   ++h2;
   --h;
   ++h;
   h2 -= 1.f;
   float f = h2, f2 = h;
   VALIDATE(1.f == f && f == f2);

   h = h2;
   h2 = 15.5f;

   f = h2, f2 = h;
   VALIDATE(15.5f == f && 1.f == f2);
   h2 *= h;
   f = h2, f2 = h;
   VALIDATE(15.5f == f && 1.f == f2);
   h2 /= h;
   f = h2, f2 = h;
   VALIDATE(15.5f == f && 1.f == f2);
   h2 += h;
   f = h2, f2 = h;
   VALIDATE(16.5f == f && 1.f == f2);
   h++;h++;h++;
   h2 = -h2;
   h2 += 17.5f;
   h2 *= h;
   f = h2, f2 = h;
   VALIDATE(4.f == f && 4.f == f2);	
   VALIDATE(h == h2);
   VALIDATE(h <= h2);
   --h;
   VALIDATE(h <= h2);
  
   h -= 250.f;
   VALIDATE(h < h2);

   h += 500.f;
   VALIDATE(h > h2);
   VALIDATE(h >= h2);

   f = h2, f2 = h;
   VALIDATE(h * h2 == (half)(f * f2));

   // addition
   // ****************************************************************************

   // identical exponents
   for (float v = 0.f; v < 1000.f; ++v)
   {
	  half one = v;
	  half two = v;
      half three = one + two;
      f2 = three;
      VALIDATE(v*2.f == f2);
   }

    // different exponents
   for (float v = 0.f, fp = 1000.f; v < 500.f; ++v, --fp)
   {
	  half one = v;
	  half two = fp;
      half three = one + two;
      f2 = three;
      VALIDATE(v+fp == f2);
   }

   // very small numbers - this is already beyond the accuracy of 16 bit floats.
   for (float v = 0.003f; v < 1000.f; v += 0.0005f)
   {
	  half one = v;
	  half two = v;
      half three = one + two;
      f2 = three;
	  float m = v*2.f;
      VALIDATE(f2 > (m-0.05*m) && f2 < (m+0.05*m));
   }

 
   // subtraction
   // ****************************************************************************

   // identical exponents
   for (float v = 0.f; v < 1000.f; ++v)
   {
	  half one = v;
	  half two = v;
      half three = one - two;
      f2 = three;
      VALIDATE(0.f == f2);
   }

   // different exponents
   for (float v = 0.f, fp = 1000.f; v < 500.f; ++v, --fp)
   {
	  half one = v;
	  half two = fp;
      half three = one - two;
      f2 = three;
      VALIDATE(v-fp == f2);
   }
   return 0;		
}

