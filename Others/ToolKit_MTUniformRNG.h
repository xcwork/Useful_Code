#ifndef _TOOLKIT_MTUNIFORMRNG_H
#define _TOOLKIT_MTUNIFORMRNG_H





#include <vector>


namespace ToolKit 
{

   class MTUniformRNG 
   {
   private:
      static const unsigned int N=624; // state size
      static const unsigned int M=397; // shift size

   public:
      explicit MTUniformRNG (unsigned long seed = 0);
      explicit MTUniformRNG (const std::vector<unsigned long>& seeds);

      double next() const //! return a random number in the (0.0, 1.0)-interval
      {
         return (double(nextInt32()) + 0.5)/4294967296.0;
      }

      unsigned long nextInt32() const  //! return a random integer in the [0,0xffffffff]-interval
      {
         if (mti==N)
            twist(); /* generate N words at a time */

         unsigned long y = mt[mti++];

         /* Tempering */
         y ^= (y >> 11);
         y ^= (y << 7) & 0x9d2c5680UL;
         y ^= (y << 15) & 0xefc60000UL;
         y ^= (y >> 18);
         return y;
      }

   private:
      void seedInitialization(unsigned long seed);
      void twist() const;
      mutable unsigned long mt[N];
      mutable unsigned int mti;
      static const unsigned long MATRIX_A, UPPER_MASK, LOWER_MASK;
   };

}



#endif // _TOOLKIT_MTUNIFORMRNG_H