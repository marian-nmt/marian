#ifndef EMULATOR
#define EMULATOR 0
#endif

#define VOCABSIZE 85120  //good multiple of 16 and 128
#define LAYER_DIM 512 // assuming to be multiple of 16

#define P 16 //should be multiple 16 for B loading logic to work
#define TILECOUNT (VOCABSIZE / P) //VOCABSIZE will be a good multiple of P 

#define WLOADTIME ((P * LAYER_DIM) >> 4) //using float16
#define BLOADTIME (P >> 4) //using float16

__attribute__((max_global_work_dim(0)))
__kernel void OutputLayer_float(
				__global float * restrict W,
				__global float * restrict X,
				__global float * restrict B,
				__global float * restrict Y,
				unsigned batchsize
                  )
{
#if EMULATOR == 1
    printf("OpenCL: OutputLayer_float, batchsize=%d \n",batchsize);
#endif

	__global volatile float16* restrict ddr_access_pointer;
	__global volatile float16* restrict Wpointer_prev;
	__global volatile float16* restrict Bpointer_prev;

    float Wlocal[P][LAYER_DIM];
    float Blocal[P];


	Wpointer_prev = (__global volatile float16 *)W;
	Bpointer_prev = (__global volatile float16 *)B;
	
	for (unsigned tile=0; tile < TILECOUNT; tile++) {
		ddr_access_pointer = (__global volatile float16 *)Wpointer_prev;
	
		unsigned wr_index=0;
		//fetch W and B to local
		for (unsigned i=0; i < (WLOADTIME + BLOADTIME); i++) {
	
			float16 temp_val = *ddr_access_pointer;
			if (i < WLOADTIME) {
				#pragma unroll 
				for (char u=0; u < 16; u++) {
					Wlocal[wr_index >> 5][(wr_index & 0x1F)*16+u]=temp_val[u]; // good for LAYER_DIM 512 (512/16=32)
				}
				wr_index++;
			}
			else {
				#pragma unroll 
				for (char u=0; u < 16; u++) {
					Blocal[wr_index*16+u]=temp_val[u]; // good for P as a multiple of 16
				}		
				wr_index++;
			}
			ddr_access_pointer++;
		
			if (i==(WLOADTIME-1)) { //we should keep track of W for the next batch
				Wpointer_prev = ddr_access_pointer;
				ddr_access_pointer = (__global volatile float16 *)Bpointer_prev; //would byte aligning be a problem?
				wr_index = 0;
			}
		}
		
		//do the matrix multiplication of tile with X
		
		
		
	}
		

}
	
 

