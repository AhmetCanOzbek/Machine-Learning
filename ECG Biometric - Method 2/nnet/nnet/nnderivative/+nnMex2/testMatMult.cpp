#include <math.h>
#include <stdio.h>
#include "mex.h"
#include "blas.h"

#define INT64 long long
 
void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray * prhs[])
  {
  // Inputs (A,B)
  double * const A = mxGetPr(prhs[0]); // A is MxK
  double * const B = mxGetPr(prhs[1]); // B is KxN
  
  const mwSize * Adimensions = mxGetDimensions(prhs[0]);
  const INT64 M = Adimensions[0];
  const INT64 K = Adimensions[1];
  const mwSize * Bdimensions = mxGetDimensions(prhs[1]);
  const INT64 N = Bdimensions[1];
  
  // Outputs C
  plhs[0] = mxCreateDoubleMatrix((mwSize) M,(mwSize) N,mxREAL);
  double * const C = mxGetPr(plhs[0]);
  
  #if 1
    double one = 1.0;
    double zero = 0.0;
    char Nchar = 'N';
    dgemm(&Nchar,&Nchar,(ptrdiff_t *)&M,(ptrdiff_t *)&N,(ptrdiff_t *)&K,&one,A,(ptrdiff_t *)&M,B,(ptrdiff_t *)&K,&zero,C,(ptrdiff_t *) &M);
  #else
    for (int i=0; i<M; i++)
      for (int j=0; j<N; j++)
        {
        double sum = 0;
        for (int k=0; k<K; k++)
          sum += A[i+k*M] * B[k+j*K]; // A[i,k] * B[k,j]
        C[i+j*M] = sum; // C[i,j]
        }
  #endif
  }
