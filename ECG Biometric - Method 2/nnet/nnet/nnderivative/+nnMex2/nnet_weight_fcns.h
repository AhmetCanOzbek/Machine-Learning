
// ======================================================= WEIGHT FUNCTIONS

// Useful MATLAB functions:

// mxGetNaN() - returns NaN
// mxGetInf() - returns inf

// mxIsNaN(double) - true if NaN
// mxIsInf(double) - true if inf

// mxIsFinite(double) - false if NaN, Inf or -Inf

// ------------------------------------------------------- DOTPROD

//#include <C:\Program Files (x86)\Intel\Composer XE 2011 SP1\mkl\include\mkl.h> 

// DGEMM(trasposeA?,transposeB?,M,N,K,alpha,A,strideA,B,strideB,beta,C,strideC)
// C = alpha*A*B + beta*C

inline static void dotprod_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION sum = 0.0;
    for (INT64 j=0; j<R; j++)
      sum += w[i+j*S]*p[j];
    z[i] = sum;
    }
  }

inline static void dotprod_netsum_apply(PRECISION * const N, PRECISION * const W, PRECISION * const P,
  const INT64 R, const INT64 S, const INT64 Q, const INT64 pStride)
  {
  // N(SxQ) += W(SxR) * P(RxQ)
  // A <- W
  // B <- P
  // C <- N
  // M <- S
  // N <- Q
  // K <- R
  #if 1
    dgemm(&blasN, &blasN, (ptrdiff_t *) &S,  (ptrdiff_t *) &Q,  (ptrdiff_t *) &R, &blas1, W, (ptrdiff_t *) &S, P, (ptrdiff_t *) &pStride, &blas1, N, (ptrdiff_t *) &S);
  #else
    for (INT64 q=0; q<Q; q++)
      {
      for (INT64 i=0; i<S; i++)
        {
        PRECISION sum = N[i+q*S];
        for (INT64 j=0; j<R; j++)
          sum += W[i+j*S]*P[j+q*pStride];
        N[i+q*S] += sum;
        }
      }
  #endif
  }

inline static void dotprod_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 j=0; j<R; j++)
    {
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++)
      dpj += w[i+S*j]*dz[i];
    dp[j] = dpj;
    }
  }

// DGEMM(trasposeA?,transposeB?,M,N,K,alpha,A,strideA,B,strideB,beta,C,strideC)
// C = alpha*A*B + beta*C

//(dXd,dz,w,ad,z,Rd,S,Qb)

inline static void dotprod_backprop_batch(PRECISION * const dP, PRECISION * const dZ, PRECISION * const W, PRECISION * const P, PRECISION * const Z,
  const INT64 R, const INT64 S, const INT64 Q)
  {
  // dP(RxQ) = W(SxR)' * dZ(SxQ)
  // A <- W (transpose)
  // B <- dZ
  // C <- dP
  // M <- R
  // N <- Q
  // K <- S
  #if 1
     dgemm(&blasT, &blasN, (ptrdiff_t *) &R, (ptrdiff_t *) &Q, (ptrdiff_t *) &S, &blas1, W, (ptrdiff_t *) &S, dZ, (ptrdiff_t *) &S, &blas0, dP, (ptrdiff_t *) &R);
  #else
    for (INT64 q=0; q<Q; q++)
      for (INT64 j=0; j<R; j++)
        {
        PRECISION dpj = 0;
        for (INT64 i=0; i<S; i++)
          dpj += W[i+S*j]*dZ[i+q*S];
        dP[j+q*R] = dpj;
        }
  #endif
  }

inline static void dotprod_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
      dw[i+S*j] += dz[i] * p[j];
  }

inline static void dotprod_backstop_batch(PRECISION * const dW, PRECISION * const dZ, PRECISION * const W, PRECISION * const P, PRECISION * const Z,
  const INT64 R, const INT64 S, const INT64 Q, const INT64 pStride)
  {
  // dW(SxR) += dZ(SxQ) * P(RxQ)'
  // A <- dZ
  // B <- P (transpose)
  // C <- dW
  // M <- S
  // N <- Q
  // K <- R
  #if 1
     dgemm(&blasN,&blasT, (ptrdiff_t *) &S, (ptrdiff_t *) &R, (ptrdiff_t *) &Q, &blas1, dZ, (ptrdiff_t *) &S, P, (ptrdiff_t *) &pStride, &blas1, dW, (ptrdiff_t *) &S);
  #else
    for (INT64 q=0; q<Q; q++)
      for (INT64 i=0; i<S; i++)
        for (INT64 j=0; j<R; j++)
          dW[i+S*j] += dZ[i+q*S] * P[j+q*pStride];
  #endif
  }

inline static void dotprod_forwardstart_batch(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 Q, INT64 pStride)
  {
  for (INT64 q=0; q<Q; q++)
    for (INT64 j=0; j<R; j++)
      {
      PRECISION pj = p[j+q*pStride];
      for (INT64 i=0; i<S; i++)
        {
        INT64 col = i+j*S;
        dz[i + (q + col*Q)*S] += pj; // dz[i + col*Q*S + q*S
        }
      }
  }

inline static void dotprod_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 j=0; j<R; j++)
    {
    PRECISION pj = p[j];
    for (INT64 i=0; i<S; i++)
      {
      INT64 k = i+j*S;
      dz[i + k*kStride] += pj;
      }
    }
  }

inline static void dotprod_forwardprop_batch(PRECISION *const dz,
  PRECISION * const dp, PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, const INT64 Q, const INT64 Nj)
  {
  const INT64 QNj = Q*Nj;
  #if 1
     dgemm(&blasN, &blasN, (ptrdiff_t *) &S, (ptrdiff_t *) &QNj, (ptrdiff_t *) &R, &blas1, w, (ptrdiff_t *) &S, dp, (ptrdiff_t *) &R, &blas1, dz, (ptrdiff_t *) &S);
  #else
  for (INT64 k=0; k<Nj; k++)
    for (INT64 q=0; q<Q; q++)
      for (INT64 i=0; i<S; i++)
        {
        PRECISION dzi = 0;
        for (INT64 j=0; j<R; j++)
          dzi += w[i+j*S] * dp[j+(q+k*Q)*R];
        dz[i+(q+k*Q)*S] += dzi;
        }
  #endif
  }

inline static void dotprod_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION dzi = 0;
    for (INT64 j=0; j<R; j++)
      dzi += w[i+j*S] * dp[j];
    dz[i] += dzi;
    }
  }

// ------------------------------------------------------- BOXDIST

inline static void boxdist_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION max = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION pj = p[j];
      if (mxIsNaN(pj))
        {
        max = pj;
        break;
        }
      PRECISION diff = fabs(w[i+j*S]-pj);
      if (diff > max) max = diff;
      }
    z[i] = max;
    }
  }

inline static void boxdist_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  PRECISION *d = new PRECISION[static_cast<size_t>(S*R)];
  for (INT64 i=0; i<S; i++)
    {
    PRECISION z3 = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      z1 = p[j] - w[i+S*j];
      if (fabs(z1) > z3) z3 = fabs(z1);
      }
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      z1 = p[j] - w[i+S*j];
      d[i+j*S] = (fabs(z1) == z3) ? ((z1 == z3) ? 1 :- 1) : 0;
      }
    }
  for (INT64 j=0; j<R; j++)
    {
    PRECISION pj = p[j];
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++) dpj += d[i+j*S]*dz[i]; 
    dp[j] = dpj;
    }
  delete[] d;
  }

inline static void boxdist_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION z3 = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      z1 = w[i+S*j] - p[j];
      if (fabs(z1) > z3) z3 = fabs(z1);
      }
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      PRECISION d;
      z1 = w[i+S*j] - p[j];
      d = (fabs(z1) == z3) ? ((z1 == z3) ? 1 :- 1) : 0;
      dw[i+S*j] += dz[i] * d;
      }        
    }
  }

inline static void boxdist_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION z3 = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      INT64 k = i+j*S;
      PRECISION z1;
      z1 = w[k] - p[j];
      if (fabs(z1) > z3) z3 = fabs(z1);
      }
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      PRECISION d;
      INT64 k = i+j*S;
      z1 = w[k] - p[j];
      d = (fabs(z1) == z3) ? ((z1 == z3) ? 1 :- 1) : 0;
      dz[i + k*kStride] += d;
      }
    }
  }

inline static void boxdist_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION z3 = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      z1 = p[j] - w[i+S*j];
      if (fabs(z1) > z3) z3 = fabs(z1);
      }
    for (INT64 j=0; j<R; j++)
      {
      PRECISION z1;
      PRECISION d;
      z1 = p[j] - w[i+S*j];
      d = (fabs(z1) == z3) ? ((z1 == z3) ? 1 :- 1) : 0;
      dz[i] += d * dp[j];
      }
    }
  }

// ------------------------------------------------------- CONVWF

inline static void convwf_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  INT64 N = R-S+1;
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = 0.0;
    for (INT64 j=0; j<N; j++)
      zi += w[j]*p[i+j];
    z[i] = zi;
    }
  }

inline static void convwf_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  INT64 N = R-S+1;
  for (INT64 j=0; j<R; j++)
    {
     PRECISION dpj = 0.0;
     for (INT64 i=0; i<N; i++)
       {
       INT64 ji = j-i;
       if ((ji >= 0)&&(ji < S)) dpj += w[i]*dz[ji];
       }
     dp[j] = dpj;
    }
  }

inline static void convwf_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  INT64 N = R-S+1;
  for (INT64 i=0; i<N; i++)
    {
    PRECISION dwi = 0.0;
    for (INT64 j=0; j<S; j++)
      dwi += p[i+j] * dz[j];
    dw[i] += dwi;
    }
  } 

inline static void convwf_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
 {
  INT64 N = R-S+1;
  for (INT64 i=0; i<S; i++)
    for (INT64 k=0; k<N; k++)
      dz[i+k*kStride] += p[i+k];
 } 

inline static void convwf_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
 {
  INT64 N = R-S+1;
  for (INT64 i=0; i<S; i++)
  {
    PRECISION dni = 0.0;
    for (INT64 j=0; j<N; j++)
       dni += w[j] * dp[i+j];
    dz[i] += dni;
  }
 }

// ------------------------------------------------------- DIST

inline static void dist_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION sum = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION diff = w[i+j*S]-p[j];
      sum += diff*diff;
      }
    z[i] = sqrt(sum);
    }
  }

inline static void dist_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 j=0; j<R; j++)
    {
    PRECISION pj = p[j];
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++)
      {
      PRECISION dij = (pj - w[i+S*j])/z[i];
      if (mxIsNaN(dij)) dij = 0;
      dpj += dij * dz[i];
      }
    dp[j] = dpj;
    }
  }

inline static void dist_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    for (INT64 j=0; j<R; j++)
      {
      PRECISION dij = (w[i+S*j] - p[j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dw[i+S*j] += dz[i] * dij;
      }
    }
  }

inline static void dist_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    for (INT64 j=0; j<R; j++)
      {
      INT64 k = i+j*S;
      PRECISION dij = (w[k] - p[j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dz[i + k*kStride] += dij;
      }
    }
  }

inline static void dist_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    PRECISION dzi = 0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION dij = (p[j] - w[i+S*j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dzi += dij * dp[j];
      }
    dz[i] += dzi;
    }
  }

// ------------------------------------------------------- LINKDIST

inline static void linkdist_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    z[i] = 0;
  }

inline static void linkdist_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 j=0; j<R; j++)
    dp[j] = 0.0;
  }

inline static void linkdist_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
 {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
      dw[i+S*j] = 0.0; 
  }

inline static void linkdist_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
      {
      INT64 k=i+j*S;
      dz[i + k*kStride] += 0.0;
      }
  }

inline static void linkdist_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    dz[i] += 0.0;
  }

// ------------------------------------------------------- MANDIST

inline static void mandist_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION sum = 0.0;
    for (INT64 j=0; j<R; j++)
      sum += fabs(w[i+j*S]-p[j]);
    z[i] = sum;
    }
  }

inline static void mandist_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 j=0; j<R; j++)
    {
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++) 
      {
      PRECISION zj;
      PRECISION d;
      zj = p[j] - w[i+S*j];
      d = (zj > 0) - (zj < 0);
      dpj += d*dz[i];
      }
    dp[j] = dpj;
    }
  }

inline static void mandist_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
        {
        PRECISION zj;
        PRECISION d;
        zj = w[i+S*j] - p[j];
        d = (zj > 0) - (zj < 0);
        dw[i+S*j] += dz[i] * d;
        }
  }

inline static void mandist_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
   {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
      {
      PRECISION zj;
      PRECISION d;
      INT64 k = i+j*S;
      zj = w[k] - p[j];
      d = (zj > 0) - (zj < 0);
      dz[i + k*kStride] += d;
      }
  }


inline static void mandist_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
    {
      PRECISION zj;
      PRECISION d;
      zj = p[j] - w[i+S*j];
      d = (zj > 0) - (zj < 0);
      dz[i] += d * dp[j];
    }
  }

// ------------------------------------------------------- NEGDIST

inline static void negdist_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION sum = 0.0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION diff = w[i+j*S]-p[j];
      sum += diff*diff;
      }
    z[i] = -sqrt(sum);
    }
  }

inline static void negdist_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 j=0; j<R; j++)
    {
    PRECISION pj = p[j];
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++)
      {
      PRECISION dij = (pj - w[i+S*j])/z[i];
      if (mxIsNaN(dij)) dij = 0;
      dpj += dij * dz[i];
      }
    dp[j] = dpj;
    }
  }

inline static void negdist_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    for (INT64 j=0; j<R; j++)
      {
      PRECISION dij = (w[i+S*j] - p[j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dw[i+S*j] += dz[i] * dij;
      }
    }
  }

inline static void negdist_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    for (INT64 j=0; j<R; j++)
      {
      INT64 k = i+j*S;
      PRECISION dij = (w[k] - p[j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dz[i + k*kStride] += dij;
      }
    }
  }

inline static void negdist_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    PRECISION dzi = 0;
    for (INT64 j=0; j<R; j++)
      {
      PRECISION dij = (p[j] - w[i+S*j])/zi;
      if (mxIsNaN(dij)) dij = 0;
      dzi += dij * dp[j];
      }
    dz[i] += dzi;
    }
  }

// ------------------------------------------------------- NORMPROD

inline static void normprod_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  if (R == 0)
    for (INT64 i=0; i<S; i++) z[i] = 0.0;
  else
    {
    PRECISION sum = 0.0;
    for (INT64 j=0; j<R; j++) 
      {
      PRECISION pj = p[j];
      sum += fabs(pj + 1e-20*(( pj > 0 ) - ( pj < 0 )));
      }
    if (mxIsNaN(sum))
      {
      for (INT64 i=0; i<S; i++)
        z[i] = sum;
      }
    else if (sum == 0)
      {
      for (INT64 i=0; i<S; i++)
        z[i] = 0;
      }
    else
      {
      PRECISION denom = 1.0/sum;
      for (INT64 i=0; i<S; i++)
        {
        PRECISION zi = 0;
        for (INT64 j=0; j<R; j++)
          {
          PRECISION pj = p[j];
          zi += w[i+j*S]*((p[j]+ 1e-20*(( pj > 0 ) - ( pj < 0 )))*denom);
          }
        z[i] = zi;
        }
      }
    }
  }

inline static void normprod_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  PRECISION sum = 0.0;
  for (INT64 j=0; j<R; j++) sum += fabs(p[j]);
  PRECISION denom = 0.0;
  if (sum != 0)
    denom = 1.0/sum;
  for (INT64 j=0; j<R; j++)
    {
    PRECISION pj = p[j];
    PRECISION dpj = 0;
    for (INT64 i=0; i<S; i++) dpj += (w[i+S*j]-z[i]*(( pj > 0 ) - ( pj < 0 )))*dz[i];
    dp[j] = dpj*denom;
    }
  }

inline static void normprod_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  PRECISION sum = 0.0;
  for (INT64 j=0; j<R; j++) sum += fabs(p[j]);
  PRECISION denom = 0.0;
  if (sum != 0)
    denom = 1.0/sum;
  for (INT64 i=0; i<S; i++)
      for (INT64 j=0; j<R; j++)
        dw[i+S*j] += dz[i] * (p[j]*denom);
  }

inline static void normprod_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  PRECISION sum = 0.0;
  for (INT64 j=0; j<R; j++) sum += fabs(p[j]);
  PRECISION denom = 0.0;
  if (sum != 0)
    denom = 1.0/sum;
  for (INT64 i=0; i<S; i++)
    for (INT64 j=0; j<R; j++)
      {
      INT64 k = i+j*S;
      dz[i + k*kStride] += p[j]*denom;
      }
  }

inline static void normprod_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  PRECISION sum = 0.0;
  for (INT64 j=0; j<R; j++) sum += fabs(p[j]);
  PRECISION denom = 0.0;
  if (sum != 0)
    denom = 1.0/sum;
  for (INT64 i=0; i<S; i++)
    {
    PRECISION zi = z[i];
    PRECISION dni = 0;
    for (INT64 j=0; j<R; j++) dni += (w[i+S*j]-zi*(( p[j] > 0 ) - ( p[j] < 0 )))*dp[j];
    dz[i] += dni*denom;
    }
  }

// ------------------------------------------------------- SCALPROD

inline static void scalprod_apply(PRECISION * const z, const INT64 S, const INT64 R, PRECISION * const w, PRECISION * const p)
  {
  PRECISION ws = *w;
  for (INT64 i=0; i<S; i++)
    z[i] = ws*p[i];
  }

inline static void scalprod_backprop(PRECISION * const dp, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  PRECISION ws = *w;
  for (INT64 i=0; i<S; i++)
    dp[i] = ws*dz[i];
  }

inline static void scalprod_backstop(PRECISION * const dw, PRECISION * const dz, const INT64 S, const INT64 R,
PRECISION * const w, PRECISION * const p, PRECISION * const z)
  {
  PRECISION sum = 0.0;
  for (INT64 i=0; i<S; i++)
      sum += dz[i] * p[i];
  *dw += sum;
  }

inline static void scalprod_forwardstart(PRECISION *const dz,
PRECISION * const w, PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S, INT64 kStride)
  {
  for (INT64 i=0; i<S; i++)
    dz[i] += p[i];
  }

inline static void scalprod_forwardprop(PRECISION *const dz,
  PRECISION * const dp,PRECISION * const w,PRECISION * const p, PRECISION * const z,
  const INT64 R, const INT64 S)
  {
  PRECISION ws = *w;
  for (INT64 i=0; i<S; i++)
    dz[i] += ws * dp[i];
  }

// =======================================================
