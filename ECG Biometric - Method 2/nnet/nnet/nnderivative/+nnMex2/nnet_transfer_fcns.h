
// ======================================================= TRANSFER FUNCTIONS

// Useful MATLAB functions:

// mxGetNaN() - returns NaN
// mxGetInf() - returns inf

// mxIsNaN(double) - true if NaN
// mxIsInf(PRECISION) - true if inf

// mxIsFinite(doule) - false if NaN, Inf or -Inf

// ------------------------------------------------------- COMPET

// Copyright 2012-2013 The MathWorks, Inc.

        
inline static void compet_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  if (S > 0)
    {
    INT64 maxInd = 0;
    PRECISION maxn = n[0];
    a[0] = 0;
    for (INT64 i=1; i<S; i++)
      {
      PRECISION ni = n[i];
      if (mxIsNaN(ni)) { maxn = ni; break; }
      if (maxn < ni) { maxn = ni; maxInd = i; }
      a[i] = 0;
      }
    if (mxIsNaN(maxn))
      for (INT64 i=0; i<S; i++) a[i] = maxn;
    else
      a[maxInd] = 1;
    }
  }

inline static void compet_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    dn[i] = 0.0;
  }

inline static void compet_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 ii=0; ii<S; ii++)
    da[ii] = 0.0; 
  }

// ------------------------------------------------------- HARDLIM

inline static void hardlim_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = mxIsNaN(ni) ? ni : (ni >= 0);
    }
  }

inline static void hardlim_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    dn[i] = 0.0;
  }

inline static void hardlim_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 ii=0; ii<S; ii++)
      da[ii] = 0.0;
  }

// ------------------------------------------------------- ELLIOTSIG

// a = n ./ (1 + abs( n));

inline static void elliotsig_apply(PRECISION *a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = ni / (1+fabs(ni));
    }
  }

// d = 1 ./ ((1+abs(n)).^2);

inline static void elliotsig_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  //  d = (1-abs(a)).^2;
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    PRECISION temp = 1-fabs(ai);
    PRECISION dni = temp*temp;
    dn[i] = da[i]*dni;
    }
  }

inline static void elliotsig_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    PRECISION temp = 1-fabs(ai);
    PRECISION dai = temp*temp;
    da[i] = dn[i]*dai;
    }
  }

// ------------------------------------------------------- ELLIOTSIG2

inline static void elliot2sig_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
	  PRECISION ni2 = ni*ni;
	  a[i] = mxIsNaN(ni) ? ni : (((ni >= 0) ? 1 : -1)*ni2/(1+ni2));
    }
  }

inline static void elliot2sig_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
   {
	 PRECISION ni = n[i];
	 PRECISION ni2 = 1+ni*ni;
	 PRECISION dni = 2*((ni >= 0) ? 1 : -1)*ni/(ni2*ni2);
	 dn[i] = da[i]*dni;
   }
 }

inline static void elliot2sig_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
	  PRECISION ni = n[i];
	  PRECISION ni2 = 1+ni*ni;
	  PRECISION dai = 2*((ni >= 0) ? 1 : -1)*ni/(ni2*ni2);
	  da[i] = dn[i]*dai;
    }
  }

// ------------------------------------------------------- HARDLIMS

inline static void hardlims_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = mxIsNaN(ni) ? ni : ((ni >= 0) ? 1 : -1);
    }
  }

inline static void hardlims_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 i=0; i<S; i++)
    dn[i] = 0.0;
  }

inline static void hardlims_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 ii=0; ii<S; ii++)
      da[ii] = 0.0;
  }

// ------------------------------------------------------- LOGSIG

inline static void logsig_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    a[i] = 1.0 / (1.0 + exp(-n[i]));
  }

inline static void logsig_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    dn[i] = da[i] * (ai*(1-ai));
    }
  }

inline static void logsig_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 ii=0; ii<S; ii++)
    {
    PRECISION aii = a[ii];
    PRECISION da_dn = (aii*(1-aii));
    da[ii] = dn[ii]*da_dn;
    }
  }

// ------------------------------------------------------- NETINV

inline static void netinv_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    PRECISION e = (ni>=0) ? EPS : NEGEPS;
    ni += e;
    a[i] = 1/ni;
    }
  }

inline static void netinv_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    dn[i] = da[i]*-(ai*ai);
    }
  }

inline static void netinv_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    da[i] = dn[i]*-(ai*ai);
    }
  }

// ------------------------------------------------------- POSLIN

inline static void poslin_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = mxIsNaN(ni) ? ni : ((ni > 0.0) ? ni : 0);
    }
  }

inline static void poslin_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 i=0; i<S; i++)
  {
  PRECISION ni = n[i];
    dn[i] = da[i]*((ni >= 0.0) ? 1 : 0);
  }
  }

inline static void poslin_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
    for (INT64 ii=0; ii<S; ii++)
    {
    PRECISION nii = n[ii];
  PRECISION da_dn = ((nii >= 0.0) ? 1 : 0);
    da[ii] = dn[ii]*da_dn;
    }
  }

// ------------------------------------------------------- PURELIN

inline static void purelin_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    a[i] = n[i];
  }

inline static void purelin_apply2(PRECISION *a,const PRECISION * const n,
  const INT64 S, const INT64 Q)
  {
  for (INT64 q=0; q<Q; q++)
    for (INT64 i=0; i<S; i++)
      a[i+q*S] = n[i+q*S];
  }

inline static void purelin_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    dn[i] = da[i];
  }

inline static void purelin_backprop2(PRECISION * const dn,
  const PRECISION * const da, const PRECISION * const n, const PRECISION * const a,
  const INT64 S, const INT64 Q)
  {
  for (INT64 q=0; q<Q; q++)
    for (INT64 i=0; i<S; i++)
      dn[i+q*S] = da[i+q*S];
  }

/*inline static void purelin_forwardprop(PRECISION * const dA, const PRECISION * const dN, const PRECISION * const N, const PRECISION * const A,
  const INT64 S, const INT64 Q, const INT64 Nj)
  {
  const INT64 numElements = S*Q*Nj;
  for (INT64 k=0; k<numElements; k++)
    dA[k] = dN[k];
  }*/

inline static void purelin_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    da[i] = dn[i];
  }

// ------------------------------------------------------- RADBAS

inline static void radbas_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = exp(-(ni*ni));
    }
  }

inline static void radbas_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    dn[i] = da[i] * (-2*a[i]*n[i]);
  }

inline static void radbas_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    da[i] = dn[i] * (-2*a[i]*n[i]);
  }

// ------------------------------------------------------- RADBASN

inline static void radbasn_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  if (S == 0) return;
  
  // Find minAbsN
  PRECISION minAbsN = fabs(n[0]);
  for (INT64 i=1; i<S; i++)
    {
    PRECISION absNi = fabs(n[i]);
    if (absNi < minAbsN) minAbsN = absNi;
    }
  
  // Exp(-n^2) and sum of those
  PRECISION minNSquared = minAbsN*minAbsN;
  PRECISION suma = 0.0;
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    PRECISION ai = exp(-(ni*ni)+minNSquared);
    suma += ai;
    a[i] = ai;
    }
  
  // Normalize or set to zero
  if (suma == 0)
    { for (INT64 i=0; i<S; i++) a[i] = 0; }
  else
    { for (INT64 i=0; i<S; i++) a[i] /= suma; }
  }

inline static void radbasn_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION dni = 0.0;
    PRECISION nai = a[i]*n[i];
    for (INT64 j=0; j<S; j++)
      {
      if (i == j)
        dni += -nai*(1-a[j])*da[j];
      else
        dni += nai*a[j]*da[j];
      }
    dn[i] = 2*dni;
    }
  }

inline static void radbasn_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION dai = 0.0;
    PRECISION ai = a[i];
    for (INT64 j=0; j<S; j++)
      {
    PRECISION naj = a[j]*n[j];
      if (i == j)
        dai += -naj*(1-ai)*dn[j];
      else
        dai += naj*ai*dn[j];
      }
    da[i] = 2*dai;
    }
  }

// ------------------------------------------------------- SATLIN

inline static void satlin_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = mxIsNaN(ni) ? ni : ((ni < 0.0) ? 0.0 : ((ni > 1.0) ? 1.0 : ni));
    }
  }

inline static void satlin_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    dn[i] = da[i]*(((ni >= 0.0) && (ni <= 1)) ? 1 : 0);
    }
  }

inline static void satlin_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 ii=0; ii<S; ii++)
    {
    PRECISION nii = n[ii];
    PRECISION da_dn = (((nii >= 0.0) && (nii <= 1)) ? 1 : 0);
    da[ii] = dn[ii]*da_dn;
    }
  }

// ------------------------------------------------------- SATLINS

inline static void satlins_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    a[i] = mxIsNaN(ni) ? ni : ((ni < -1.0) ? -1.0 : ((ni > 1.0) ? 1.0 : ni));
    }
  }

inline static void satlins_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    dn[i] = da[i]*(((ni >= -1) && (ni <= 1)) ? 1 : 0);
    }
  }

inline static void satlins_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 ii=0; ii<S; ii++)
    {
    PRECISION nii = n[ii];
    PRECISION da_dn = (((nii >= -1) && (nii <= 1)) ? 1 : 0);
    da[ii] = dn[ii]*da_dn;
    }
  }

// ------------------------------------------------------- SOFTMAX

inline static void softmax_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  PRECISION nmax = 0;
  for (INT64 i=0; i<S; i++)
    if (n[i] > nmax) nmax = n[i];

  PRECISION denom = 0.0;
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = exp(n[i]-nmax);
    denom += ai;
    a[i] = ai;
    }
  
  if (mxIsNaN(denom))
    for (INT64 i=0; i<S; i++) a[i] = denom;
  else
    {
    if (denom == 0) denom = 1;
    for (INT64 i=0; i<S; i++)
      a[i] /= denom;
    }
  }

inline static void softmax_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION dni = 0.0;
    PRECISION ai = a[i];
    for (INT64 j=0; j<S; j++)
      {
      if (i == j)
        dni += ai*(1-ai)*da[j];
      else
        dni += -ai*a[j]*da[j];
      }
    dn[i] = dni;
    }
  }

inline static void softmax_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION dai = 0.0;
    PRECISION ai = a[i];
    for (INT64 j=0; j<S; j++)
      {
      if (i == j)
        dai += ai*(1-ai)*dn[j];
      else
        dai += -a[j]*ai*dn[j];
      }
    da[i] = dai;
    }
  }

// ------------------------------------------------------- TANSIG

inline static void tansig_apply(PRECISION *a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    a[i] = 2 / (1 + exp(-2*n[i])) - 1;
  }

inline static void tansig_apply2(PRECISION *a,const PRECISION * const n,
  const INT64 S, const INT64 Q)
  {
  for (INT64 q=0; q<Q; q++)
    for (INT64 i=0; i<S; i++)
      a[i+q*S] = 2 / (1 + exp(-2*n[i+q*S])) - 1;
  }

inline static void tansig_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    dn[i] = da[i] * (1 - (ai*ai));
    }
  }

inline static void tansig_backprop2(PRECISION * const dn,
  const PRECISION * const da, const PRECISION * const n, const PRECISION * const a,
  const INT64 S, const INT64 Q)
  {
  for (INT64 q=0; q<Q; q++)
    //tansig_backprop(dn+q*S,da+q*S,S,n+q*S,a+q*S);
    for (INT64 i=0; i<S; i++)
      {
      PRECISION ai = a[i+q*S];
      dn[i+q*S] = da[i+q*S] * (1 - (ai*ai));
      }
  }

/*
inline static void tansig_forwardprop2(PRECISION * const dA, const PRECISION * const dN, const PRECISION * const N, const PRECISION * const A,
  const INT64 S, const INT64 Q, const INT64 Nj)
  {
  PRECISION * da_dn = new PRECISION[S*Q];
  for (INT64 q=0; q<Q; q++)
    for (INT64 i=0; i<S; i++)
      {
      PRECISION ai = A[i+q*S];
      da_dn[i+q*S] = (1 - (ai*ai));
      }
  
  for (INT64 k=0; k<Nj; k++)
    for (INT64 q=0; q<Q; q++)
      for (INT64 i=0; i<S; i++)
        {
        INT64 ii = i+q*S;
        INT64 jj = ii + k*S*Q;
        dA[jj] = dN[jj] * da_dn[ii];
        }
  
  free(da_dn);
  }
*/

inline static void tansig_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ai = a[i];
    da[i] = dn[i] * (1 - (ai*ai));
    }
  }

// ------------------------------------------------------- TRIBAS

inline static void tribas_apply(PRECISION * const a, const INT64 S, const PRECISION * const n)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    PRECISION absn = fabs(ni);
    a[i] = mxIsNaN(ni) ? ni : ((absn<1) ? (1-absn) : 0);
    }
  }

inline static void tribas_backprop(PRECISION * const dn, const PRECISION * const da, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 i=0; i<S; i++)
    {
    PRECISION ni = n[i];
    dn[i] = da[i]*(((ni >= -1) && (ni <= 1)) ? ((ni <= 0) ? 1 : -1) : 0);
    }
  }

inline static void tribas_forwardprop(PRECISION * const da, const PRECISION * const dn, const INT64 S, const PRECISION * const n, const PRECISION * const a)
  {
  for (INT64 ii=0; ii<S; ii++)
    {
    PRECISION nii = n[ii];
    PRECISION da_dn = (((nii >= -1) && (nii <= 1)) ? ((nii <= 0) ? 1 : -1) : 0);
    da[ii] = dn[ii]*da_dn;
    }
  }

// =======================================================

