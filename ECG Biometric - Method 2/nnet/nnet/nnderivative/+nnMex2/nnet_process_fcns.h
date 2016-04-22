
// ======================================================= PROCESSING FUNCTIONS

// Useful MATLAB functions:

// mxGetNaN() - returns NaN
// mxGetInf() - returns inf

// mxIsNaN(PRECISION) - true if NaN
// mxIsInf(PRECISION) - true if inf

// mxIsFinite(doule) - false if NaN, Inf or -Inf

// ------------------------------------------------------- MAPMINMAX

// Floating Point Parameters:
//   gain - Xsize elements
//   xoffset - Xsize elements
//   yoffset - 1 element

// Integer Parameters: None

// Dimensions: Xsize == Ysize

inline static void mapminmax_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize,
  const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;
  const PRECISION * xoffset = floatParam + Xsize;
  const PRECISION ymin = *(floatParam + 2*Xsize);

  for (INT64 i=0; i<Xsize; i++)
    y[i] = ((x[i] - xoffset[i]) * gain[i]) + ymin;
  }

inline static void mapminmax_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize,
  const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;
  const PRECISION * xoffset = floatParam + Xsize;
  const PRECISION ymin = *(floatParam + 2*Xsize);

  for (INT64 i=0; i<Xsize; i++)
    x[i] = ((y[i] - ymin) / gain[i]) + xoffset[i];
  }

inline static void mapminmax_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize,
  const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;
  
  for (INT64 i=0; i<Xsize; i++)
    dy[i] = dx[i] / gain[i];
  }

inline static void mapminmax_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize,
  const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;

  for (INT64 i=0; i<Xsize; i++)
    dx[i] = dy[i] / gain[i];
  }

// ------------------------------------------------------- FIXUNKNOWNS

// Floating Point Parameters: None
//   xmean - Xsize elements

// Integer Parameters
//   shift - Xsize elements, indicating shifted position of each x in y
//   unknown - (Ysize-Xsize) indices of expanded elements

// Dimensions: Xsize != Ysize

inline static void fixunknowns_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 numUnknown = Ysize-Xsize;
  const PRECISION * const xmean = floatParam;
  const INT64 * const shift = intParam;
  const INT64 * const unknown = intParam + Xsize;

  for (INT64 i=0; i<Xsize; i++)
    {
    PRECISION xi = x[i];
    y[i+shift[i]] = mxIsNaN(xi) ? xmean[i] : xi;
    }
  for (INT64 i=0; i<numUnknown; i++)
    {
    INT64 unknown_ind = unknown[i];
    y[unknown_ind+shift[unknown_ind]+1] = (PRECISION) mxIsNaN(x[unknown_ind]);
    }
  }

inline static void fixunknowns_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 numUnknown = Ysize-Xsize;
  const INT64 * const shift = intParam;
  const INT64 * const unknown = intParam + Xsize;
  for (INT64 i=0; i<Xsize; i++)
    x[i] = y[i+shift[i]];
  for (INT64 i=0; i<numUnknown; i++)
    {
    INT64 unknown_ind = unknown[i];
    if (y[unknown_ind + shift[unknown_ind] + 1] < 0.5)
      x[unknown_ind] = mxGetNaN();
    }
  }

inline static void fixunknowns_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 numUnknown = Ysize-Xsize;
  const PRECISION * const xmean = floatParam;
  const INT64 * const shift = intParam;
  const INT64 * const unknown = intParam + Xsize;

  for (INT64 i=0; i<Xsize; i++)
    dy[i+shift[i]] = mxIsNaN(x[i]) ? 0 : dx[i];
  for (INT64 i=0; i<numUnknown; i++)
    {
    INT64 unknown_ind = unknown[i];
    dy[unknown_ind+shift[unknown_ind]+1] = 0;
    }
  }

inline static void fixunknowns_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const shift = intParam;
  
  for (INT64 i=0; i<Xsize; i++)
    dx[i] = mxIsNaN(x[i]) ? 0 : dy[i + shift[i]];
  }

// ------------------------------------------------------- LVQOUTPUTS

// Floating Point Parameters: None
// Integer Parameters: None

// Dimensions: Xsize != Ysize

// This processing function has no effect: Y == X

inline static void lvqoutputs_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  for (INT64 i=0; i<Xsize; i++) y[i] = x[i];
  }

inline static void lvqoutputs_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  for (INT64 i=0; i<Xsize; i++) x[i] = y[i];
  }

inline static void lvqoutputs_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  for (INT64 i=0; i<Xsize; i++) dy[i] = dx[i];
  }

inline static void lvqoutputs_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  for (INT64 i=0; i<Xsize; i++) dx[i] = dy[i];
  }

// ------------------------------------------------------- MAPSTD

// Floating Point Parameters:
//   gain - Xsize elements
//   xoffset - Xsize elements
//   yoffset - 1 element

// Integer Parameters: None

// Dimensions: Xsize == Ysize

inline static void mapstd_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;
  const PRECISION * xoffset = floatParam + Xsize;
  const PRECISION ymean = *(floatParam + 2*Xsize);

  for (INT64 i=0; i<Xsize; i++)
    y[i] = ((x[i] - xoffset[i]) * gain[i]) + ymean;
  }

inline static void mapstd_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;
  const PRECISION * xoffset = floatParam + Xsize;
  const PRECISION ymean = *(floatParam + 2*Xsize);

  for (INT64 i=0; i<Xsize; i++)
    x[i] = ((y[i] - ymean) / gain[i]) + xoffset[i];
  }

inline static void mapstd_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const gain = floatParam;

  for (INT64 i=0; i<Xsize; i++)
    dy[i] = dx[i] / gain[i];
  }

inline static void mapstd_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
 {
  const PRECISION * const gain = floatParam;

  for (INT64 i=0; i<Xsize; i++)
    dx[i] = dy[i] / gain[i];
  }

// ------------------------------------------------------- PROCESSPCA

// Floating Point Parameters:
//   transform - Ysize x Xsize (transform[i,j] = transform[i+j*Ysize])
//   inverseTransform - Xsize x Ysize (inverseTransform[i,j] = invTransform[i+j*Xsize])

// Integer Parameters: None

// Dimensions: Xsize >= Ysize

inline static void processpca_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const transform = floatParam;
  
  for (INT64 i=0; i<Ysize; i++)
    {
	  PRECISION sum = 0.0;
	  for (INT64 j=0; j<Xsize; j++)
		  sum += transform[i+j*Ysize]*x[j];
	  y[i] = sum;
    }
  }

inline static void processpca_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const invTransform = floatParam + Ysize*Xsize;
  
  for (INT64 i=0; i<Xsize; i++)
    {
	  PRECISION sum = 0.0;
	  for (INT64 j=0; j<Ysize; j++)
		  sum += invTransform[i+j*Xsize]*y[j];
	  x[i] = sum;
    }
  }

inline static void processpca_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const invTransform = floatParam + Ysize*Xsize;
  
  for (INT64 j=0; j<Ysize; j++)
    {
	  PRECISION sum = 0.0;
	  for (INT64 i=0; i<Xsize; i++)
		  sum += invTransform[i+j*Xsize]*dx[i];
	  dy[j] = sum;
    }
  }

inline static void processpca_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const PRECISION * const invTransform = floatParam + Ysize*Xsize;
  
  for (INT64 i=0; i<Xsize; i++)
    {
	  PRECISION sum = 0.0;
	  for (INT64 j=0; j<Ysize; j++)
		  sum += invTransform[i+j*Xsize]*dy[j];
	  dx[i] = sum;
    }
  }

// ------------------------------------------------------- REMOVECONSTANTROWS

// Floating Point Parameters:
//   removeValues - (Xsize-Ysize) elements indicating constant values in X

// Integer Parameters:
//   keepInd - Xsize elements (indices of X that are kept in Y)
//   removeInd - (Xsize-Ysize) elements indicating removed elements of X

// Dimensions: Xsize >= Ysize

inline static void removeconstantrows_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;

  for (INT64 i=0; i<Ysize; i++)
    y[i] = x[keepInd[i]];
  }

inline static void removeconstantrows_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  const INT64 * const removeInd = intParam + Ysize;
  const PRECISION * const removeValues = floatParam;

  for (INT64 i=0; i<Ysize; i++)
    x[keepInd[i]] = y[i];
  for (INT64 i=0; i<(Xsize-Ysize); i++)
    x[removeInd[i]] = removeValues[i];
  }

inline static void removeconstantrows_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  
  for (INT64 i=0; i<Ysize; i++)
    dy[i] = dx[keepInd[i]];
  }

inline static void removeconstantrows_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  const INT64 * const removeInd = intParam + Ysize;

  for (INT64 i=0; i<Ysize; i++)
    dx[keepInd[i]] = dy[i];
  for (INT64 i=0; i<(Xsize-Ysize); i++)
    dx[removeInd[i]] = 0;
  }

// ------------------------------------------------------- REMOVEROWS

// Floating Point Parameters: None

// Integer Parameters:
//   keepInd - Ysize elements (indices of X that are kept in Y)
//   removeInd - (Xsize-Ysize) elements indicating removed elements of X

// Dimensions: Xsize >= Ysize

inline static void removerows_apply(PRECISION * const y, const PRECISION * const x, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;

  for (INT64 i=0; i<Ysize; i++)
    y[i] = x[keepInd[i]];
  }

inline static void removerows_reverse(PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  const INT64 * const removeInd = intParam + Ysize;

  for (INT64 i=0; i<Ysize; i++)
    x[keepInd[i]] = y[i];
  for (INT64 i=0; i<(Xsize-Ysize); i++)
    x[removeInd[i]] = mxGetNaN();
  }

inline static void removerows_backpropReverse(PRECISION * const dy, const PRECISION * const dx, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  
  for (INT64 i=0; i<Ysize; i++)
    dy[i] = dx[keepInd[i]];
  }

inline static void removerows_forward_reverse(PRECISION * const dx, const PRECISION * const dy, PRECISION * const x, const PRECISION * const y, const INT64 Xsize, INT64 Ysize, const PRECISION * const floatParam, const INT64 * const intParam)
  {
  const INT64 * const keepInd = intParam;
  const INT64 * const removeInd = intParam + Ysize;

  for (INT64 i=0; i<Ysize; i++)
    dx[keepInd[i]] = dy[i];
  for (INT64 i=0; i<(Xsize-Ysize); i++)
    dx[removeInd[i]] = 0;
  }

// =======================================================

