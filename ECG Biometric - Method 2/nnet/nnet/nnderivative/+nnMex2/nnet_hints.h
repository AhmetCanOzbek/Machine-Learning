// Copyright 2013 The MathWorks, Inc.

// 64 Bit Integer

#define INT64 long long int
#define INT64_STR "long"

#define PRECISION double
#define PRECISION_STR "double"
#define CREATE_MATLAB_MATRIX mxCreateDoubleMatrix
#define EPS ((PRECISION) (2.2204460492503130808e-16))
#define NEGEPS (-EPS)

// Useful macro

#define SIGN1(x) (((x) > 0.0) - ((x) < 0.0))

// BLAS CONSTANTS
static double blas1 = 1.0;
static double blas0 = 0.0;
static INT64 blasInt1 = 1;
static char blasN = 'N';
static char blasT = 'T';


// Structures used to control and optimize network simulation
// and derivative calculations:

typedef struct
  {
  INT64 numAllWeightElements;
  INT64 numLearningWeightElements;
  INT64 numInputs;
  INT64 numLayers;
  INT64 numOutputs;
  INT64 numInputElements;
  INT64 numProcessedInputElements;
  INT64 numLayerElements;
  INT64 maxLayerSize;
  INT64 totalLayerSize;
  INT64 maxLayerZSize;
  INT64 totalZSize;
  INT64 maxSignalSize;
  INT64 numOutputElements;
  INT64 maxDelayedElements;
  INT64 maxIWSizeByS;
  INT64 maxNumLWByS;
  INT64 numInputDelays;
  INT64 numLayerDelays;
  INT64 numSimLayers;
  INT64 seriesInputProcElements;
  INT64 maxOutProcXElements;
  INT64 maxOutProcYElements;
  INT64 maxOutputSize;
  INT64 fcnInfoSize;
  INT64 doubleSize;
  INT64 perfFcn;
  INT64 DUMMY1;
  INT64 forwardLayerDelays;
  INT64 intInfoSize;
  } NetHints;
  
typedef struct
  {
  INT64 fcn;
  INT64 xOffset;
  INT64 yOffset;
  INT64 xSize;
  INT64 ySize;
  INT64 floatParamOffset;
  INT64 intParamOffset;
  } Process;
  
typedef struct
  {
  INT64 size;
  INT64 offset;
  INT64 processedSize;
  INT64 processedOffset;
  INT64 numProc;
  INT64 procInfoPos;
  } Input;
  
typedef struct
  {
  INT64 size;
  INT64 offset;
  INT64 netInputFcn;
  INT64 transferFcn;
  INT64 outputIndex;
  } Layer;
  
typedef struct
  {
  INT64 connect;
  INT64 outputIndex;
  INT64 size;
  INT64 offset;
  INT64 processedSize;
  INT64 processedOffset;
  INT64 numProc;
  INT64 procInfoPos;
  INT64 doErrNorm;
  INT64 errNormOffset;
  } Output;

typedef struct
  {
  INT64 connect;
  INT64 numel;
  INT64 allWBOffset;
  INT64 learnWBOffset;
  INT64 learn;
  } Bias;

typedef struct
  {
  INT64 connect;
  INT64 numel;
  INT64 allWBOffset;
  INT64 learnWBOffset;
  INT64 inLayerIWPos;
  INT64 inLayerLWIndex;
  INT64 inLayerZOffset;
  INT64 zOffset;
  INT64 weightFcn;
  INT64 numDelays;
  INT64 noDelay;
  INT64 singleDelay;
  INT64 delaysOffset;
  INT64 learn;
  } Weight;

typedef struct
  {
  INT64 doProcessInputs;
  INT64 doDelayedInputs;
  INT64 doEW;
  INT64 M_EW;
  INT64 Q_EW;
  INT64 TS_EW;
  } DataHints;


