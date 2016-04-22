
#include <math.h>
#include <stdio.h>

//#if !defined(_WIN32)
//#define dgemm dgemm_
//#endif

#include "mex.h"
#include "blas.h"

#include "nnet_hints.h"
#include "nnet_process_fcns.h"
#include "nnet_weight_fcns.h"
#include "nnet_transfer_fcns.h"

// Copyright 2013 The MathWorks, Inc.
        
void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray * prhs[])
  {
  // function [dWB,Perfs,PerfN] = bg(WB,X,Xi,Pc,Pd,Ai,T,EW,masks,Q,TS,numMasks,netHints)
  
  // ========= INPUT ARGUMENTS
  
  PRECISION * const WB = mxGetPr(prhs[0]);
  PRECISION * const X = mxGetPr(prhs[1]);
  PRECISION * const Xi = mxGetPr(prhs[2]);
  PRECISION * const Pc0 = mxGetPr(prhs[3]);
  PRECISION * const Pd = mxGetPr(prhs[4]);
  PRECISION * const Ai = mxGetPr(prhs[5]);
  PRECISION * const T = mxGetPr(prhs[6]);
  PRECISION * const EW = mxGetPr(prhs[7]);
  PRECISION * const masks = mxGetPr(prhs[8]);
  const INT64 Q = (INT64) (mxGetPr(prhs[9])[0]);
  const INT64 TS = (INT64) (mxGetPr(prhs[10])[0]);
  const INT64 numMasks = (INT64) (mxGetPr(prhs[11])[0]);
  PRECISION * const TEMP = mxGetPr(prhs[13]);
  const INT64 B = (INT64) (mxGetPr(prhs[14])[0]);
  
  // Long Hints
  const char *HintsL = (const char *) mxGetData(mxGetField(prhs[12],0,INT64_STR));
  const DataHints *const dataHints = (DataHints *) HintsL; HintsL += sizeof(DataHints);
  const INT64 doProcessInputs = dataHints->doProcessInputs;
  const INT64 doEW = dataHints->doEW;
  const INT64 M_EW = dataHints->M_EW;
  const INT64 Q_EW = dataHints->Q_EW;
  const INT64 TS_EW = dataHints->TS_EW;
  const INT64 * const N_EW = (INT64 *) HintsL; HintsL += sizeof(INT64)*M_EW;
  
  const NetHints * const netHints = (NetHints *) HintsL; HintsL += sizeof(NetHints);
  const INT64 numInputs = netHints->numInputs;
  const INT64 numLayers = netHints->numLayers;
  const INT64 numOutputs = netHints->numOutputs;
  const INT64 numInputDelays = netHints->numInputDelays;
  const INT64 numLayerDelays = netHints->numLayerDelays;
  const INT64 numSimLayers = netHints->numSimLayers;
  const INT64 numInputElements = netHints->numInputElements;
  const INT64 numProcessedInputElements = netHints->numProcessedInputElements;
  const INT64 numLayerElements = netHints->numLayerElements;
  const INT64 numOutputElements = netHints->numOutputElements;
  const INT64 seriesInputProcElements = netHints->seriesInputProcElements;
  const INT64 totalZSize = netHints->totalZSize;
  
  const INT64 * const layerOrder = (INT64 *) HintsL; HintsL += sizeof(INT64)*numSimLayers;
  const Input * const inputs = (Input *) HintsL; HintsL += sizeof(Input)*numInputs;
  const Layer * const layers = (Layer *) HintsL; HintsL += sizeof(Layer)*numLayers;
  const Output * const outputs = (Output *) HintsL; HintsL += sizeof(Output)*numLayers;
  const Bias * const biases = (Bias *) HintsL; HintsL += sizeof(Bias)*numLayers;
  const Weight * const inputWeights = (Weight *) HintsL; HintsL += sizeof(Weight)*numLayers*numInputs;
  const Weight * const layerWeights = (Weight *) HintsL; HintsL += sizeof(Weight)*numLayers*numLayers;
  const INT64 * const fcnInfo = (INT64 *) HintsL; HintsL += sizeof(INT64)*netHints->fcnInfoSize;
  const INT64 * const intInfo = (INT64 *) HintsL;
  
  // Double Hints
  const char *HintsD = (const char *) mxGetData(mxGetField(prhs[12],0,PRECISION_STR));
  PRECISION * const paramDouble = (PRECISION *) HintsD; HintsD += sizeof(PRECISION)*netHints->doubleSize;
  
  // ========= ALLOCATE OUTPUT ARGUMENTS
  
  plhs[0] = CREATE_MATLAB_MATRIX((mwSize) netHints->numLearningWeightElements,1,mxREAL);
  PRECISION * const dWB = mxGetPr(plhs[0]);
  
  PRECISION *Perfs = 0;
  if (nlhs >= 2)
    {
    plhs[1] = CREATE_MATLAB_MATRIX(1,(mwSize) numMasks,mxREAL);
    Perfs = mxGetPr(plhs[1]);
    }
  else
    Perfs = (PRECISION *) malloc(static_cast<size_t>(sizeof(PRECISION)*numMasks));
  
  PRECISION *PerfN = 0;
  if (nlhs >= 3)
    {
    plhs[2] = CREATE_MATLAB_MATRIX(1,(mwSize) numMasks,mxREAL);
    PerfN = mxGetPr(plhs[2]);
    }
  else
    PerfN = (PRECISION *) malloc(static_cast<size_t>(sizeof(PRECISION)*numMasks));
  
  // ========= ALLOCATE TEMPORARY
  
  INT64 P_size = (doProcessInputs) ? (sizeof(PRECISION)*seriesInputProcElements*(numInputDelays+TS)*B) : 0; // ==>numInpProcElements
  INT64 Pc_size = (doProcessInputs) ? (sizeof(PRECISION **)*numInputs*(numInputDelays+TS)*B) : 0;
  INT64 Xd_size = sizeof(PRECISION)*netHints->maxDelayedElements*B;
  INT64 Z_size = sizeof(PRECISION)*totalZSize*TS*B;
  INT64 N_size = sizeof(PRECISION)*numLayerElements*TS*B;
  INT64 Ac_size = sizeof(PRECISION)*numLayerElements*(numLayerDelays+TS)*B;
  INT64 Ap_size = sizeof(PRECISION)*netHints->maxOutProcXElements*B;
  INT64 E_size = sizeof(PRECISION)*netHints->maxOutputSize*B;
  INT64 PERF_size = sizeof(PRECISION)*netHints->maxOutputSize*B;
  
  INT64 dY_size = sizeof(PRECISION)*netHints->maxOutputSize*B;
  INT64 dYp_size = sizeof(PRECISION)*netHints->maxOutProcYElements*B;
  INT64 dAi_size = sizeof(PRECISION)*numLayerElements*numLayerDelays*B; // Spill over from dA backprop
  INT64 dA_size = sizeof(PRECISION)*numLayerElements*TS*B;
  INT64 dN_size = sizeof(PRECISION)*numLayerElements*B;
  INT64 dZ_size = sizeof(PRECISION)*netHints->maxLayerSize*B;
  INT64 dXd_size = sizeof(PRECISION)*netHints->maxDelayedElements*B;
  
  INT64 temp_size = P_size + Pc_size + Xd_size + Z_size + N_size + Ac_size + Ap_size + E_size + PERF_size
     + dY_size + dYp_size + dAi_size + dA_size + dN_size + dZ_size + dXd_size;
  char * temp = (char *) TEMP;
  
  PRECISION * const P = (PRECISION *) temp; temp += P_size;
  PRECISION **const Pc = (PRECISION **) temp; temp += Pc_size;
  PRECISION * const Xd = (PRECISION *) temp; temp += Xd_size;
  PRECISION * const Z = (PRECISION *) temp; temp += Z_size;
  PRECISION * const N = (PRECISION *) temp; temp += N_size;
  PRECISION * const Ac = (PRECISION *) temp; temp += Ac_size;
  PRECISION * const Ap = (PRECISION *) temp; temp += Ap_size;
  PRECISION * const E = (PRECISION *) temp; temp += E_size;
  PRECISION * const PERF = (PRECISION *) temp; temp += PERF_size;
  
  PRECISION * const dY = (PRECISION *) temp; temp += dY_size;
  PRECISION * const dYp = (PRECISION *) temp; temp += dYp_size;
  temp += dAi_size;
  PRECISION * const dA = (PRECISION *) temp; temp += dA_size;
  PRECISION * const dN = (PRECISION *) temp; temp += dN_size;
  PRECISION * const dZ = (PRECISION *) temp; temp += dZ_size;
  PRECISION * const dXd = (PRECISION *) temp; temp += dXd_size;
    
 // ========= INITIALIZE
  
  for (INT64 i=0; i<netHints->numLearningWeightElements; i++) dWB[i] = 0;
  for (INT64 k=0; k<numMasks; k++) { Perfs[k] = 0; PerfN[k] = 0; }

  // ========= SIMULATE
  
  for (INT64 q1=0; q1<Q; q1+=B) // Iterate over batches
    {
    const INT64 q2 = ((q1 + B) < Q) ? (q1 + B) : Q;
    const INT64 Qb = q2-q1;

    // Clear dA
    INT64 dALength = numLayerElements*Qb*TS;
    for (INT64 k=0; k<dALength; k++) { dA[k] = 0; }
    
    // Process Input Delay States
    /*if (doProcessInputs)
      for (INT64 ts=0; ts<numInputDelays; ts++)
        {
        for (INT64 i=0; i<numInputs; i++)
          {
          const Input * const input = inputs + i;
          const INT64 R = input->size;
          PRECISION * pi = Xi + (ts*Q+q)*numInputElements + input->offset;
          for (INT64 j=0; j<input->numProc; j++)
            {
            const Process * const inputProc = (const Process *) (fcnInfo + input->procInfoPos);
            PRECISION *nextpi = P + (ts*seriesInputProcElements) + inputProc->yOffset;
            PRECISION * const floatParam = paramDouble + inputProc->floatParamOffset;
            const INT64 * const intParam = intInfo + inputProc->floatParamOffset;
            switch (inputProc->fcn)
              {
              case 1: fixunknowns_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 2: mapminmax_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 3: mapstd_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 4: processpca_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 5: removeconstantrows_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 6: removerows_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              // case 7: lvqoutputs_apply - not applicable for inputs
              }
            pi = nextpi;
            }
          Pc[ts*numInputs+i] = pi;
          }
        }
    */
    
    // Initial Layer States
    for (INT64 ts=0; ts<numLayerDelays; ts++)
      {
      for (INT64 i=0; i<numLayers; i++)
        {
        const Layer * const layer = layers + i;
        const INT64 S = layer->size;
        PRECISION *ai = Ai + layer->offset + (q1+ts*Q)*numLayerElements;
        PRECISION *ac = Ac + (layer->offset + ts*numLayerElements)*Qb;
        for (INT64 qb=0; qb<Qb; qb++)
          for (INT64 ii=0; ii<S; ii++)
            ac[ii + qb*S] = ai[ii + qb*numLayerElements];
        }
      }
    
    for (INT64 ts=0; ts<TS; ts++)
      {
      // Process Inputs
      /*if (doProcessInputs)
        for (INT64 i=0; i<numInputs; i++)
          {
          const Input * const input = inputs + i;
          const INT64 R = input->size;
          PRECISION *pi = X + (ts*Q+q)*numInputElements + input->offset;
          for (INT64 j=0; j<input->numProc; j++)
            {
            const Process * const inputProc = ((const Process *) (fcnInfo + input->procInfoPos))+j;
            PRECISION *nextpi = P + ((numInputDelays+ts)*seriesInputProcElements) + inputProc->yOffset;
            PRECISION * const floatParam = paramDouble + inputProc->floatParamOffset;
            const INT64 * const intParam = intInfo + inputProc->floatParamOffset;
            switch (inputProc->fcn)
              {
              case 1: fixunknowns_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 2: mapminmax_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 3: mapstd_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 4: processpca_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 5: removeconstantrows_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              case 6: removerows_apply(nextpi,pi,inputProc->xSize,inputProc->ySize,floatParam,intParam); break;
              // case 7: lvqoutputs_apply - not applicable for inputs
              }
            pi = nextpi;
            }
          Pc[i+(numInputDelays+ts)*numInputs] = pi;
          }*/
      
      // Simulate Layers Forward
      for (INT64 isim=0; isim<numSimLayers; isim++)
        {
        INT64 i = layerOrder[isim];
        const Layer * const layer = layers + i;
        const INT64 S = layer->size;
        PRECISION *n = N + ((ts*numLayerElements) + layer->offset)*Qb;

        // Biases
        const Bias *bias = biases + i;
        if (bias->connect)
          {
          PRECISION * const b = WB + bias->allWBOffset;
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) n[ii+qb*S] = b[ii];
          }
        else
          {
          PRECISION base = (layer->netInputFcn == 1) ? 0  /* netsum */ : 1; /* netprod */
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) n[ii+qb*S] = base;
          }

        // Input Weights
        for (INT64 j=0; j<numInputs; j++)
          {
          const Weight * const weight = inputWeights+i+j*numLayers;
          if (weight->connect)
            {
            const Input * const source = inputs + j;
            const INT64 R = source->processedSize;
            const INT64 Rd = R * weight->numDelays;
            PRECISION * const w = WB + weight->allWBOffset;
            PRECISION * const z = Z + (ts*totalZSize + weight->zOffset)*Qb;

            // Delayed Processed Inputs
            PRECISION *pd;
            INT64 pStride;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              INT64 p_ts = numInputDelays+ts-delay;
              pd = /*(doProcessInputs)
                ? Pc[p_ts*numInputs+j]
                : */ Pc0 + ((q1 + p_ts*Q)*numProcessedInputElements) + source->processedOffset;
              pStride = numProcessedInputElements;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              PRECISION *xd = Xd;
              for (INT64 qb=0; qb<Qb; qb++)
                {
                const INT64 q = q1+qb;
                for (INT64 k=0; k<weight->numDelays; k++)
                  {
                  INT64 p_ts = numInputDelays+ts-delays[k];
                  PRECISION *pc = Pc0 + source->processedOffset + (q + p_ts*Q)*numProcessedInputElements;
                  for (INT64 m=0; m<R; m++)
                    *(xd++) = *(pc++);
                  }
                }
              pd = Xd;
              pStride = Rd;
              }
            
            if ((weight->weightFcn == 2) && (layer->netInputFcn == 1)) // DOTPROD, NETSUM
              dotprod_netsum_apply(n,w,pd,Rd,S,Qb,pStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = z + qb*S;
                PRECISION * const pdq = pd + qb*pStride;
                switch (weight->weightFcn)
                  {
                  case 1:   convwf_apply(zq,S,Rd,w,pdq); break;
                  case 2:  dotprod_apply(zq,S,Rd,w,pdq); break;
                  case 3:  negdist_apply(zq,S,Rd,w,pdq); break;
                  case 4: normprod_apply(zq,S,Rd,w,pdq); break;
                  case 5: scalprod_apply(zq,S,Rd,w,pdq); break;
                  case 6:  boxdist_apply(zq,S,Rd,w,pdq); break;
                  case 7:     dist_apply(zq,S,Rd,w,pdq); break;
                  case 8: linkdist_apply(zq,S,Rd,w,pdq); break;
                  case 9:  mandist_apply(zq,S,Rd,w,pdq); break;
                  }
                PRECISION * const nq = n + qb*S;
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // NETSUM
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // NETPROD
                }
              }
            } // input weight i j
          } // from input j
   
        // Layer Weights
        for (INT64 j=0; j<numLayers; j++)
          {
          const Weight * const weight = layerWeights+i+j*numLayers;
          if (weight->connect)
            {
            const Layer * const source = layers + j;
            const INT64 R = source->size;
            const INT64 Rd = R * weight->numDelays;
            PRECISION * const w = WB + weight->allWBOffset;
            PRECISION * const z = Z + (ts*totalZSize + weight->zOffset)*Qb;

            // Delayed Layer Outputs
            PRECISION *ad;
            const INT64 aStride = Rd;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              const INT64 a_ts = (numLayerDelays+ts-delay);
              ad = Ac + (source->offset + (a_ts*numLayerElements))*Qb;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              const INT64 numDelays = weight->numDelays;
              for (INT64 k=0; k<numDelays; k++)
                {
                const INT64 a_ts = (numLayerDelays+ts-delays[k]);
                PRECISION * const ac = Ac + (source->offset + (a_ts*numLayerElements))*Qb;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<R; ii++)
                    Xd[(ii + k*R) + qb*Rd] = ac[ii + qb*R];
                }
              ad = Xd;
              }
            
            if ((weight->weightFcn == 2) && (layer->netInputFcn == 1)) // DOTPROD, NETSUM
              dotprod_netsum_apply(n,w,ad,Rd,S,Qb,aStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = z + qb*S;
                PRECISION * const adq = ad + qb*Rd;
                switch (weight->weightFcn)
                  {
                  case 1:   convwf_apply(zq,S,Rd,w,adq); break;
                  case 2:  dotprod_apply(zq,S,Rd,w,adq); break;
                  case 3:  negdist_apply(zq,S,Rd,w,adq); break;
                  case 4: normprod_apply(zq,S,Rd,w,adq); break;
                  case 5: scalprod_apply(zq,S,Rd,w,adq); break;
                  case 6:  boxdist_apply(zq,S,Rd,w,adq); break;
                  case 7:     dist_apply(zq,S,Rd,w,adq); break;
                  case 8: linkdist_apply(zq,S,Rd,w,adq); break;
                  case 9:  mandist_apply(zq,S,Rd,w,adq); break;
                  }
                PRECISION * const nq = n + qb*S;
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // NETSUM
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // NETPROD
                }
              }
            }
          }

        // Transfer Function
        INT64 a_ts = numLayerDelays+ts;
        PRECISION *a = Ac + ((a_ts*numLayerElements) + layer->offset) * Qb;
        const INT64 transferFcn = layer->transferFcn;
        for (INT64 qb=0; qb<Qb; qb++)
          {
          PRECISION * const nq = n + qb*S;
          PRECISION * const aq = a + qb*S;
          switch (transferFcn)
            {
            case  1:     compet_apply(aq,S,nq); break;
            case  2:    hardlim_apply(aq,S,nq); break;
            case  3:   hardlims_apply(aq,S,nq); break;
            case  4:     logsig_apply(aq,S,nq); break;
            case  5:     netinv_apply(aq,S,nq); break;
            case  6:     poslin_apply(aq,S,nq); break;
            case  7:    purelin_apply(aq,S,nq); break;
            case  8:     radbas_apply(aq,S,nq); break;
            case  9:    radbasn_apply(aq,S,nq); break;
            case 10:     satlin_apply(aq,S,nq); break;
            case 11:    satlins_apply(aq,S,nq); break;
            case 12:    softmax_apply(aq,S,nq); break;
            case 13:     tansig_apply(aq,S,nq); break;
            case 14:     tribas_apply(aq,S,nq); break;
            case 15:  elliotsig_apply(aq,S,nq); break;
            case 16: elliot2sig_apply(aq,S,nq); break;
            }
          } // qb
        } // forward layer
      } // forward ts

    for (INT64 ts=TS-1; ts>=0; ts--)
      {
      for (INT64 isim=numSimLayers-1; isim>=0; isim--)
        {
        INT64 i = layerOrder[isim];
        const Layer * const layer = layers + i;
        INT64 S = layer->size;
        INT64 a_ts = numLayerDelays+ts;
        PRECISION * const a = Ac + ((a_ts*numLayerElements) + layer->offset) * Qb;
        PRECISION * const da = dA + ((ts*numLayerElements) + layer->offset) * Qb;
        
        // Outputs
        const Output * const output = outputs + i;
        if (output->connect)
          {          
          const INT64 So = output->size;
          
          // Output Processing
          PRECISION *ap = a;
          for (INT64 j=output->numProc-1; j >= 0; j--)
            {
            const Process *outputProc = ((const Process *) (fcnInfo + output->procInfoPos))+j;
            const INT64 xSize = outputProc->xSize;
            const INT64 ySize = outputProc->ySize;
            PRECISION * const floatParam = paramDouble + outputProc->floatParamOffset;
            const INT64 * const intParam = intInfo + outputProc->intParamOffset;
            PRECISION *nextap = Ap + outputProc->xOffset * Qb;
            INT64 fcn = outputProc->fcn;
            for (INT64 qb=0; qb<Qb; qb++)
              {
              PRECISION * const nextapq = nextap + qb*xSize;
              PRECISION * const apq = ap + qb*ySize;
              switch (fcn)
                {
                case 1: fixunknowns_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 2: mapminmax_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 3: mapstd_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 4: processpca_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 5: removeconstantrows_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 6: removerows_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                case 7: lvqoutputs_reverse(nextapq,apq,xSize,ySize,floatParam,intParam); break;
                }
              }
            ap = nextap;
            }
          
          for (INT64 qb = 0; qb<Qb; qb++)
            {
            const INT64 q = q1 + qb;
            
            // Error
            PRECISION * const y = ap + qb*So;
            PRECISION * const t = T + (ts*Q+q)*netHints->numOutputElements + output->offset;
            for (INT64 ii=0; ii<So; ii++) E[ii] = t[ii] - y[ii];

            // Error Normalization
            if (output->doErrNorm)
              {
              PRECISION * const errNorm = paramDouble + output->errNormOffset;
              for (INT64 ii=0; ii<So; ii++) E[ii] *= errNorm[ii];
              }

            // Performance
            switch (netHints->perfFcn)
              {
              case 1: /* mae */ for (INT64 ii=0; ii<So; ii++) { PRECISION e = E[ii]; PERF[ii] = (e>0) ? e : (-e); } break;
              case 2: /* mse */ for (INT64 ii=0; ii<So; ii++) { PRECISION e = E[ii]; PERF[ii] = e*e; } break;
              case 3: /* sae */ for (INT64 ii=0; ii<So; ii++) { PRECISION e = E[ii]; PERF[ii] = (e>0) ? e : (-e); } break;
              case 4: /* sse */ for (INT64 ii=0; ii<So; ii++) { PRECISION e = E[ii]; PERF[ii] = e*e; } break;
              case 5: /* crossentropy */
                for (INT64 ii=0; ii<So; ii++)
                  {
                  PRECISION yii = y[ii];
                  if (yii<0) yii = 0; else if (yii>1) yii = 1;
                  PRECISION tii = t[ii];
                  if (tii<0) tii = 0; else if (tii>1) tii = 1;
                  PRECISION safey = (yii<EPS) ? EPS : yii;
                  PRECISION safe1my = ((1-yii)<EPS) ? EPS : (1-yii);
                  PERF[ii] =  -(tii*log(safey) + (1-tii)*log(safe1my));
                  }
                break;
              }

            // Performance Weights
            PRECISION * ew = 0;
            PRECISION ewii = 1;
            char isVectorEW = 0;
            if (doEW)
              {
              INT64 EWi = (M_EW==1) ? 0 : output->outputIndex;
              INT64 EWi_offset = (M_EW==1) ? 0 : output->offset;
              INT64 EWQ_size = (M_EW==1) ? N_EW[0] : netHints->numOutputElements;
              INT64 EWts = (TS_EW==1) ? 0 : ts;
              INT64 EWq = (Q_EW==1) ? 0 : q;
              ew = EW + (EWts*Q_EW+EWq)*EWQ_size + EWi_offset;
              isVectorEW = (N_EW[EWi] != 1);
              if (!isVectorEW) ewii = ew[0];
              }

            // dY
            PRECISION * const dy = dY + qb*So;
            for (INT64 k=0; k<numMasks; k++)
              {
              PRECISION * const maskk = masks + (k*TS*Q+ts*Q+q)*netHints->numOutputElements + output->offset;
              for (INT64 ii=0; ii<So; ii++)
                {
                PRECISION perfii = PERF[ii] * maskk[ii];
                if (mxIsNaN(perfii))
                  { if (k==0) dy[ii] = 0; }
                else
                  {
                  if (isVectorEW) ewii = ew[ii];
                  Perfs[k] += perfii * ewii;
                  PerfN[k]++;
                  if (k == 0)
                    {
                    switch (netHints->perfFcn)
                      {
                      case 1: /* mae */ dy[ii] = SIGN1(E[ii])*ewii; break;
                      case 2: /* mse */ dy[ii] = 2*E[ii]*ewii; break;
                      case 3: /* sae */ dy[ii] = SIGN1(E[ii])*ewii; break;
                      case 4: /* sse */ dy[ii] = 2*E[ii]*ewii; break;
                      case 5: /* crossentropy */
                        PRECISION yii = y[ii];
                        if (yii<0) yii = 0; else if (yii>1) yii = 1;
                        PRECISION tii = t[ii];
                        if (tii<0) tii = 0; else if (tii>1) tii = 1;
                        dy[ii] = ((tii/(yii+EPS)) + ((tii-1)/(1-yii+EPS)))*ewii;
                        break;
                      }

                    // Error Normalization
                    if (output->doErrNorm)
                      {
                      PRECISION * const errNorm = paramDouble + output->errNormOffset;
                      dy[ii] *= errNorm[ii];
                      }
                    }
                  }
                }
              }
            } // qb, q

          // Output Processing
          PRECISION *dyp = dY;
          for (INT64 j=0; j<output->numProc; j ++)
            {
            const Process *outputProc = ((const Process *) (fcnInfo + output->procInfoPos))+j;
            const INT64 xSize = outputProc->xSize;
            const INT64 ySize = outputProc->ySize;
            PRECISION * const floatParam = paramDouble + outputProc->floatParamOffset;
            const INT64 * const intParam = intInfo + outputProc->intParamOffset;
            PRECISION *lastap = Ap + outputProc->xOffset * Qb;
            PRECISION *prevap;
            if (j == 0)
              prevap = a;
            else
              {
              const Process *prevOutputProc = ((const Process *) (fcnInfo + output->procInfoPos))+(j-1);
              prevap = Ap + prevOutputProc->xOffset * Qb;
              }

            PRECISION *nextdyp = dYp + outputProc->yOffset * Qb;
            for (INT64 qb=0; qb<Qb; qb++)
              {
              PRECISION * const nextdypq = nextdyp + qb*ySize;
              PRECISION * const dypq = dyp + qb*xSize;
              PRECISION * const lastapq = lastap + qb*xSize;
              PRECISION * const prevapq = prevap + qb*ySize;
              switch (outputProc->fcn)
                {
                case 1: fixunknowns_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 2: mapminmax_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 3: mapstd_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 4: processpca_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 5: removeconstantrows_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 6: removerows_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                case 7: lvqoutputs_backpropReverse(nextdypq,dypq,lastapq,prevapq,xSize,ySize,floatParam,intParam); break;
                }
               }
            dyp = nextdyp;
            }

          // Layer Output
          for (INT64 qb=0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) da[ii+qb*S] += dyp[ii+qb*S];
          }
        
        // Transfer Function
        PRECISION * const n = N + ((ts*numLayerElements) + layer->offset)*Qb;
        PRECISION * const dn = dN + layer->offset;
        const INT64 transferFcn = layer->transferFcn;
        for (INT64 qb=0; qb<Qb; qb++)
          {
          PRECISION * const nq = n + qb*S;
          PRECISION * const aq = a + qb*S;
          PRECISION * const dnq = dn + qb*S;
          PRECISION * const daq = da + qb*S;
          switch (layer->transferFcn)
            {
            case  1:     compet_backprop(dnq,daq,S,nq,aq); break;
            case  2:    hardlim_backprop(dnq,daq,S,nq,aq); break;
            case  3:   hardlims_backprop(dnq,daq,S,nq,aq); break;
            case  4:     logsig_backprop(dnq,daq,S,nq,aq); break;
            case  5:     netinv_backprop(dnq,daq,S,nq,aq); break;
            case  6:     poslin_backprop(dnq,daq,S,nq,aq); break;
            case  7:    purelin_backprop(dnq,daq,S,nq,aq); break;
            case  8:     radbas_backprop(dnq,daq,S,nq,aq); break;
            case  9:    radbasn_backprop(dnq,daq,S,nq,aq); break;
            case 10:     satlin_backprop(dnq,daq,S,nq,aq); break;
            case 11:    satlins_backprop(dnq,daq,S,nq,aq); break;
            case 12:    softmax_backprop(dnq,daq,S,nq,aq); break;
            case 13:     tansig_backprop(dnq,daq,S,nq,aq); break;
            case 14:     tribas_backprop(dnq,daq,S,nq,aq); break;
            case 15:  elliotsig_backprop(dnq,daq,S,nq,aq); break;
            case 16: elliot2sig_backprop(dnq,daq,S,nq,aq); break;
            }
          }

        // Layer Weights
        for (INT64 j=0; j<numLayers; j++)
          {
          const Weight * const weight = layerWeights+i+j*numLayers;
          if (weight->connect)
            {
            const Layer * const source = layers + j;
            const INT64 R = source->size;
            const INT64 Rd = R * weight->numDelays;
            PRECISION * const w = WB + weight->allWBOffset;
            PRECISION * const z = Z + (ts*totalZSize + weight->zOffset)*Qb;
            PRECISION * const dw = dWB + weight->learnWBOffset;
            INT64 weightFcn = weight->weightFcn;
                
            // Net Input -> Layer Weight
            PRECISION * const dz = (layer->netInputFcn == 1)
              ? dn // NETSUM
              : dZ; // NETPROD
            if (layer->netInputFcn == 2) // NETPROD
              {
              for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] = dn[ii+qb*S];
              const Bias *bias = biases + i;
              if (bias->connect)
                {
                PRECISION * const b = WB + bias->allWBOffset;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= b[ii];
                }
              for (INT64 jj=0; jj<numInputs; jj++)
                {
                const Weight * const weightjj = inputWeights+i+jj*numLayers;
                if (weightjj->connect)
                  {
                  PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                  for (INT64 qb=0; qb<Qb; qb++)
                      for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                  }
                }
              for (INT64 jj=0; jj<numLayers; jj++) if (jj != j) 
                {
                const Weight * const weightjj = layerWeights+i+jj*numLayers;
                if (weightjj->connect)
                  {
                  PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                  for (INT64 qb=0; qb<Qb; qb++)
                    for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                  }
                }
              }

            // Delayed Layer Outputs
            PRECISION *ad;
            const INT64 aStride = Rd;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              const INT64 a_ts = (numLayerDelays+ts-delay);
              ad = Ac + ((a_ts*numLayerElements) + source->offset)*Qb;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              for (INT64 k=0; k<weight->numDelays; k++)
                {
                const INT64 a_ts = (numLayerDelays+ts-delays[k]);
                PRECISION * const ac = Ac + ((a_ts*numLayerElements) + source->offset)*Qb;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 m=0; m<R; m++)
                    Xd[(m + k*R) + qb*Rd] = ac[m + qb*R];
                }
              ad = Xd;
              }
            
            // Backprop to Layer Weight
            if (weight->learn)
              {
              if (weightFcn == 2) // DOTPROD
                dotprod_backstop_batch(dw,dz,w,ad,z,Rd,S,Qb,Rd);
              else
                {
                for (INT64 qb=0; qb<Qb; qb++)
                  {
                  PRECISION * const zq = z + qb*S;
                  PRECISION * const dzq = dz + qb*S;
                  PRECISION * const adq = ad + qb*Rd;
                  switch(weightFcn)
                    {
                    case 1:   convwf_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 2:  dotprod_backstop(dw,dzq,S,Rd,w,adq,zq); break; // Handled above
                    case 3:  negdist_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 4: normprod_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 5: scalprod_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 6:  boxdist_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 7:     dist_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 8: linkdist_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    case 9:  mandist_backstop(dw,dzq,S,Rd,w,adq,zq); break;
                    }
                  }
                }
              }
            
            // Backprop through Layer Weights
            if (weightFcn == 2) // DOTPROD
              dotprod_backprop_batch(dXd,dz,w,ad,z,Rd,S,Qb);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = z + qb*S;
                PRECISION * const dzq = dz + qb*S;
                PRECISION * const adq = ad + qb*Rd;
                PRECISION * const dXdq = dXd + qb*Rd;
                switch (weightFcn)
                  {
                  case 1:   convwf_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 2:  dotprod_backprop(dXdq,dzq,S,Rd,w,adq,zq); break; // Handled above
                  case 3:  negdist_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 4: normprod_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 5: scalprod_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 6:  boxdist_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 7:     dist_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 8: linkdist_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  case 9:  mandist_backprop(dXdq,dzq,S,Rd,w,adq,zq); break;
                  }
                }
              }
                      
            const INT64 * const delays = intInfo + weight->delaysOffset;
            const INT64 numDelays = weight->numDelays;
            for (INT64 k=0; k<numDelays; k++)
              {
              INT64 a_ts = (ts-delays[k]);
              PRECISION *dad = dXd + k*R;
              PRECISION *dak = dA + (source->offset + (a_ts*numLayerElements))*Qb;
              for (INT64 qb=0; qb<Qb; qb++)
                for (INT64 ii=0; ii<R; ii++)
                  dak[ii+qb*R] += dad[ii+qb*Rd];
              } // delay k
            
            } // layer weight->connect
          } // layer weight j

        // Input Weights
        for (INT64 j=0; j<numInputs; j++) 
          {
          const Weight * const weight = inputWeights+i+j*numLayers;
          if (weight->learn)
            {
            const Input * const source = inputs + j;
            const INT64 R = source->processedSize;
            const INT64 Rd = R * weight->numDelays;
            PRECISION * const w = WB + weight->allWBOffset;
            PRECISION * const z = Z + (ts*totalZSize + weight->zOffset)*Qb;
            PRECISION * const dw = dWB + weight->learnWBOffset;
            INT64 weightFcn = weight->weightFcn;
            
            // Delayed Processed Inputs
            PRECISION *pd;
            INT64 pStride = numProcessedInputElements;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              INT64 p_ts = numInputDelays+ts-delay;
              pd = /*(doProcessInputs)
                ? Pc[p_ts*numInputs+j]
                : */ Pc0 + ((q1 + p_ts*Q)*numProcessedInputElements) + source->processedOffset;
              pStride = numProcessedInputElements;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              PRECISION *xd = Xd;
              for (INT64 qb=0; qb<Qb; qb++)
                {
                const INT64 q = q1+qb;
                for (INT64 k=0; k<weight->numDelays; k++)
                  {
                  INT64 p_ts = numInputDelays+ts-delays[k];
                  PRECISION *pc = Pc0 + source->processedOffset + (q + p_ts*Q)*numProcessedInputElements;
                  for (INT64 m=0; m<R; m++)
                    *(xd++) = *(pc++);
                  }
                }
              pd = Xd;
              pStride = Rd;
              }

            // Net Input -> Weighted Input
            PRECISION * const dz = (layer->netInputFcn == 1)
              ? dn // NETSUM
              : dZ; // NETPROD
            if (layer->netInputFcn == 2) // NETPROD
              {
              for (INT64 qb=0; qb<Qb; qb++)
                for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] = dn[ii+qb*S];
              const Bias *bias = biases + i;
              if (bias->connect)
                {
                PRECISION * const b = WB + bias->allWBOffset;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= b[ii];
                }
              for (INT64 jj=0; jj<numInputs; jj++)  if (jj != j) 
                {
                const Weight * const weightjj = inputWeights+i+jj*numLayers;
                if (weightjj->connect)
                  {
                  PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                  for (INT64 qb=0; qb<Qb; qb++)
                    for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                  }
                }
              for (INT64 jj=0; jj<numLayers; jj++)
                {
                const Weight * const weightjj = layerWeights+i+jj*numLayers;
                if (weightjj->connect)
                  {
                  PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                  for (INT64 qb=0; qb<Qb; qb++)
                    for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                  }
                }
              }

            // Backprop to Input Weight
            if (weightFcn == 2) // DOTPROD
              dotprod_backstop_batch(dw,dz,w,pd,z,Rd,S,Qb,pStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = z + qb*S;
                PRECISION * const dzq = dz + qb*S;
                PRECISION * const pdq = pd + qb*pStride;
                switch (weightFcn)
                  {
                  case 1:   convwf_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 2:  dotprod_backstop(dw,dzq,S,Rd,w,pdq,zq); break; // Handled above
                  case 3:  negdist_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 4: normprod_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 5: scalprod_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 6:  boxdist_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 7:     dist_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 8: linkdist_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  case 9:  mandist_backstop(dw,dzq,S,Rd,w,pdq,zq); break;
                  }
                }  
              } // to input weight
            } // input weight->learn
          } // weight from input j

        // Bias
        const Bias *bias = biases + i;
        if (bias->learn)
          {
          // Net Input -> Bias
          PRECISION * const dz = (layer->netInputFcn == 1)
            ? dn // NETSUM
            : dZ; // NETPROD
          if (layer->netInputFcn == 2) // NETPROD
            {
            for (INT64 qb=0; qb<Qb; qb++)
               for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] = dn[ii+qb*S];
            for (INT64 jj=0; jj<numInputs; jj++) 
              {
              const Weight * const weightjj = inputWeights+i+jj*numLayers;
              if (weightjj->connect)
                {
                PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                }
              }
            for (INT64 jj=0; jj<numLayers; jj++) 
              {
              const Weight * const weightjj = layerWeights+i+jj*numLayers;
              if (weightjj->connect)
                {
                PRECISION * const z = Z + (ts*totalZSize + weightjj->zOffset)*Qb;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<S; ii++) dZ[ii+qb*S] *= z[ii+qb*S];
                }
              }
            }
          
          // Backprop to Bias
          PRECISION * const db = dWB + bias->learnWBOffset;
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++)
              db[ii] += dz[ii+qb*S];
          
          } // bias->learn
        } // backward layer
      } // backward ts
    } // q1, q2, Qb
  } // function
