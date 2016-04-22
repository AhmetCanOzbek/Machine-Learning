
#include <math.h>
#include <stdio.h>

//#if !defined(_WIN32)
//  #define dgemm dgemm_
//#endif

#include "mex.h"
#include "blas.h"

#include "nnet_hints.h"
#include "nnet_process_fcns.h"
#include "nnet_weight_fcns.h"
#include "nnet_transfer_fcns.h"

void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray * prhs[])
  {
  // function [dWB,Perfs,PerfN] = perfs(WB,X,Xi,Pc,Pd,Ai,T,EW,masks,Q,TS,numMasks,netHints,TEMP)
  
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
  const double * const TEMP = mxGetPr(prhs[13]);
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
  const INT64 numLearningWeightElements = netHints->numLearningWeightElements;
  const INT64 maxLayerSize = netHints->maxLayerSize;
  
  const INT64* const layerOrder = (INT64 *) HintsL; HintsL += sizeof(INT64)*numSimLayers;
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
  
  plhs[0] = CREATE_MATLAB_MATRIX((mwSize) numLearningWeightElements,1,mxREAL);
  PRECISION *JE = mxGetPr(plhs[0]);
  
  plhs[1] = CREATE_MATLAB_MATRIX((mwSize) numLearningWeightElements,(mwSize) numLearningWeightElements,mxREAL);
  PRECISION *JJ = mxGetPr(plhs[1]);
  
  plhs[2] = CREATE_MATLAB_MATRIX(1,(mwSize) numMasks,mxREAL);
  PRECISION *Perfs = mxGetPr(plhs[2]);
  
  plhs[3] = CREATE_MATLAB_MATRIX(1,(mwSize) numMasks,mxREAL);
  PRECISION *PerfN = mxGetPr(plhs[3]);
  
  // ========= ALLOCATE TEMPORARY
  
  INT64 P_size = (doProcessInputs) ? (sizeof(PRECISION)*seriesInputProcElements*(numInputDelays+TS)*B) : 0; // ==>numInpProcElements
  INT64 Pc_size = (doProcessInputs) ? (sizeof(PRECISION **)*numInputs*(numInputDelays+TS)*B) : 0;
  INT64 Xd_size = sizeof(PRECISION)*netHints->maxDelayedElements*B;
  INT64 Z_size = sizeof(PRECISION)*netHints->maxLayerZSize*B;
  INT64 N_size = sizeof(PRECISION)*maxLayerSize*B;
  INT64 Ac_size = sizeof(PRECISION)*numLayerElements*(numLayerDelays+1)*B;
  INT64 Ap_size = sizeof(PRECISION)*netHints->maxOutProcXElements*B;
  INT64 E_size = sizeof(PRECISION)*netHints->maxOutputSize*B;
  INT64 PERF_size = sizeof(PRECISION)*netHints->maxOutputSize*B;
  
  INT64 dXd_size = sizeof(PRECISION)*netHints->maxDelayedElements*numLearningWeightElements*B;
  INT64 dIWZ_size = sizeof(PRECISION)*netHints->maxIWSizeByS*B;
  INT64 dLWZ_size = sizeof(PRECISION)*netHints->maxNumLWByS*numLearningWeightElements*B;
  INT64 dN_size = sizeof(PRECISION)*maxLayerSize*numLearningWeightElements*B;
  INT64 dA_size = sizeof(PRECISION)*numLayerElements*numLearningWeightElements*(numLayerDelays+1)*B;
  INT64 dAp_size = sizeof(PRECISION)*netHints->maxOutProcXElements*numLearningWeightElements*B;
    
  INT64 temp_size = P_size + Pc_size + Xd_size + Z_size + N_size + Ac_size + Ap_size + E_size + PERF_size
    + dXd_size + dIWZ_size + dLWZ_size + dN_size + dA_size + dAp_size;
  
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
  
  PRECISION * const dXd = (PRECISION *) temp; temp += dXd_size;
  PRECISION * const dIWZ = (PRECISION *) temp; temp += dIWZ_size;
  PRECISION * const dLWZ = (PRECISION *) temp; temp += dLWZ_size;
  PRECISION * const dN = (PRECISION *) temp; temp += dN_size;
  PRECISION * const dA = (PRECISION *) temp; temp += dA_size;
  PRECISION * const dAp = (PRECISION *) temp; temp += dAp_size;
    
  // ========= INITIALIZE
  
  for (INT64 i=0; i<numLearningWeightElements; i++) JE[i] = 0;
  for (INT64 i=0; i<numLearningWeightElements*numLearningWeightElements; i++) JJ[i] = 0;
  for (INT64 k=0; k<numMasks; k++) { Perfs[k] = 0; PerfN[k] = 0; }
  
  // ========= SIMULATE
  
  for (INT64 q1=0; q1<Q; q1+=B) // Iterate over batches
    {
    const INT64 q2 = ((q1 + B) < Q) ? (q1 + B) : Q;
    const INT64 Qb = q2-q1;

    // Process Input Delay States
    /*
    if (doProcessInputs)
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
    
    INT64 dALength = numLayerElements*(numLayerDelays+1)*numLearningWeightElements*Qb;
    for (INT64 k=0; k<dALength; k++) { dA[k] = 0; }
      
    for (INT64 ts=0; ts<TS; ts++)
      {
      // Process Inputs
      /*
      if (doProcessInputs)
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
              }
            pi = nextpi;
            }
          Pc[i+(numInputDelays+ts)*numInputs] = pi;
          }
      */
      
      // Simulate Layers Forward
      for (INT64 isim=0; isim<numSimLayers; isim++)
        {
        INT64 i = layerOrder[isim];
        const Layer * const layer = layers + i;
        const INT64 S = layer->size;
        
        // Clear dN
        const INT64 dNLength = S*Qb*numLearningWeightElements;
        for (INT64 k=0; k<dNLength; k++) dN[k] = 0;

        // Biases
        const Bias *bias = biases + i;
        if (bias->connect)
          {
          PRECISION * const b = WB + bias->allWBOffset;
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) N[ii+qb*S] = b[ii];
          // ---------
          if ((bias->learn) && (layer->netInputFcn == 1)) // NETSUM
            {
            PRECISION * const dn = dN + (bias->learnWBOffset)*S*Qb;
            for (INT64 qb = 0; qb<Qb; qb++)
              for (INT64 ii=0; ii<S; ii++)
                dn[ii + (qb + ii*Qb)*S] = 1;
            }
          }
        else
          {
          PRECISION base = (layer->netInputFcn == 1) ? 0  /* netsum */ : 1; /* netprod */
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) N[ii+qb*S] = base;
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
            PRECISION * const z = Z + weight->inLayerZOffset*Qb;
            const INT64 weightFcn = weight->weightFcn;
            
            // Delayed Processed Inputs
            PRECISION *pd;
            INT64 pStride;;
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
            
            if ((layer->netInputFcn == 1) && (weightFcn == 2))
              dotprod_netsum_apply(N,w,pd,Rd,S,Qb,pStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const nq = N + qb*S;
                PRECISION * const zq = z + qb*S;
                PRECISION * const pdq = pd + qb*pStride;
                switch (weightFcn)
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
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // NETSUM
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // NETPROD
                }
              }
            // -------
            if (weight->learn)
              {
              // Forwardprop from input weight
              PRECISION * dz;
              if (layer->netInputFcn == 1) // NETSUM
                dz = dN + S*Qb*weight->learnWBOffset;
              else // NETPROD
                {
                dz = dIWZ + S*weight->inLayerIWPos;
                const INT64 dIWZsize = S*weight->numel*Qb;
                for (INT64 ii=0; ii<dIWZsize; ii++) dz[ii] = 0;
                }
              if (weightFcn == 2)
                dotprod_forwardstart_batch(dz,w,pd,NULL,Rd,S,Qb,pStride);
              else
                {
                for (INT64 qb=0; qb<Qb; qb++)
                  {
                  PRECISION * const zq = z + qb*S;
                  PRECISION * const pdq = pd + qb*pStride;
                  PRECISION * const dzq = dz + qb*S;
                  const INT64 kStride = Qb*S;
                  switch (weightFcn)
                    {
                    case 1: convwf_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 2: dotprod_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 3: negdist_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 4: normprod_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 5: scalprod_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 6: boxdist_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 7: dist_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 8: linkdist_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    case 9: mandist_forwardstart(dzq,w,pdq,zq,Rd,S,kStride); break;
                    }
                  }
                }
              }
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
            const INT64 weightFcn = weight->weightFcn;
            
            // Delayed Layer Outputs
            PRECISION *ad;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              const INT64 a_ts = (numLayerDelays+ts-delay) % (numLayerDelays+1);
              ad = Ac + ((a_ts*numLayerElements) + source->offset)*Qb;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              for (INT64 k=0; k<weight->numDelays; k++)
                {
                const INT64 a_ts = (numLayerDelays+ts-delays[k]) % (numLayerDelays+1);
                PRECISION * const ac = Ac + ((a_ts*numLayerElements) + source->offset)*Qb;
                for (INT64 qb=0; qb<Qb; qb++)
                  for (INT64 ii=0; ii<R; ii++)
                    Xd[(ii + k*R) + qb*Rd] = ac[ii + qb*R];
                }
              ad = Xd;
              }
            
            PRECISION * const z = Z + weight->inLayerZOffset*Qb;
            if ((layer->netInputFcn == 1) && (weightFcn == 2))
              dotprod_netsum_apply(N,w,ad,Rd,S,Qb,Rd);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const nq = N + qb*S;
                PRECISION * const zq = z + qb*S;
                PRECISION * const adq = ad + qb*Rd;
                switch (weightFcn)
                  {
                  case 1: convwf_apply(zq,S,Rd,w,adq); break;
                  case 2: dotprod_apply(zq,S,Rd,w,adq); break;
                  case 3: negdist_apply(zq,S,Rd,w,adq); break;
                  case 4: normprod_apply(zq,S,Rd,w,adq); break;
                  case 5: scalprod_apply(zq,S,Rd,w,adq); break;
                  case 6: boxdist_apply(zq,S,Rd,w,adq); break;
                  case 7: dist_apply(zq,S,Rd,w,adq); break;
                  case 8: linkdist_apply(zq,S,Rd,w,adq); break;
                  case 9: mandist_apply(zq,S,Rd,w,adq); break;
                  }
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // Netsum
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // Netprod
                }
              }
              // -------
            PRECISION *dad;
            if (weight->singleDelay)
              {
              const INT64 delay = *(intInfo + weight->delaysOffset);
              const INT64 a_ts = (numLayerDelays+ts-delay) % (numLayerDelays+1);
              dad = dA + (source->offset + a_ts*numLayerElements)*Qb*numLearningWeightElements;
              }
            else
              {
              const INT64 * const delays = intInfo + weight->delaysOffset;
              const INT64 numDelays = weight->numDelays;
              for (INT64 k=0; k<numDelays; k++)
                {
                const INT64 a_ts = (numLayerDelays+ts-delays[k]) % (numLayerDelays+1);
                PRECISION * const da = dA + (source->offset + a_ts*numLayerElements)*Qb*numLearningWeightElements;
                for (INT64 kk=0; kk<numLearningWeightElements; kk++)
                  for (INT64 qb=0; qb<Qb; qb++)
                    for (INT64 ii=0; ii<R; ii++)
                      dXd[(ii + k*R) + (qb + kk*Qb)*Rd] = da[ii + (qb + kk*Qb)*R];
                }
              dad = dXd;
              }
            
            PRECISION * dz;
            if (layer->netInputFcn == 1) // NETSUM
              dz = dN;
            else // NETPROD
              {
              const INT64 dLWZsize = S*numLearningWeightElements*Qb;
              dz = dLWZ + dLWZsize*weight->inLayerLWIndex*Qb;
              for (INT64 ii=0; ii<dLWZsize; ii++) dz[ii] = 0;
              }
            // Forward prop layer derivatives through layer weight
            if (weightFcn == 2) // DOTPROD
              dotprod_forwardprop_batch(dz,dad,w,ad,z,Rd,S,Qb,numLearningWeightElements);
            else
              {
              for (INT64 k=0; k<numLearningWeightElements; k++)
                for (INT64 qb=0; qb<Qb; qb++)
                  {
                  INT64 col = qb + k*Qb;
                  PRECISION *dadq = dad + col*Rd;
                  PRECISION *dzq = dz + col*S;
                  PRECISION *adq = ad + qb*Rd;
                  PRECISION *zq = z + qb*S;
                  switch (weightFcn)
                    {
                    case 1: convwf_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 2: dotprod_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 3: negdist_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 4: normprod_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 5: scalprod_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 6: boxdist_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 7: dist_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 8: linkdist_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    case 9: mandist_forwardprop(dzq,dadq,w,adq,zq,Rd,S); break;
                    }
                  }
              }

            // Forward prop from layer weight
            if (weight->learn)
              {
              PRECISION *dz2;
              if (layer->netInputFcn == 1)
                dz2 = dN + S*Qb*weight->learnWBOffset;
              else
                {
                dz2 = dN; // TODO
                }
              if (weightFcn == 2)
                dotprod_forwardstart_batch(dz2,w,ad,NULL,Rd,S,Qb,Rd);
              else
                {
                for (INT64 qb=0; qb<Qb; qb++)
                  {
                  PRECISION * const zq = z + qb*S;
                  PRECISION * const adq = ad + qb*Rd;
                  PRECISION * const dzq = dz2 + qb*S;
                  const INT64 kStride = Qb*S;
                  switch (weightFcn)
                    {
                    case 1: convwf_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 2: dotprod_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 3: negdist_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 4: normprod_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 5: scalprod_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 6: boxdist_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 7: dist_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 8: linkdist_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    case 9: mandist_forwardstart(dzq,w,adq,zq,Rd,S,kStride); break;
                    }
                  }
                }
              } // layer weight -> learn
            } // layer weight -> connnect
          } // from layer j
                
        // Net Input - NETPROD
        if (layer->netInputFcn == 2)
          {
          if (bias->learn)
            {
            PRECISION * const dn = dN + S*Qb*(bias->learnWBOffset);
            for (INT64 ii=0; ii<S; ii++)
              {
              // bias derivative
              PRECISION d = 1.0;
              // multiply by all weighted inputs
              for (INT64 jj=0; jj<numInputs; jj++)
                {
                const Weight * const weight = inputWeights+i+jj*numLayers;
                if (weight->connect)
                  {
                  PRECISION * const z = Z + weight->inLayerZOffset;
                  d *= z[ii];
                  }
                }
              // multiply by all weighted layers
              for (INT64 jj=0; jj<numLayers; jj++)
                {
                const Weight * const weight = layerWeights+i+jj*numLayers;
                if (weight->connect)
                  {
                  PRECISION * const z = Z + weight->inLayerZOffset;
                  d *= z[ii];
                  }
                }
              dn[ii + ii*S] = d;
              }
            }

          // Input Weight -> Net Input
          for (INT64 j=0; j<numInputs; j++)
            {
            const Weight * const weight = inputWeights+i+j*numLayers;
            if (weight->learn)
              {
              const INT64 dIWZsize = S*weight->numel;
              PRECISION * const dz = dIWZ + S*weight->inLayerIWPos;
              PRECISION * const dn = dN + (weight->learnWBOffset)*S;
              for (INT64 ii=0; ii<S; ii++)
                {
                for (INT64 jj=0; jj<weight->numel; jj++)
                  {
                  // weighted input derivative
                  INT64 kk = ii + jj*S;
                  PRECISION d = dz[kk];
                  // multiply by bias
                  if (bias->connect)
                    {
                    PRECISION * const b = WB + bias->allWBOffset;
                    d *= b[ii];
                    }
                  // multiply by other weighted inputs
                  for (INT64 jj=0; jj<numInputs; jj++) if (jj != j)
                    {
                    const Weight * const weight = inputWeights+i+jj*numLayers;
                    if (weight->connect)
                      {
                      PRECISION * const z = Z + weight->inLayerZOffset;
                      d *= z[ii];
                      }
                    }
                  // multiply by all weighted layers
                  for (INT64 jj=0; jj<numLayers; jj++)
                    {
                    const Weight * const weight = layerWeights+i+jj*numLayers;
                    if (weight->connect)
                      {
                      PRECISION * const z = Z + weight->inLayerZOffset;
                      d *= z[ii];
                      }
                    }
                  dn[kk] = d;
                  }
                }
              }
            }
           
          for (INT64 j=0; j<numLayers; j++)
            {
            const Weight * const weight = layerWeights+i+(j*numLayers);
            if (weight->connect)
               {
               const INT64 dLWZsize = S*numLearningWeightElements;
               PRECISION * const dz = dLWZ + dLWZsize*weight->inLayerLWIndex;
               for (INT64 ii=0; ii<S; ii++)
                 {
                 for (INT64 jj=0; jj<numLearningWeightElements; jj++)
                   {
                   // weighted layer derivative
                   INT64 kk = ii + jj*S;
                   PRECISION d = dz[kk];
                   // multiply derivative by bias
                   if (bias->connect)
                      {
                      PRECISION * const b = WB + bias->allWBOffset;
                      d *= b[ii];
                      }
                    // multiply by all weighted inputs
                    for (INT64 jj=0; jj<numInputs; jj++)
                      {
                      const Weight * const weight = inputWeights+i+jj*numLayers;
                      if (weight->connect)
                        {
                        PRECISION * const z = Z + weight->inLayerZOffset;
                        d *= z[ii];
                        }
                      }
                    // multiply by other weighted layers
                    for (INT64 jj=0; jj<numLayers; jj++) if (jj != j)
                      {
                      const Weight * const weight = layerWeights+i+jj*numLayers;
                      if (weight->connect)
                        {
                        PRECISION * const z = Z + weight->inLayerZOffset;
                        d *= z[ii];
                        }
                      }
                   dN[kk] += d;
                   }
                 }
               }
            }
          }
        
        // Transfer Function
        const INT64 a_ts = (numLayerDelays+ts) % (numLayerDelays+1);
        PRECISION * const a = Ac + ((a_ts*numLayerElements) + layer->offset)*Qb;
        for (INT64 qb=0; qb<Qb; qb++)
          {
          PRECISION * const nq = N + qb*S;
          PRECISION * const aq = a + qb*S;
          switch (layer->transferFcn)
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
          }
        // --
        PRECISION * const da = dA + (layer->offset + a_ts*numLayerElements)*Qb*numLearningWeightElements;
        for (INT64 k=0; k<numLearningWeightElements; k++)
          for (INT64 qb=0; qb<Qb; qb++)
          {
          PRECISION * const dak = da + k*S*Qb + qb*S;
          PRECISION * const dnk = dN + k*S*Qb + qb*S;
          PRECISION * const aq = a + qb*S;
          PRECISION * const nq = N + qb*S;
          switch (layer->transferFcn)
            {
            case  1:     compet_forwardprop(dak,dnk,S,nq,aq); break;
            case  2:    hardlim_forwardprop(dak,dnk,S,nq,aq); break;
            case  3:   hardlims_forwardprop(dak,dnk,S,nq,aq); break;
            case  4:     logsig_forwardprop(dak,dnk,S,nq,aq); break;
            case  5:     netinv_forwardprop(dak,dnk,S,nq,aq); break;
            case  6:     poslin_forwardprop(dak,dnk,S,nq,aq); break;
            case  7:    purelin_forwardprop(dak,dnk,S,nq,aq); break;
            case  8:     radbas_forwardprop(dak,dnk,S,nq,aq); break;
            case  9:    radbasn_forwardprop(dak,dnk,S,nq,aq); break;
            case 10:     satlin_forwardprop(dak,dnk,S,nq,aq); break;
            case 11:    satlins_forwardprop(dak,dnk,S,nq,aq); break;
            case 12:    softmax_forwardprop(dak,dnk,S,nq,aq); break;
            case 13:     tansig_forwardprop(dak,dnk,S,nq,aq); break;
            case 14:     tribas_forwardprop(dak,dnk,S,nq,aq); break;
            case 15:  elliotsig_forwardprop(dak,dnk,S,nq,aq); break;
            case 16: elliot2sig_forwardprop(dak,dnk,S,nq,aq); break;
            }
          }

        // Output
        const Output * const output = outputs + i;
        if (output->connect)
          {
          INT64 So = output->size;
          PRECISION *ap = a;
          PRECISION * dY;
          if (output->numProc == 0)
            {
            dY = dAp;
            INT64 dYLength = So*Qb*numLearningWeightElements;
            for (INT64 k=0; k<dYLength; k++) dY[k] = da[k];
            }
          else
            {
            PRECISION *dap = da;
            for (INT64 j=output->numProc-1; j >= 0; j--)
              {
              const Process *outputProc = ((const Process *) (fcnInfo + output->procInfoPos))+j;
              const INT64 xSize = outputProc->xSize;
              const INT64 ySize = outputProc->ySize;
              PRECISION * const floatParam = paramDouble + outputProc->floatParamOffset;
              const INT64 * const intParam = intInfo + outputProc->intParamOffset;
              PRECISION *nextap = Ap + outputProc->xOffset*Qb;
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
                  }
                }
              ap = nextap;
              // --------
              PRECISION *nextdap = dAp + outputProc->xOffset*Qb*numLearningWeightElements;
              for (INT64 k=0; k<numLearningWeightElements; k++)
                {
                for (INT64 qb=0; qb<Qb; qb++)
                  {
                  INT64 col = qb + k*Qb;
                  PRECISION * const nextapq = nextap + qb*xSize;
                  PRECISION * const apq = ap + qb*ySize;
                  PRECISION * const nextdapq = nextdap + col*xSize;
                  PRECISION * const dapq = dap + col*ySize;
                  switch (fcn)
                    {
                    case 1: fixunknowns_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break;
                    case 2: mapminmax_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break;
                    case 3: mapstd_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break;
                    case 4: processpca_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break;
                    case 5: removeconstantrows_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break; break;
                    case 6: removerows_forward_reverse(nextdapq,dapq,nextapq,apq,xSize,ySize,floatParam,intParam); break;
                    }
                  }
                }
              dap = nextdap;
              }
            dY = dap;
            }
          
         for (INT64 qb = 0; qb<Qb; qb++)
            {
            const INT64 q = q1 + qb;

            // Error
            PRECISION * const y = ap + qb*So;
            PRECISION * const t = T + (ts*Q+q)*netHints->numOutputElements + output->offset;
            PRECISION * const errNorm = paramDouble + output->errNormOffset;
            for (INT64 ii=0; ii<So; ii++)
              {
              PRECISION e = y[ii] - t[ii]; // - y[ii];

              // Error Normalization
              if (output->doErrNorm) e *= errNorm[ii];
              
              // Wait to apply errNorm to dY as dY may actually be dA, and needed for later layers/timesteps
              
              // Performance: Must be MSE or SSE
              E[ii+qb*So] = e;
              PERF[ii] = e*e;
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

            for (INT64 k=numMasks-1; k>=0; k--)
              {
              PRECISION * const maskk = masks + (k*TS*Q+ts*Q+q)*netHints->numOutputElements + output->offset;
              for (INT64 ii=0; ii<So; ii++)
                {
                PRECISION perfii = PERF[ii] * maskk[ii];
                if (mxIsNaN(perfii))
                  {
                  if (k==0)
                    {
                    E[ii+qb*So] = 0;                
                    for (INT64 jj=0; jj<numLearningWeightElements; jj++)
                      dY[ii+(qb+jj*Qb)*So] = 0;
                    }
                  }
                else
                  {
                  if (isVectorEW) ewii = ew[ii];
                  Perfs[k] += perfii * ewii;
                  PerfN[k]++;
                  if (k == 0)
                    {
                    // Error Normalization
                    PRECISION enii = (output->doErrNorm) ? errNorm[ii] : 1.0;
                    
                    // dY and E both need to be multiplied by sqrt(ewii)
                    // So dY*E and dY*dY are multiplied by ewii.
                    if (ewii != 1.0)
                      E[ii+qb*So] *= sqrt(ewii);
                    if ((ewii != 1) || (enii != 1.0))
                      {
                      PRECISION temp = sqrt(ewii) * enii;
                      for (INT64 jj=0; jj<numLearningWeightElements; jj++)
                        dY[ii+(qb+jj*Qb)*So] *= temp;
                      }
                    } // k == 0
                  } // ! mxIsNaN(perfii)
                } // ii
              } // k
            } // q, qb
                      
          // JE(1,numLearningWeightElements) =  E(1,(So,Qb)) * dY((So,Qb),numWeightElements)
          const INT64 SoQb = So * Qb;
          dgemm(&blasN,&blasN,(ptrdiff_t *)&blasInt1,(ptrdiff_t *)&numLearningWeightElements,
                  (ptrdiff_t *)&SoQb,&blas1,E,(ptrdiff_t *)&blasInt1,dY,(ptrdiff_t *)&SoQb,
                  &blas1,JE,(ptrdiff_t *)&blasInt1);
          
          // JJ(numLearningWeightElements,numLearningWeightElements) = dY((So,Qb),numWeightElements)' * dY((So,Qb),numWeightElements)
          dgemm(&blasT,&blasN,(ptrdiff_t *)&numLearningWeightElements,
                  (ptrdiff_t *)&numLearningWeightElements,(ptrdiff_t *)&SoQb,&blas1,
                  dY,(ptrdiff_t *)&SoQb,dY,(ptrdiff_t *)&SoQb,&blas1,JJ,
                  (ptrdiff_t *)&numLearningWeightElements);
          } // output
        } // layer i        
      } // ts
    } // q1, Qb
  
  // Copy transposed half of JJ
  //for (INT64 j1=0; j1<numLearningWeightElements; j1++)
  //  for (INT64 j2=0; j2<j1; j2++)
  //    JJ[j1+j2*numLearningWeightElements] = JJ[j2+j1*numLearningWeightElements];
  }
