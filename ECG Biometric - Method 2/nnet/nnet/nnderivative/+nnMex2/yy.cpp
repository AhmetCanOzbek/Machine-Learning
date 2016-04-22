
#include <math.h>
#include <stdio.h>
#include <string.h>

//#if !defined(_WIN32)
//#if defined(_WIN64)
//#define dgemm dgemm_
//#endif

#include "mex.h"
#include "blas.h"

#include "nnet_hints.h"
#include "nnet_process_fcns.h"
#include "nnet_weight_fcns.h"
#include "nnet_transfer_fcns.h"
  
void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray * prhs[])
  {
  // function [Y,Af] = y(WB,X,Xi,Pc,Pd,Ai,Q,TS,netHints,TEMP,B)
  
  // ========= INPUT ARGUMENTS
  
  PRECISION * const WB = mxGetPr(prhs[0]);
  PRECISION * const X = mxGetPr(prhs[1]);
  PRECISION * const Xi = mxGetPr(prhs[2]);
  PRECISION * const Pc0 = mxGetPr(prhs[3]);
  PRECISION * const Pd = mxGetPr(prhs[4]);
  PRECISION * const Ai = mxGetPr(prhs[5]);
  const INT64 Q = (INT64) (mxGetPr(prhs[6])[0]);
  const INT64 TS = (INT64) (mxGetPr(prhs[7])[0]);
  PRECISION * const TEMP = mxGetPr(prhs[9]);
  const INT64 B = (INT64) (mxGetPr(prhs[10])[0]);
  
  // Long Hints
  const char *HintsL = (const char *) mxGetData(mxGetField(prhs[8],0,INT64_STR));
  const DataHints *const dataHints = (DataHints *) HintsL; HintsL += sizeof(DataHints);
  HintsL += sizeof(INT64)*dataHints->M_EW;
  
  const INT64 doProcessInputs = dataHints->doProcessInputs;
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
  const INT64 maxLayerSize = netHints->maxLayerSize;
  
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
  const char *HintsD = (const char *) mxGetData(mxGetField(prhs[8],0,PRECISION_STR));
  PRECISION * const paramDouble = (PRECISION *) HintsD; HintsD += sizeof(PRECISION)*netHints->doubleSize;
  
  // ========= ALLOCATE OUTPUT ARGUMENTS
  
  plhs[0] = CREATE_MATLAB_MATRIX((mwSize) numOutputElements,(mwSize) (Q*TS),mxREAL);
  PRECISION * const Y = mxGetPr(plhs[0]);
  
  PRECISION *Af = 0;
  if (nlhs >= 2)
    {
    plhs[1] = CREATE_MATLAB_MATRIX((mwSize) numLayerElements,(mwSize) (numLayerDelays*Q),mxREAL);
    Af = mxGetPr(plhs[1]);
    }
  
  // ========= ALLOCATE TEMPORARY

  INT64 P_size = (doProcessInputs) ? (sizeof(PRECISION)*seriesInputProcElements*(numInputDelays+TS)) : 0; // ==>numInpProcElements
  INT64 Pc_size = (doProcessInputs) ? (sizeof(PRECISION **)*numInputs*(numInputDelays+TS)) : 0;
  INT64 Xd_size = sizeof(PRECISION)*netHints->maxDelayedElements * B;;
  INT64 Z_size = sizeof(PRECISION)*maxLayerSize * B;
  INT64 N_size = sizeof(PRECISION) * maxLayerSize * B;
  INT64 Ac_size = sizeof(PRECISION)*numLayerElements*(numLayerDelays+1) * B;
  INT64 Ap_size = sizeof(PRECISION)*netHints->maxOutProcXElements * B;
  INT64 totalSize = P_size + Pc_size + Xd_size + Z_size + N_size + Ac_size + Ap_size;
  
  char * temp = (char *) TEMP;
  PRECISION * const P = (PRECISION *) temp; temp += P_size;
  PRECISION **const Pc = (PRECISION **) temp; temp += Pc_size;
  PRECISION * const Xd = (PRECISION *) temp; temp += Xd_size;
  PRECISION * const Z = (PRECISION *) temp; temp += Z_size;
  PRECISION * const N = (PRECISION *) temp; temp += N_size;
  PRECISION * const Ac = (PRECISION *) temp; temp += Ac_size;
  PRECISION * const Ap = (PRECISION *) temp; temp += Ap_size;
                 
  // ========= SIMULATE
  
  for (INT64 q1=0; q1<Q; q1+=B) // Iterate over batches
    {
    const INT64 q2 = ((q1 + B) < Q) ? (q1 + B) : Q;
    const INT64 Qb = q2-q1;
    
    /*
    // Preprocess Input Delay States
    if (doProcessInputs)
      {
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
      /*
      if (doProcessInputs)
        {
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
          }
        }
      */

      // For each layer i, in layerOrder
      for (INT64 isim=0; isim<numSimLayers; isim++)
        {
        INT64 i = layerOrder[isim];
        const Layer * const layer = layers + i;
        const INT64 S = layer->size;

        // Biases
        const Bias *bias = biases + i;
        if (bias->connect)
          {
          PRECISION * const b = WB + bias->allWBOffset;
          for (INT64 qb = 0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++) N[ii+qb*S] = b[ii];
          //for (INT64 qb = 0; qb<Qb; qb++)
          //  memcpy((char *) N+qb*S,(char *) b,S*sizeof(PRECISION));
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
              dotprod_netsum_apply(N,w,pd,Rd,S,Qb,pStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = Z + qb*S;
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
                PRECISION * const nq = N + qb*S;
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // NETSUM
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // NETPROD
                } // qb
              }
            } // input weight i,j
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
            
            // Delayed Layer Outputs
            PRECISION *ad;
            const INT64 aStride = Rd;
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
                  for (INT64 m=0; m<R; m++)
                    Xd[(m + k*R) + qb*Rd] = ac[m + qb*R];
                }
              ad = Xd;
              }
            
            if ((weight->weightFcn == 2) && (layer->netInputFcn == 1)) // DOTPROD, NETSUM
              dotprod_netsum_apply(N,w,ad,Rd,S,Qb,aStride);
            else
              {
              for (INT64 qb=0; qb<Qb; qb++)
                {
                PRECISION * const zq = Z + qb*S;
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
                PRECISION * const nq = N + qb*S;
                if (layer->netInputFcn == 1)
                  for (INT64 ii=0; ii<S; ii++) nq[ii] += zq[ii]; // NETSUM
                else
                  for (INT64 ii=0; ii<S; ii++) nq[ii] *= zq[ii]; // NETPROD
                } // qb
              }
            } // layer weight i,j
          } // from layer j

        // Transfer Function
        const INT64 a_ts = (numLayerDelays+ts) % (numLayerDelays+1);
        PRECISION * const a = Ac + ((a_ts*numLayerElements) + layer->offset)*Qb;
        const INT64 transferFcn = layer->transferFcn;
        for (INT64 qb=0; qb<Qb; qb++)
          {
          PRECISION * const nq = N + qb*S;
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
          }
        
        // Output
        const Output * const output = outputs + i;
        if (output->connect)
          {
          const INT64 a_ts = (numLayerDelays+ts) % (numLayerDelays+1);
          INT64 So = output->size;
          
          PRECISION *ap = a;
          for (INT64 j=output->numProc-1; j >= 0; j--)
            {
            const Process *outputProc = ((const Process *) (fcnInfo + output->procInfoPos))+j;
            PRECISION *nextap;
            INT64 xStride;
            if (j==0)
              {
              nextap = Y + (ts*Q+q1)*netHints->numOutputElements + output->offset;
              xStride = netHints->numOutputElements;
              }
            else
              {
              nextap = Ap + outputProc->xOffset * Qb;
              xStride = outputProc->xSize;
              }
            const INT64 xSize = outputProc->xSize;
            const INT64 ySize = outputProc->ySize;
            PRECISION * const floatParam = paramDouble + outputProc->floatParamOffset;
            const INT64 * const intParam = intInfo + outputProc->intParamOffset;
            for (INT64 qb=0; qb<Qb; qb++)
              {
              PRECISION * const nextapq = nextap + qb*xStride;
              PRECISION * const apq = ap + qb*ySize;
              switch (outputProc->fcn)
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
        
          if (output->numProc == 0)
            {
            PRECISION * const y = Y + (ts*Q+q1)*netHints->numOutputElements + output->offset;
            for (INT64 qb=0; qb<Qb; qb++)
              {
              PRECISION *aq = a + qb*S;
              PRECISION *yq = y + qb*netHints->numOutputElements;
              for (int ii=0; ii<S; ii++) yq[ii] = aq[ii];
              }
            }
          } // output i
        } //layer i
      
      } //ts
    
    // Af
    if (Af != 0)
      {
      for (INT64 ts=0; ts<numLayerDelays; ts++)
        {
        const INT64 a_ts = (TS+ts) % (numLayerDelays+1);
        for (INT64 i=0; i<numLayers; i++)
          {
          const Layer * const layer = layers + i;
          const INT64 S = layer->size;
          PRECISION *ac = Ac + (layer->offset + a_ts*numLayerElements)*Qb;
          PRECISION *af = Af + layer->offset + (q1+ts*Q)*numLayerElements;
          for (INT64 qb=0; qb<Qb; qb++)
            for (INT64 ii=0; ii<S; ii++)
              af[ii + qb*numLayerElements] = ac[ii + qb*S];
          }
        }
      } // Af
    } // q1
  }

