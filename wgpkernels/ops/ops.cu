// Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
// Standalone CUDA activation and valid Conv3D kernels. No PyTorch/CUTLASS dependency.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
#include <type_traits>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
#define DEV __device__ __forceinline__
template<class T> DEV float fp(T x) { return float(x); }
template<class T> DEV float rounded(float x) { return fp(T(x)); }
DEV float warp_sum(float x) {
    #pragma unroll
    for (int d=16; d; d/=2) x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
template<int THREADS> DEV float block_sum(float x, float* sm) {
    x=warp_sum(x);
    if (!(threadIdx.x&31)) sm[threadIdx.x/32]=x;
    __syncthreads();
    x=threadIdx.x<THREADS/32?sm[threadIdx.x]:0.f;
    if (threadIdx.x<32) x=warp_sum(x);
    if (!threadIdx.x) sm[0]=x;
    __syncthreads();
    return sm[0];
}

// Every launch shape keeps the original 256-logical-lane sum tree.
// Changing the physical block size must not change AdaLN's output.
template<class T,int THREADS,bool VAR> DEV float adaln_sum(const T* x,int width,float mean,float* sm) {
    #pragma unroll
    for(int group=0;group<(256+THREADS-1)/THREADS;++group) {
        int logical=threadIdx.x+group*THREADS;
        float sum=0;
        if(logical<256) {
            for(int c=logical;c<width;c+=256) {
                float value=fp(x[c]);
                if(VAR) { value=__fsub_rn(value,mean);value=__fmul_rn(value,value); }
                sum=__fadd_rn(sum,value);
            }
        }
        sum=warp_sum(sum);
        if(logical<256 && !(threadIdx.x&31))sm[logical/32]=sum;
    }
    __syncthreads();
    float sum=threadIdx.x<8?sm[threadIdx.x]:0.f;
    if(threadIdx.x<32)sum=warp_sum(sum);
    if(!threadIdx.x)sm[0]=sum;
    __syncthreads();
    return sm[0];
}

// LayerNorm output, 1+scale, product and shift each round as separate native
// dtype PyTorch operations. Two-pass FP32 statistics avoid E[x^2]-E[x]^2.
template<class T,int THREADS=256> __global__ void adaln_kernel(const T* x,const T* scale,const T* shift,T* y,
    int64_t rows,int width,int inner,int64_t s0,int64_t s1,int64_t h0,int64_t h1,float eps) {
    __shared__ float sm[8];
    int64_t row=blockIdx.x;
    float mean=__fdiv_rn(adaln_sum<T,THREADS,false>(x+row*width,width,0.f,sm),float(width));
    __syncthreads();
    float var=__fdiv_rn(adaln_sum<T,THREADS,true>(x+row*width,width,mean,sm),float(width));
    float inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(var,eps)));
    int64_t so=(row/inner)*s0+(row%inner)*s1, ho=(row/inner)*h0+(row%inner)*h1;
    for(int c=threadIdx.x;c<width;c+=THREADS) {
        float n=rounded<T>(__fmul_rn(__fsub_rn(fp(x[row*width+c]),mean),inv));
        float gain=rounded<T>(__fadd_rn(1.f,fp(scale[so+c])));
        float v=rounded<T>(__fmul_rn(n,gain));
        y[row*width+c]=T(__fadd_rn(v,fp(shift[ho+c])));
    }
}

// One warp per output pixel. Arbitrary input strides; channels-last output.
// Retain every PixelNorm rounding point, including x*x before FP32 summation.
template<class T> __global__ void pixel_kernel(const T* x,T* y,int64_t pixels,int c,int t,int h,int w,
    int64_t sb,int64_t sc,int64_t st,int64_t sh,int64_t sw,float eps) {
    int lane=threadIdx.x&31;
    int64_t p=int64_t(blockIdx.x)*4+threadIdx.x/32;
    if(p>=pixels) return;
    int64_t base=(p%w)*sw+(p/w%h)*sh+(p/(int64_t(w)*h)%t)*st+(p/(int64_t(w)*h*t))*sb;
    float sum=0;
    for(int ch=lane;ch<c;ch+=32) {
        float v=fp(x[base+ch*sc]);
        sum=__fadd_rn(sum,rounded<T>(__fmul_rn(v,v)));
    }
    float mean=rounded<T>(__fmul_rn(warp_sum(sum),1.f/c));
    float rms=rounded<T>(__fsqrt_rn(rounded<T>(__fadd_rn(mean,eps))));
    for(int ch=lane;ch<c;ch+=32) {
        float v=rounded<T>(__fdiv_rn(fp(x[base+ch*sc]),rms));
        y[p*c+ch]=T(__fdiv_rn(v,__fadd_rn(1.f,expf(-v))));
    }
}

template<class T,int C,int L> __global__ void pixel_specialized(const T* x,T* y,int64_t pixels,int unused_c,int t,int h,int w,
    int64_t sb,int64_t sc,int64_t st,int64_t sh,int64_t sw,float eps) {
    int lane=threadIdx.x%L;
    int p=int(blockIdx.x)*(128/L)+threadIdx.x/L;
    bool valid=p<pixels;
    bool cl=sc==1 && sw==C && sh==int64_t(w)*C && st==int64_t(h)*w*C && sb==int64_t(t)*h*w*C;
    int64_t base=cl ? int64_t(p)*C : (p%w)*sw+(p/w%h)*sh+(p/(w*h)%t)*st+(p/(w*h*t))*sb;
    float values[C/L],partial[32/L]={};
    #pragma unroll
    for(int i=0;i<C/L;++i) {
        float v=valid ? fp(x[base+(lane+i*L)*sc]) : 0;
        values[i]=v;
        partial[i%(32/L)]=__fadd_rn(partial[i%(32/L)],rounded<T>(__fmul_rn(v,v)));
    }
    #pragma unroll
    for(int d=16/L;d;d/=2) {
        #pragma unroll
        for(int j=0;j<d;++j)partial[j]=__fadd_rn(partial[j],partial[j+d]);
    }
    float sum=partial[0];
    #pragma unroll
    for(int d=L/2;d;d/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,d,L));
    float mean=rounded<T>(__fmul_rn(sum,1.f/C));
    float rms=rounded<T>(__fsqrt_rn(rounded<T>(__fadd_rn(mean,eps))));
    #pragma unroll
    for(int i=0;i<C/L;++i) {
        float v=rounded<T>(__fdiv_rn(values[i],rms));
        if(valid)y[int64_t(p)*C+lane+i*L]=T(__fdiv_rn(v,__fadd_rn(1.f,expf(-v))));
    }
}

// Implicit GEMM: no im2col allocation. 128x64 output tile, FP32 tensor-core
// accumulators. Input and weights have channels-last layout. Rounded convolution
// plus bias matches the established 16-bit convolution epilogue.
DEV void copy16(void* dst,const void* src,bool valid) {
    unsigned smem=unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem),"l"(src),"r"(valid?16:0));
}
template<class T,int BK,int BM> __global__ void conv_kernel(const T* x,const T* wt,const T* bias,T* y,
    int64_t pixels,int cin,int cout,int t,int h,int w,int tp,int hp,int wp,int kt,int kh,int kw) {
    using namespace nvcuda;
    constexpr int LD=BK+8, AV=BM*BK/8/128, BV=64*BK/8/128, WM=BM/2, FM=WM/16;
    __shared__ __align__(32) union {
        struct { T a[2][BM][LD]; T b[2][64][LD]; } tiles;
        float out[4][WM][32];
    } sm;
    int warp=threadIdx.x/32, wm=warp/2, wn=warp%2;
    int64_t m0=int64_t(blockIdx.x)*BM;
    int n0=blockIdx.y*64;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[FM][2];
    #pragma unroll
    for(int i=0;i<FM;++i) for(int j=0;j<2;++j) wmma::fill_fragment(acc[i][j],0.f);
    int ksize=kt*kh*kw*cin;
    const T* xp[AV];
    const T* wpointers[BV];
    bool xok[AV], wok[BV];
    #pragma unroll
    for(int v=0;v<AV;++v) {
        int id=threadIdx.x+v*128,r=id/(BK/8),k=id%(BK/8)*8;
        int64_t p=m0+r;
        xok[v]=p<pixels;
        int iw=p%w,ih=p/w%h,it=p/(int64_t(w)*h)%t;
        int64_t batch=p/(int64_t(w)*h*t);
        xp[v]=x+(xok[v]?(((batch*tp+it)*hp+ih)*wp+iw)*cin+k:0);
    }
    #pragma unroll
    for(int v=0;v<BV;++v) {
        int id=threadIdx.x+v*128,r=id/(BK/8),k=id%(BK/8)*8;
        wok[v]=n0+r<cout;
        wpointers[v]=wt+(wok[v]?int64_t(n0+r)*ksize+k:0);
    }
    int taps_row=kw*cin, step=0;
    // Input kw taps are contiguous; only the kh/kt boundaries need pointer gaps.
    // Pipelined asynchronous 16-byte loads avoid register round-trips.
    for(int tt=0;tt<kt;++tt) for(int th=0;th<kh;++th) for(int k=0;k<taps_row;k+=BK) {
        int slot=step&1;
        int64_t offset=(int64_t(tt)*hp+th)*wp*cin+k;
        int flat=(tt*kh+th)*taps_row+k;
        if(step==0) {
            #pragma unroll
            for(int v=0;v<AV;++v) {
                int id=threadIdx.x+v*128,r=id/(BK/8),col=id%(BK/8)*8;
                copy16(&sm.tiles.a[slot][r][col],xp[v]+offset,xok[v]);
            }
            #pragma unroll
            for(int v=0;v<BV;++v) {
                int id=threadIdx.x+v*128,r=id/(BK/8),col=id%(BK/8)*8;
                copy16(&sm.tiles.b[slot][r][col],wpointers[v]+flat,wok[v]);
            }
            asm volatile("cp.async.commit_group;\n");
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        bool next=flat+BK<ksize;
        if(next) {
            int nt=tt,nh=th,nk=k+BK;
            if(nk==taps_row) { nk=0; if(++nh==kh) { nh=0;++nt; } }
            int64_t next_offset=(int64_t(nt)*hp+nh)*wp*cin+nk;
            #pragma unroll
            for(int v=0;v<AV;++v) {
                int id=threadIdx.x+v*128,r=id/(BK/8),col=id%(BK/8)*8;
                copy16(&sm.tiles.a[slot^1][r][col],xp[v]+next_offset,xok[v]);
            }
            #pragma unroll
            for(int v=0;v<BV;++v) {
                int id=threadIdx.x+v*128,r=id/(BK/8),col=id%(BK/8)*8;
                copy16(&sm.tiles.b[slot^1][r][col],wpointers[v]+flat+BK,wok[v]);
            }
            asm volatile("cp.async.commit_group;\n");
        }
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af[FM];
            wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf[2];
            #pragma unroll
            for(int i=0;i<FM;++i) wmma::load_matrix_sync(af[i],&sm.tiles.a[slot][wm*WM+i*16][kk],LD);
            #pragma unroll
            for(int j=0;j<2;++j) wmma::load_matrix_sync(bf[j],&sm.tiles.b[slot][wn*32+j*16][kk],LD);
            #pragma unroll
            for(int i=0;i<FM;++i) for(int j=0;j<2;++j) wmma::mma_sync(acc[i][j],af[i],bf[j],acc[i][j]);
        }
        __syncthreads();
        ++step;
    }
    #pragma unroll
    for(int i=0;i<FM;++i) for(int j=0;j<2;++j) wmma::store_matrix_sync(&sm.out[warp][i*16][j*16],acc[i][j],32,wmma::mem_row_major);
    __syncwarp();
    for(int v=threadIdx.x%32;v<WM*32;v+=32) {
        int64_t m=m0+wm*WM+v/32;
        int n=n0+wn*32+v%32;
        if(m<pixels && n<cout) {
            float z=rounded<T>(sm.out[warp][v/32][v%32]);
            y[m*cout+n]=T(bias?__fadd_rn(z,fp(bias[n])):z);
        }
    }
}

API int wops_abi() { return 1; }
API int wops_adaln(const void* x,const void* scale,const void* shift,void* y,int dtype,int64_t rows,int width,int inner,
    int64_t s0,int64_t s1,int64_t h0,int64_t h1,float eps,void* stream) {
    if(rows<=0 || rows>2147483647 || width<=0 || inner<=0) return int(cudaErrorInvalidValue);
    if(dtype==0) adaln_kernel<<<unsigned(rows),256,0,(cudaStream_t)stream>>>((const __half*)x,(const __half*)scale,(const __half*)shift,(__half*)y,rows,width,inner,s0,s1,h0,h1,eps);
    else if(dtype==1) adaln_kernel<<<unsigned(rows),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)scale,(const __nv_bfloat16*)shift,(__nv_bfloat16*)y,rows,width,inner,s0,s1,h0,h1,eps);
    else return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}
API int wops_pixel(const void* x,void* y,int dtype,int64_t pixels,int c,int t,int h,int w,
    int64_t sb,int64_t sc,int64_t st,int64_t sh,int64_t sw,float eps,void* stream) {
    if(pixels<=0 || (pixels+3)/4>2147483647 || c<=0) return int(cudaErrorInvalidValue);
    if(dtype==0) pixel_kernel<<<unsigned((pixels+3)/4),128,0,(cudaStream_t)stream>>>((const __half*)x,(__half*)y,pixels,c,t,h,w,sb,sc,st,sh,sw,eps);
    else if(dtype==1) pixel_kernel<<<unsigned((pixels+3)/4),128,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)y,pixels,c,t,h,w,sb,sc,st,sh,sw,eps);
    else return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}
API int wops_conv(const void* x,const void* wt,const void* bias,void* y,int dtype,int64_t pixels,int cin,int cout,
    int t,int h,int w,int tp,int hp,int wp,int kt,int kh,int kw,void* stream) {
    if(pixels<=0 || (pixels+63)/64>2147483647 || cin%32 || cout%32 || cin<=0 || cout<=0 || cout>65535*64) return int(cudaErrorInvalidValue);
    dim3 grid(unsigned((pixels+127)/128),(cout+63)/64);
    #define LAUNCH(T) conv_kernel<T,32,128><<<grid,128,0,(cudaStream_t)stream>>>((const T*)x,(const T*)wt,(const T*)bias,(T*)y,pixels,cin,cout,t,h,w,tp,hp,wp,kt,kh,kw)
    if(dtype==0) { LAUNCH(__half); }
    else if(dtype==1) { LAUNCH(__nv_bfloat16); }
    else return int(cudaErrorInvalidValue);
    #undef LAUNCH
    return int(cudaGetLastError());
}

// Configuration entry points retain the legacy ABI above for older callers.
API int wops_tuning_version() { return 1; }
API int wops_adaln_config(const void* x,const void* scale,const void* shift,void* y,int dtype,int64_t rows,int width,int inner,
    int64_t s0,int64_t s1,int64_t h0,int64_t h1,float eps,void* stream,int config) {
    if(config==0)return wops_adaln(x,scale,shift,y,dtype,rows,width,inner,s0,s1,h0,h1,eps,stream);
    if(rows<=0 || rows>2147483647 || width<=0 || inner<=0 || config<1 || config>2)return int(cudaErrorInvalidValue);
    #define LAUNCH(T,N) adaln_kernel<T,N><<<unsigned(rows),N,0,(cudaStream_t)stream>>>((const T*)x,(const T*)scale,(const T*)shift,(T*)y,rows,width,inner,s0,s1,h0,h1,eps)
    #define CONFIG(T) if(config==1){LAUNCH(T,128);}else{LAUNCH(T,512);}
    if(dtype==0){CONFIG(__half)}else if(dtype==1){CONFIG(__nv_bfloat16)}else return int(cudaErrorInvalidValue);
    #undef CONFIG
    #undef LAUNCH
    return int(cudaGetLastError());
}
API int wops_pixel_config(const void* x,void* y,int dtype,int64_t pixels,int c,int t,int h,int w,
    int64_t sb,int64_t sc,int64_t st,int64_t sh,int64_t sw,float eps,void* stream,int config) {
    if(config==0)return wops_pixel(x,y,dtype,pixels,c,t,h,w,sb,sc,st,sh,sw,eps,stream);
    if(pixels<=0 || pixels>2147483616 || int64_t(w)*h*t>2147483647)return int(cudaErrorInvalidValue);
    #define CASE(T,C,L) case C: if constexpr(C/L<=64) {pixel_specialized<T,C,L><<<unsigned((pixels+128/L-1)/(128/L)),128,0,(cudaStream_t)stream>>>((const T*)x,(T*)y,pixels,c,t,h,w,sb,sc,st,sh,sw,eps);}else{return int(cudaErrorInvalidValue);}break;
    #define CHANNELS(T,L) switch(c){CASE(T,32,L) CASE(T,64,L) CASE(T,128,L) CASE(T,256,L) CASE(T,512,L) CASE(T,1024,L) default:return int(cudaErrorInvalidValue);}
    #define CONFIG(T) switch(config){case 1:{CHANNELS(T,4)}break;case 2:{CHANNELS(T,8)}break;case 3:{CHANNELS(T,16)}break;case 4:{CHANNELS(T,32)}break;default:return int(cudaErrorInvalidValue);}
    if(dtype==0){CONFIG(__half)}else if(dtype==1){CONFIG(__nv_bfloat16)}else return int(cudaErrorInvalidValue);
    #undef CONFIG
    #undef CHANNELS
    #undef CASE
    return int(cudaGetLastError());
}
API int wops_conv_config(const void* x,const void* wt,const void* bias,void* y,int dtype,int64_t pixels,int cin,int cout,
    int t,int h,int w,int tp,int hp,int wp,int kt,int kh,int kw,void* stream,int config) {
    if(config==0)return wops_conv(x,wt,bias,y,dtype,pixels,cin,cout,t,h,w,tp,hp,wp,kt,kh,kw,stream);
    if(pixels<=0 || (pixels+31)/32>2147483647 || cin<=0 || cout<=0 || cin%32 || cout%32 || cout>65535*64 || config<1 || config>3 || (config>=2 && (kw*cin)%64))return int(cudaErrorInvalidValue);
    #define LAUNCH(T,BK,BM) conv_kernel<T,BK,BM><<<dim3(unsigned((pixels+BM-1)/BM),(cout+63)/64),128,0,(cudaStream_t)stream>>>((const T*)x,(const T*)wt,(const T*)bias,(T*)y,pixels,cin,cout,t,h,w,tp,hp,wp,kt,kh,kw)
    #define CONFIG(T) switch(config){case 1:LAUNCH(T,32,64);break;case 2:LAUNCH(T,64,64);break;case 3:LAUNCH(T,64,32);break;}
    if(dtype==0){CONFIG(__half)}else if(dtype==1){CONFIG(__nv_bfloat16)}else return int(cudaErrorInvalidValue);
    #undef CONFIG
    #undef LAUNCH
    return int(cudaGetLastError());
}
