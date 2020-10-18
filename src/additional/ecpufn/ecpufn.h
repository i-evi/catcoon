/* some experimental implementations of NN functions on cpu */
#ifndef _ECPUFN_
#define _ECPUFN_

#ifdef _MSC_VER
	#error Can not compile on a MSC compiler for now.
#endif

#ifdef __cplusplus
	extern "C" {
#endif

typedef               char           i8;
typedef      unsigned char           u8;
typedef               short          i16;
typedef      unsigned short          u16;
typedef               int            i32;
typedef      unsigned int            u32;
typedef               long long      i64;
typedef      unsigned long long      u64;
typedef               float          f32;
typedef               double         f64;

/* void ecpu_conv2d_xxx(xxx *in, xxx *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, xxx *k, i32 kw) */
#define ECPU_CONV2D_DECLARATION(dtype) \
void  ecpu_conv2d_ ## dtype (const dtype *in, dtype *out, i32 ix, i32 iy, \
	i32 ox, i32 oy, i32 sx, i32 sy, const dtype *k, i32 kw);          \
void naive_conv2d_ ## dtype (const dtype *in, dtype *out, i32 ix, i32 iy, \
	i32 ox, i32 oy, i32 sx, i32 sy, const dtype *k, i32 kw);

ECPU_CONV2D_DECLARATION  (i8);
ECPU_CONV2D_DECLARATION  (u8);
ECPU_CONV2D_DECLARATION  (i16);
ECPU_CONV2D_DECLARATION  (u16);
ECPU_CONV2D_DECLARATION  (i32);
ECPU_CONV2D_DECLARATION  (u32);
ECPU_CONV2D_DECLARATION  (i64);
ECPU_CONV2D_DECLARATION  (u64);
ECPU_CONV2D_DECLARATION  (f32);
ECPU_CONV2D_DECLARATION  (f64);

#if defined(__x86_64) && defined(__SSE__)
  #define ALT_CONV2D_F32_K1S1  sse_conv2d_f32_k1s1
  #define ALT_CONV2D_F32_K2SX  sse_conv2d_f32_k2sx
  #define ALT_CONV2D_F32_K3S1  sse_conv2d_f32_k3s1
  #define ALT_CONV2D_F32_K3SX  sse_conv2d_f32_k3sx
#endif

void sse_conv2d_f32_k1s1(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw);
void sse_conv2d_f32_k2sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw);
void sse_conv2d_f32_k3s1(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw);
void sse_conv2d_f32_k3sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw);

/* void ecpu_dot_prod_xxx(xxx *in, xxx *out, xxx *w, i32 iw); */
#define ECPU_DOTPROD_DECLARATION(dtype) \
void ecpu_dot_prod_  ## dtype (                               \
	const dtype *in, dtype *out, const dtype *w, i32 iw); \
void naive_dot_prod_ ## dtype (                               \
	const dtype *in, dtype *out, const dtype *w, i32 iw);

ECPU_DOTPROD_DECLARATION  (i8);
ECPU_DOTPROD_DECLARATION  (u8);
ECPU_DOTPROD_DECLARATION  (i16);
ECPU_DOTPROD_DECLARATION  (u16);
ECPU_DOTPROD_DECLARATION  (i32);
ECPU_DOTPROD_DECLARATION  (u32);
ECPU_DOTPROD_DECLARATION  (i64);
ECPU_DOTPROD_DECLARATION  (u64);
ECPU_DOTPROD_DECLARATION  (f32);
ECPU_DOTPROD_DECLARATION  (f64);

#if defined(__x86_64)
  #if   defined(__AVX__)
    #define ALT_DOTPROD_F32   avx_dot_prod_f32
  #elif defined(__SSE__)
    #define ALT_DOTPROD_F32  sse_dot_prod_f32
  #endif
#endif

void  avx_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw);
void sse_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw);

/* void ecpu_max_pool2d_xxx(xxx *in, xxx *out, i32 x, i32 y, i32 s); */
#define ECPU_MAXPOOL2D_DECLARATION(dtype) \
void  ecpu_max_pool2d_ ## dtype (                          \
	const dtype *in, dtype *out, i32 x, i32 y, i32 s); \
void naive_max_pool2d_ ## dtype (                          \
	const dtype *in, dtype *out, i32 x, i32 y, i32 s);

ECPU_MAXPOOL2D_DECLARATION  (i8);
ECPU_MAXPOOL2D_DECLARATION  (u8);
ECPU_MAXPOOL2D_DECLARATION  (i16);
ECPU_MAXPOOL2D_DECLARATION  (u16);
ECPU_MAXPOOL2D_DECLARATION  (i32);
ECPU_MAXPOOL2D_DECLARATION  (u32);
ECPU_MAXPOOL2D_DECLARATION  (i64);
ECPU_MAXPOOL2D_DECLARATION  (u64);
ECPU_MAXPOOL2D_DECLARATION  (f32);
ECPU_MAXPOOL2D_DECLARATION  (f64);

#if defined(__x86_64) && defined(__SSE__)
  #define ALT_MAXPOOL2D_F32_S2  sse_max_pool2d_f32_s2
#endif

void sse_max_pool2d_f32_s2(const f32 *in, f32 *out, i32 x, i32 y, i32 s);

#ifdef __cplusplus
	}
#endif

#endif /* _ECPUFNC_ */
