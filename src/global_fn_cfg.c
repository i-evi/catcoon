#include <assert.h>

#include "cc_cpufn.h"
#include "global_fn_cfg.h"

#ifdef ENABLE_ECPUFN
	#include "additional/ecpufn/ecpufn.h"
#endif

void __gfn_check__(void) {return;}

fn_array_set    _array_set    = cc_cpu_array_set;

fn_array_clip_by_value
	_array_clip_by_value  = cc_cpu_array_clip_by_value;

fn_array_add_by _array_add_by = cc_cpu_array_add_by;
fn_array_sub_by _array_sub_by = cc_cpu_array_sub_by;
fn_array_mul_by _array_mul_by = cc_cpu_array_mul_by;
fn_array_div_by _array_div_by = cc_cpu_array_div_by;

fn_array_add_ew _array_add_ew = cc_cpu_array_add_ew;
fn_array_sub_ew _array_sub_ew = cc_cpu_array_sub_ew;
fn_array_mul_ew _array_mul_ew = cc_cpu_array_mul_ew;
fn_array_div_ew _array_div_ew = cc_cpu_array_div_ew;

#ifdef ENABLE_ECPUFN
void cc_ecpu_dot_prod_wrap(const void *in,
	const void *w, int iw, void *out, int dt);
fn_array_dot_prod  _array_dot_prod = cc_ecpu_dot_prod_wrap;
#else
fn_array_dot_prod  _array_dot_prod = cc_cpu_array_dot_prod;
#endif /* ENABLE_ECPUFN */
fn_array_sum       _array_sum      = cc_cpu_array_sum;
fn_array_mean      _array_mean     = cc_cpu_array_mean;

#define GLOBAL_FN_SET_ARRAY_CAST(dtype) \
fn_array_cast_ ## dtype _array_cast_ ## dtype = \
	cc_cpu_array_cast_ ## dtype;

GLOBAL_FN_SET_ARRAY_CAST  (uint8)
GLOBAL_FN_SET_ARRAY_CAST  (uint16)
GLOBAL_FN_SET_ARRAY_CAST  (uint32)
GLOBAL_FN_SET_ARRAY_CAST  (uint64)
GLOBAL_FN_SET_ARRAY_CAST  (int8)
GLOBAL_FN_SET_ARRAY_CAST  (int16)
GLOBAL_FN_SET_ARRAY_CAST  (int32)
GLOBAL_FN_SET_ARRAY_CAST  (int64)
GLOBAL_FN_SET_ARRAY_CAST  (float32)
GLOBAL_FN_SET_ARRAY_CAST  (float64)

fn_activation_relu    _activation_relu    = cc_cpu_activation_relu;
fn_activation_relu6   _activation_relu6   = cc_cpu_activation_relu6;
fn_activation_softmax _activation_softmax = cc_cpu_activation_softmax;

#ifdef ENABLE_ECPUFN
void cc_ecpu_conv2d_wrap(const void *in, void *out, cc_int32 ix,
	cc_int32 iy, cc_int32 ox, cc_int32 oy, cc_int32 sx, cc_int32 sy,
	const void *k, cc_int32 kw, cc_dtype dt);
fn_conv2d          _conv2d          = cc_ecpu_conv2d_wrap;
#else
fn_conv2d          _conv2d          = cc_cpu_conv2d;
#endif /* ENABLE_ECPUFN */
fn_max_pool2d      _max_pool2d      = cc_cpu_max_pool2d;
fn_max_pool2d      _avg_pool2d      = cc_cpu_avg_pool2d;
fn_batch_norm      _batch_norm      = cc_cpu_batch_norm;

#ifdef ENABLE_ECPUFN
void cc_ecpu_dot_prod_wrap(
	const void *in, const void *w, int iw, void *out, int dt)
{
	switch (dt) {
	case CC_UINT8:
		ecpu_dot_prod_u8((cc_uint8*)in,
			(cc_uint8*)out, (cc_uint8*)w, iw);
		break;
	case CC_UINT16:
		ecpu_dot_prod_u16((cc_uint16*)in,
			(cc_uint16*)out, (cc_uint16*)w, iw);
		break;
	case CC_UINT32:
		ecpu_dot_prod_u32((cc_uint32*)in,
			(cc_uint32*)out, (cc_uint32*)w, iw);
		break;
	case CC_UINT64:
		ecpu_dot_prod_u64((cc_uint64*)in,
			(cc_uint64*)out, (cc_uint64*)w, iw);
		break;
	case CC_INT8:
		ecpu_dot_prod_i8((cc_int8*)in,
			(cc_int8*)out, (cc_int8*)w, iw);
		break;
	case CC_INT16:
		ecpu_dot_prod_i16((cc_int16*)in,
			(cc_int16*)out, (cc_int16*)w, iw);
		break;
	case CC_INT32:
		ecpu_dot_prod_i32((cc_int32*)in,
			(cc_int32*)out, (cc_int32*)w, iw);
		break;
	case CC_INT64:
		ecpu_dot_prod_i64((cc_int64*)in,
			(cc_int64*)out, (cc_int64*)w, iw);
		break;
	case CC_FLOAT32:
		ecpu_dot_prod_f32((cc_float32*)in,
			(cc_float32*)out, (cc_float32*)w, iw);
		break;
	case CC_FLOAT64:
		ecpu_dot_prod_f64((cc_float64*)in,
			(cc_float64*)out, (cc_float64*)w, iw);
		break;
	default:
		assert(0);
		break;
	}
}

void cc_ecpu_conv2d_wrap(const void *in, void *out, cc_int32 ix,
	cc_int32 iy, cc_int32 ox, cc_int32 oy, cc_int32 sx, cc_int32 sy,
	const void *k, cc_int32 kw, cc_dtype dt)
{
	switch (dt) {
	case CC_UINT8:
		ecpu_conv2d_u8((cc_uint8*)in, (cc_uint8*)out,
			ix, iy, ox, oy, sx, sy, (cc_uint8*)k, kw);
		break;
	case CC_UINT16:
		ecpu_conv2d_u16((cc_uint16*)in, (cc_uint16*)out,
			ix, iy, ox, oy, sx, sy, (cc_uint16*)k, kw);
		break;
	case CC_UINT32:
		ecpu_conv2d_u32((cc_uint32*)in, (cc_uint32*)out,
			ix, iy, ox, oy, sx, sy, (cc_uint32*)k, kw);
		break;
	case CC_UINT64:
		ecpu_conv2d_u64((cc_uint64*)in, (cc_uint64*)out,
			ix, iy, ox, oy, sx, sy, (cc_uint64*)k, kw);
		break;
	case CC_INT8:
		ecpu_conv2d_i8((cc_int8*)in, (cc_int8*)out,
			ix, iy, ox, oy, sx, sy, (cc_int8*)k, kw);
		break;
	case CC_INT16:
		ecpu_conv2d_i16((cc_int16*)in, (cc_int16*)out,
			ix, iy, ox, oy, sx, sy, (cc_int16*)k, kw);
		break;
	case CC_INT32:
		ecpu_conv2d_i32((cc_int32*)in, (cc_int32*)out,
			ix, iy, ox, oy, sx, sy, (cc_int32*)k, kw);
		break;
	case CC_INT64:
		ecpu_conv2d_i64((cc_int64*)in, (cc_int64*)out,
			ix, iy, ox, oy, sx, sy, (cc_int64*)k, kw);
		break;
	case CC_FLOAT32:
		ecpu_conv2d_f32((cc_float32*)in, (cc_float32*)out,
			ix, iy, ox, oy, sx, sy, (cc_float32*)k, kw);
		break;
	case CC_FLOAT64:
		ecpu_conv2d_f64((cc_float64*)in, (cc_float64*)out,
			ix, iy, ox, oy, sx, sy, (cc_float64*)k, kw);
		break;
	default:
		assert(0);
		break;
	}
}
#endif /* ENABLE_ECPUFN */