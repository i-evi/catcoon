/*
 * Global function configuration
 */

#ifndef _GLOBAL_FN_CFG_H_
#define _GLOBAL_FN_CFG_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_dtype.h"
#include "cc_cpufn.h"

void (*_activation_relu)(void *inp, cc_int32 elems, cc_dtype dt)
	=  cc_cpu_activation_relu;

void (*_activation_relu6)(void *inp, cc_int32 elems, cc_dtype dt)
	=  cc_cpu_activation_relu6;

void (*_activation_softmax)(void *inp, cc_int32 elems, cc_dtype dt)
	= cc_cpu_activation_softmax;

void (*_max_pool2d)(void *inp, void *oup, cc_int32 x, cc_int32 y,
	cc_int32 s, cc_dtype dt) = cc_cpu_max_pool2d;

void (*_conv2d)(void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 oup_x, cc_int32 oup_y,
	cc_int32 sx, cc_int32 sy, void *filter, cc_int32 fw,
	cc_dtype dt) = cc_cpu_conv2d;

void (*_fully_connected)(void *inp, void *oup, void *w, void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt) = cc_cpu_fully_connected;

void (*_batch_norm)(void *inp, cc_int32 len,
	void *bnpara, cc_dtype dt) = cc_cpu_batch_norm;

/*
 * cc_array functions' cfg, we do not use a standard BLAS directly
 */
#include "cc_array.h"

#define GLOBAL_FN_SET_ARRAY_CAST(dtype) \
void (*_array_cast_ ## dtype)(void *dst, void *src,   \
	int arrlen, int dt) = cc_array_cast_ ## dtype;

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

void (*_array_set)(
	void *arr, int arrlen, void *x, int dt) = cc_array_set;

void (*_array_clip_by_value)(void *arr, int arrlen,
	void *min, void *max, int dt) = cc_array_clip_by_value;

void (*_array_add_by)(void *arr,
	int arrlen, void *x, int dt) = cc_array_add_by;
void (*_array_sub_by)(void *arr,
	int arrlen, void *x, int dt) = cc_array_sub_by;
void (*_array_mul_by)(void *arr,
	int arrlen, void *x, int dt) = cc_array_mul_by;
void (*_array_div_by)(void *arr,
	int arrlen, void *x, int dt) = cc_array_div_by;

void (*_array_add_ew)(void *oup,
	int arrlen, void *a, void *b, int dt) = cc_array_add_ew;
void (*_array_sub_ew)(void *oup,
	int arrlen, void *a, void *b, int dt) = cc_array_sub_ew;
void (*_array_mul_ew)(void *oup,
	int arrlen, void *a, void *b, int dt) = cc_array_mul_ew;
void (*_array_div_ew)(void *oup,
	int arrlen, void *a, void *b, int dt) = cc_array_div_ew;

#ifdef __cplusplus
	}
#endif

#endif /* _GLOBAL_FN_CFG_H_ */
