/*
 * Global function configuration
 */

#ifndef _GLOBAL_FN_CFG_H_
#define _GLOBAL_FN_CFG_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_dtype.h"

typedef void (*fn_array_set)(
	void *arr, int arrlen, const void *x, int dt);

typedef void (*fn_array_clip_by_value)(void *arr,
	int arrlen, const void *min, const void *max, int dt);

typedef void (*fn_array_add_by)(void *oup,
	int arrlen, const void *a, const void *x, int dt); 
typedef void (*fn_array_sub_by)(void *oup,
	int arrlen, const void *a, const void *x, int dt);
typedef void (*fn_array_mul_by)(void *oup,
	int arrlen, const void *a, const void *x, int dt);
typedef void (*fn_array_div_by)(void *oup,
	int arrlen, const void *a, const void *x, int dt);

typedef void (*fn_array_add_ew)(void *oup,
	int arrlen, const void *a, const void *b, int dt);
typedef void (*fn_array_sub_ew)(void *oup,
	int arrlen, const void *a, const void *b, int dt);
typedef void (*fn_array_mul_ew)(void *oup,
	int arrlen, const void *a, const void *b, int dt);
typedef void (*fn_array_div_ew)(void *oup,
	int arrlen, const void *a, const void *b, int dt);

typedef void (*fn_array_dot_prod)(
	const void *a, const void *b, int arrlen, void *x, int dt);

typedef void (*fn_array_sum )(
	const void *arr, int arrlen, void *x, int dt);
typedef void (*fn_array_mean)(
	const void *arr, int arrlen, void *x, int dt);

#define TYPEDEF_FN_ARRAY_CAST(dtype) \
typedef void (*fn_array_cast_ ## dtype)(                \
	void *dst, const void *src, int arrlen, int dt);

TYPEDEF_FN_ARRAY_CAST  (uint8)
TYPEDEF_FN_ARRAY_CAST  (uint16)
TYPEDEF_FN_ARRAY_CAST  (uint32)
TYPEDEF_FN_ARRAY_CAST  (uint64)
TYPEDEF_FN_ARRAY_CAST  (int8)
TYPEDEF_FN_ARRAY_CAST  (int16)
TYPEDEF_FN_ARRAY_CAST  (int32)
TYPEDEF_FN_ARRAY_CAST  (int64)
TYPEDEF_FN_ARRAY_CAST  (float32)
TYPEDEF_FN_ARRAY_CAST  (float64)

typedef void (*fn_activation_relu)(
	void *inp, cc_int32 elems, cc_dtype dt);
typedef void (*fn_activation_relu6)(
	void *inp, cc_int32 elems, cc_dtype dt);

typedef void (*fn_activation_softmax)(
	void *inp, cc_int32 elems, cc_dtype dt);

typedef void (*fn_pool2d)(const void *inp, void *oup, cc_int32 x,
	cc_int32 y, cc_int32 sx, cc_int32 sy, cc_int32 kw, cc_dtype dt);

typedef void (*fn_conv2d)(const void *inp, void *oup,
	cc_int32 x,cc_int32 y, cc_int32 sx, cc_int32 sy,
	const void *filter, cc_int32 fw, cc_dtype dt);

typedef void (*fn_batch_norm)(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt);

#ifdef __cplusplus
	}
#endif

#endif /* _GLOBAL_FN_CFG_H_ */
