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

typedef void (*fn_activation_relu)(
	void *inp, cc_int32 elems, cc_dtype dt);
typedef void (*fn_activation_relu6)(
	void *inp, cc_int32 elems, cc_dtype dt);

typedef void (*fn_activation_softmax)(
	void *inp, cc_int32 elems, cc_dtype dt);

typedef void (*fn_max_pool2d)(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

typedef void (*fn_avg_pool2d)(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

typedef void (*fn_conv2d)(const void *inp, void *oup,
	cc_int32 x,cc_int32 y, cc_int32 oup_x, cc_int32 oup_y,
	cc_int32 sx, cc_int32 sy, const void *filter,
	cc_int32 fw, cc_dtype dt);

typedef void (*fn_fully_connected)(const void *inp,
		void *oup, const void *w, const void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt);

typedef void (*fn_batch_norm)(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt);

/*
 * cc_array functions' cfg, we do not use a standard BLAS directly
 */
#include "cc_array.h"

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

#define GLOBAL_FN_DEF_ARRAY_CAST(dtype) \
typedef void (*fn_array_cast_ ## dtype)(                \
	void *dst, const void *src, int arrlen, int dt);

GLOBAL_FN_DEF_ARRAY_CAST  (uint8)
GLOBAL_FN_DEF_ARRAY_CAST  (uint16)
GLOBAL_FN_DEF_ARRAY_CAST  (uint32)
GLOBAL_FN_DEF_ARRAY_CAST  (uint64)
GLOBAL_FN_DEF_ARRAY_CAST  (int8)
GLOBAL_FN_DEF_ARRAY_CAST  (int16)
GLOBAL_FN_DEF_ARRAY_CAST  (int32)
GLOBAL_FN_DEF_ARRAY_CAST  (int64)
GLOBAL_FN_DEF_ARRAY_CAST  (float32)
GLOBAL_FN_DEF_ARRAY_CAST  (float64)

#ifdef __cplusplus
	}
#endif

#endif /* _GLOBAL_FN_CFG_H_ */
