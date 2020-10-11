#ifndef _CC_CPUFN_H_
#define _CC_CPUFN_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_dtype.h"

void cc_cpu_activation_relu (void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_activation_relu6(void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_activation_softmax(void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_max_pool2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

void cc_cpu_avg_pool2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

void cc_cpu_conv2d(const void *inp, void *oup, cc_int32 x, cc_int32 y,
		cc_int32 oup_x, cc_int32 oup_y, cc_int32 sx, cc_int32 sy,
	const void *filter, cc_int32 fw, cc_dtype dt);

void cc_cpu_fully_connected(const void *inp,
		void *oup, const void *w, const void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt);

void cc_cpu_batch_norm(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt);

/*
 * int <---> cc_int32
 */

#define CC_CPU_ARRAY_CAST_DEFINITION(dtype) \
void cc_cpu_array_cast_ ## dtype(                        \
	void *dst, const void *src, int arrlen, int dt);

CC_CPU_ARRAY_CAST_DEFINITION  (uint8)
CC_CPU_ARRAY_CAST_DEFINITION  (uint16)
CC_CPU_ARRAY_CAST_DEFINITION  (uint32)
CC_CPU_ARRAY_CAST_DEFINITION  (uint64)
CC_CPU_ARRAY_CAST_DEFINITION  (int8)
CC_CPU_ARRAY_CAST_DEFINITION  (int16)
CC_CPU_ARRAY_CAST_DEFINITION  (int32)
CC_CPU_ARRAY_CAST_DEFINITION  (int64)
CC_CPU_ARRAY_CAST_DEFINITION  (float32)
CC_CPU_ARRAY_CAST_DEFINITION  (float64)

void cc_cpu_array_set(void *arr, int arrlen, const void *x, int dt);

void cc_cpu_array_clip_by_value(void *arr,
	int arrlen, const void *min, const void *max, int dt);

void cc_cpu_array_add_by(void *oup,
	int arrlen, const void *a, const void *x, int dt);
void cc_cpu_array_sub_by(void *oup,
	int arrlen, const void *a, const void *x, int dt);
void cc_cpu_array_mul_by(void *oup,
	int arrlen, const void *a, const void *x, int dt);
void cc_cpu_array_div_by(void *oup,
	int arrlen, const void *a, const void *x, int dt);

void cc_cpu_array_add_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt);
void cc_cpu_array_sub_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt);
void cc_cpu_array_mul_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt);
void cc_cpu_array_div_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt);

void cc_cpu_array_sum (const void *arr, int arrlen, void *x, int dt);
void cc_cpu_array_mean(const void *arr, int arrlen, void *x, int dt);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_CPUFN_H_ */
