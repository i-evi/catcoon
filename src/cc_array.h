#ifndef _CC_ARRAY_H_
#define _CC_ARRAY_H_

#ifdef __cplusplus
	extern "C" {
#endif

/*
 * int <---> cc_int32
 */

#define CC_ARRAY_CAST_DEFINITION(dtype) \
void cc_array_cast_ ## dtype(void *dst, void *src, int arrlen, int dt);

CC_ARRAY_CAST_DEFINITION  (uint8)
CC_ARRAY_CAST_DEFINITION  (uint16)
CC_ARRAY_CAST_DEFINITION  (uint32)
CC_ARRAY_CAST_DEFINITION  (uint64)
CC_ARRAY_CAST_DEFINITION  (int8)
CC_ARRAY_CAST_DEFINITION  (int16)
CC_ARRAY_CAST_DEFINITION  (int32)
CC_ARRAY_CAST_DEFINITION  (int64)
CC_ARRAY_CAST_DEFINITION  (float32)
CC_ARRAY_CAST_DEFINITION  (float64)

void cc_array_set(void *arr, int arrlen, void *x, int dt);

void cc_array_clip_by_value(
	void *arr, int arrlen, void *min, void *max, int dt);

void cc_array_add_by(void *arr, int arrlen, void *x, int dt);
void cc_array_sub_by(void *arr, int arrlen, void *x, int dt);
void cc_array_mul_by(void *arr, int arrlen, void *x, int dt);
void cc_array_div_by(void *arr, int arrlen, void *x, int dt);

void cc_array_add_ew(void *oup, int arrlen, void *a, void *b, int dt);
void cc_array_sub_ew(void *oup, int arrlen, void *a, void *b, int dt);
void cc_array_mul_ew(void *oup, int arrlen, void *a, void *b, int dt);
void cc_array_div_ew(void *oup, int arrlen, void *a, void *b, int dt);

#define cc_add_to_array_ew(dst, src, arrlen, dt) \
	cc_array_add_ew(dst, arrlen, dst, src, dt)
#define cc_sub_to_array_ew(dst, src, arrlen, dt) \
	cc_array_sub_ew(dst, arrlen, dst, src, dt)
#define cc_mul_to_array_ew(dst, src, arrlen, dt) \
	cc_array_mul_ew(dst, arrlen, dst, src, dt)
#define cc_div_to_array_ew(dst, src, arrlen, dt) \
	cc_array_div_ew(dst, arrlen, dst, src, dt)

void cc_array_sum(void *arr, int arrlen, void *x, int dt);
void cc_array_mean(void *arr, int arrlen, void *x, int dt);

void cc_print_array(void *a, int arrlen, int dt, void *stream);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_ARRAY_H_ */
