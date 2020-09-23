#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_log.h"
#include "cc_dtype.h"
#include "cc_array.h"

#define ARRAY_SC_OPS(op, oup, arr, elem, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                                    \
		*((dtype*)oup + i) = *((dtype*)arr + i) op *(dtype*)elem; \
	}

#define ARRAY_ELEM_SET(arr, elem, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {              \
		*((dtype*)arr + i) = *(dtype*)elem; \
	}

#define ARRAY_ELEM_CLIP(arr, min, max, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                              \
		if (min) {                                          \
			*((dtype*)arr + i) =                        \
				*((dtype*)arr + i) < *(dtype*)min ? \
			*(dtype*)min : *((dtype*)arr + i);          \
		}                                                   \
		if (max) {                                          \
			*((dtype*)arr + i) =                        \
				*((dtype*)arr + i) > *(dtype*)max ? \
			*(dtype*)max : *((dtype*)arr + i);          \
		}                                                   \
	}

#define ARRAY_EW_OPS(op, oup, a, b, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                   \
		*((dtype*)oup + i) = *((dtype*)a + i) op \
		*((dtype*)b + i);                        \
	}

#define ARRAY_SUM(arr, arrlen, dtype, sum) \
	*(dtype*)sum = 0;                           \
	for (i = 0; i < arrlen; ++i) {              \
		*(dtype*)sum += *((dtype*)arr + i); \
	}

#define ARRAY_CAST_CASE(_DT, _srcdt, _dstdt) \
case _DT:                                                          \
	for (i = 0; i < arrlen; ++i)                               \
		*((_dstdt*)dst + i) = (_dstdt)*((_srcdt*)src + i); \
	break;

#define CC_ARRAY_CAST_IMPLEMENTATION(dtype) \
void cc_array_cast_ ## dtype(                                            \
	void *dst, const void *src, int arrlen, int dt)                  \
{                                                                        \
	cc_int32 i;                                                      \
	switch (dt) {                                                    \
		ARRAY_CAST_CASE(CC_UINT8, cc_uint8, cc_ ## dtype);       \
		ARRAY_CAST_CASE(CC_UINT16, cc_uint16, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_UINT32, cc_uint32, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_UINT64, cc_uint64, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_INT8, cc_int8, cc_ ## dtype);         \
		ARRAY_CAST_CASE(CC_INT16, cc_int16, cc_ ## dtype);       \
		ARRAY_CAST_CASE(CC_INT32, cc_int32, cc_ ## dtype);       \
		ARRAY_CAST_CASE(CC_INT64, cc_int64, cc_ ## dtype);       \
		ARRAY_CAST_CASE(CC_FLOAT32, cc_float32, cc_ ## dtype);   \
		ARRAY_CAST_CASE(CC_FLOAT64, cc_float64, cc_ ## dtype);   \
		default:                                                 \
			utlog_format(UTLOG_ERR,                          \
				"cc_array: unsupported dtype %x\n", dt); \
			break;                                           \
	}                                                                \
}

#define ARRAY_SET_CASE(_DT, _dt) \
case _DT:                                   \
	ARRAY_ELEM_SET(arr, x, arrlen, _dt) \
	break;
void cc_array_set(void *arr, int arrlen, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SET_CASE(CC_UINT8, cc_uint8);
		ARRAY_SET_CASE(CC_UINT16, cc_uint16);
		ARRAY_SET_CASE(CC_UINT32, cc_uint32);
		ARRAY_SET_CASE(CC_UINT64, cc_uint64);
		ARRAY_SET_CASE(CC_INT8, cc_int8);
		ARRAY_SET_CASE(CC_INT16, cc_int16);
		ARRAY_SET_CASE(CC_INT32, cc_int32);
		ARRAY_SET_CASE(CC_INT64, cc_int64);
		ARRAY_SET_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SET_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_CLIP_CASE(_DT, _dt) \
case _DT:                                     \
	ARRAY_ELEM_CLIP(arr, min, max, arrlen, _dt); \
	break;
void cc_array_clip_by_value(
	void *arr, int arrlen, const void *min, const void *max, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_CLIP_CASE(CC_UINT8, cc_uint8);
		ARRAY_CLIP_CASE(CC_UINT16, cc_uint16);
		ARRAY_CLIP_CASE(CC_UINT32, cc_uint32);
		ARRAY_CLIP_CASE(CC_UINT64, cc_uint64);
		ARRAY_CLIP_CASE(CC_INT8, cc_int8);
		ARRAY_CLIP_CASE(CC_INT16, cc_int16);
		ARRAY_CLIP_CASE(CC_INT32, cc_int32);
		ARRAY_CLIP_CASE(CC_INT64, cc_int64);
		ARRAY_CLIP_CASE(CC_FLOAT32, cc_float32);
		ARRAY_CLIP_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

CC_ARRAY_CAST_IMPLEMENTATION  (uint8)
CC_ARRAY_CAST_IMPLEMENTATION  (uint16)
CC_ARRAY_CAST_IMPLEMENTATION  (uint32)
CC_ARRAY_CAST_IMPLEMENTATION  (uint64)
CC_ARRAY_CAST_IMPLEMENTATION  (int8)
CC_ARRAY_CAST_IMPLEMENTATION  (int16)
CC_ARRAY_CAST_IMPLEMENTATION  (int32)
CC_ARRAY_CAST_IMPLEMENTATION  (int64)
CC_ARRAY_CAST_IMPLEMENTATION  (float32)
CC_ARRAY_CAST_IMPLEMENTATION  (float64)

#define ARRAY_ADD_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(+, oup, a, x, arrlen, _dt); \
	break;
void cc_array_add_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_ADD_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_ADD_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_ADD_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_ADD_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_ADD_BY_CASE(CC_INT8, cc_int8);
		ARRAY_ADD_BY_CASE(CC_INT16, cc_int16);
		ARRAY_ADD_BY_CASE(CC_INT32, cc_int32);
		ARRAY_ADD_BY_CASE(CC_INT64, cc_int64);
		ARRAY_ADD_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_ADD_BY_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_SUB_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(-, oup, a, x, arrlen, _dt); \
	break;
void cc_array_sub_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUB_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUB_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUB_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUB_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUB_BY_CASE(CC_INT8, cc_int8);
		ARRAY_SUB_BY_CASE(CC_INT16, cc_int16);
		ARRAY_SUB_BY_CASE(CC_INT32, cc_int32);
		ARRAY_SUB_BY_CASE(CC_INT64, cc_int64);
		ARRAY_SUB_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUB_BY_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_MUL_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(*, oup, a, x, arrlen, _dt); \
	break;
void cc_array_mul_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MUL_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_MUL_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_MUL_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_MUL_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_MUL_BY_CASE(CC_INT8, cc_int8);
		ARRAY_MUL_BY_CASE(CC_INT16, cc_int16);
		ARRAY_MUL_BY_CASE(CC_INT32, cc_int32);
		ARRAY_MUL_BY_CASE(CC_INT64, cc_int64);
		ARRAY_MUL_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MUL_BY_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_DIV_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(/, oup, a, x, arrlen, _dt); \
	break;
void cc_array_div_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_DIV_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_DIV_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_DIV_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_DIV_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_DIV_BY_CASE(CC_INT8, cc_int8);
		ARRAY_DIV_BY_CASE(CC_INT16, cc_int16);
		ARRAY_DIV_BY_CASE(CC_INT32, cc_int32);
		ARRAY_DIV_BY_CASE(CC_INT64, cc_int64);
		ARRAY_DIV_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_DIV_BY_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_ADD_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(+, oup, a, b, arrlen, _dt); \
	break;
void cc_array_add_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_ADD_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_ADD_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_ADD_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_ADD_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_ADD_EW_CASE(CC_INT8, cc_int8);
		ARRAY_ADD_EW_CASE(CC_INT16, cc_int16);
		ARRAY_ADD_EW_CASE(CC_INT32, cc_int32);
		ARRAY_ADD_EW_CASE(CC_INT64, cc_int64);
		ARRAY_ADD_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_ADD_EW_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_SUB_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(-, oup, a, b, arrlen, _dt); \
	break;
void cc_array_sub_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUB_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUB_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUB_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUB_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUB_EW_CASE(CC_INT8, cc_int8);
		ARRAY_SUB_EW_CASE(CC_INT16, cc_int16);
		ARRAY_SUB_EW_CASE(CC_INT32, cc_int32);
		ARRAY_SUB_EW_CASE(CC_INT64, cc_int64);
		ARRAY_SUB_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUB_EW_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_MUL_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(*, oup, a, b, arrlen, _dt); \
	break;
void cc_array_mul_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MUL_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_MUL_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_MUL_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_MUL_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_MUL_EW_CASE(CC_INT8, cc_int8);
		ARRAY_MUL_EW_CASE(CC_INT16, cc_int16);
		ARRAY_MUL_EW_CASE(CC_INT32, cc_int32);
		ARRAY_MUL_EW_CASE(CC_INT64, cc_int64);
		ARRAY_MUL_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MUL_EW_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_DIV_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(/, oup, a, b, arrlen, _dt); \
	break;
void cc_array_div_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_DIV_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_DIV_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_DIV_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_DIV_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_DIV_EW_CASE(CC_INT8, cc_int8);
		ARRAY_DIV_EW_CASE(CC_INT16, cc_int16);
		ARRAY_DIV_EW_CASE(CC_INT32, cc_int32);
		ARRAY_DIV_EW_CASE(CC_INT64, cc_int64);
		ARRAY_DIV_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_DIV_EW_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_SUM_CASE(_DT, _dt) \
case _DT:                               \
	ARRAY_SUM(arr, arrlen, _dt, x); \
	break;
void cc_array_sum(const void *arr, int arrlen, void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUM_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUM_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUM_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUM_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUM_CASE(CC_INT8, cc_int8);
		ARRAY_SUM_CASE(CC_INT16, cc_int16);
		ARRAY_SUM_CASE(CC_INT32, cc_int32);
		ARRAY_SUM_CASE(CC_INT64, cc_int64);
		ARRAY_SUM_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUM_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define ARRAY_MEAN_CASE(_DT, _dt) \
case _DT:                               \
	ARRAY_SUM(arr, arrlen, _dt, x); \
	*(_dt*)x /= arrlen;             \
	break;
void cc_array_mean(const void *arr, int arrlen, void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MEAN_CASE(CC_UINT8, cc_uint8);
		ARRAY_MEAN_CASE(CC_UINT16, cc_uint16);
		ARRAY_MEAN_CASE(CC_UINT32, cc_uint32);
		ARRAY_MEAN_CASE(CC_UINT64, cc_uint64);
		ARRAY_MEAN_CASE(CC_INT8, cc_int8);
		ARRAY_MEAN_CASE(CC_INT16, cc_int16);
		ARRAY_MEAN_CASE(CC_INT32, cc_int32);
		ARRAY_MEAN_CASE(CC_INT64, cc_int64);
		ARRAY_MEAN_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MEAN_CASE(CC_FLOAT64, cc_float64);
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
}

#define PRINT_ARRAY_CASE(_DT, _dt) \
case _DT:                                 \
	fprintf(fp, pat, *((_dt*)a + i)); \
	break;
void cc_print_array(const void *a, int arrlen, int dt, void *stream)
{	
	cc_int32 i;
	const char *pat = NULL;
	FILE *fp = stream ? (FILE*)stream : stdout;
	switch (dt) {
		case CC_FLOAT32:
			pat = "%f ";
			break;
		case CC_FLOAT64:
			pat = "%lf ";
			break;
		case CC_UINT8:
		case CC_UINT16:
		case CC_UINT32:
			pat = "%d ";
			break;
		case CC_UINT64:
			pat = "%ld ";
			break;
		case CC_INT8:
		case CC_INT16:
		case CC_INT32:
			pat = "%d ";
			break;
		case CC_INT64:
			pat = "%ld ";
			break;
		default:
			utlog_format(UTLOG_ERR,
				"cc_array: unsupported dtype %x\n", dt);
			break;
	}
	for (i = 0; i < arrlen; ++i) {
		switch(dt) {
			PRINT_ARRAY_CASE(CC_UINT8, cc_uint8);
			PRINT_ARRAY_CASE(CC_UINT16, cc_uint16);
			PRINT_ARRAY_CASE(CC_UINT32, cc_uint32);
			PRINT_ARRAY_CASE(CC_UINT64, cc_uint64);
			PRINT_ARRAY_CASE(CC_INT8, cc_int8);
			PRINT_ARRAY_CASE(CC_INT16, cc_int16);
			PRINT_ARRAY_CASE(CC_INT32, cc_int32);
			PRINT_ARRAY_CASE(CC_INT64, cc_int64);
			PRINT_ARRAY_CASE(CC_FLOAT32, cc_float32);
			PRINT_ARRAY_CASE(CC_FLOAT64, cc_float64);
			default:
				utlog_format(UTLOG_ERR,
					"cc_array: unsupported dtype %x\n", dt);
				break;
		}
	}
	printf("\n");
}
