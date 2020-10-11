#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_log.h"
#include "cc_dtype.h"
#include "cc_array.h"

#include "global_fn_cfg.h"
#define EXT_ARRAY_CAST_DEFINITION(dtype) \
extern fn_array_cast_ ## dtype _array_cast_ ## dtype;

EXT_ARRAY_CAST_DEFINITION  (uint8)
EXT_ARRAY_CAST_DEFINITION  (uint16)
EXT_ARRAY_CAST_DEFINITION  (uint32)
EXT_ARRAY_CAST_DEFINITION  (uint64)
EXT_ARRAY_CAST_DEFINITION  (int8)
EXT_ARRAY_CAST_DEFINITION  (int16)
EXT_ARRAY_CAST_DEFINITION  (int32)
EXT_ARRAY_CAST_DEFINITION  (int64)
EXT_ARRAY_CAST_DEFINITION  (float32)
EXT_ARRAY_CAST_DEFINITION  (float64)

extern fn_array_set    _array_set;
extern fn_array_clip_by_value _array_clip_by_value;

extern fn_array_add_by _array_add_by;
extern fn_array_sub_by _array_sub_by;
extern fn_array_mul_by _array_mul_by;
extern fn_array_div_by _array_div_by;

extern fn_array_add_ew _array_add_ew;
extern fn_array_sub_ew _array_sub_ew;
extern fn_array_mul_ew _array_mul_ew;
extern fn_array_div_ew _array_div_ew;

extern fn_array_sum  _array_sum;
extern fn_array_mean _array_mean;

#define CC_ARRAY_CAST_IMPLEMENTATION(dtype) \
void cc_array_cast_ ## dtype(                           \
	void *dst, const void *src, int arrlen, int dt) \
{                                                       \
	_array_cast_ ## dtype(dst, src, arrlen, dt);    \
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

void cc_array_set(void *arr, int arrlen, const void *x, int dt)
{
	_array_set(arr, arrlen, x, dt);
}

void cc_array_clip_by_value(void *arr,
	int arrlen, const void *min, const void *max, int dt)
{
	_array_clip_by_value(arr, arrlen, min, max, dt);
}

void cc_array_add_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	_array_add_by(oup, arrlen, a, x, dt);
}

void cc_array_sub_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	_array_sub_by(oup, arrlen, a, x, dt);
}

void cc_array_mul_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	_array_mul_by(oup, arrlen, a, x, dt);
}

void cc_array_div_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	_array_div_by(oup, arrlen, a, x, dt);
}

void cc_array_add_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	_array_add_ew(oup, arrlen, a, b, dt);
}

void cc_array_sub_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	_array_sub_ew(oup, arrlen, a, b, dt);
}

void cc_array_mul_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	_array_mul_ew(oup, arrlen, a, b, dt);
}

void cc_array_div_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	_array_div_ew(oup, arrlen, a, b, dt);
}

void cc_array_sum (const void *arr, int arrlen, void *x, int dt)
{
	_array_sum(arr, arrlen, x, dt);
}

void cc_array_mean(const void *arr, int arrlen, void *x, int dt)
{
	_array_mean(arr, arrlen, x, dt);
}

#define PRINT_ARRAY_CASE(_DT, _dt) \
case _DT:                                 \
	fprintf(fp, pat, *((_dt*)a + i)); \
	if (i < (arrlen - 1))             \
		fputc(' ', fp);           \
	break;
void cc_print_array(const void *a, int arrlen, int dt, void *stream)
{	
	cc_int32 i;
	const char *pat = NULL;
	FILE *fp = stream ? (FILE*)stream : stdout;
	switch (dt) {
	case CC_FLOAT32:
		pat = "%f";
		break;
	case CC_FLOAT64:
		pat = "%lf";
		break;
	case CC_UINT8:
	case CC_UINT16:
	case CC_UINT32:
		pat = "%d";
		break;
	case CC_UINT64:
		pat = "%ld";
		break;
	case CC_INT8:
	case CC_INT16:
	case CC_INT32:
		pat = "%d";
		break;
	case CC_INT64:
		pat = "%ld";
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
}
