#include "cc_dtype.h"
#include "util_log.h"

#define COMPILER_ASSERT(condition)\
		((void)sizeof(char[1 - 2*!!(condition)]))
/*============ compile-time test ============*/
void __________cc_datatype_check__________()
{
	COMPILER_ASSERT(CC_1B_LEN - sizeof(cc_uint8));
	COMPILER_ASSERT(CC_2B_LEN - sizeof(cc_uint16));
	COMPILER_ASSERT(CC_4B_LEN - sizeof(cc_uint32));
	COMPILER_ASSERT(CC_8B_LEN - sizeof(cc_uint64));
	COMPILER_ASSERT(CC_4B_LEN - sizeof(cc_float32));
	COMPILER_ASSERT(CC_8B_LEN - sizeof(cc_float64));
}

int cc_dtype_size(cc_dtype dt)
{
	switch (dt) {
		case CC_INT8:
		case CC_UINT8:
			return CC_1B_LEN;
		case CC_INT16:
		case CC_UINT16:
			return CC_2B_LEN;
		case CC_INT32:
		case CC_UINT32:
		case CC_FLOAT32:
			return CC_4B_LEN;
		case CC_INT64:
		case CC_UINT64:
		case CC_FLOAT64:
			return CC_8B_LEN;
		default:
			return 0;
	}
}

const char *cc_dtype_to_string(cc_dtype dt)
{
	switch (dt) {
		case CC_UINT8:
			return "cc_uint8";
		case CC_UINT16:
			return "cc_uint16";
		case CC_UINT32:
			return "cc_uint32";
		case CC_UINT64:
			return "cc_uint64";
		case CC_INT8:
			return "cc_int8";
		case CC_INT16:
			return "cc_int16";
		case CC_INT32:
			return "cc_int32";
		case CC_INT64:
			return "cc_int64";
		case CC_FLOAT32:
			return "cc_float32";
		case CC_FLOAT64:
			return "cc_float64";
		default:
			return "unsupported dtype";
	}
}
