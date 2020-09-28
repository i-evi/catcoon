#ifndef _CC_DTYPE_H_
#define _CC_DTYPE_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define _DT_UINT8   unsigned char
#define _DT_INT8    char
#define _DT_UINT16  unsigned short
#define _DT_INT16   short
#define _DT_UINT32  unsigned int
#define _DT_INT32   int
#ifdef _MSC_VER
	#define _DT_UINT64 unsigned __int64
	#define _DT_INT64  __int64
#else  /* 64bit integer for gcc, clang... */
	#define _DT_UINT64 unsigned long long
	#define _DT_INT64  long long
#endif
/* #define _DT_FLOAT16 */
#define _DT_FLOAT32 float
#define _DT_FLOAT64 double

typedef _DT_UINT8   cc_dtype;
typedef _DT_UINT8   cc_uint8;
typedef _DT_UINT16  cc_uint16;
typedef _DT_UINT32  cc_uint32;
typedef _DT_UINT64  cc_uint64;
typedef _DT_INT8    cc_int8;
typedef _DT_INT16   cc_int16;
typedef _DT_INT32   cc_int32;
typedef _DT_INT64   cc_int64;
/* typedef _DT_FLOAT32 cc_float16; */
typedef _DT_FLOAT32 cc_float32;
typedef _DT_FLOAT64 cc_float64;

/*
 * 7 6 5 4 3  ~  0
 * | | | | |_____|
 * | | | |    |____sub-class code
 * | | |
 * | |
 * | |______integer or float(0/1)
 * |
 * |______unsigned or signed(0/1)
 */

#define CC_DT_FLAG_POINT 0x40
#define CC_DT_FLAG_SIGN  0x80

#define CC_UINT8    0x03
#define CC_UINT16   0x04
#define CC_UINT32   0x05
#define CC_UINT64   0x06
#define CC_INT8     0x83
#define CC_INT16    0x84
#define CC_INT32    0x85
#define CC_INT64    0x86
/* #define CC_FLOAT16  0xC4 */
#define CC_FLOAT32  0xC5
#define CC_FLOAT64  0xC6

#define CC_1B_LEN 1
#define CC_2B_LEN 2
#define CC_4B_LEN 4
#define CC_8B_LEN 8

#define CC_INT8_LEN    CC_1B_LEN
#define CC_INT16_LEN   CC_2B_LEN
#define CC_INT32_LEN   CC_4B_LEN
#define CC_INT64_LEN   CC_8B_LEN
#define CC_UINT8_LEN   CC_1B_LEN
#define CC_UINT16_LEN  CC_2B_LEN
#define CC_UINT32_LEN  CC_4B_LEN
#define CC_UINT64_LEN  CC_8B_LEN
/* #define CC_FLOAT16_LEN CC_2B_LEN */
#define CC_FLOAT32_LEN CC_4B_LEN
#define CC_FLOAT64_LEN CC_8B_LEN

#define cc_dtype_check(dt1, dt2) (!(dt1 - dt2))

int cc_dtype_size(cc_dtype dt);

const char *cc_dtype_to_string(cc_dtype dt);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_DTYPE_H_ */
