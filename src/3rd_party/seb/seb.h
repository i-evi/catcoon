#ifndef _SEB_H_
#define _SEB_H_

#ifdef __cplusplus
	extern "C" {
#endif
/*
 * Sequential Encoded Buffers
 *
 * | Magic | length | buflen | const blklen | block 0 | block 1 ...
 *   seb-  4        8        12             16
 *      |____Encode type(SEB_ENCODE_TYPE)
 *
 * Block
 * | blklen | data
 *          4
 *
 */

/* #define SEB_ENCODE_BASE64  10 */
/* #define SEB_ENCODE_ZLIB    20 */

#define SEB_SET_FASTLZ /* Set by default */

#ifdef SEB_SET_FASTLZ
	#define SEB_ENCODE_FASTLZ  21
#endif /* SEB_SET_FASTLZ */

#define SEB_MAGIC_BEGIN    0
#define SEB_ENCODE_TYPE    3
#define SEB_LENGTH_BEGIN   4
#define SEB_BUFLEN_BEGIN   8
#define SEB_C_BLKLEN_BEGIN 12
#define SEB_BLOCK_BEGIN    16

#define SEB_BUFFER_LENGTH (16 * 1024) /* Buffer length by default */

#ifndef byte
	#define SEB_TYPE_BYTE
	#define byte unsigned char
#endif
#ifndef uint32
	#define SEB_TYPE_UINT32
	#define uint32 unsigned int
#endif

enum {
	sebFILE_MODE_RO,
	sebFILE_MODE_WO
};

enum {
	SEB_GLOBAL_ENCODE_CTRL,
	SEB_GLOBAL_DECODE_CTRL,
	SEB_GLOBAL_CONST_BLKLEN,
	SEB_GLOBAL_BUFFER_LENGTH,
	SEB_GLOBAL_ENCDAT_LENGTH
};

typedef int (seb_encode_function)(int ctrl,
			void *input, int input_length,
		void *output, int max_out_length);

typedef int (seb_decode_function)(int ctrl,
			void *input, int input_length,
		void *output, int max_out_length);

/* #if `stdio.h` has been included */
#ifndef FOPEN_MAX
#define SEB_FPTR_TYPE void*
#else
#define SEB_FPTR_TYPE FILE*
#endif

typedef struct {
	byte  *buffer;        /* Data buffer */
	byte  *encdat;        /* Encoded data */
	uint32 buflen;
	uint32 blklen;
	uint32 cursor;        /* Current buffer cursor */
	uint32 length;        /* File length */
	uint32 offset;        /* File length */
	uint32 const_blklen;  /* 0 if not constant */
	int mode;
	int encode_ctrl;
	int decode_ctrl;
	seb_encode_function *encode;
	seb_decode_function *decode;
	SEB_FPTR_TYPE fp; /* FILE* */
} sebFILE;

void seb_global_encoder(int enctype);

void seb_global_parameter(int para, int v);

sebFILE *sebfopen(const char *pathname, const char *mode);
void     sebfclose(sebFILE *sebfp);
uint32   sebflush(sebFILE *sebfp);
uint32   sebfwrite(void *ptr, uint32 size, uint32 nmemb, sebFILE *sebfp);
uint32   sebfread(void *ptr, uint32 size, uint32 nmemb, sebFILE *sebfp);

#define sebfilelen(sebfp) (sebfp->length)

#ifdef SEB_TYPE_BYTE
	#undef byte
#endif
#ifdef SEB_TYPE_UINT32
	#undef uint32
#endif

#ifdef __cplusplus
	}
#endif

#endif /* _SEB_H_ */
