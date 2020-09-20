#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "seb.h"

#ifdef SEB_SET_FASTLZ
	#include "fastlz.h"
	#define FASTLZ_COMPRESS_LEVEL_DEF 1
	int encoder_fastlz(int ctrl,
				void *input, int input_length,
			void *output, int max_out_length)
	{
		return fastlz_compress_level(ctrl,
			input, input_length, output);
	}

	int decoder_fastlz(int ctrl,
				void *input, int input_length,
			void *output, int max_out_length)
	{
		return fastlz_decompress(input, input_length,
			 output, max_out_length);
	}
#endif /* SEB_SET_FASTLZ */

#ifndef byte
	#define SEB_TYPE_BYTE
	#define byte unsigned char
#endif
#ifndef uint32
	#define SEB_TYPE_UINT32
	#define uint32 unsigned int
#endif

static int seb_buffer_length = SEB_BUFFER_LENGTH;
static int seb_encdat_length = (SEB_BUFFER_LENGTH * 2);
static int seb_const_blklen  = 0;
static int seb_encode_ctrl = 1;
static int seb_decode_ctrl = 1;
static unsigned char seb_magic_encode_type = SEB_ENCODE_FASTLZ;
static seb_encode_function *seb_encoder = encoder_fastlz;
static seb_decode_function *seb_decoder = decoder_fastlz;

static unsigned char seb_magic[] = {'s', 'e', 'b', '\0'};

static int _seb_set_encoder(sebFILE *sebfp, int enctype)
{
	switch (enctype) {
		case SEB_ENCODE_FASTLZ:
			sebfp->encode = encoder_fastlz;
			sebfp->decode = decoder_fastlz;
			sebfp->encode_ctrl = FASTLZ_COMPRESS_LEVEL_DEF;
			sebfp->decode_ctrl = FASTLZ_COMPRESS_LEVEL_DEF;
			return 0;
		default:
			return -1;
	}
}

void seb_global_encoder(int enctype)
{
	switch (enctype) {
		case SEB_ENCODE_FASTLZ:
			seb_magic_encode_type = SEB_ENCODE_FASTLZ;
			seb_encoder = encoder_fastlz;
			seb_decoder = decoder_fastlz;
			break;
	}
}

void seb_global_parameter(int para, int v)
{
	switch (para) {
		case SEB_GLOBAL_ENCODE_CTRL:
			seb_encode_ctrl = v;
			break;
		case SEB_GLOBAL_DECODE_CTRL:
			seb_decode_ctrl = v;
			break;
		case SEB_GLOBAL_CONST_BLKLEN:
			seb_const_blklen = v;
			break;
		case SEB_GLOBAL_BUFFER_LENGTH:
			seb_buffer_length = v;
			break;
		case SEB_GLOBAL_ENCDAT_LENGTH:
			seb_encdat_length = v;
			break;
	}
}

/* Notice: little endian */
static void _write_magic(FILE *fp, byte enctype)
{
	uint32 fpoff = ftell(fp);
	seb_magic[SEB_ENCODE_TYPE] = enctype;
	fseek(fp, SEB_MAGIC_BEGIN, SEEK_SET);
	fwrite(seb_magic, sizeof(seb_magic), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
}

static void _write_length(FILE *fp, uint32 off)
{
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_LENGTH_BEGIN, SEEK_SET);
	fwrite(&off, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
}

static void _write_buflen(FILE *fp, uint32 buflen)
{
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_BUFLEN_BEGIN, SEEK_SET);
	fwrite(&buflen, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
}

static void _write_const_blklen(FILE *fp, uint32 blklen)
{
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_C_BLKLEN_BEGIN, SEEK_SET);
	fwrite(&blklen, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
}

static byte _read_encode_type(FILE *fp)
{
	byte enctype;
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_ENCODE_TYPE, SEEK_SET);
	fread(&enctype, sizeof(byte), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
	return enctype;
}

static uint32 _read_length(FILE *fp)
{
	uint32 length;
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_LENGTH_BEGIN, SEEK_SET);
	fread(&length, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
	return length;
}

static uint32 _read_buflen(FILE *fp)
{
	uint32 buflen;
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_BUFLEN_BEGIN, SEEK_SET);
	fread(&buflen, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
	return buflen;
}

static uint32 _read_const_blklen(FILE *fp)
{
	uint32 blklen;
	uint32 fpoff = ftell(fp);
	fseek(fp, SEB_C_BLKLEN_BEGIN, SEEK_SET);
	fread(&blklen, sizeof(uint32), 1, fp);
	fseek(fp, fpoff, SEEK_SET);
	return blklen;
}

static sebFILE *_sebfopen_read_only(sebFILE *sebfp, const char *pathname)
{
	sebfp->fp = fopen(pathname, "rb");
	if (!sebfp->fp) {
		free(sebfp);
		return NULL;
	}
	sebfp->mode = sebFILE_MODE_RO;
	sebfp->length = _read_length(sebfp->fp);
	sebfp->buflen = _read_buflen(sebfp->fp);
	if (_seb_set_encoder(sebfp, _read_encode_type(sebfp->fp))) {
		free(sebfp);
		return NULL;
	}
	sebfp->cursor = sebfp->buflen;
	sebfp->const_blklen = _read_const_blklen(sebfp->fp);
	if (sebfp->const_blklen) {
		sebfp->blklen = sebfp->const_blklen;
		sebfp->encdat = (byte*)malloc(sebfp->blklen);
		if (!sebfp->encdat) {
			fclose(sebfp->fp);
			free(sebfp);
			return NULL;
		}
	} else { /* Initializing by default */
		sebfp->blklen = sebfp->buflen * 2;
		sebfp->encdat = (byte*)malloc(sebfp->blklen);
		if (!sebfp->encdat) {
			fclose(sebfp->fp);
			free(sebfp);
			return NULL;
		}
	}
	sebfp->buffer = (byte*)malloc(sebfp->buflen);
	if (!sebfp->buffer) {
		free(sebfp->encdat);
		fclose(sebfp->fp);
		free(sebfp);
		return NULL;
	}
	fseek(sebfp->fp, SEB_BLOCK_BEGIN, SEEK_SET);
	return sebfp;
}

static sebFILE *_sebfopen_write_only(sebFILE *sebfp, const char *pathname)
{
	sebfp->fp = fopen(pathname, "wb");
	if (!sebfp->fp) {
		free(sebfp);
		return NULL;
	}
	sebfp->mode = sebFILE_MODE_WO;
	sebfp->buflen = seb_buffer_length;
	sebfp->blklen = seb_encdat_length;
	sebfp->encode_ctrl = seb_encode_ctrl;
	sebfp->decode_ctrl = seb_decode_ctrl;
	sebfp->buffer = (byte*)malloc(sebfp->buflen);
	if (!sebfp->buffer) {
		fclose(sebfp->fp);
		free(sebfp);
		return NULL;
	}
	sebfp->encdat = (byte*)malloc(sebfp->blklen);
	if (!sebfp->encdat) {
		fclose(sebfp->fp);
		free(sebfp->buffer);
		free(sebfp);
		return NULL;
	}
	_write_magic(sebfp->fp, seb_magic_encode_type);
	_write_buflen(sebfp->fp, sebfp->buflen);
	sebfp->const_blklen = seb_const_blklen;
	_write_const_blklen(sebfp->fp, sebfp->const_blklen);
	sebfp->encode = seb_encoder;
	sebfp->decode = seb_decoder;
	fseek(sebfp->fp, SEB_BLOCK_BEGIN, SEEK_SET);
	return sebfp;
}

sebFILE *sebfopen(const char *pathname, const char *mode)
{
	sebFILE *sebfp = (sebFILE*)malloc(sizeof(sebFILE));
	if (!sebfp)
		return NULL;
	memset(sebfp, 0, sizeof(sebFILE));
	switch (*mode) { /* Support Read/Write Only */
		case 'r':
			sebfp = _sebfopen_read_only(sebfp, pathname);
			break;
		case 'w':
			sebfp = _sebfopen_write_only(sebfp, pathname);
			break;
		default:
			free(sebfp);
			return NULL;
	}
	return sebfp;
}

void sebclose(sebFILE *sebfp)
{
	fclose(sebfp->fp);
	free(sebfp->buffer);
	free(sebfp);
}

uint32 sebflush(sebFILE *sebfp)
{
	uint32 l;
	if (sebfp->mode != sebFILE_MODE_WO)
		return 0;
	l = sebfp->encode(sebfp->encode_ctrl,
			sebfp->buffer, sebfp->cursor,
		sebfp->encdat, sebfp->buflen);
	sebfp->cursor = 0;
	if (!sebfp->const_blklen)
		fwrite(&l, sizeof(uint32), 1, sebfp->fp);
	_write_length(sebfp->fp, sebfp->length);
	fwrite(sebfp->encdat, l, 1, sebfp->fp);
	return fflush(sebfp->fp);
}

uint32 sebfwrite(void *ptr, uint32 size, uint32 nmemb, sebFILE *sebfp)
{
	uint32 cnt, sum, avail;
	byte *pos = (byte*)ptr;
	sum = size * nmemb;
	while (sum) {
		avail = sebfp->buflen - sebfp->cursor;
		if (!avail)
			sebflush(sebfp);
		cnt = sum > avail ? avail : sum;
		memcpy(sebfp->buffer + sebfp->cursor, pos, cnt);
		pos += cnt;
		sum -= cnt;
		sebfp->length += cnt;
		sebfp->cursor += cnt;
	}
	return nmemb;
}

/* Notice: little endian */
static void _seb_read_block(sebFILE *sebfp)
{
	uint32 l;
	byte *p;
	if (!sebfp->const_blklen) {
		fread(&l, sizeof(uint32), 1, sebfp->fp);
		if (l > sebfp->blklen) {
			sebfp->blklen = l;
			p = (byte*)realloc(sebfp->encdat, sebfp->blklen);
			sebfp->encdat = p ? p : sebfp->encdat;
		}
	} else {
		l = sebfp->const_blklen;
	}
	fread(sebfp->encdat, l, 1, sebfp->fp);
	sebfp->decode(sebfp->decode_ctrl,
			sebfp->encdat, sebfp->blklen,
		sebfp->buffer, sebfp->buflen);
	sebfp->cursor = 0;
}

uint32 sebfread(void *ptr, uint32 size, uint32 nmemb, sebFILE *sebfp)
{
	uint32 cnt, sum, avail;
	byte *pos = (byte*)ptr;
	if (!size)
		return 0;
	sum = size * nmemb;
	avail = sebfp->length - sebfp->offset;
	sum = avail > sum ? sum : avail;
	while (sum) {
		avail = sebfp->buflen - sebfp->cursor;
		if (!avail)
			_seb_read_block(sebfp);
		cnt = sum > avail ? avail : sum;
		memcpy(pos, sebfp->buffer + sebfp->cursor, cnt);
		pos += cnt;
		sum -= cnt;
		sebfp->cursor += cnt;
		sebfp->offset += cnt;
	}
	return (pos - ((byte*)ptr)) / size;
}

void sebfclose(sebFILE *sebfp)
{
	if (sebfp->mode == sebFILE_MODE_WO)
		sebflush(sebfp);
	free(sebfp->buffer);
	free(sebfp->encdat);
	fclose(sebfp->fp);
	free(sebfp);
}

#ifdef SEB_TYPE_BYTE
	#undef byte
#endif
#ifdef SEB_TYPE_UINT32
	#undef uint32
#endif
