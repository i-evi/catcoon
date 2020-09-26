/*
 * #define CONFIG_STD_C89
 * #define RUN_TIME_ENDIAN_CHECK
 * #define BYTE_ORDER_BIG_ENDIAN
 * #define USE3RD_STB_IMAGE
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_image.h"

#ifndef byte
	#define byte unsigned char
#endif

#define ASSERT_ON_BUILD(condition)\
	((void)sizeof(char[1 - 2*!!(condition)]))
/*=========== compile-time test ===========*/
void ____utim_datatype_test____()
{
	/*
	 * Ref: utim_write(), utim_write_ctrl()
	 * At simple bmp operations...
	 */
	ASSERT_ON_BUILD(sizeof(utim_int32_t) - 4);
	/*
	 * Psf2 font file support
	 */
	ASSERT_ON_BUILD(sizeof(struct psf2_header) - 32);
}
#ifdef RUN_TIME_ENDIAN_CHECK
	static int _is_little_endian()
	{
		unsigned int i = 1;
		unsigned char *c = (unsigned char*)&i;
		if (*c)
			return 1;
		return 0;
	}

	static void _memrev(void *i, unsigned int s)
	{
		unsigned int j = s >> 1;
		unsigned char *c = (unsigned char*)i;
		for (s--; j > 0; j--) {
			*c ^= *(c + s);
			*(c + s) ^= *c;
			*c ^= *(c + s);
			c++, s -= 2;
		}
	}
#endif

#ifndef CONFIG_STD_C89
	inline
#endif
static utim_int32_t _index_calc(int xsize, int ysize, int x, int y)
{
	utim_int32_t index;
	if (x >= xsize || y >= ysize || x < 0 || y < 0)
		return -1;
	index = xsize * y + x;
	return index;
}

static const char *file_extension(const char *filename)
{
	const char *p = filename + strlen(filename);
	while (filename != p--)
		if (*p == '.')
			break;
	return filename < p ? p + 1 : NULL;
}

#ifndef USE3RD_STB_IMAGE
/* 8, 24, 32 bit NON-COMPRESSED BMP supported */
static byte *_bmp_read(const char *filename,
	int *xsize, int *ysize, int *channels)
{
	byte header[54];
	byte color_dp;
	utim_uint32_t pxl_begin, *x;
	utim_uint32_t r, tmp, windows_byte_cnt = 0;
	byte *image_tail, *pixels;
	FILE *fp;
	if (!(fp = fopen(filename, "rb"))) 
		return NULL;
	fread(header, sizeof(byte), 54, fp);
	x = (utim_uint32_t*)(header + 18);
	*xsize = x[0];
	*ysize = x[1];
	pxl_begin = *(utim_uint32_t*)(header + 10);
#if   defined(RUN_TIME_ENDIAN_CHECK)
	if (!_is_little_endian) {
		_memrev(xsize, sizeof(utim_uint32_t));
		_memrev(ysize, sizeof(utim_uint32_t));
		_memrev(&pxl_begin, sizeof(utim_uint32_t));
	}
#elif defined(BYTE_ORDER_BIG_ENDIAN)
	_memrev(xsize, sizeof(utim_uint32_t));
	_memrev(ysize, sizeof(utim_uint32_t));
	_memrev(&pxl_begin, sizeof(utim_uint32_t));
#endif
	color_dp = *(byte*)(header + 28);
	switch (color_dp) {
	case 8:
	case 24:
	case 32:
		*channels = color_dp >> 3;
		break;
	default:
		fclose(fp);
		return NULL;
	}
	if (!(((*xsize) * (*channels)) % 4)) {
		windows_byte_cnt = 0;
	} else {
		tmp = (*xsize * (*channels)) / 4;
		windows_byte_cnt = ((tmp + 1) * 4) - *xsize * (*channels);
	}
	pixels = (byte*)malloc((*xsize) * (*ysize) * (*channels));
	if (!pixels) {
		fclose(fp);
		return NULL;
	}
	image_tail = pixels + ((*ysize) * (*xsize) * (*channels));
	fseek(fp, pxl_begin, SEEK_SET);
	for (r = 0; r < (*ysize); ++r) {
		image_tail -= ((*xsize) * (*channels));
		fread(image_tail , sizeof(byte), (*xsize) * (*channels), fp);
		fseek(fp, windows_byte_cnt, SEEK_CUR);
	}
	fclose(fp);
	return pixels;
}

static void _write_palette_gray(FILE *fp)
{
	int i;
	byte c[4] = {0};
	for (i = 0; i < 256; ++i) {
		memset(c, i, 3);
		fwrite(c, sizeof(4), 1, fp);
	}
}

static int _bmp_write(const char *filename,
	byte *pixels, int xsize, int ysize, int channels)
{
	byte header[54] = {
		0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
		54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0
	};
	FILE *fp;
	byte *image_tail;
	utim_uint32_t windows_byte_cnt = 0;
	utim_uint32_t windows_byte_set = 0;
	utim_uint32_t tmp, r = (xsize * channels) % 4;
	utim_uint32_t pxl_begin, *x;
	switch (channels) {
	case 1:
		pxl_begin = 54 + 1024;
		break;
	case 3:
	case 4:
		pxl_begin = 54;
		break;
	default:
		return -1;
	}
	if (!r) {
		windows_byte_cnt = 0;
	} else {
		tmp = (xsize * channels ) / 4;
		windows_byte_cnt = ((tmp + 1) * 4) - xsize * channels;
	}
	x = (utim_uint32_t*)(header + 2);
	x[0] = (xsize * channels + windows_byte_cnt) * ysize + 54;
	x = (utim_uint32_t*)(header + 18);
	x[0] = xsize;
	x[1] = ysize;
	*(utim_uint32_t*)(header + 10) = pxl_begin;
#if   defined(RUN_TIME_ENDIAN_CHECK)
	if (!_is_little_endian) {
		_memrev(x + 0, sizeof(utim_uint32_t));
		_memrev(x + 1, sizeof(utim_uint32_t));
		_memrev((utim_uint32_t*)(header + 10),
			sizeof(utim_uint32_t));  /* pxl_begin */
	}
#elif defined(BYTE_ORDER_BIG_ENDIAN)
	_memrev(x + 0, sizeof(utim_uint32_t));
	_memrev(x + 1, sizeof(utim_uint32_t));
	_memrev((utim_uint32_t*)(header + 10),
		sizeof(utim_uint32_t));  /* pxl_begin */
#endif
	*(byte*)(header + 28) = channels << 3;
	if (!(fp = fopen(filename, "wb+")))
		return -1;
	fwrite(header, sizeof(byte), 54, fp);
	image_tail = pixels + (ysize * xsize * channels);
	if (channels == 1) { /* 8bit palette, support gray image only */
		_write_palette_gray(fp);
	}
	fseek(fp, pxl_begin, SEEK_SET);
	for (r = 0; r < ysize; ++r) {
		image_tail -= (xsize * channels);
		fwrite(image_tail, sizeof(byte) * channels, xsize, fp);
		fwrite(&windows_byte_set, 1, windows_byte_cnt, fp);
	}
	fclose(fp);
	return 0;
}
#endif /* not defined USE3RD_STB_IMAGE */

/*
 * Use stb image. stb_image.h, stb_image_write.h are included. 
 */
#ifdef USE3RD_STB_IMAGE
	#define STB_IMAGE_IMPLEMENTATION
	#include "stb_image.h"
	#define STB_IMAGE_WRITE_IMPLEMENTATION
	#include "stb_image_write.h"
	#define STB_IMAGE_RESIZE_IMPLEMENTATION
	#include "stb_image_resize.h"
	int jpg_default_quality = 80;
	int png_default_compress = 8;
#endif

UTIM_IMG *utim_read(const char*filename)
{
#ifndef USE3RD_STB_IMAGE
	const char *t;
#endif
	UTIM_IMG *img = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!img)
		return NULL;
#ifdef USE3RD_STB_IMAGE
	img->pixels = stbi_load(filename, &img->xsize, &img->ysize,
		&img->channels, 0);
	if (!img->pixels) {
		free(img);
		return NULL;
	}
#else /* Support 24bit bmp image only */
	t = file_extension(filename);
	if (strcmp(t, "bmp")) {
		free(img);
		return NULL;
	}
	img->pixels = _bmp_read(filename,
		&img->xsize, &img->ysize, &img->channels);
	if (!img->pixels) {
		free(img);
		return NULL;
	}
#endif
	return img;
}

int utim_write(const char *filename, UTIM_IMG *img)
{
	return utim_write_ctrl(filename, img, 0, 0);
}

int utim_write_ctrl(const char *filename,
	UTIM_IMG *img, int comp, int quality)
{
	const char *t;
	char tbuf[8];
	char *p = tbuf;
#ifdef USE3RD_STB_IMAGE
	int state;
#endif
	t = file_extension(filename);
	strcpy(tbuf, t);
	while ((*p = tolower(*p))) ++p;
	if (strlen(t) < 3)
		return -1; /* Not supported file format */
#if   defined(RUN_TIME_ENDIAN_CHECK)
	if (!_is_little_endian())
		_memrev(tbuf, strlen(tbuf) + 1);
#elif defined(BYTE_ORDER_BIG_ENDIAN)
	_memrev(tbuf, strlen(tbuf) + 1);
#endif
#ifdef USE3RD_STB_IMAGE
	switch (*(utim_int32_t*)tbuf) {
	case UTIM_BMP:
		if (!comp)
			comp = img->channels;
		state = stbi_write_bmp(filename, img->xsize,
			img->ysize, comp, img->pixels);
		if (state != 1)
			return state;
		return 0;
	case UTIM_PNG:
		if (!comp)
			comp = img->channels;
		quality = quality ? quality : png_default_compress;
		stbi_write_png_compression_level = quality;
		state = stbi_write_png(filename, img->xsize,
			img->ysize, img->channels, img->pixels,
			img->xsize * img->channels);
		if (state != 1)
			return state;
		return 0;
	case UTIM_JPG:
		if (!comp)
			comp = img->channels;
		quality = quality ? quality : jpg_default_quality;
		state = stbi_write_jpg(filename, img->xsize,
			img->ysize, comp, img->pixels, quality);
		if (state != 1)
			return state;
		return 0;
	case UTIM_TGA:
		if (!comp)
			comp = img->channels;
		state = stbi_write_tga(filename, img->xsize,
			img->ysize, comp, img->pixels);
		if (state != 1)
			return state;
		return 0;
	default:
		return -1; /* Not supported file format */
	}
#else /* Support 8, 24, 32 bit bmp image only */
	if (strcmp(tbuf, "bmp"))
		return -1;  /* Not supported file format */
	switch (img->channels) {
	case 1:
	case 3:
	case 4:
		break;
	default:
		return -1;
	}
	return _bmp_write(filename,
		img->pixels, img->xsize, img->ysize, img->channels);
#endif
}

void utim_free_image(UTIM_IMG *img)
{
	if (!img)
		return;
	if (img->pixels)
		free(img->pixels);
	free(img);
}

UTIM_IMG *utim_clone(UTIM_IMG *img)
{
	UTIM_IMG *clone;
	int size = img->xsize * img->ysize * img->channels;
	if (!size)
		return NULL;
	clone = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!clone)
		return NULL;
	memcpy(clone, img, sizeof(UTIM_IMG));
	clone->pixels = (byte*)malloc(size);
	if (!clone->pixels) {
		free(clone);
		return NULL;
	}
	memcpy(clone->pixels, img->pixels, size);
	return clone;
}

static byte *_image_resize_nearest(UTIM_IMG *img, int x, int y)
{
	byte *p;
	int i, j, ch_size = x * y;
	p = (byte*)malloc(ch_size * img->channels);
	if (!p)
		return NULL;
	for (j = 0; j < y; ++j) {
		for (i = 0; i < x; ++i) {
			memcpy(p + _index_calc(x, y, i, j) * img->channels,
				img->pixels + _index_calc
						(img->xsize, img->ysize,
				i * img->xsize / x, j * img->ysize / y) * 
				img->channels,img->channels);
		}
	}
	return p;
}

#ifdef USE3RD_STB_IMAGE
static byte *_image_resize_linear(UTIM_IMG *img, int x, int y)
{
	byte *p;
	int ch_size = x * y;
	p = (byte*)malloc(ch_size * img->channels);
	if (!p)
		return NULL;
	stbir_resize_uint8(img->pixels,
		img->xsize, img->ysize, 0, p, x, y, 0, img->channels);
	return p;
}
#endif

UTIM_IMG *utim_resize(UTIM_IMG *img, int x, int y, int mode)
{
	UTIM_IMG *resize;
	resize = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!resize)
		return NULL;
	switch (mode) {
	case UTIM_RESIZE_NEAREST:
		resize->pixels = _image_resize_nearest(img, x, y);
		if (!resize->pixels) {
			free(resize);
			return NULL;
		}
		break;
#ifdef USE3RD_STB_IMAGE
	case UTIM_RESIZE_LINEAR:
		resize->pixels = _image_resize_linear(img, x, y);
		if (!resize->pixels) {
			free(resize);
			return NULL;
		}
		break;
#endif
	default:
		return NULL;
	}
	resize->xsize = x;
	resize->ysize = y;
	resize->channels = img->channels;
	return resize;
}

int utim_swap_chl(UTIM_IMG *img, int a, int b)
{
	byte c;
	int i, ch_size;
	if (a == b || a >= img->channels || b >= img->channels)
		return UTIM_ERR_BAD_ARG;
	ch_size = img->xsize * img->ysize;
	for (i = 0; i < ch_size; ++i) {
		c = *(img->pixels + img->channels * i + a);
		*(img->pixels + img->channels * i + a) =
			*(img->pixels + img->channels * i + b);
		*(img->pixels + img->channels * i + b) = c;
	}
	return 0;
}

int utim_img2rgb(UTIM_IMG *img)
{
	switch (img->channels) {
	case 1:
		return utim_gray2rgb(img);
	case 3:
		return 0;
	case 4:
		return utim_rgba2rgb(img);
	default:
		return UTIM_ERR_BAD_ARG;
	}
}

int utim_img2gray(UTIM_IMG *img)
{
	switch (img->channels) {
	case 1:
		return 0;
	case 3:
		return utim_rgb2gray(img);
	case 4:
		return utim_rgba2gray(img);
	default:
		return UTIM_ERR_BAD_ARG;
	}
}

int utim_img2rgba(UTIM_IMG *img)
{
	switch (img->channels) {
	case 1:
		return utim_gray2rgba(img);
	case 3:
		return utim_rgb2rgba(img);
	case 4:
		return 0;
	default:
		return UTIM_ERR_BAD_ARG;
	}
}

int utim_rgba2rgb(UTIM_IMG *rgba)
{
	byte *new_pixels, *p;
	int i, ch_size = rgba->xsize * rgba->ysize;
	if (!ch_size || rgba->channels != 4)
		return UTIM_ERR_BAD_ARG;
	new_pixels = (byte*)malloc(ch_size * 3);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	p = new_pixels;
	for (i = 0; i < ch_size; ++i) {
		*p++ = *(rgba->pixels + i * 4);
		*p++ = *(rgba->pixels + i * 4 + 1);
		*p++ = *(rgba->pixels + i * 4 + 2);
	}
	free(rgba->pixels);
	rgba->pixels = new_pixels;
	rgba->channels = 3;
	return 0;
}

UTIM_IMG *utim_rgb_by_rgba(UTIM_IMG *rgba)
{
	UTIM_IMG *rgb;
	byte *p;
	int i, ch_size = rgba->xsize * rgba->ysize;
	if (!ch_size || rgba->channels != 4)
		return NULL;
	rgb = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!rgb)
		return NULL;
	rgb->pixels = (byte*)malloc(ch_size * 3);
	if (!rgb->pixels) {
		free(rgb);
		return NULL;
	}
	p = rgb->pixels;
	for (i = 0; i < ch_size; ++i) {
		*p++ = *(rgba->pixels + i * 4);
		*p++ = *(rgba->pixels + i * 4 + 1);
		*p++ = *(rgba->pixels + i * 4 + 2);
	}
	rgb->xsize = rgba->xsize;
	rgb->ysize = rgba->ysize;
	rgb->channels = 3;
	return rgb;
}

int utim_rgb2bgr(UTIM_IMG *rgb)
{
	int n;
	byte *p1, *p2;
	if (rgb->channels != 3)
		return UTIM_ERR_BAD_ARG;
	n = rgb->xsize * rgb->ysize;
	p1 = rgb->pixels;
	p2 = rgb->pixels + 2;
	while (n--) {
		*(p1) ^= *(p2);
		*(p2) ^= *(p1);
		*(p1) ^= *(p2);
		p1 += 3;
		p2 += 3;
	}
	return 0;
}

UTIM_IMG *utim_bgr_by_rgb(UTIM_IMG *rgb)
{
	UTIM_IMG *bgr = utim_clone(rgb);
	if (!bgr)
		return NULL;
	if (utim_rgb2bgr(bgr)) {
		utim_free_image(bgr);
		return NULL;
	}
	return bgr;
}

int utim_rgb2gray(UTIM_IMG *rgb)
{
	byte *new_pixels, *p;
	int i, sum, ch_size;
	if (rgb->channels != 3)
		return UTIM_ERR_BAD_ARG;
	ch_size = rgb->xsize * rgb->ysize;
	new_pixels = (byte*)malloc(ch_size);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	p = new_pixels;
	for (i = 0; i < ch_size; ++i) {
		sum = 0;
		sum += *(rgb->pixels + i * 3);
		sum += *(rgb->pixels + i * 3 + 1);
		sum += *(rgb->pixels + i * 3 + 2);
		*p++ = sum / 3;
	}
	free(rgb->pixels);
	rgb->pixels = new_pixels;
	rgb->channels = 1;
	return 0;
}

int utim_rgba2gray(UTIM_IMG *rgba)
{
	byte *new_pixels, *p;
	int i, sum, ch_size;
	if (rgba->channels != 4)
		return UTIM_ERR_BAD_ARG;
	ch_size = rgba->xsize * rgba->ysize;
	new_pixels = (byte*)malloc(ch_size);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	p = new_pixels;
	for (i = 0; i < ch_size; ++i) {
		sum = 0;
		sum += *(rgba->pixels + i * 4);
		sum += *(rgba->pixels + i * 4 + 1);
		sum += *(rgba->pixels + i * 4 + 2);
		*p++ = sum / 3;
	}
	free(rgba->pixels);
	rgba->pixels = new_pixels;
	rgba->channels = 1;
	return 0;
}

UTIM_IMG *utim_gray_by_rgb(UTIM_IMG *rgb)
{
	byte *p;
	UTIM_IMG *gray;
	int i, sum, ch_size;
	if (rgb->channels != 3)
		return NULL;
	gray = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!gray)
		return NULL;
	ch_size = rgb->xsize * rgb->ysize;
	gray->pixels = (byte*)malloc(ch_size);
	if (!gray->pixels) {
		free(gray);
		return NULL;
	}
	p = gray->pixels;
	for (i = 0; i < ch_size; ++i) {
		sum = 0;
		sum += *(rgb->pixels + i * 3);
		sum += *(rgb->pixels + i * 3 + 1);
		sum += *(rgb->pixels + i * 3 + 2);
		*p++ = sum / 3;
	}
	gray->xsize = rgb->xsize;
	gray->ysize = rgb->ysize;
	gray->channels = 1;
	return gray;
}

int utim_gray2rgb(UTIM_IMG *gray)
{
	byte *new_pixels;
	int i, off, ch_size;
	if (gray->channels != 1)
		return UTIM_ERR_BAD_ARG;
	ch_size = gray->xsize * gray->ysize;
	new_pixels = (byte*)malloc(ch_size * 3);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	for (i = 0; i < ch_size; ++i) {
		off = i * 3;
		*(new_pixels + off) = *(gray->pixels + i);
		*(new_pixels + off + 1) = *(gray->pixels + i);
		*(new_pixels + off + 2) = *(gray->pixels + i);
	}
	free(gray->pixels);
	gray->pixels = new_pixels;
	gray->channels = 3;
	return 0;
}

int utim_gray2rgba(UTIM_IMG *gray)
{
	byte *new_pixels;
	int i, off, ch_size;
	if (gray->channels != 1)
		return UTIM_ERR_BAD_ARG;
	ch_size = gray->xsize * gray->ysize;
	new_pixels = (byte*)malloc(ch_size * 4);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	for (i = 0; i < ch_size; ++i) {
		off = i * 4;
		*(new_pixels + off) = *(gray->pixels + i);
		*(new_pixels + off + 1) = *(gray->pixels + i);
		*(new_pixels + off + 2) = *(gray->pixels + i);
		*(new_pixels + off + 3) = 255;
	}
	free(gray->pixels);
	gray->pixels = new_pixels;
	gray->channels = 4;
	return 0;
}

int utim_rgb2rgba(UTIM_IMG *rgb)
{
	byte *new_pixels;
	int i, off_rgba, off_rgb, ch_size;
	if (rgb->channels != 3)
		return UTIM_ERR_BAD_ARG;
	ch_size = rgb->xsize * rgb->ysize;
	new_pixels = (byte*)malloc(ch_size * 4);
	if (!new_pixels)
		return UTIM_ERR_ALLOC;
	for (i = 0; i < ch_size; ++i) {
		off_rgb  = i * 3;
		off_rgba = i * 4;
		*(new_pixels + off_rgba) = *(rgb->pixels + off_rgb);
		*(new_pixels + off_rgba + 1) = *(rgb->pixels + off_rgb + 1);
		*(new_pixels + off_rgba + 2) = *(rgb->pixels + off_rgb + 2);
		*(new_pixels + off_rgba + 3) = 255;
	}
	free(rgb->pixels);
	rgb->pixels = new_pixels;
	rgb->channels = 4;
	return 0;
}

UTIM_IMG *utim_rgb_by_gray(UTIM_IMG *gray)
{
	UTIM_IMG *rgb;
	int i, off, ch_size;
	if (gray->channels != 1)
		return NULL;
	rgb = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!rgb)
		return NULL;
	ch_size = gray->xsize * gray->ysize;
	rgb->pixels = (byte*)malloc(ch_size * 3);
	if (!rgb->pixels) {
		free(rgb);
		return NULL;
	}
	for (i = 0; i < ch_size; ++i) {
		off = i * 3;
		*(rgb->pixels + off + 0) = *(gray->pixels + i);
		*(rgb->pixels + off + 1) = *(gray->pixels + i);
		*(rgb->pixels + off + 2) = *(gray->pixels + i);
	}
	rgb->xsize = gray->xsize;
	rgb->ysize = gray->ysize;
	rgb->channels = 3;
	return rgb;
}

UTIM_IMG *utim_create(int x, int y, int nch, int c)
{
	UTIM_IMG *img;
	int size = x * y * nch;
	if (!size || nch > UTIM_MAX_CHANNELS)
		return NULL;
	img = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!img)
		return NULL;
	img->pixels = (byte*)malloc(size);
	if (!img->pixels) {
		free(img);
		return NULL;
	}
	memset(img->pixels, c, size);
	img->xsize = x;
	img->ysize = y;
	img->channels = nch;
	return img;
}

UTIM_IMG *utim_stack(UTIM_IMG **chx, int nch)
{
	byte *p;
	UTIM_IMG *img;
	int i, j, ch_size = (*chx)->xsize * (*chx)->ysize;
	if (nch > UTIM_MAX_CHANNELS || nch <= 0)
		return NULL;
	for (i = 1; i < nch; ++i) { /* size check */
		if (chx[0]->xsize != chx[i]->xsize ||
			chx[0]->ysize != chx[i]->ysize)
			return NULL;
	}
	img = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!img)
		return NULL;
	img->pixels = (byte*)malloc(ch_size * nch);
	if (!img->pixels) {
		free(img);
		return NULL;
	}
	p = img->pixels;
	for (i = 0; i < ch_size; ++i) {
		for (j = 0; j < nch; ++j)
			*(p + i * nch + j) = chx[j]->pixels[i];
	}
	img->xsize = chx[0]->xsize;
	img->ysize = chx[0]->ysize;
	img->channels = nch;
	return img;
}

UTIM_IMG *utim_pick_chl(UTIM_IMG *img, int ich)
{
	byte *p;
	UTIM_IMG *ch;
	int i, ch_size;
	if (!(ich < img->channels))
		return NULL;
	ch = (UTIM_IMG*)malloc(sizeof(UTIM_IMG));
	if (!ch)
		return NULL;
	ch_size = img->xsize * img->ysize;
	ch->pixels = (byte*)malloc(ch_size);
	if (!ch->pixels) {
		free(ch);
		return NULL;
	}
	p = ch->pixels;
	for (i = 0; i < ch_size; ++i)
		*p++ = *(img->pixels + i * img->channels + ich);
	ch->xsize = img->xsize;
	ch->ysize = img->ysize;
	ch->channels = 1;
	return ch;
}

int utim_set_opacity(UTIM_IMG *img, int opacity)
{
	byte *c;
	int i, sum, ch_size;
	switch (img->channels) {
	case 1:
		if (!utim_gray2rgba(img))
			return UTIM_ERR_BAD_ARG;
	case 3:
		if (!utim_rgb2rgba(img))
			return UTIM_ERR_BAD_ARG;
	case 4: /* RGBA */
		break;
	default:
		return UTIM_ERR_BAD_ARG;
	}
	ch_size = img->xsize * img->ysize;
	c = img->pixels;
	for (i = 0; i < ch_size; ++i) {
		sum = 0;
		sum += *c++;
		sum += *c++;
		sum += *c++;
		if (sum)
			*c = opacity;
		c++;
	}
	return 0;
}

UTIM_IMG *utim_set_chl(UTIM_IMG *img, int ich, int color)
{
	byte *c;
	int i, ch_size;
	if (ich >= img->channels)
		return NULL;
	ch_size = img->xsize * img->ysize;
	c = img->pixels;
	for (i = 0; i < ch_size; ++i) {
		*(c + ich) = color;
		c += img->channels;
	}
	return img;
}

void utim_negative_color(UTIM_IMG *img)
{
	int i, j, ch_size = img->xsize * img->ysize;
	for (i = 0; i < ch_size; ++i) {
		for (j = 0; j < img->channels && j < UTIM_COLOR_A; ++j) {
			img->pixels[i * img->channels + j] =
				255 - img->pixels[i * img->channels + j];
		}
	}
}

/* _compose_pixel: used for implementing utim_superpose only */
#ifndef CONFIG_STD_C89
	inline
#endif
static void _compose_pixel(UTIM_IMG *bg,
	UTIM_IMG *im, int i, int j, UTIM_POINT p)
{
	UTIM_COLOR c_bg, c_im;
	int k, index_bg, index_im;
	byte *rb = &c_bg[UTIM_COLOR_R];
	byte *gb = &c_bg[UTIM_COLOR_G];
	byte *bb = &c_bg[UTIM_COLOR_B];
	byte *ab = &c_bg[UTIM_COLOR_A];
	byte *ri = &c_im[UTIM_COLOR_R];
	byte *gi = &c_im[UTIM_COLOR_G];
	byte *bi = &c_im[UTIM_COLOR_B];
	byte *ai = &c_im[UTIM_COLOR_A];
	index_bg = _index_calc(bg->xsize, bg->ysize, i, j);
	index_im = _index_calc(im->xsize, im->ysize,
			i - p[UTIM_POINT_X], j - p[UTIM_POINT_Y]);
	if (index_im <0)
		return;
	if (bg->channels == 4) { /* RGBA */
		for (k = 0; k < bg->channels; ++k)
			c_bg[k] = bg->pixels[index_bg * bg->channels + k];
		memset(c_im, UTIM_BACK_COLOR, sizeof(c_im));
		for (k = 0; k < im->channels; ++k)
			c_im[k] = im->pixels[index_im * bg->channels + k];
                *rb = ((255 - *ai) * (*rb) + (*ai) * (*ri)) / 255;
                *gb = ((255 - *ai) * (*gb) + (*ai) * (*gi)) / 255;
                *bb = ((255 - *ai) * (*bb) + (*ai) * (*bi)) / 255;
                *ab = *ab + ((255 - *ab) * (*ai) / 255);
		for (k = 0; k < bg->channels; ++k)
			bg->pixels[index_bg * bg->channels + k] = c_bg[k];

	} else { /* Direct copy */
		for (k = 0; k < bg->channels; ++k)
			bg->pixels[index_bg + k] = im->pixels[index_im + k];
	}
}

int utim_superpose(UTIM_IMG *bg,
				UTIM_IMG *img, UTIM_POINT p)
{
	int i, j;
	if (p[UTIM_POINT_X] >= bg->xsize ||
		p[UTIM_POINT_Y] >= bg->ysize)
		return 0;
	for (j = p[UTIM_POINT_Y]; j < bg->ysize; ++j) {
		for (i = p[UTIM_POINT_X]; i < bg->xsize; ++i)
			_compose_pixel(bg, img, i, j, p);
	}
	return 0;
}

/*
 * Simple drawing functions
 */

static int (*utim_draw_point_fn)
	(UTIM_IMG*, UTIM_POINT, UTIM_COLOR) = utim_draw_point;

int utim_set_pixel(UTIM_IMG *img, UTIM_POINT p, UTIM_COLOR c)
{
	int i, index = _index_calc(img->xsize,
			img->ysize, p[UTIM_POINT_X], p[UTIM_POINT_Y]);
	if (index < 0)
		return -1;
	for (i = 0; i < img->channels; ++i)
		img->pixels[index * img->channels + i] = c[i];
	return 0;
}

int utim_draw_point(UTIM_IMG *img, UTIM_POINT p, UTIM_COLOR c)
{
	UTIM_COLOR c_img;
	byte *ri = &c[UTIM_COLOR_R];
	byte *gi = &c[UTIM_COLOR_G];
	byte *bi = &c[UTIM_COLOR_B];
	byte *ai = &c[UTIM_COLOR_A];
	byte *rb = &c_img[UTIM_COLOR_R];
	byte *gb = &c_img[UTIM_COLOR_G];
	byte *bb = &c_img[UTIM_COLOR_B];
	byte *ab = &c_img[UTIM_COLOR_A];
	int k, index = _index_calc(img->xsize,
			img->ysize, p[UTIM_POINT_X], p[UTIM_POINT_Y]);
	if (index < 0)
		return -1;
	if (img->channels == 4) { /* RGBA */
		for (k = 0; k < img->channels; ++k)
			c_img[k] = img->pixels[index * img->channels + k];
                *rb = ((255 - *ai) * (*rb) + (*ai) * (*ri)) / 255;
                *gb = ((255 - *ai) * (*gb) + (*ai) * (*gi)) / 255;
                *bb = ((255 - *ai) * (*bb) + (*ai) * (*bi)) / 255;
                *ab = *ab + ((255 - *ab) * (*ai) / 255);
		for (k = 0; k < img->channels; ++k)
			img->pixels[index * img->channels + k] = c_img[k];
	} else { /* Direct copy */
		for (k = 0; k < img->channels; ++k)
			img->pixels[index + k] = c[k];
	}
	return 0;
}

void utim_set_draw_point_fn(int (*fn)
	(UTIM_IMG*, UTIM_POINT, UTIM_COLOR))
{
	utim_draw_point_fn = fn;
	return;
}

static void _image_draw_rect_w1(UTIM_IMG *img,
	UTIM_POINT a, int w, int h, UTIM_COLOR c)
{
	int i;
	UTIM_POINT p;
	if (!w || !h)
		return;
	p[UTIM_POINT_X] = a[UTIM_POINT_X];
	p[UTIM_POINT_Y] = a[UTIM_POINT_Y];
	for (i = 0; i < w; ++i) {
		utim_draw_point_fn(img, p, c);
		p[UTIM_POINT_X]++;
	}
	for (i = 0; i < h; ++i) {
		utim_draw_point_fn(img, p, c);
		p[UTIM_POINT_Y]++;
	}
	for (i = 0; i < w; ++i) {
		utim_draw_point_fn(img, p, c);
		p[UTIM_POINT_X]--;
	}
	for (i = 0; i < h; ++i) {
		utim_draw_point_fn(img, p, c);
		p[UTIM_POINT_Y]--;
	}
	return;
}

void utim_draw_rect(UTIM_IMG *img,
	UTIM_POINT a, int w, int h, UTIM_COLOR c, int width)
{
	int i, half;
	UTIM_POINT p;
	if (!w || !h || !width)
		return;
	half = width >> 1;
	w -= width; h -= width;
	p[UTIM_POINT_X] = a[UTIM_POINT_X] + half;
	p[UTIM_POINT_Y] = a[UTIM_POINT_Y] + half;
	for (i = 0; i < width; ++i) {
		_image_draw_rect_w1(img, p, w, h, c);
		w += 2; h += 2;
		p[UTIM_POINT_X]--;
		p[UTIM_POINT_Y]--;
	}
	return;
}

void utim_draw_line(UTIM_IMG *img,
	UTIM_POINT a, UTIM_POINT b, UTIM_COLOR c, int width)
{
	int dx, dy, orientation, xa, xb, ya, yb, t;
	UTIM_POINT p;
	if (!width)
		return;
	width = width >  0 ? width : -width;
	dx = b[UTIM_POINT_X] - a[UTIM_POINT_X];
	dx = dx < 0 ? -dx : dx;
	dy = b[UTIM_POINT_Y] - a[UTIM_POINT_Y];
	dy = dy < 0 ? -dy : dy;
	if ((dx == 0) && (dy == 0)) {
		utim_draw_point_fn(img, a, c);
		return;
	}
	orientation = dx < dy ? 0 : 1;
	xa = a[UTIM_POINT_X]; ya = a[UTIM_POINT_Y];
	xb = b[UTIM_POINT_X]; yb = b[UTIM_POINT_Y];
	if (orientation) {
		if (a[UTIM_POINT_X] > b[UTIM_POINT_X]) {
			t = xa; xa = xb; xb = t;
			t = ya; ya = yb; yb = t;
		}
		for (p[UTIM_POINT_X] = xa;
			p[UTIM_POINT_X] <= xb; ++p[UTIM_POINT_X]) {
			p[UTIM_POINT_Y] =
				ya + (yb - ya) * (p[UTIM_POINT_X] - xa) / dx;
			if (width == 1)
				utim_draw_point_fn(img, p, c);
			else
				utim_draw_filled_circle(img, p, width, c);
		}
	} else {
		if (a[UTIM_POINT_Y] > b[UTIM_POINT_Y]) {
			t = xa; xa = xb; xb = t;
			t = ya; ya = yb; yb = t;
		}
		for (p[UTIM_POINT_Y] = ya;
			p[UTIM_POINT_Y] <= yb; ++p[UTIM_POINT_Y]) {
			p[UTIM_POINT_X] =
				xa + (xb - xa) * (p[UTIM_POINT_Y] - ya) / dy;
			if (width == 1)
				utim_draw_point_fn(img, p, c);
			else
				utim_draw_filled_circle(img, p, width, c);
		}
	}
}

#ifndef CONFIG_STD_C89
	inline
#endif
static void _draw_circle_point(UTIM_IMG *img, UTIM_POINT p,
	UTIM_POINT center, int x, int y, UTIM_COLOR color, int width)
{
	p[UTIM_POINT_X] += x;
	p[UTIM_POINT_Y] += y;
	if (width == 1)
		utim_draw_point_fn(img, p, color);
	else
		utim_draw_filled_circle(img, p, width, color);
	p[UTIM_POINT_X] = center[UTIM_POINT_X];
	p[UTIM_POINT_Y] = center[UTIM_POINT_Y];
}
void utim_draw_circle(UTIM_IMG *img,
	UTIM_POINT center, int radius, UTIM_COLOR color, int width)
{
	UTIM_POINT p;
	int x, y, dx, dy, err;
	if (!width)
		return;
	x = radius - 1; y = 0;
	dx = 1; dy = 1; err = dx - (radius << 1);
	p[UTIM_POINT_X] = center[UTIM_POINT_X];
	p[UTIM_POINT_Y] = center[UTIM_POINT_Y];
	width = width >  0 ? width : -width;
	while(x >= y) {
		_draw_circle_point(img, p, center,  x,  y, color, width);
		_draw_circle_point(img, p, center,  y,  x, color, width);
		_draw_circle_point(img, p, center, -y,  x, color, width);
		_draw_circle_point(img, p, center, -x,  y, color, width);
		_draw_circle_point(img, p, center, -x, -y, color, width);
		_draw_circle_point(img, p, center, -y, -x, color, width);
		_draw_circle_point(img, p, center,  y, -x, color, width);
		_draw_circle_point(img, p, center,  x, -y, color, width);
		if (err <= 0) {
			y++;
			err += dy;
			dy += 2;
		}
		if (err > 0) {
			x--;
			dx += 2;
			err += dx - (radius << 1);
		}
	}
}

void utim_draw_filled_circle(UTIM_IMG *img,
	UTIM_POINT center, int radius, UTIM_COLOR color)
{
	int x, y, r2;
	UTIM_POINT p, bounds[2];
	if (radius <= 0) return;
	bounds[0][UTIM_POINT_X] = center[UTIM_POINT_X] - radius;
	bounds[1][UTIM_POINT_X] = center[UTIM_POINT_X] + radius;
	bounds[0][UTIM_POINT_Y] = center[UTIM_POINT_Y] - radius;
	bounds[1][UTIM_POINT_Y] = center[UTIM_POINT_Y] + radius;

	for (x = bounds[0][UTIM_POINT_X];
		x <= bounds[1][UTIM_POINT_X]; ++x) {
		for (y = bounds[0][UTIM_POINT_Y];
			y <= bounds[1][UTIM_POINT_Y]; ++y) {
			p[UTIM_POINT_X] = x;
			p[UTIM_POINT_Y] = y;
			r2 = radius * radius;
			if(((p[UTIM_POINT_X] - center[UTIM_POINT_X]) *
				(p[UTIM_POINT_X] - center[UTIM_POINT_X]) +
				(p[UTIM_POINT_Y] - center[UTIM_POINT_Y]) *
				(p[UTIM_POINT_Y] - center[UTIM_POINT_Y])) < r2)
				utim_draw_point_fn(img, p, color);
		}
	}
}

/*
 * PSF2 fonts
 * View https://www.win.tue.nl/~aeb/linux/kbd/font-formats-1.html
 */
#define PSF2_MAGIC0     0x72
#define PSF2_MAGIC1     0xb5
#define PSF2_MAGIC2     0x4a
#define PSF2_MAGIC3     0x86

/* bits used in flags */
#define PSF2_HAS_UNICODE_TABLE 0x01

/* max version recognized so far */
#define PSF2_MAXVERSION 0

/* UTF8 separators */
#define PSF2_SEPARATOR  0xFF
#define PSF2_STARTSEQ   0xFE

static int _chk_psf2_header(struct psf2_header header)
{
	if (header.magic[0]!=PSF2_MAGIC0) return 0;
	if (header.magic[1]!=PSF2_MAGIC1) return 0;
	if (header.magic[2]!=PSF2_MAGIC2) return 0;
	if (header.magic[3]!=PSF2_MAGIC3) return 0;
	if (header.version > PSF2_MAXVERSION) return 0;
	return 1;
}

static unsigned int _read_utf8_value(byte *character, int *nb)
{
	int i;
	unsigned int first, scalar;
	*nb = 1;
	first = character[0];
	scalar = first;
	if (first <= 127)
		return first;
	if (first >> 5 == 6)
		*nb = 2;
	else if (first >> 4 == 14)
		*nb = 3;
	else if (first >> 3 == 30)
		*nb = 4;
	for (i = 1; i < *nb; ++i)
		scalar = scalar * 256 + character[i];
	return scalar;
}

static void _init_utf8_table(unsigned int *utf8index,
	int nb_glyphs, unsigned char *data, int data_len)
{
	unsigned int v;
	int b, i = 0, pos = 0;
	while (pos < data_len && i < nb_glyphs) {
		v = _read_utf8_value(data + pos, &b);
		utf8index[i] = v;
		if (v != 255)
			pos += b;
		while (pos < data_len && data[pos] != PSF2_SEPARATOR)
			pos++;
		pos++; i++;
	}
}

UTIM_FONT *utim_load_font(const char *filename)
{
	FILE *fp;
	byte *utf_data;
	UTIM_FONT *font;
	int dsize, state, pos, end;
	fp = fopen(filename, "rb");
	if (!fp)
		return NULL;
	font = (UTIM_FONT*)malloc(sizeof(UTIM_FONT));
	if (!font) {
		fclose(fp);
		return NULL;
	}
	font->utf8index = NULL;
	/* _ARRT_ */
	state = fread(&(font->header), sizeof(struct psf2_header), 1, fp);
	/* CHK */
	if ((state < 1) || (!_chk_psf2_header(font->header))) {
		free(font);
		fclose(fp);
		return NULL;
	}
	dsize = font->header.length * font->header.charsize;
	font->glyphs = (byte*)malloc(dsize);
	if (!font->glyphs) {
		free(font);
		fclose(fp);
		return NULL;
	}
	state = fread(font->glyphs, dsize, 1, fp);
	if (state < 1) {
		free(font->glyphs);
		free(font);
		fclose(fp);
		return NULL;
	}
	if (font->header.flags & PSF2_HAS_UNICODE_TABLE) {
		font->utf8index = (unsigned int*)malloc(
			font->header.length * sizeof(unsigned int));
		if (!font->utf8index) {
			free(font->glyphs);
			free(font);
			fclose(fp);
			return NULL;
		}
		pos = ftell(fp);
		fseek(fp, 0, SEEK_END);
		end = ftell(fp);
		fseek(fp, pos, SEEK_SET);
		utf_data = (byte*)malloc(end - pos);
		if (!utf_data) {
			free(font->utf8index);
			free(font->glyphs);
			free(font);
			fclose(fp);
			return NULL;
		}
		state = fread(utf_data, end - pos, 1, fp);
		if (!state) {
			font->header.flags = font->header.flags &
				(0xffffffff - PSF2_HAS_UNICODE_TABLE);
			free(font->utf8index);
			font->utf8index = NULL;
			free(utf_data);
			fclose(fp);
			return font;
		}
		_init_utf8_table(font->utf8index,
			font->header.length, utf_data, end - pos);
		free(utf_data);
	}
	fclose(fp);
	return font;
}

void utim_free_font(UTIM_FONT *font)
{
	if (!font)
		return;
	if (font->glyphs)
		free(font->glyphs);
	if (font->utf8index)
		free(font->utf8index);
	free(font);
}

byte *get_character(UTIM_FONT *font, unsigned int number)
{
	if (number >= font->header.length)
		return NULL;
	if (!font->glyphs)
		return NULL;
	if (number > 0)
        	return(font->glyphs + (font->header.charsize) * number);
	/* If not found, use the default glyph */
	return(font->glyphs);
}

static int _glyph_index(UTIM_FONT *font, unsigned int v)
{
	unsigned int i;
	if (!(font->header.flags & PSF2_HAS_UNICODE_TABLE)) {
		/* Only 7-bits ASCII is supported */
        	if (v <= 32 || v > 127) {
			/* use a default glyph */
			return -1;
		}
		return v;
	}
	for (i = 0; i < font->header.length; ++i) {
		if (font->utf8index[i] == v) {
			return i;
		}
	}
	return -1;
}

byte *get_glyph(UTIM_FONT *font, char *character, int *nb){
    return get_character(font,
    	_glyph_index(font, _read_utf8_value((byte*)character, nb)));
}

#define UTIM_TEXT_DRAW_GLYPH_PROC \
for (j = 0; j < charlen; ++j) {                                \
	for (k = 7; k > 0; --k) {                              \
		t = ((1 << k) &charp[j]);                      \
		if (t) {                                       \
			point[UTIM_POINT_X] =                  \
			(pos * font->header.width) + (7 - k);  \
			point[UTIM_POINT_Y] = j;               \
			utim_draw_point_fn(img, point, color); \
		}                                              \
	}                                                      \
}

#define UTIM_TEXT_DEFAULT_GLYPH_PROC \
for (j = 0; j < charlen; ++j) {                                \
	for (k = 7; k > 0; --k) {                              \
		point[UTIM_POINT_X] =                          \
		(pos * font->header.width) + (7 - k);          \
		point[UTIM_POINT_Y] = j;                       \
		utim_draw_point_fn(img, point, color);         \
	}                                                      \
}

UTIM_IMG *utim_text(UTIM_FONT *font, char *text, UTIM_COLOR color)
{
	byte *charp;
	UTIM_IMG *img;
	UTIM_POINT point;
	int i, j, k, t, l, charlen, nb, pos = 0;
	if (!font || !text)
		return NULL;
	l = strlen(text);
	img = utim_create(font->header.width * l, font->header.height, 4, 0);
	for (i = 0; i < l; i = i + nb) {
		charlen = font->header.charsize;
		charp = get_glyph(font, text + i, &nb);
		if (charp) {
			UTIM_TEXT_DRAW_GLYPH_PROC;
		} else { /* failled getting the character */
			UTIM_TEXT_DEFAULT_GLYPH_PROC;
		}
		pos++;
	}
	return img;
}
