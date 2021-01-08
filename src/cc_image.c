#include <stdio.h>
#include <stdlib.h>

#include "cc_assert.h"
#include "cc_array.h"
#include "cc_basic.h"
#include "util_log.h"
#define _CC_IMAGE_C_
#include "cc_image.h"

cc_tensor_t *cc_imread(const char *pathname, cc_ssize h, cc_ssize w,
	int mode, enum cc_colorspace color, const char *name)
{
	UTIM_IMG *rd, *img;
	cc_tensor_t *tensor;
	cc_assert_ptr(rd = utim_read(pathname));
	switch (color) {
	case CC_GRAY:
		utim_img2gray(rd);
		break;
	case CC_RGB:
		utim_img2rgb (rd);
		break;
	case CC_RGBA:
		utim_img2rgba(rd);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_image: unsupported color space\n");
		break;
	}
	if (rd->xsize != w || rd->ysize != h) {
		img = utim_resize(rd, w, h, mode);
		utim_free_image(rd);
	} else {
		img = rd;
	}
	tensor = cc_image2tensor(img, name);
	utim_free_image(img);
	tensor = cc_cast(tensor, CC_FLOAT32, name);
	return tensor;
}

int cc_imsave(const char *pathname, cc_ssize h, cc_ssize w,
	int mode, enum cc_colorspace color, cc_tensor_t *tensor)
{
	int ret;
	UTIM_IMG *img, *tsrimg;
	tsrimg = cc_tensor2image(tensor);
	switch (color) {
	case CC_GRAY:
		utim_img2gray(tsrimg);
		break;
	case CC_RGB:
		utim_img2rgb (tsrimg);
		break;
	case CC_RGBA:
		utim_img2rgba(tsrimg);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_image: unsupported color space\n");
		break;
	}
	if (tsrimg->xsize != w || tsrimg->ysize != h) {
		img = utim_resize(tsrimg, w, h, mode);
		utim_free_image(tsrimg);
	} else {
		img = tsrimg;
	}
	ret = utim_write(pathname, img);
	utim_free_image(img);
	return ret;
}

cc_tensor_t *cc_image2tensor(const UTIM_IMG *img, const char *name)
{
	cc_uint8 *ptr;
	cc_ssize i, j, ch_size;
	cc_ssize shape[4] = {0}; /* [C, H, W, \0] */
	cc_tensor_t *tensor;
	shape[CC_IM_SHAPE_C] = img->channels;
	shape[CC_IM_SHAPE_H] = img->ysize;
	shape[CC_IM_SHAPE_W] = img->xsize;
	cc_assert_ptr(tensor = cc_create(shape, CC_UINT8, name));
	ch_size = img->xsize * img->ysize;
	ptr = tensor->data;
	for (i = 0; i < ch_size; ++i) {
		for (j = 0; j < img->channels; ++j) {
			*(ptr + ch_size * j) =
				*(img->pixels + i * img->channels + j);
		}
		ptr++;
	}
	return tensor;
}

UTIM_IMG *cc_tensor2image(const cc_tensor_t *tensor)
{
	UTIM_IMG *img;
	cc_uint8 *pxls, *ptr;
	cc_ssize i, j, ch_size, npxls;
	const cc_ssize *sptr = tensor->shape;
	if (cc_dimension(tensor) != CC_IM_DIM) /* [C, H, W] */
		return NULL;
	img = utim_create(sptr[CC_IM_SHAPE_W],
		sptr[CC_IM_SHAPE_H], sptr[CC_IM_SHAPE_C], 0);
	ch_size = img->xsize * img->ysize;
	npxls = ch_size * img->channels;
	cc_assert_alloc(pxls = (cc_uint8*)malloc(npxls));
	cc_array_cast_uint8(pxls, tensor->data, npxls, *tensor->dtype);
	ptr = img->pixels;
	for (i = 0; i < ch_size; ++i) {
		for (j = 0; j < img->channels; ++j)
			*ptr++ = *(pxls + j * ch_size + i);
	}
	free(pxls);
	return img;
}

static void cc_image_norm_minmax_rgb(cc_tensor_t *tsrimg)
{
	cc_float32 v = 255.f;
	/* experimental function for fp32 only */
	cc_assert(*tsrimg->dtype == CC_FLOAT32);
	tsrimg = cc_scalar(tsrimg, '/', &v, tsrimg->name);
}

static const cc_float32 im_zs_rgb_mean[] = {0.485, 0.456, 0.406};
static const cc_float32 im_zs_rgb_std [] = {0.229, 0.224, 0.225};

static void cc_image_norm_zscore_rgb(cc_tensor_t *tsrimg)
{
	cc_ssize i, ch_size;
	cc_float32 *obj;
	const cc_float32 *mp, *sp;
	/* experimental function for fp32 only */
	cc_assert(*tsrimg->dtype == CC_FLOAT32);
	cc_assert(cc_dimension(tsrimg) == CC_IM_DIM);
	cc_assert(tsrimg->shape[CC_IM_SHAPE_C] == 3); /* RGB only */
	ch_size = tsrimg->shape[CC_IM_SHAPE_H] *
			tsrimg->shape[CC_IM_SHAPE_W];
	for (i = 0; i < 3; ++i) {
		obj = ((cc_float32*)tsrimg->data) + ch_size * i;
		mp = im_zs_rgb_mean + i;
		sp = im_zs_rgb_std  + i;
		cc_array_sub_by(obj, ch_size, obj, mp, CC_FLOAT32);
		cc_array_div_by(obj, ch_size, obj, sp, CC_FLOAT32);
	}
}

void cc_image_norm(cc_tensor_t *tsrimg, enum cc_image_norm_mode mode)
{
	switch (mode) {
	case CC_IM_NORM_MINMAX_RGB:
		cc_image_norm_minmax_rgb(tsrimg);
		break;
	case CC_IM_NORM_ZSCORE_RGB:
		cc_image_norm_zscore_rgb(tsrimg);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_image: unsupported normalization mode\n");
		break;
	}
}
