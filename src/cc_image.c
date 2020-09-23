#include <stdio.h>
#include <stdlib.h>

#include "cc_assert.h"
#include "cc_array.h"
#include "cc_basic.h"
#define _CC_IMAGE_C_
#include "cc_image.h"

cc_tensor_t *cc_image2tensor(const UTIM_IMG *img, const char *name)
{
	cc_uint8 *ptr;
	cc_int32 i, j, ch_size;
	cc_int32 shape[4] = {0}; /* [C, H, W, \0] */
	cc_tensor_t *tensor;
	shape[CC_IT_SHAPE_C] = img->channels;
	shape[CC_IT_SHAPE_H] = img->ysize;
	shape[CC_IT_SHAPE_W] = img->xsize;
	cc_assert_ptr(tensor =
		cc_create_tensor(shape, CC_UINT8, name));
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
	cc_int32 i, j, ch_size, npxls;
	const cc_int32 *sptr = tensor->shape;
	if (cc_tensor_dimension(tensor) != 3) /* [C, H, W] */
		return NULL;
	img = utim_create(sptr[CC_IT_SHAPE_W],
		sptr[CC_IT_SHAPE_H], sptr[CC_IT_SHAPE_C], 0);
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
