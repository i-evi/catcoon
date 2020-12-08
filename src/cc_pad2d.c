#include <stdio.h>
#include <string.h>

#include "cc_dtype.h"
#include "cc_fmap2d.h"
#include "cc_pad2d.h"
#include "cc_tsrmgr.h"

#define PAD_MEMCPY \
	memcpy((pad->data) + p_ch_mem_size * c +                         \
	(i + p - poffset) * p_row_mem_size + (p + j - poffset) * dtsize, \
	inp->data + i_ch_mem_size * c + i * i_row_mem_size + j * dtsize, \
	dtsize);

cc_tensor_t *cc_pad2d(const cc_tensor_t *inp,
	cc_ssize p, cc_ssize offset, const char *name)
{
	cc_tensor_t *pad = NULL;
	cc_ssize shape[CC_CNN2D_SHAPE] = {0};
	cc_ssize soffset = offset ? 1 : 0;
	cc_ssize poffset = offset > 0 ? 1 : 0;
	cc_ssize i, j, c, dtsize = cc_dtype_size(*inp->dtype);
	cc_ssize i_ch_size, i_ch_mem_size, i_row_mem_size,
		p_ch_size, p_ch_mem_size, p_row_mem_size;
	i_ch_size = inp->shape[CC_CNN2D_SHAPE_W] *
			inp->shape[CC_CNN2D_SHAPE_H];
	i_ch_mem_size  = i_ch_size * dtsize;
	i_row_mem_size = inp->shape[CC_CNN2D_SHAPE_W] * dtsize;
#ifdef AUTO_TSRMGR
	pad = cc_tsrmgr_get(name);
#endif
	if (!pad) {
		shape[CC_CNN2D_SHAPE_C] = inp->shape[CC_CNN2D_SHAPE_C];
		shape[CC_CNN2D_SHAPE_H] =
			inp->shape[CC_CNN2D_SHAPE_H] + p + p - soffset;
		shape[CC_CNN2D_SHAPE_W] = 
			inp->shape[CC_CNN2D_SHAPE_W] + p + p - soffset;
		pad = cc_create(shape, *inp->dtype, name);
	}
	p_ch_size = pad->shape[CC_CNN2D_SHAPE_W] *
			pad->shape[CC_CNN2D_SHAPE_H];
	p_ch_mem_size  = p_ch_size * dtsize;
	p_row_mem_size = pad->shape[CC_CNN2D_SHAPE_W] * dtsize;
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(c, i, j)
#endif
	for (c = 0; c < inp->shape[CC_CNN2D_SHAPE_C]; ++c) {
		for (i = 0; i < inp->shape[CC_CNN2D_SHAPE_H]; ++i) {
			for (j = 0; j < inp->shape[CC_CNN2D_SHAPE_W]; ++j)
				PAD_MEMCPY;
		}
	}
	return pad;
}
