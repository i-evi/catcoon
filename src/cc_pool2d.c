#include <stdio.h>

#include "cc_fmap2d.h"
#include "cc_tsrmgr.h"
#include "cc_pad2d.h"
#include "cc_pool2d.h"

#include "global_fn_cfg.h"
extern fn_pool2d _max_pool2d;
extern fn_pool2d _avg_pool2d;

cc_ssize cc_pool2d_shape_calc(
	cc_ssize i, cc_ssize k, cc_ssize s, cc_ssize p)
{
	return (cc_ssize)((i - k + 2 * p) / s) + 1;
}

cc_tensor_t *cc_pool2d(const cc_tensor_t *inp, cc_ssize k,
	cc_ssize s, cc_ssize p, cc_ssize off, fn_pool2d pool2d, const char *name)
{
	cc_tensor_t *pool = NULL;
	const cc_tensor_t *inp_pad;
	cc_ssize i, i_ch_size, i_ch_mem_size, o_ch_size, o_ch_mem_size;
	cc_ssize shape[CC_CNN2D_SHAPE] = {0};
	char pad_name[CC_TSR_NAME_LEN];
	if (p) {
		sprintf(pad_name, "%s%s",
			inp->name, CC_POOL2D_PAD_NAME_SURFFIX);
		inp_pad = cc_pad2d(inp, p, off, pad_name);
	} else {
		inp_pad = inp;
	}
#ifdef AUTO_TSRMGR
	pool = cc_tsrmgr_get(name);
#endif
	if (!pool) {
		shape[CC_CNN2D_SHAPE_C] = inp_pad->shape[CC_CNN2D_SHAPE_C];
		shape[CC_CNN2D_SHAPE_H] = cc_pool2d_shape_calc(
			inp_pad->shape[CC_CNN2D_SHAPE_H], k, s, p);
		shape[CC_CNN2D_SHAPE_W] = cc_pool2d_shape_calc(
			inp_pad->shape[CC_CNN2D_SHAPE_W], k, s, p);
		pool = cc_create(shape, *inp->dtype, name);
	}
	i_ch_size = inp_pad->shape[CC_CNN2D_SHAPE_H] *
			inp_pad->shape[CC_CNN2D_SHAPE_W];
	i_ch_mem_size = i_ch_size * cc_dtype_size(*inp_pad->dtype);
	o_ch_size = pool->shape[CC_CNN2D_SHAPE_H] *
			pool->shape[CC_CNN2D_SHAPE_W];
	o_ch_mem_size = o_ch_size * cc_dtype_size(*pool->dtype);
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < inp_pad->shape[CC_CNN2D_SHAPE_C]; ++i) {
		pool2d(inp_pad->data + i_ch_mem_size * i,
				pool->data + o_ch_mem_size * i,
				inp_pad->shape[CC_CNN2D_SHAPE_W],
				inp_pad->shape[CC_CNN2D_SHAPE_H],
				s, s, k, *inp->dtype);
	}
#ifndef AUTO_TSRMGR
	if (p)
		cc_free((cc_tensor_t*)inp_pad);
#endif
	return pool;
}

cc_tensor_t *cc_max_pool2d(const cc_tensor_t *inp, cc_ssize k,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name)
{
	return cc_pool2d(inp, k, s, p, off, _max_pool2d, name);
}

cc_tensor_t *cc_avg_pool2d(const cc_tensor_t *inp, cc_ssize k,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name)
{
	return cc_pool2d(inp, k, s, p, off, _avg_pool2d, name);
}
