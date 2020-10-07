#include <stdio.h>

#include "cc_fmap2d.h"
#include "cc_tsrmgr.h"
#include "cc_pool2d.h"

#include "global_fn_cfg.h"
extern fn_max_pool2d _max_pool2d;
extern fn_avg_pool2d _avg_pool2d;

cc_tensor_t *cc_max_pool2d(
	const cc_tensor_t *inp, cc_int32 s, const char *name)
{
	cc_int32 i, i_ch_size, i_ch_mem_size, o_ch_size, o_ch_mem_size;
	cc_int32 shape[CC_CNN2D_SHAPE] = {0};
	cc_tensor_t *pool = NULL;
#ifdef AUTO_TSRMGR
	pool = cc_tsrmgr_get(name);
#endif
	if (!pool) {
		shape[CC_CNN2D_SHAPE_C] = inp->shape[CC_CNN2D_SHAPE_C];
		shape[CC_CNN2D_SHAPE_H] = inp->shape[CC_CNN2D_SHAPE_H] / s;
		shape[CC_CNN2D_SHAPE_W] = inp->shape[CC_CNN2D_SHAPE_W] / s;
		pool = cc_create(shape, *inp->dtype, name);
	}
	i_ch_size = inp->shape[CC_CNN2D_SHAPE_H] *
			inp->shape[CC_CNN2D_SHAPE_W];
	i_ch_mem_size = i_ch_size * cc_dtype_size(*inp->dtype);
	o_ch_size = pool->shape[CC_CNN2D_SHAPE_H] *
			pool->shape[CC_CNN2D_SHAPE_W];
	o_ch_mem_size = o_ch_size * cc_dtype_size(*pool->dtype);
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < inp->shape[CC_CNN2D_SHAPE_C]; ++i) {
		_max_pool2d(inp->data + i_ch_mem_size * i,
				pool->data + o_ch_mem_size * i,
				inp->shape[CC_CNN2D_SHAPE_W],
				inp->shape[CC_CNN2D_SHAPE_H], s,
				*inp->dtype);
	}
	return pool;
}

cc_tensor_t *cc_avg_pool2d(
	const cc_tensor_t *inp, cc_int32 s, const char *name)
{
	cc_int32 i, i_ch_size, i_ch_mem_size, o_ch_size, o_ch_mem_size;
	cc_int32 shape[CC_CNN2D_SHAPE] = {0};
	cc_tensor_t *pool = NULL;
#ifdef AUTO_TSRMGR
	pool = cc_tsrmgr_get(name);
#endif
	if (!pool) {
		shape[CC_CNN2D_SHAPE_C] = inp->shape[CC_CNN2D_SHAPE_C];
		shape[CC_CNN2D_SHAPE_H] = inp->shape[CC_CNN2D_SHAPE_H] / s;
		shape[CC_CNN2D_SHAPE_W] = inp->shape[CC_CNN2D_SHAPE_W] / s;
		pool = cc_create(shape, *inp->dtype, name);
	}
	i_ch_size = inp->shape[CC_CNN2D_SHAPE_H] *
			inp->shape[CC_CNN2D_SHAPE_W];
	i_ch_mem_size = i_ch_size * cc_dtype_size(*inp->dtype);
	o_ch_size = pool->shape[CC_CNN2D_SHAPE_H] *
			pool->shape[CC_CNN2D_SHAPE_W];
	o_ch_mem_size = o_ch_size * cc_dtype_size(*pool->dtype);
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < inp->shape[CC_CNN2D_SHAPE_C]; ++i) {
		_avg_pool2d(inp->data + i_ch_mem_size * i,
				pool->data + o_ch_mem_size * i,
				inp->shape[CC_CNN2D_SHAPE_W],
				inp->shape[CC_CNN2D_SHAPE_H], s,
				*inp->dtype);
	}
	return pool;
}
