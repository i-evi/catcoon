#include <stdio.h>
#include <string.h>

#ifdef ENABLE_OPENMP
	#include <omp.h>
#endif

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_fmap2d.h"
#include "cc_tsrmgr.h"

#include "global_fn_cfg.h"
extern fn_array_add_by _array_add_by;

cc_tensor_t *cc_fmap2d_bias(cc_tensor_t *inp,
	const cc_tensor_t *bias, const char *name)
{
	cc_tensor_t *fmap;
	cc_int32 i, ch_size, ch_mem_size, dt_size;
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(*inp->dtype - *bias->dtype);
	cc_assert_zero(inp->shape[CC_CNN2D_SHAPE_C]
			- bias->shape[CC_CNN2D_SHAPE_C]);
	cc_assert_zero(bias->shape[CC_CNN2D_SHAPE_H]); /* [C, \0] */
#endif
	if (!name || !strcmp(name, inp->name))
		fmap = inp;
	else
		fmap = cc_copy(inp, name);
	dt_size = cc_dtype_size(*fmap->dtype);
	ch_size = fmap->shape[CC_CNN2D_SHAPE_H] *
			fmap->shape[CC_CNN2D_SHAPE_W];
	ch_mem_size = ch_size * dt_size;
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < bias->shape[CC_CNN2D_SHAPE_C]; ++i) {
		_array_add_by(fmap->data + ch_mem_size * i,
			ch_size, fmap->data + ch_mem_size * i,
			bias->data + dt_size * i, *fmap->dtype);
	}
	return fmap;
}

cc_tensor_t *cc_fmap2d_flat(cc_tensor_t *inp, const char *name)
{
	cc_tensor_t *flat = NULL;
	cc_uint8 *sptr, *dptr;
	cc_int32 shape[CC_CNN2D_SHAPE] = {0};
	cc_int32 i, j ,ch_size, dt_size;
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_dimension(inp) - CC_CNN2D_DIM);
#endif
#ifdef AUTO_TSRMGR
	flat = cc_tsrmgr_get(name);
#endif
	if (!flat) {
		shape[CC_CNN2D_SHAPE_C] = cc_elements(inp);
		shape[CC_CNN2D_SHAPE_H] = 1;
		shape[CC_CNN2D_SHAPE_W] = 1;
		cc_assert_ptr(flat =
			cc_create(shape, *inp->dtype, name));
	}
	sptr = (cc_uint8*)inp->data;
	dptr = (cc_uint8*)flat->data;
	dt_size = cc_dtype_size(*inp->dtype);
	ch_size = inp->shape[CC_CNN2D_SHAPE_H] *
			inp->shape[CC_CNN2D_SHAPE_W];
	for (i = 0; i < ch_size; ++i) {
		for (j = 0; j < inp->shape[CC_CNN2D_SHAPE_C]; ++j) {
			memcpy((dptr + 
				(i * inp->shape[CC_CNN2D_SHAPE_C] + j) *
				dt_size), (sptr + (j * ch_size + i) *
				dt_size), dt_size);
		}
	}
	return flat;
}
