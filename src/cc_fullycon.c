#include <stdio.h>

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_conv2d.h"
#include "cc_fmap2d.h"
#include "cc_tsrmgr.h"
#include "cc_fullycon.h"

#include "global_fn_cfg.h"
extern fn_array_dot_prod _array_dot_prod;

cc_tensor_t *cc_fully_connected(const cc_tensor_t *inp,
	const cc_tensor_t *w, const cc_tensor_t *b, const char *name)
{
	cc_ssize i, mmsize, dtsize;
	cc_tensor_t *oup = NULL;
	cc_ssize shape[CC_CNN2D_SHAPE] = {0};
#ifdef ENABLE_CC_ASSERT
	cc_assert((cc_dimension(w) == CC_CONV2D_KERNEL_DIM) ||
		(cc_dimension(w) == CC_FULLYCON_KERNEL_DIM));
	cc_assert_zero(cc_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(*inp->dtype - *w->dtype);
	cc_assert_zero(*inp->dtype - *b->dtype);
	cc_assert_zero(inp->shape[CC_CNN2D_SHAPE_C]
			- w->shape[CC_CONV2D_KERNEL_I]);
#endif
#ifdef AUTO_TSRMGR
	oup = cc_tsrmgr_get(name);
#endif
	if (!oup) {
		shape[CC_CNN2D_SHAPE_C] = w->shape[CC_CONV2D_KERNEL_O];
		shape[CC_CNN2D_SHAPE_H] = 1;
		shape[CC_CNN2D_SHAPE_W] = 1;
		oup = cc_create(shape, *inp->dtype, name);
	}
	dtsize = cc_dtype_size(*inp->dtype);
	mmsize = inp->shape[CC_CNN2D_SHAPE_C] * dtsize;
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < w->shape[CC_CONV2D_KERNEL_O]; ++i) {
		_array_dot_prod(inp->data, w->data + i * mmsize,
				inp->shape[CC_CNN2D_SHAPE_C],
			oup->data + i * dtsize, *inp->dtype);
	}
	if (b)
		oup = cc_fmap2d_bias(oup, b, oup->name);
	return oup;
}
