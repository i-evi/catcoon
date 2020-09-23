#include <stdio.h>

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_conv2d.h"
#include "cc_fmap2d.h"
#include "cc_tsrmgr.h"
#include "cc_fullycon.h"

#include "global_fn_cfg.h"
extern fn_fully_connected _fully_connected;

cc_tensor_t *cc_fully_connected(const cc_tensor_t *inp,
	const cc_tensor_t *w, const cc_tensor_t *b, const char *name)
{
	cc_tensor_t *oup = NULL;
	cc_int32 shape[CC_CNN2D_SHAPE_LEN] = {0};
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_tensor_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(cc_tensor_dimension(w) - CC_CONV2D_KERNEL_DIM);
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
		oup = cc_create_tensor(shape, *inp->dtype, name);
	}
	if (b)
		_fully_connected(inp->data, oup->data,
			w->data, b->data, w->shape[CC_CONV2D_KERNEL_I],
			w->shape[CC_CONV2D_KERNEL_O], *inp->dtype);
	else
		_fully_connected(inp->data, oup->data,
			w->data, NULL, w->shape[CC_CONV2D_KERNEL_I],
			w->shape[CC_CONV2D_KERNEL_O], *inp->dtype);
	return oup;
}
