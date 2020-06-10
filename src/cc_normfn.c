#include <string.h>

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_fmap2d.h"
#include "cc_normfn.h"

/* #include "global_fn_cfg.h" */
extern void (*_batch_norm)(
	void *inp, cc_int32 len, void *bnpara, cc_dtype dt);

cc_tensor_t *cc_batch_norm2d(cc_tensor_t *inp,
	cc_tensor_t *para, const char *name)
{
	cc_tensor_t *oup;
	cc_int32 i, dt_size, ch_size, ch_mem_size;
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_tensor_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(*inp->dtype - *para->dtype);
#endif
	if (!name || !strcmp(name, inp->name))
		oup = inp;
	else
		oup = cc_copy_tensor(inp, name);
	dt_size = cc_dtype_size(*inp->dtype);
	ch_size = inp->shape[CC_CNN2D_SHAPE_H] *
			inp->shape[CC_CNN2D_SHAPE_W];
	ch_mem_size = ch_size * dt_size;
	for (i = 0; i < inp->shape[CC_CNN2D_SHAPE_C]; ++i) {
		_batch_norm(inp->data + ch_mem_size * i,
				ch_size, para->data + CC_BN_PARAMETERS * 
			dt_size * i, *para->dtype);
	}
	return oup;
}
