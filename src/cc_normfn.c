#include <string.h>

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_fmap2d.h"
#include "cc_normfn.h"

#include "global_fn_cfg.h"
extern fn_batch_norm _batch_norm;

static cc_float32 cc_bn_epsilon_dfl_fp32 = CC_BN_EPSILON_DFL_FP32;

cc_tensor_t *cc_load_bin_bnpara(const char *w_path, const char *b_path,
		const char *m_path, const char *v_path, const char *e_path,
	cc_int32 nchl, cc_dtype dtype, const char *name)
{
	static cc_int32 shape[] = {0, 1, 1, 0};
	cc_int32 i;
	cc_tensor_t *para;
	cc_tensor_t *tsr[CC_BN_PARAMETERS];
	shape[0] = nchl;
	tsr[CC_BN_OFFSET_GAMMA] = cc_load_bin(w_path, shape, dtype, NULL);
	tsr[CC_BN_OFFSET_BETA]  = cc_load_bin(b_path, shape, dtype, NULL);
	tsr[CC_BN_OFFSET_MEAN]  = cc_load_bin(m_path, shape, dtype, NULL);
	tsr[CC_BN_OFFSET_VAR]   = cc_load_bin(v_path, shape, dtype, NULL);
	if (e_path) {
		tsr[CC_BN_OFFSET_EPSILON] =
			cc_load_bin(e_path, shape, dtype, NULL);
	} else {
		tsr[CC_BN_OFFSET_EPSILON] =
			cc_create_tensor(shape, dtype, NULL);
		switch (dtype) {
			case CC_FLOAT32:
				cc_set_tensor(tsr[CC_BN_OFFSET_EPSILON],
				 	&cc_bn_epsilon_dfl_fp32);
				break;
			default:
				utlog_format(UTLOG_ERR,
					"[%s: %d] Unsupported dtype %x\n",
					__FILE__, __LINE__, dtype);
				break;
		}
	}
	para = cc_tensor_stack(tsr, CC_BN_PARAMETERS, 1, name);
	for (i = 0; i < CC_BN_PARAMETERS; ++i)
		cc_free_tensor(tsr[i]);
	return para;
}

cc_tensor_t *cc_batch_norm2d(cc_tensor_t *inp,
	const cc_tensor_t *para, const char *name)
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
