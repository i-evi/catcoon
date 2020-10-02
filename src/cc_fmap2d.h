#ifndef _CC_FMAP2D_H_
#define _CC_FMAP2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

enum cc_cnn2d_shape {
	CC_CNN2D_SHAPE_C,
	CC_CNN2D_SHAPE_H,
	CC_CNN2D_SHAPE_W,
	CC_CNN2D_DIM,
	CC_CNN2D_SHAPE
};

cc_tensor_t *cc_fmap2d_bias(cc_tensor_t *inp,
	const cc_tensor_t *bias, const char *name);

cc_tensor_t *cc_fmap2d_flat(cc_tensor_t *inp, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_FMAP2D_H_ */
