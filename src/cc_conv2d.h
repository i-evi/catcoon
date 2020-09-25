#ifndef _CC_CONV2D_H_
#define _CC_CONV2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define CC_CONV2D_PAD_NAME_LEN     128
#define CC_CONV2D_PAD_NAME_SURFFIX "_pad"

enum cc_conv2d_kernel {
	CC_CONV2D_KERNEL_O,
	CC_CONV2D_KERNEL_I,
	CC_CONV2D_KERNEL_H,
	CC_CONV2D_KERNEL_W,
	CC_CONV2D_KERNEL_DIM
};

#include "cc_tensor.h"

cc_int32 cc_conv2d_shape_calc(
	cc_int32 i, cc_int32 k, cc_int32 s, cc_int32 p);

cc_tensor_t *cc_conv2d(const cc_tensor_t *inp,
		const cc_tensor_t *kernel, const cc_tensor_t *bias,
	cc_int32 s, cc_int32 p, cc_int32 off, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_CONV2D_H_ */
