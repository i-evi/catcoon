#ifndef _CC_CONV2D_H_
#define _CC_CONV2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define CC_CONV2D_PAD_NAME_SURFFIX "_pad"

enum cc_conv2d_kernel {
	CC_CONV2D_KERNEL_O,
	CC_CONV2D_KERNEL_I,
	CC_CONV2D_KERNEL_H,
	CC_CONV2D_KERNEL_W,
	CC_CONV2D_KERNEL_DIM
};

#include "cc_tensor.h"

cc_ssize cc_conv2d_shape_calc(
	cc_ssize i, cc_ssize k, cc_ssize s, cc_ssize p);

cc_tensor_t *cc_conv2d(const cc_tensor_t *inp,
		const cc_tensor_t *kernel, const cc_tensor_t *bias,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name);

/* Depth-wise convolution 2d */
cc_tensor_t *cc_dw_conv2d(cc_tensor_t *inp,
		const cc_tensor_t *kernel, const cc_tensor_t *bias,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name);

/* Point-wise convolution 2d */
cc_tensor_t *cc_pw_conv2d(cc_tensor_t *inp, const cc_tensor_t *kernel,
	const cc_tensor_t *bias, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_CONV2D_H_ */
