#ifndef _CC_FULLYCON_H_
#define _CC_FULLYCON_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

enum cc_fullycon_kernel {
	CC_FULLYCON_KERNEL_O,
	CC_FULLYCON_KERNEL_I,
	CC_FULLYCON_KERNEL_DIM
};

cc_tensor_t *cc_fully_connected(const cc_tensor_t *inp,
	const cc_tensor_t *w, const cc_tensor_t *b, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_FULLYCON_H_ */
