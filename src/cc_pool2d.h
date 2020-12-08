#ifndef _CC_POOL2D_H_
#define _CC_POOL2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_tensor_t *cc_max_pool2d(
	const cc_tensor_t *inp, cc_ssize s, const char *name);

cc_tensor_t *cc_avg_pool2d(
	const cc_tensor_t *inp, cc_ssize s, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_POOL2D_H_ */
