#ifndef _CC_POOL2D_H_
#define _CC_POOL2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"
#include "global_fn_cfg.h"

#define CC_POOL2D_PAD_NAME_SURFFIX "_pad"

cc_ssize cc_pool2d_shape_calc(
	cc_ssize i, cc_ssize k, cc_ssize s, cc_ssize p);

cc_tensor_t *cc_pool2d(const cc_tensor_t *inp, cc_ssize k, cc_ssize s,
	cc_ssize p, cc_ssize off, fn_pool2d pool2d, const char *name);

cc_tensor_t *cc_max_pool2d(const cc_tensor_t *inp, cc_ssize k,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name);

cc_tensor_t *cc_avg_pool2d(const cc_tensor_t *inp, cc_ssize k,
	cc_ssize s, cc_ssize p, cc_ssize off, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_POOL2D_H_ */
