#ifndef _CC_PAD2D_H_
#define _CC_PAD2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_tensor_t *cc_pad2d(const cc_tensor_t *inp,
	cc_int32 p, cc_int32 offset, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_PAD2D_H_ */
