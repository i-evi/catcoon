#ifndef _CC_ACTFN_H_
#define _CC_ACTFN_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_tensor_t *cc_relu (cc_tensor_t *tensor, const char *name);
cc_tensor_t *cc_relu6(cc_tensor_t *tensor, const char *name);

cc_tensor_t *cc_softmax(cc_tensor_t *tensor, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_ACTFN_H_ */
