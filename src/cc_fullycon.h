#ifndef _CC_FULLYCON_H_
#define _CC_FULLYCON_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_tensor_t *cc_fully_connected(cc_tensor_t *inp,
		cc_tensor_t *w, cc_tensor_t *b, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_FULLYCON_H_ */
