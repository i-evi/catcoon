#ifndef _CC_NORMFN_H_
#define _CC_NORMFN_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

/*
 * Batch Normalization parameters offset
 * bnpara:
 * GAMMA | BETA | MEAN | VAR | EPSILON
 * cc_int32 shape[] = {ch, CC_BN_PARAMETERS, 1, 0}
 *                      \        \___ number of parameters(5)
 *                       \___________ number of channels
 * cc_tensor_t *bnpara = cc_create_tensor(shape, dt, "name");
 */
#ifndef CC_BN_OFFSET_CFG
#define CC_BN_OFFSET_CFG
enum cc_batch_norm_paraoff {
	CC_BN_OFFSET_GAMMA,
	CC_BN_OFFSET_BETA,
	CC_BN_OFFSET_MEAN,
	CC_BN_OFFSET_VAR,
	CC_BN_OFFSET_EPSILON,
	CC_BN_PARAMETERS
};
#endif

cc_tensor_t *cc_batch_norm2d(cc_tensor_t *inp,
	cc_tensor_t *para, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_NORMFN_H_ */
