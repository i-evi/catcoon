#ifndef _CC_NORMFN_H_
#define _CC_NORMFN_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

/*
 * Normalization parameters offset
 * 
 * GAMMA | BETA | MEAN | VAR | EPSILON
 * cc_int32 shape[] = {ch, CC_BN_PARAMETERS, 1, 0}
 *                      \        \___ number of parameters(5)
 *                       \___________ number of channels
 * cc_tensor_t *para = cc_create_tensor(shape, dt, "name");
 */
#ifndef CC_NORM_PARA_CFG
#define CC_NORM_PARA_CFG
enum cc_norm_para {
	CC_NORM_GAMMA = 0,
	CC_NORM_BETA,
	CC_NORM_MEAN,
	CC_NORM_VAR,
	CC_NORM_EPSILON,
	CC_NORM_PARAMETERS
};
#endif

#define CC_NORM_EPSILON_DFL_FP32 1e-3

cc_tensor_t *cc_load_bin_norm_para(
		const char *w_path,             /* Gamma   */
		const char *b_path,             /* Beta    */
		const char *m_path,             /* Mean    */
		const char *v_path,             /* Var     */
		const char *e_path,             /* Epsilon */
		cc_ssize    nchl,               /* Channel */
		cc_dtype    dtype,
		const char *name);

cc_tensor_t *cc_batch_norm2d(cc_tensor_t *inp,
	const cc_tensor_t *para, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_NORMFN_H_ */
