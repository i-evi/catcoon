#ifndef _CC_CPUFN_H_
#define _CC_CPUFN_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_dtype.h"

void cc_cpu_activation_relu (void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_activation_relu6(void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_activation_softmax(void *inp, cc_int32 elems, cc_dtype dt);

void cc_cpu_max_pool2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

void cc_cpu_avg_pool2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt);

void cc_cpu_conv2d(const void *inp, void *oup, cc_int32 x, cc_int32 y,
		cc_int32 oup_x, cc_int32 oup_y, cc_int32 sx, cc_int32 sy,
	const void *filter, cc_int32 fw, cc_dtype dt);

void cc_cpu_fully_connected(const void *inp,
		void *oup, const void *w, const void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt);

void cc_cpu_batch_norm(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_CPUFN_H_ */
