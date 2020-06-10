/*
 * Global function configuration
 */

#ifndef _GLOBAL_FN_CFG_H_
#define _GLOBAL_FN_CFG_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_dtype.h"
#include "cc_cpufn.h"

void (*_activation_relu)(void *inp, cc_int32 elems, cc_dtype dt)
	=  cc_cpu_activation_relu;

void (*_activation_relu6)(void *inp, cc_int32 elems, cc_dtype dt)
	=  cc_cpu_activation_relu6;

void (*_activation_softmax)(void *inp, cc_int32 elems, cc_dtype dt)
	= cc_cpu_activation_softmax;

void (*_max_pool2d)(void *inp, void *oup, cc_int32 x, cc_int32 y,
	cc_int32 s, cc_dtype dt) = cc_cpu_max_pool2d;

void (*_conv2d)(void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 oup_x, cc_int32 oup_y,
	cc_int32 sx, cc_int32 sy, void *filter, cc_int32 fw,
	cc_dtype dt) = cc_cpu_conv2d;

void (*_fully_connected)(void *inp, void *oup, void *w, void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt) = cc_cpu_fully_connected;

void (*_batch_norm)(void *inp, cc_int32 len,
	void *bnpara, cc_dtype dt) = cc_cpu_batch_norm;

#ifdef __cplusplus
	}
#endif

#endif /* _GLOBAL_FN_CFG_H_ */
