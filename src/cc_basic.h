#ifndef _CC_BASIC_H_
#define _CC_BASIC_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_int32 cc_elements (const cc_tensor_t *tensor);

cc_int32 cc_dimension(const cc_tensor_t *tensor);

void cc_shape_fix(cc_int32 *shape, cc_int32 elems);

/*
 * Before reshape, `cc_reshape` will check and fix shape
 * e.g. For a tensor shape with `[3, 3, 3]`, it can be reshaped through
 * the argument `shape = [-1, 3]`. After reshaping, the tensor's shape
 * should be `[9, 3]`, and `shape` will be modified to `[9, 3]`.
 */
cc_tensor_t *cc_reshape(cc_tensor_t *tensor, cc_int32 *shape);

int cc_compare_by_shape(const cc_tensor_t *a, const cc_tensor_t *b);

cc_tensor_t *cc_stack(cc_tensor_t **tsr,
	cc_int32 ntsr, cc_int32 axis, const char *name);

void cc_print(const cc_tensor_t *tensor);

void cc_set_value(cc_tensor_t *tensor, void *v);

cc_tensor_t *cc_cast(cc_tensor_t *tensor,
		cc_dtype dtype, const char *name);

cc_tensor_t *cc_scalar(cc_tensor_t *tensor,
	char op, const void *data, const char *name);

/*
 * Element wise operations
 * a = op(a, b) if name not set(NULL)
 * c = op(a, b) if set name for c
 */
cc_tensor_t *cc_elemwise(cc_tensor_t *a,
	cc_tensor_t *b, char op, const char *name);

cc_tensor_t *cc_clip_by_value(cc_tensor_t *tensor,
	const void *min, const void *max, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_BASIC_H_ */
