#ifndef _CC_BASIC_H_
#define _CC_BASIC_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

cc_int32 cc_tensor_elements(cc_tensor_t *tensor);

cc_int32 cc_tensor_dimension(cc_tensor_t *tensor);

void cc_tensor_shape_fix(cc_int32 *shape, cc_int32 elems);

cc_tensor_t *cc_tensor_reshape(cc_tensor_t *tensor, cc_int32 *shape);

int cc_tsrcmp_by_shape(cc_tensor_t *a, cc_tensor_t *b);

void cc_print_tensor(cc_tensor_t *tensor);

cc_tensor_t *cc_cast_tensor(cc_tensor_t *tensor,
		cc_dtype dtype, const char *name);

cc_tensor_t *cc_tensor_by_scalar(cc_tensor_t *tensor,
	char op, void *data, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_BASIC_H_ */
