#ifndef _CC_TENSOR_H_
#define _CC_TENSOR_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "util_list.h"
#include "cc_dtype.h"

enum cc_tensor_items {	
	CC_TENSOR_DATA,  /* 0 */
	CC_TENSOR_SHAPE, /* 1 */
	CC_TENSOR_DTYPE, /* 2 */
	CC_TENSOR_ITEMS  /* 3 */
};
/*
 * container____ data   (0)
 *     |     |__ shape  (1)
 *     |     |__ dtype  (2)
 *     |_____ name
 */

typedef struct {
	struct list *container;
	const char     *name;
	unsigned char  *data;
	const cc_dtype *dtype;
	const cc_int32 *shape;
} cc_tensor_t;

cc_tensor_t *cc_create_tensor(const cc_int32 *shape,
			cc_dtype dtype, const char *name);

cc_tensor_t *cc_copy_tensor(const cc_tensor_t *tensor, const char *name);

cc_tensor_t *cc_load_tensor(const char *filename);

cc_tensor_t *cc_load_bin(const char *filename,
	const cc_int32 *shape, cc_dtype dtype, const char *name);

cc_tensor_t *cc_tensor_from_array(void *arr,
	const cc_int32 *shape, cc_dtype dtype, const char *name);

void cc_save_tensor(const cc_tensor_t *tensor, const char *filename);

void cc_free_tensor(cc_tensor_t *tensor);

void cc_print_tensor_property(const cc_tensor_t *tensor);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_TENSOR_H_ */
