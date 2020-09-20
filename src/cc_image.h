#ifndef _CC_IMAGE_H_
#define _CC_IMAGE_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"
#include "util_image.h"

/*
 * Create tensor from image, [C, H, W] format
 */
#define CC_IT_SHAPE_C 0
#define CC_IT_SHAPE_H 1
#define CC_IT_SHAPE_W 2

cc_tensor_t *cc_image2tensor(UTIM_IMG *img, const char *name);

UTIM_IMG *cc_tensor2image(cc_tensor_t *tensor);

#ifndef _CC_IMAGE_C_
	#undef CC_IT_SHAPE_C
	#undef CC_IT_SHAPE_H
	#undef CC_IT_SHAPE_W
#endif

#ifdef __cplusplus
	}
#endif

#endif /* _CC_IMAGE_H_ */
