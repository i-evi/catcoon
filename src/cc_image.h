#ifndef _CC_IMAGE_H_
#define _CC_IMAGE_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"
#include "util_image.h"

#define CC_RESIZE_NEAREST UTIM_RESIZE_NEAREST
#define CC_RESIZE_LINEAR  UTIM_RESIZE_LINEAR

/*
 * Create tensor from image, [C, H, W] format
 */
#define CC_IM_DIM 3
#define CC_IM_SHAPE_C 0
#define CC_IM_SHAPE_H 1
#define CC_IM_SHAPE_W 2

enum cc_colorspace {
	CC_GRAY,
	CC_RGB,
	CC_RGBA
};

cc_tensor_t *cc_imread(const char *pathname, cc_ssize h, cc_ssize w,
	int mode, enum cc_colorspace color, const char *name);

int cc_imsave(const char *pathname, cc_ssize h, cc_ssize w,
	int mode, enum cc_colorspace color, cc_tensor_t *tensor);

cc_tensor_t *cc_image2tensor(const UTIM_IMG *img, const char *name);

UTIM_IMG *cc_tensor2image(const cc_tensor_t *tensor);

enum cc_image_norm_mode {
	/* CC_IM_NORM_MINMAX, */
	CC_IM_NORM_MINMAX_RGB,
	/* CC_IM_NORM_ZSCORE, */
	CC_IM_NORM_ZSCORE_RGB
};

void cc_image_norm(cc_tensor_t *tsrimg, enum cc_image_norm_mode mode);

#ifndef _CC_IMAGE_C_
	#undef CC_IM_DIM
	#undef CC_IM_SHAPE_C
	#undef CC_IM_SHAPE_H
	#undef CC_IM_SHAPE_W
#endif

#ifdef __cplusplus
	}
#endif

#endif /* _CC_IMAGE_H_ */
