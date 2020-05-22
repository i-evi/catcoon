#ifndef _CARCOON_H_
#define _CARCOON_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_conv2d.h"
#include "cc_fmap2d.h"
#include "cc_fullycon.h"
#include "cc_image.h"
#include "cc_pad2d.h"
#include "cc_pool2d.h"
#include "cc_tsrmgr.h"
#include "cc_tensor.h"

#define cc_clear() \
	cc_tsrmgr_clear();

#define CATCOON_VER 0.0F

void cc_print_info(void);

#ifdef __cplusplus
	}
#endif

#endif /* _CARCOON_H_ */
