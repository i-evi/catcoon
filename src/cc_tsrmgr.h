#ifndef _CC_TSRMGR_H_
#define _CC_TSRMGR_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "cc_tensor.h"

void cc_tsrmgr_init(void);
void cc_tsrmgr_clear(void);

void cc_tsrmgr_reg(cc_tensor_t *tensor);
void cc_tsrmgr_del(const char *name);
void cc_tsrmgr_replace(cc_tensor_t *tensor);

int cc_tsrmgr_status(void);

cc_tensor_t *cc_tsrmgr_get(const char *name);

void cc_tsrmgr_list(void);

/* AUTO_TSRMGR */
#define cc_tsrmgr_auto_reg(tensor) \
	if (!cc_tsrmgr_status())   \
		cc_tsrmgr_init();  \
	cc_tsrmgr_reg(tensor);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_TSRMGR_H_ */
