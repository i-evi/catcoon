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

cc_tensor_t *cc_tsrmgr_get(const char *name);

int cc_tsrmgr_status(void);

void cc_tsrmgr_list(void);

/*
 * To pack a tensor's container:
 * name | dtype | shape | data 
 */
struct list *cc_tsrmgr_pack();
void cc_tsrmgr_unpack(struct list *tls);

void cc_tsrmgr_export(const char *filename);
void cc_tsrmgr_import(const char *filename);

/* AUTO_TSRMGR */
#define cc_tsrmgr_auto_reg(tensor) \
	if (!cc_tsrmgr_status())   \
		cc_tsrmgr_init();  \
	cc_tsrmgr_reg(tensor);

enum cc_tsrmgr_ctrl {
	CC_GC_CLEAN
};

void cc_tsrmgr_gc(enum cc_tsrmgr_ctrl ctrl);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_TSRMGR_H_ */
