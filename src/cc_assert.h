#ifndef _CC_ASSERT_H_
#define _CC_ASSERT_H_

#ifdef __cplusplus
	extern "C" {
#endif

#include "util_log.h"

#if defined(ENABLE_CC_ASSERT) /* Assertions */
#define cc_assert(expr) \
	if (!(expr))                                             \
		utlog_format(UTLOG_ERR,                          \
		"Assertion failed @["#expr "] in %s line: %d\n", \
		__FILE__, __LINE__);

#define cc_assert_zero(expr) \
	if (expr)                                                \
		utlog_format(UTLOG_ERR,                          \
		"Assertion failed @["#expr "] in %s line: %d\n", \
		__FILE__, __LINE__);

#define cc_assert_ptr(ptr) \
	if (!(ptr))                                             \
		utlog_format(UTLOG_ERR,                         \
		"Assertion failed @["#ptr "] in %s line: %d\n", \
		__FILE__, __LINE__);

#define cc_assert_alloc(ptr) \
	if (!(ptr))                                                \
		utlog_format(UTLOG_ERR,                            \
		"Memory alloc failed @["#ptr "] in %s line: %d\n", \
		__FILE__, __LINE__);
#else /* not defined(ENABLE_CC_ASSERT) */

#define cc_assert(expr)      (expr)
#define cc_assert_zero(expr) (expr)
#define cc_assert_ptr(ptr)   (ptr)
#define cc_assert_alloc(ptr) (ptr)

#endif

#ifdef __cplusplus
	}
#endif

#endif /* _CC_ASSERT_H_ */
