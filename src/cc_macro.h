#ifndef _CC_MACRO_H_
#define _CC_MACRO_H_

#ifdef CONFIG_STD_C89
	#define CC_INLINE
#else
	#define CC_INLINE inline
#endif

#define CC_ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

#define CC_ALLOC(type) ((type*)calloc(1, sizeof(type)))

#endif /* _CC_MACRO_H_ */
