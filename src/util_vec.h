#ifndef _UTIL_VECTOR_H_
#define _UTIL_VECTOR_H_

#ifdef CONFIG_STD_C89
	#define size_t  unsigned int
	#define uint8_t unsigned char
#else
	#include <stdint.h>
#endif

#define VECTOR_ALLOC(type) ((type*)calloc(1, sizeof(type)))
#define VECTOR_GROWTH_RATE 2.0

struct vector {
	size_t   esize;
	size_t   size;
	size_t   capacity;
	size_t   cap_back;
	float    gw_rate;
	uint8_t *data;
};

#define vector_size(vec)     (vec->size)
#define vector_capacity(vec) (vec->capacity)
#define vector_elemsize(vec) (vec->esize)

struct vector *vector_new(size_t elem_size);

void vector_del(struct vector *vec);

void vector_clear(struct vector *vec);

void vector_push_back(struct vector *vec, void *elem);

void vector_pop_back(struct vector *vec);

void *vector_index(struct vector *vec, size_t index);

void vector_erase(struct vector *vec, size_t index);

void vector_insert(struct vector *vec, int index, void *elem);

#ifdef CONFIG_STD_C89
	#undef size_t
	#undef uint8_t
#endif

#endif /* _UTIL_VECTOR_H_ */
