#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "util_vec.h"

#ifdef CONFIG_STD_C89
	#define size_t  unsigned int
	#define uint8_t unsigned char
#endif

struct vector *vector_new(size_t elem_size)
{
	struct vector *vec;
	assert((vec = VECTOR_ALLOC(struct vector)));
	vec->esize = elem_size;
	vec->capacity = 1;
	vec->cap_back = 1;
	vec->gw_rate  = VECTOR_GROWTH_RATE;
	assert((vec->data = (uint8_t*)malloc(vec->esize)));
	return vec;
}

void vector_del(struct vector *vec)
{
	if (vec) {
		if (vec->data)
			free(vec->data);
		free(vec);
	}
}

void vector_clear(struct vector *vec)
{
	free(vec->data);
	vec->capacity = 1;
	vec->cap_back = 1;
	vec->gw_rate  = VECTOR_GROWTH_RATE;
	assert((vec->data = (uint8_t*)malloc(vec->esize)));
	vec->size = 0;
}

void vector_push_back(struct vector *vec, void *elem)
{
	if (vec->size == vec->capacity) {
		vec->cap_back = vec->capacity;
		vec->capacity = ceil(((float)vec->capacity) * vec->gw_rate);
		assert((vec->data = (uint8_t*)
			realloc(vec->data, vec->capacity * vec->esize)));
	}
	memcpy(vec->data + vec->size * vec->esize, elem, vec->esize);
	vec->size++;
}

void vector_pop_back(struct vector *vec)
{
	if (!vec->size)
		return;
	vec->size--;
	if (vec->size < vec->cap_back) {
		vec->capacity = vec->cap_back;
		assert((vec->data = (uint8_t*)
			realloc(vec->data, vec->capacity * vec->esize)));
	}
}

void *vector_index(struct vector *vec, size_t index)
{
	return (vec->data + vec->esize * index);
}

void vector_erase(struct vector *vec, size_t index)
{
	if (index >= vec->size)
		return;
	memmove(vec->data + vec->esize * index,
		vec->data + vec->esize * (index + 1),
		vec->esize * (vec->size - index));
	vec->size--;
	if (vec->size < vec->cap_back) {
		vec->capacity = vec->cap_back;
		assert((vec->data = (uint8_t*)
			realloc(vec->data, vec->capacity * vec->esize)));
	}
}

void vector_insert(struct vector *vec, int index, void *elem)
{
	if (vec->size == vec->capacity) {
		vec->cap_back = vec->capacity;
		vec->capacity = ceil(((float)vec->capacity) * vec->gw_rate);
		assert((vec->data = (uint8_t*)
			realloc(vec->data, vec->capacity * vec->esize)));
	}
	memmove(vec->data + vec->esize * (index + 1),
			vec->data + vec->esize * index,
		vec->esize * (vec->size - index));
	memcpy(vec->data + index * vec->esize, elem, vec->esize);
	vec->size++;
}
