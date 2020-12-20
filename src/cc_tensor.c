#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cc_macro.h"
#include "cc_assert.h"
#include "cc_tensor.h"
#include "cc_tsrmgr.h"

cc_tensor_t *cc_create(const cc_ssize *shape,
			cc_dtype dtype, const char *name)
{
	const cc_ssize *sptr = shape;
	cc_ssize memsize, elems = *shape;
	cc_tensor_t *tensor;
	while (*++sptr)
		elems *= *sptr;
	memsize = cc_dtype_size(dtype) * elems;
	if (!memsize)
		return NULL;
	cc_assert_alloc(tensor = CC_ALLOC(cc_tensor_t));
	cc_assert_ptr(
		tensor->container = list_new(CC_TENSOR_ITEMS, 0));
	cc_assert_alloc(
		list_alloc(tensor->container, CC_TENSOR_DATA, memsize));
	tensor->data = (unsigned char*)
		list_index(tensor->container, CC_TENSOR_DATA);
	memset(tensor->data, 0, memsize);
	cc_assert_ptr(
		list_set_data(tensor->container, CC_TENSOR_SHAPE,
			shape, (sptr - shape + 1) * sizeof(cc_ssize)));
	cc_assert_ptr(
		tensor->shape = (cc_ssize*)
			list_index(tensor->container, CC_TENSOR_SHAPE));
	cc_assert_ptr(
		list_set_data(tensor->container,
			CC_TENSOR_DTYPE, &dtype, sizeof(cc_dtype)));
	cc_assert_ptr(
		tensor->dtype = (cc_dtype*)
			list_index(tensor->container, CC_TENSOR_DTYPE));
	cc_assert_zero(list_rename(tensor->container, name));
	tensor->name = tensor->container->name;
#ifdef AUTO_TSRMGR
	if (name) {
		cc_tsrmgr_auto_reg(tensor);
	}
#endif
	return tensor;
}

cc_tensor_t *cc_copy(const cc_tensor_t *tensor, const char *name)
{
	cc_tensor_t *copied;
	cc_assert_alloc(copied = CC_ALLOC(cc_tensor_t));
	cc_assert_ptr(
		copied->container = list_clone(tensor->container));
	cc_assert_ptr(
		copied->data = (unsigned char*)
			list_index(copied->container, CC_TENSOR_DATA));
	cc_assert_ptr(
		copied->shape = (cc_ssize*)
			list_index(copied->container, CC_TENSOR_SHAPE));
	cc_assert_ptr(
		copied->dtype = (cc_dtype*)
			list_index(copied->container, CC_TENSOR_DTYPE));
	cc_assert_zero(list_rename(copied->container, name));
	copied->name = copied->container->name;
#ifdef AUTO_TSRMGR
	if (name) {
		cc_tsrmgr_auto_reg(copied);
	}
#endif
	return copied;
}

cc_tensor_t *cc_load(const char *filename)
{
	cc_tensor_t *tensor;
	cc_assert_alloc(tensor = CC_ALLOC(cc_tensor_t));
	if (!(tensor->container = list_import(filename))) {
		free(tensor);
		return NULL;
	}
	cc_assert_ptr(
		tensor->data = (unsigned char*)
			list_index(tensor->container, CC_TENSOR_DATA));
	cc_assert_ptr(
		tensor->shape = (cc_ssize*)
			list_index(tensor->container, CC_TENSOR_SHAPE));
	cc_assert_ptr(
		tensor->dtype = (cc_dtype*)
			list_index(tensor->container, CC_TENSOR_DTYPE));
	tensor->name = tensor->container->name;
#ifdef AUTO_TSRMGR
	cc_tsrmgr_del(tensor->name);
	cc_tsrmgr_auto_reg(tensor);
#endif
	return tensor;
}

cc_tensor_t *cc_load_bin(const char *filename,
	const cc_ssize *shape, cc_dtype dtype, const char *name)
{
	FILE *fp;
	cc_ssize memsize;
	cc_tensor_t *tensor;
	cc_assert_ptr(tensor = cc_create(shape, dtype, name));
	cc_assert_ptr(fp = fopen(filename, "rb"));
	memsize = list_getlen(tensor->container, CC_TENSOR_DATA);
	cc_assert(fread(tensor->data, memsize, 1, fp));
	fclose(fp);
	return tensor;
}

void cc_save(const cc_tensor_t *tensor, const char *filename)
{
	list_export(tensor->container, filename);
}

void cc_free(cc_tensor_t *tensor)
{
	if (tensor) {
		if (tensor->owner)
			*tensor->owner = NULL;
		list_del(tensor->container);
		free(tensor);
	}
}

#define BUFLEN 128
#define BUFLIM 100
void cc_property(const cc_tensor_t *tensor)
{
	char buf[BUFLEN];
	char *bptr = buf;
	const cc_ssize *sptr;
	if (!tensor) {
		utlog_format(UTLOG_WARN, "invalid tensor\n");
		return;
	}
	sptr = tensor->shape;
	while (*sptr) {
		bptr += sprintf(bptr, "%lld, ", *sptr++);
		if ((bptr - buf) > BUFLIM) {
			sprintf(bptr, "..., ");
			break;
		}
	}
	buf[strlen(buf) - 2] = '\0'; /* clear ", " */
	utlog_format(UTLOG_INFO,
		"tensor: \"%s\", dtype: \"%s\", shape: [%s]\n",
		tensor->name, cc_dtype_to_string(*tensor->dtype), buf);
}

cc_ftype cc_getflag(cc_tensor_t *tensor)
{
	return tensor->flag;
}

void cc_setflag(cc_tensor_t *tensor, cc_ftype flag)
{
	tensor->flag = flag;
}

void cc_ptr_bind(cc_tensor_t *tensor, cc_tensor_t **owner)
{
	/* cc_assert_zero(tensor->owner) */
	if (tensor->owner) {
		return;
	}
	tensor->owner = owner;
}

void cc_ptr_unbind(cc_tensor_t *tensor)
{
	tensor->owner = NULL;
}
