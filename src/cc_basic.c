#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cc_array.h"
#include "cc_assert.h"
#include "cc_dtype.h"
#include "cc_tensor.h"
#include "cc_tsrmgr.h"
#include "util_list.h"
#include "util_log.h"
#include "cc_basic.h"

/* #include "global_fn_cfg.h" */
extern void (*_activation_relu)
	(void *inp, cc_int32 elems, cc_dtype dt);
extern void (*_activation_relu6)
	(void *inp, cc_int32 elems, cc_dtype dt);
extern void (*_activation_softmax)
	(void *inp, cc_int32 elems, cc_dtype dt);

static cc_int32 _calc_elems(const cc_int32 *shape)
{
	cc_int32 elems;
	elems = *shape;
	while (*++shape)
		elems *= *shape;
	return elems;
}

cc_int32 cc_tensor_elements(cc_tensor_t *tensor)
{
	cc_int32 elems;
	if (!tensor)
		return 0;
	elems = _calc_elems(tensor->shape);
	return elems;
}

void cc_tensor_shape_fix(cc_int32 *shape, cc_int32 elems)
{
	cc_int32 v, i = 0, f = 0, s = 1;
	const cc_int32 *sptr = shape;
	while ((v = *sptr)) {
		if (v == -1) {
#ifdef ENABLE_CC_ASSERT
			cc_assert_zero(f);
#endif
			i = sptr - shape;
			f = (v = 1);
		}
		s *= v;
		sptr++;
	}
	if (s != elems) {
#ifdef ENABLE_CC_ASSERT
			cc_assert(f);
#endif
		shape[i] = elems / s;
	}
}

cc_int32 cc_tensor_dimension(cc_tensor_t *tensor)
{
	cc_int32 dim = 0;
	const cc_int32 *sptr;
	if (!tensor)
		return 0;
	sptr = tensor->shape;
	while (*sptr++)
		++dim;
	return dim;
}

cc_tensor_t *cc_tensor_reshape(cc_tensor_t *tensor, cc_int32 *shape)
{
	cc_int32 elems;
	const cc_int32 *sptr;
	cc_tensor_shape_fix(shape, _calc_elems(tensor->shape));
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(_calc_elems(tensor->shape) - _calc_elems(shape));
#endif
	sptr = shape;
	elems = *shape;
	while (*++sptr)
		elems *= *sptr;
	cc_assert_zero(
		list_del_record(tensor->container, CC_TENSOR_SHAPE));
	cc_assert_zero(
		list_set_record(tensor->container, CC_TENSOR_SHAPE,
			shape, (sptr - shape + 1) * sizeof(cc_int32)));
	cc_assert_ptr(
		tensor->shape = (cc_int32*)
			list_get_record(tensor->container, CC_TENSOR_SHAPE));
	return tensor;
}

int cc_tsrcmp_by_shape(cc_tensor_t *a, cc_tensor_t *b)
{
	int ret = 0;
	const cc_int32 *ptra = a->shape;
	const cc_int32 *ptrb = b->shape;
	while(!(ret = *(cc_int32*)ptra - *(cc_int32*)ptrb) && *ptra) {
		ptra++;
		ptrb++;
	}
	if (ret < 0)
		return -1;
	else if (ret > 0)
		return 1;
	return 0;
}

void cc_print_tensor(cc_tensor_t *tensor)
{
	utlog_format(UTLOG_INFO, "");
	cc_print_array(tensor->data,
		cc_tensor_elements(tensor), *tensor->dtype,
		utlog_get_ostream());
}

cc_tensor_t *cc_cast_tensor(cc_tensor_t *tensor,
		cc_dtype dtype, const char *name)
{
	cc_tensor_t *cast;
	const cc_int32 *sptr = tensor->shape;
	cc_int32 memsize, elems = *tensor->shape;
	while (*++sptr)
		elems *= *sptr;
	memsize = cc_dtype_size(dtype) * elems;
	if (!memsize)
		return NULL;
	cc_assert_ptr(cast = cc_create_tensor(
		tensor->shape, dtype, NULL));
	switch (dtype) {
		case CC_INT8:
			cc_array_cast_int8(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_UINT8:
			cc_array_cast_uint8(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_INT16:
			cc_array_cast_int16(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_UINT16:
			cc_array_cast_uint16(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_INT32:
			cc_array_cast_int32(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_UINT32:
			cc_array_cast_uint32(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_INT64:
			cc_array_cast_int64(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_UINT64:
			cc_array_cast_uint64(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_FLOAT32:
			cc_array_cast_float32(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		case CC_FLOAT64:
			cc_array_cast_float64(cast->data,
				tensor->data, elems, *tensor->dtype);
			break;
		default:
			utlog_format(UTLOG_ERR,
				"cc_cast_tensor: unsupported dtype %x\n",
				dtype);
			cc_free_tensor(cast);
			return NULL;
	}
	list_set_name(cast->container, name);
	cast->name = cast->container->name;
	if (!name || !strcmp(name, tensor->name)) {
#ifdef AUTO_TSRMGR
		cc_tsrmgr_replace(cast);
#else
		cc_free_tensor(tensor);
#endif
		return cast;
	}
#ifdef AUTO_TSRMGR
	cc_tsrmgr_auto_reg(cast);
#endif
	return cast;
}

cc_tensor_t *cc_tensor_by_scalar(cc_tensor_t *tensor,
	char op, void *data, const char *name)
{
	cc_tensor_t *yield;
	cc_int32 elems = *tensor->shape;
	const cc_int32 *sptr = tensor->shape;
	while (*++sptr)
		elems *= *sptr;
	if (!name || !strcmp(name, tensor->name))
		yield = tensor;
	else
		yield = cc_copy_tensor(tensor, name);
	switch (op) {
		case '+':
			cc_array_add_by(
				yield->data, elems, data, *tensor->dtype);
			break;
		case '-':
			cc_array_sub_by(
				yield->data, elems, data, *tensor->dtype);
			break;
		case '*':
			cc_array_mul_by(
				yield->data, elems, data, *tensor->dtype);
			break;
		case '/':
			cc_array_div_by(
				yield->data, elems, data, *tensor->dtype);
			break;
		default:
			utlog_format(UTLOG_ERR,
			"cc_tensor_by_scalar: unsupported operator [%c]\n",
			op);
#ifdef AUTO_TSRMGR
			if (strcmp(name, tensor->name))
				cc_tsrmgr_del(name);
#else
			cc_free_tensor(yield);
#endif
			return NULL;
	}
	return yield;
}

cc_tensor_t *cc_relu(cc_tensor_t *tensor, const char *name)
{
	cc_tensor_t *relu;
	if (!name || !strcmp(name, tensor->name))
		relu = tensor;
	else
		relu = cc_copy_tensor(tensor, name);
	_activation_relu(relu->data,
		cc_tensor_elements(relu), *relu->dtype);
	return relu;
}

cc_tensor_t *cc_relu6(cc_tensor_t *tensor, const char *name)
{
	cc_tensor_t *relu;
	if (!name || !strcmp(name, tensor->name))
		relu = tensor;
	else
		relu = cc_copy_tensor(tensor, name);
	_activation_relu6(relu->data,
		cc_tensor_elements(relu), *relu->dtype);
	return relu;
}

cc_tensor_t *cc_softmax(cc_tensor_t *tensor, const char *name)
{
	cc_tensor_t *softmax;
	if (!name || !strcmp(name, tensor->name))
		softmax = tensor;
	else
		softmax = cc_copy_tensor(tensor, name);
	_activation_softmax(softmax->data,
		cc_tensor_elements(softmax), *softmax->dtype);
	return softmax;
}
