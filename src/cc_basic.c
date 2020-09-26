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

#include "global_fn_cfg.h"
#define EXT_ARRAY_CAST_DEFINITION(dtype) \
extern fn_array_cast_ ## dtype _array_cast_ ## dtype;

EXT_ARRAY_CAST_DEFINITION  (uint8)
EXT_ARRAY_CAST_DEFINITION  (uint16)
EXT_ARRAY_CAST_DEFINITION  (uint32)
EXT_ARRAY_CAST_DEFINITION  (uint64)
EXT_ARRAY_CAST_DEFINITION  (int8)
EXT_ARRAY_CAST_DEFINITION  (int16)
EXT_ARRAY_CAST_DEFINITION  (int32)
EXT_ARRAY_CAST_DEFINITION  (int64)
EXT_ARRAY_CAST_DEFINITION  (float32)
EXT_ARRAY_CAST_DEFINITION  (float64)

extern fn_array_set _array_set;
extern fn_array_clip_by_value _array_clip_by_value;

extern fn_array_add_by _array_add_by;
extern fn_array_sub_by _array_sub_by;
extern fn_array_mul_by _array_mul_by;
extern fn_array_div_by _array_div_by;

extern fn_array_add_ew _array_add_ew;
extern fn_array_sub_ew _array_sub_ew;
extern fn_array_mul_ew _array_mul_ew;
extern fn_array_div_ew _array_div_ew;

static cc_int32 _calc_elems(const cc_int32 *shape)
{
	cc_int32 elems;
	elems = *shape;
	while (*++shape)
		elems *= *shape;
	return elems;
}

cc_int32 cc_tensor_elements(const cc_tensor_t *tensor)
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

cc_int32 cc_tensor_dimension(const cc_tensor_t *tensor)
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
		list_erase(tensor->container, CC_TENSOR_SHAPE));
	cc_assert_ptr(
		list_set_data(tensor->container, CC_TENSOR_SHAPE,
			shape, (sptr - shape + 1) * sizeof(cc_int32)));
	cc_assert_ptr(
		tensor->shape = (cc_int32*)
			list_index(tensor->container, CC_TENSOR_SHAPE));
	return tensor;
}

int cc_tsrcmp_by_shape(const cc_tensor_t *a, const cc_tensor_t *b)
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

cc_tensor_t *cc_tensor_stack(cc_tensor_t **tsr,
	cc_int32 ntsr, cc_int32 axis, const char *name)
{
	cc_tensor_t *yield;
	cc_int32 *shape;
	cc_int32 i, dim, size, umem, ymem;
	cc_int32 ncp = 0, nstp = 0;
	cc_int32 off = 0, unit = 1;
	dim = cc_tensor_dimension(tsr[0]);
	cc_assert_ptr(shape =
		(cc_int32*)calloc(dim + 2, sizeof(cc_int32)));
	if (dim == axis) { /* axis <= dim */
		off = 1;
		shape[0] = 1;
	}
	memcpy(shape + off,
		tsr[0]->shape, dim * sizeof(cc_int32));
	shape[dim + off - 1 - axis] *= ntsr;
	cc_assert_ptr(yield =
		cc_create_tensor(shape, *tsr[0]->dtype, name));
	for (i = 0; i < axis; ++i) {
		unit *= tsr[0]->shape[dim - i - 1];
	}
	size = cc_dtype_size(*tsr[0]->dtype);
	ymem = list_getlen(yield->container, CC_TENSOR_DATA);
	umem = unit * size;
	i = 0;
	do {
		memcpy(yield->data + ncp,
			tsr[i++]->data + nstp, umem);
		ncp += umem;
		if (i == ntsr) {
			i = 0;
			nstp += umem;
		}
	} while (ncp != ymem);
	free(shape);
	return yield;
}

static void _cc_print_tensor_indent(int n)
{
	FILE *ostream = (FILE*)utlog_get_ostream();
	while (n--)
		fputc(' ', ostream);
}

#define CC_PRINT_TENSOR_PROC \
  i = 0, fidt = 0, cidt = 0;                      \
  do {                                            \
    while (sops[i]) {                             \
      if (fidt) {                                 \
        _cc_print_tensor_indent(cidt);            \
        fidt = 0;                                 \
      }                                           \
      fputc('[', ostream);                        \
      cidt++;                                     \
      i++;                                        \
    }                                             \
    fputc('[', ostream);                          \
    cc_print_array(tensor->data + npt, lelem,     \
      *tensor->dtype, ostream);                   \
    fputc(']', ostream);                          \
    npt += lsize;                                 \
    if (i - 1 < 0) {                              \
      fputc('\n', ostream);                       \
      _cc_print_tensor_indent(cidt);              \
      break;                                      \
    }                                             \
    sops[i - 1]--;                                \
    if (sops[i - 1]) {                            \
      fputc('\n', ostream);                       \
      _cc_print_tensor_indent(cidt);              \
    } else {                                      \
      fputc(']', ostream);                        \
      cidt--;                                     \
      fidt++;                                     \
      i = i - 2;                                  \
      if (i < 0) {                                \
        fputc('\n', ostream);                     \
        break;                                    \
      }                                           \
      if (sops[i]) {                              \
        sops[i]--;                                \
        if (sops[i]) {                            \
          for (j = ++i; j < (dim - 1); ++j)       \
            sops[j] = sbak[j];                    \
        } else {                                  \
          fputc(']', ostream);                    \
          cidt--;                                 \
          fidt++;                                 \
          while (i > 0) {                         \
            if (--sops[--i]) {                    \
              for (j = ++i; j < (dim - 1); ++j)   \
                sops[j] = sbak[j];                \
                break;                            \
            } else {                              \
              fputc(']', ostream);                \
              cidt--;                             \
              fidt++;                             \
            }                                     \
          }                                       \
        }                                         \
      }                                           \
      fputc('\n', ostream);                       \
    }                                             \
  } while (sops[0]);

void cc_print_tensor(const cc_tensor_t *tensor)
{
	cc_int32 *sops, *sbak;
	int fidt, cidt;
	cc_int32 i, j, dim, lelem, lsize, esize, ssize, npt = 0;
	FILE *ostream = (FILE*)utlog_get_ostream();
	dim   = cc_tensor_dimension(tensor);
	esize = cc_dtype_size(*tensor->dtype);
	lelem = tensor->shape[dim - 1];
	lsize = lelem * esize;
	ssize = (dim + 1) * sizeof(cc_int32);
	cc_assert_ptr(sops =
		(cc_int32*)calloc(dim + 1, sizeof(cc_int32)));
	cc_assert_ptr(sbak =
		(cc_int32*)calloc(dim + 1, sizeof(cc_int32)));
	memcpy(sops, tensor->shape, ssize);
	memcpy(sbak, tensor->shape, ssize);
	sops[dim - 1] = 0;
	sbak[dim - 1] = 0;
	cc_print_tensor_property(tensor);
	CC_PRINT_TENSOR_PROC;
	free(sops);
	free(sbak);
}

void cc_set_tensor(cc_tensor_t *tensor, void *v)
{
	_array_set(tensor->data, 
		cc_tensor_elements(tensor), v, *tensor->dtype);
}

cc_tensor_t *cc_clip_by_value(cc_tensor_t *tensor,
	const void *min, const void *max, const char *name)
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
	_array_clip_by_value(tensor->data,
		cc_tensor_elements(tensor), min, max, *tensor->dtype);
	return yield;
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
		_array_cast_int8(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_UINT8:
		_array_cast_uint8(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_INT16:
		_array_cast_int16(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_UINT16:
		_array_cast_uint16(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_INT32:
		_array_cast_int32(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_UINT32:
		_array_cast_uint32(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_INT64:
		_array_cast_int64(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_UINT64:
		_array_cast_uint64(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_FLOAT32:
		_array_cast_float32(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	case CC_FLOAT64:
		_array_cast_float64(cast->data,
			tensor->data, elems, *tensor->dtype);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_cast_tensor: unsupported dtype %x\n",
			dtype);
		cc_free_tensor(cast);
		return NULL;
	}
	list_rename(cast->container, name);
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
	char op, const void *data, const char *name)
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
		_array_add_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '-':
		_array_sub_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '*':
		_array_mul_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '/':
		_array_div_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
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

cc_tensor_t *cc_tensor_ewop(cc_tensor_t *a,
	cc_tensor_t *b, char op, const char *name)
{
	cc_tensor_t *yield;
	cc_int32 elems = *a->shape;
	const cc_int32 *sptr = a->shape;
	while (*++sptr)
		elems *= *sptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_tsrcmp_by_shape(a, b));
#endif
	if (!name || !strcmp(name, a->name))
		yield = a;
	else
		yield = cc_copy_tensor(a, name);
	switch (op) {
	case '+':
		_array_add_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '-':
		_array_sub_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '*':
		_array_mul_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '/':
		_array_div_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	default:
		utlog_format(UTLOG_ERR,
		"cc_tensor_by_scalar: unsupported operator [%c]\n", op);
#ifdef AUTO_TSRMGR
		if (strcmp(name, a->name))
			cc_tsrmgr_del(name);
#else
		cc_free_tensor(yield);
#endif
		return NULL;
	}
	return yield;
}
