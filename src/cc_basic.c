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

static cc_ssize _calc_elems(const cc_ssize *shape)
{
	cc_ssize elems;
	elems = *shape;
	while (*++shape)
		elems *= *shape;
	return elems;
}

cc_ssize cc_elements(const cc_tensor_t *tensor)
{
	cc_ssize elems;
	if (!tensor)
		return 0;
	elems = _calc_elems(tensor->shape);
	return elems;
}

void cc_shape_fix(cc_ssize *shape, cc_ssize elems)
{
	cc_ssize v, i = 0, f = 0, s = 1;
	const cc_ssize *sptr = shape;
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
	if (s != elems || f) {
		shape[i] = elems / s;
	}
}

cc_ssize cc_dimension(const cc_tensor_t *tensor)
{
	cc_ssize dim = 0;
	const cc_ssize *sptr;
	if (!tensor)
		return 0;
	sptr = tensor->shape;
	while (*sptr++)
		++dim;
	return dim;
}

cc_tensor_t *cc_reshape(cc_tensor_t *tensor, cc_ssize *shape)
{
	cc_ssize elems;
	const cc_ssize *sptr;
	cc_shape_fix(shape, _calc_elems(tensor->shape));
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
			shape, (sptr - shape + 1) * sizeof(cc_ssize)));
	cc_assert_ptr(
		tensor->shape = (cc_ssize*)
			list_index(tensor->container, CC_TENSOR_SHAPE));
	return tensor;
}

int cc_compare_by_shape(const cc_tensor_t *a, const cc_tensor_t *b)
{
	int ret = 0;
	const cc_ssize *ptra = a->shape;
	const cc_ssize *ptrb = b->shape;
	while(!(ret = *(cc_ssize*)ptra - *(cc_ssize*)ptrb) && *ptra) {
		ptra++;
		ptrb++;
	}
	if (ret < 0)
		return -1;
	else if (ret > 0)
		return 1;
	return 0;
}

cc_tensor_t *cc_stack(cc_tensor_t **tsr,
	cc_ssize ntsr, cc_ssize axis, const char *name)
{
	cc_tensor_t *yield;
	cc_ssize *shape;
	cc_ssize i, dim, size, umem, ymem;
	cc_ssize ncp = 0, nstp = 0;
	cc_ssize off = 0, unit = 1;
	dim = cc_dimension(tsr[0]);
	cc_assert_ptr(shape =
		(cc_ssize*)calloc(dim + 2, sizeof(cc_ssize)));
	if (dim == axis) { /* axis <= dim */
		off = 1;
		shape[0] = 1;
	}
	memcpy(shape + off,
		tsr[0]->shape, dim * sizeof(cc_ssize));
	shape[dim + off - 1 - axis] *= ntsr;
	cc_assert_ptr(yield =
		cc_create(shape, *tsr[0]->dtype, name));
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

cc_tensor_t *cc_concat(cc_tensor_t **tsr,
	cc_ssize ntsr, cc_ssize axis, const char *name)
{
	cc_tensor_t *yield;
	cc_ssize *shape;
	cc_ssize i, j, dim, raxis, umem, cmem;
	cc_ssize ncp = 0, nseg = 1, unit = 1;
	dim = cc_dimension(tsr[0]);
	cc_assert_ptr(shape =
		(cc_ssize*)calloc(dim + 1, sizeof(cc_ssize)));
	memcpy(shape, tsr[0]->shape, dim * sizeof(cc_ssize));
	raxis = dim - 1 - axis;
	shape[raxis] = 0;
	for (i = 0; i < ntsr; ++i) {
		shape[raxis] += tsr[i]->shape[raxis];
	}
	cc_assert_ptr(yield =
		cc_create(shape, *tsr[0]->dtype, name));
	for (i = 0; i < dim - axis - 1; ++i) {
		nseg *= tsr[0]->shape[i];
	}
	for (i = 0; i < axis; ++i) {
		unit *= tsr[0]->shape[dim - i - 1];
	}
	umem = unit * cc_dtype_size(*tsr[0]->dtype);
	for (i = 0; i < nseg; ++i) {
		for (j = 0; j < ntsr; ++j) {
			cmem = umem * tsr[j]->shape[raxis];
			memcpy(yield->data + ncp,
				tsr[j]->data + i * cmem, cmem);
			ncp += cmem;
		}
	}
	free(shape);
	return yield;
}

static void _cc_print_indent(int n)
{
	FILE *ostream = (FILE*)utlog_get_ostream();
	while (n--)
		fputc(' ', ostream);
}

#define CC_PRINT_PROC \
  i = 0, fidt = 0, cidt = 0;                      \
  do {                                            \
    while (sops[i]) {                             \
      if (fidt) {                                 \
        _cc_print_indent(cidt);                   \
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
      _cc_print_indent(cidt);                     \
      break;                                      \
    }                                             \
    sops[i - 1]--;                                \
    if (sops[i - 1]) {                            \
      fputc('\n', ostream);                       \
      _cc_print_indent(cidt);                     \
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

void cc_print(const cc_tensor_t *tensor)
{
	cc_ssize *sops, *sbak;
	int fidt, cidt;
	cc_ssize i, j, dim, lelem, lsize, esize, ssize, npt = 0;
	FILE *ostream = (FILE*)utlog_get_ostream();
	dim   = cc_dimension(tensor);
	esize = cc_dtype_size(*tensor->dtype);
	lelem = tensor->shape[dim - 1];
	lsize = lelem * esize;
	ssize = (dim + 1) * sizeof(cc_ssize);
	cc_assert_ptr(sops =
		(cc_ssize*)calloc(dim + 1, sizeof(cc_ssize)));
	cc_assert_ptr(sbak =
		(cc_ssize*)calloc(dim + 1, sizeof(cc_ssize)));
	memcpy(sops, tensor->shape, ssize);
	memcpy(sbak, tensor->shape, ssize);
	sops[dim - 1] = 0;
	sbak[dim - 1] = 0;
	cc_property(tensor);
	CC_PRINT_PROC;
	free(sops);
	free(sbak);
}

void cc_set_value(cc_tensor_t *tensor, void *v)
{
	cc_array_set(tensor->data, 
		cc_elements(tensor), v, *tensor->dtype);
}

cc_tensor_t *cc_clip_by_value(cc_tensor_t *tensor,
	const void *min, const void *max, const char *name)
{
	cc_tensor_t *yield;
	cc_ssize elems = *tensor->shape;
	const cc_ssize *sptr = tensor->shape;
	while (*++sptr)
		elems *= *sptr;
	if (!name || !strcmp(name, tensor->name))
		yield = tensor;
	else
		yield = cc_copy(tensor, name);
	cc_array_clip_by_value(tensor->data,
		cc_elements(tensor), min, max, *tensor->dtype);
	return yield;
}

cc_tensor_t *cc_cast(cc_tensor_t *tensor,
		cc_dtype dtype, const char *name)
{
	cc_tensor_t *cast;
	const cc_ssize *sptr = tensor->shape;
	cc_ssize memsize, elems = *tensor->shape;
	while (*++sptr)
		elems *= *sptr;
	memsize = cc_dtype_size(dtype) * elems;
	if (!memsize)
		return NULL;
	cc_assert_ptr(cast = cc_create(tensor->shape, dtype, NULL));
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
			"cc_cast: unsupported dtype %x\n",
			dtype);
		cc_free(cast);
		return NULL;
	}
	if (!name)
		return cast;
	list_rename(cast->container, name);
	cast->name = cast->container->name;
	if (!strcmp(name, tensor->name)) {
#ifdef AUTO_TSRMGR
		cc_tsrmgr_replace(cast);
#else
		cc_free(tensor);
#endif
		return cast;
	}
#ifdef AUTO_TSRMGR
	cc_tsrmgr_auto_reg(cast);
#endif
	return cast;
}

cc_tensor_t *cc_scalar(cc_tensor_t *tensor,
	char op, const void *data, const char *name)
{
	cc_tensor_t *yield;
	cc_ssize elems = *tensor->shape;
	const cc_ssize *sptr = tensor->shape;
	while (*++sptr)
		elems *= *sptr;
	if (!name || !strcmp(name, tensor->name))
		yield = tensor;
	else
		yield = cc_copy(tensor, name);
	switch (op) {
	case '+':
		cc_array_add_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '-':
		cc_array_sub_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '*':
		cc_array_mul_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	case '/':
		cc_array_div_by(yield->data, elems,
			yield->data, data, *tensor->dtype);
		break;
	default:
		utlog_format(UTLOG_ERR,
		"cc_scalar: unsupported operator [%c]\n",
		op);
#ifdef AUTO_TSRMGR
		if (strcmp(name, tensor->name))
			cc_tsrmgr_del(name);
#else
		cc_free(yield);
#endif
		return NULL;
	}
	return yield;
}

cc_tensor_t *cc_elemwise(cc_tensor_t *a,
	cc_tensor_t *b, char op, const char *name)
{
	cc_tensor_t *yield;
	cc_ssize elems = *a->shape;
	const cc_ssize *sptr = a->shape;
	while (*++sptr)
		elems *= *sptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_compare_by_shape(a, b));
#endif
	if (!name || !strcmp(name, a->name))
		yield = a;
	else
		yield = cc_copy(a, name);
	switch (op) {
	case '+':
		cc_array_add_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '-':
		cc_array_sub_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '*':
		cc_array_mul_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	case '/':
		cc_array_div_ew(yield->data, elems,
			yield->data, b->data, *yield->dtype);
		break;
	default:
		utlog_format(UTLOG_ERR,
		"cc_scalar: unsupported operator [%c]\n", op);
#ifdef AUTO_TSRMGR
		if (strcmp(name, a->name))
			cc_tsrmgr_del(name);
#else
		cc_free(yield);
#endif
		return NULL;
	}
	return yield;
}

cc_tensor_t *cc_from_array(void *arr,
	const cc_ssize *shape, cc_dtype dtype, const char *name)
{
	cc_ssize memsize;
	cc_tensor_t *tensor;
	cc_assert_ptr(tensor = cc_create(shape, dtype, name));
	memsize = list_getlen(tensor->container, CC_TENSOR_DATA);
	memcpy(tensor->data, arr, memsize);
	return tensor;
}
