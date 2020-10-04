#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_OPENMP
	#include <omp.h>
#endif

#include "cc_assert.h"
#include "cc_basic.h"
#include "cc_fmap2d.h"
#include "cc_pad2d.h"
#include "cc_tsrmgr.h"
#include "util_list.h"
#include "util_log.h"
#include "cc_conv2d.h"
#include "cc_dsc2d.h"

#include "global_fn_cfg.h"
extern fn_conv2d       _conv2d;
extern fn_array_add_ew _array_add_ew;
extern fn_array_mul_by _array_mul_by;

cc_tensor_t *cc_dw_conv2d(cc_tensor_t *inp,
		const cc_tensor_t *kernel, const cc_tensor_t *bias,
	cc_int32 s, cc_int32 p, cc_int32 off, const char *name)
{
	cc_tensor_t *inp_pad, *oup = NULL;
	cc_int32 o_ch_size, p_ch_mem_size, o_ch_mem_size, k_ch_mem_size, i;
	cc_int32 shape[CC_CNN2D_SHAPE] = {0};
	char pad_name[CC_CONV2D_PAD_NAME_LEN];
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_tensor_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(cc_tensor_dimension(kernel) - CC_CONV2D_KERNEL_DIM);
	cc_assert_zero(*inp->dtype - *kernel->dtype);
	cc_assert_zero(inp->shape[CC_CNN2D_SHAPE_C]
			- kernel->shape[CC_CONV2D_KERNEL_O]);
#endif
	if (p) {
		sprintf(pad_name, "%s%s",
			inp->name, CC_CONV2D_PAD_NAME_SURFFIX);
		inp_pad = cc_pad2d(inp, p, off, pad_name);
	}
	else
		inp_pad = inp;
#ifdef AUTO_TSRMGR
	oup = cc_tsrmgr_get(name);
#endif
	if (!oup) {
		shape[CC_CNN2D_SHAPE_C] = kernel->shape[CC_CONV2D_KERNEL_O];
		shape[CC_CNN2D_SHAPE_H] = cc_conv2d_shape_calc(
				inp->shape[CC_CNN2D_SHAPE_H],
			kernel->shape[CC_CONV2D_KERNEL_H], s, p);
		shape[CC_CNN2D_SHAPE_W] = cc_conv2d_shape_calc(
				inp->shape[CC_CNN2D_SHAPE_W],
			kernel->shape[CC_CONV2D_KERNEL_W], s, p);
		oup = cc_create_tensor(shape, *inp->dtype, name);
	}
	o_ch_size = oup->shape[CC_CNN2D_SHAPE_W] *
			oup->shape[CC_CNN2D_SHAPE_H];
	o_ch_mem_size = o_ch_size * cc_dtype_size(*oup->dtype);
	p_ch_mem_size = inp_pad->shape[CC_CNN2D_SHAPE_W] *
			inp_pad->shape[CC_CNN2D_SHAPE_H] *
				cc_dtype_size(*inp->dtype);
	k_ch_mem_size = kernel->shape[CC_CONV2D_KERNEL_W] *
			kernel->shape[CC_CONV2D_KERNEL_H] *
			cc_dtype_size(*kernel->dtype);
#ifdef AUTO_TSRMGR
	memset(oup->data, 0,
		list_getlen(oup->container, CC_TENSOR_DATA));
#endif
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
	for (i = 0; i < kernel->shape[CC_CONV2D_KERNEL_O]; ++i) {
		_conv2d((inp_pad->data + i * p_ch_mem_size),
			oup->data + i * o_ch_mem_size,
				inp_pad->shape[CC_CNN2D_SHAPE_W],
				inp_pad->shape[CC_CNN2D_SHAPE_H],
				oup->shape[CC_CNN2D_SHAPE_W],
				oup->shape[CC_CNN2D_SHAPE_H], s, s, 
				kernel->data + (k_ch_mem_size * i),
				kernel->shape[CC_CONV2D_KERNEL_W],
			*kernel->dtype);
	}
	if (!bias){
#ifndef AUTO_TSRMGR
		if (p)
			cc_free_tensor(inp_pad);
#endif
		return oup;
	} else {
		oup = cc_fmap2d_bias(oup, bias, oup->name);
	}
#ifndef AUTO_TSRMGR
	if (p)
		cc_free_tensor(inp_pad);
#endif
	return oup;
}

cc_tensor_t *cc_pw_conv2d(cc_tensor_t *inp, const cc_tensor_t *kernel,
	const cc_tensor_t *bias, const char *name)
{
	cc_uint8 *omp_out_buf = NULL;
	cc_tensor_t *oup = NULL;
	cc_int32 o_ch_size, o_ch_mem_size,
		k_ch_mem_size, k_mem_size, num_omp_threads, i, j;
	cc_int32 shape[CC_CNN2D_SHAPE] = {0};
#ifdef ENABLE_CC_ASSERT
	cc_assert_zero(cc_tensor_dimension(inp) - CC_CNN2D_DIM);
	cc_assert_zero(cc_tensor_dimension(kernel) - CC_CONV2D_KERNEL_DIM);
	cc_assert_zero(*inp->dtype - *kernel->dtype);
	cc_assert_zero(inp->shape[CC_CNN2D_SHAPE_C]
			- kernel->shape[CC_CONV2D_KERNEL_I]);
#endif
#ifdef AUTO_TSRMGR
	oup = cc_tsrmgr_get(name);
#endif
	if (!oup) {
		shape[CC_CNN2D_SHAPE_C] = kernel->shape[CC_CONV2D_KERNEL_O];
		shape[CC_CNN2D_SHAPE_H] = inp->shape[CC_CNN2D_SHAPE_H];
		shape[CC_CNN2D_SHAPE_W] = inp->shape[CC_CNN2D_SHAPE_W];
		oup = cc_create_tensor(shape, *inp->dtype, name);
	}
	o_ch_size = oup->shape[CC_CNN2D_SHAPE_W] *
			oup->shape[CC_CNN2D_SHAPE_H];
	o_ch_mem_size = o_ch_size * cc_dtype_size(*oup->dtype);
	k_ch_mem_size = kernel->shape[CC_CONV2D_KERNEL_W] *
			kernel->shape[CC_CONV2D_KERNEL_H] *
			cc_dtype_size(*kernel->dtype);
	k_mem_size = k_ch_mem_size * kernel->shape[CC_CONV2D_KERNEL_I];
	num_omp_threads = 1;
#ifdef ENABLE_OPENMP
	num_omp_threads = omp_get_max_threads();
#endif
	cc_assert_alloc(omp_out_buf =
		(cc_uint8*)malloc(o_ch_mem_size * num_omp_threads));
#ifdef AUTO_TSRMGR
	memset(oup->data, 0,
		list_getlen(oup->container, CC_TENSOR_DATA));
#endif
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i, j)
#endif
	for (i = 0; i < kernel->shape[CC_CONV2D_KERNEL_O]; ++i) {
		for (j = 0; j < kernel->shape[CC_CONV2D_KERNEL_I]; ++j)
		{
#ifdef ENABLE_OPENMP
		_array_mul_by(
			omp_out_buf + omp_get_thread_num() * o_ch_mem_size,
			o_ch_size, inp->data + o_ch_mem_size * j,
			kernel->data + k_mem_size * i + k_ch_mem_size * j,
			*oup->dtype);
		_array_add_ew(oup->data + o_ch_mem_size * i,
			o_ch_size, oup->data + o_ch_mem_size * i,
			omp_out_buf + omp_get_thread_num() * o_ch_mem_size,
		 *oup->dtype);
#else
		_array_mul_by(omp_out_buf, o_ch_size,
			inp->data + o_ch_mem_size * j,
			kernel->data + k_mem_size * i + k_ch_mem_size * j,
			*oup->dtype);
		_array_add_ew(oup->data + o_ch_mem_size * i, o_ch_size,
				oup->data + o_ch_mem_size * i, omp_out_buf,
			*oup->dtype);
#endif
		}
	}
	free(omp_out_buf);
	if (!bias)
		return oup;
	else
		oup = cc_fmap2d_bias(oup, bias, oup->name);
	return oup;
}