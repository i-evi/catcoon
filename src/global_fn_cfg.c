#include "cc_cpufn.h"
#include "global_fn_cfg.h"

void __gfn_check__(void) {return;}

fn_array_set    _array_set    = cc_cpu_array_set;

fn_array_clip_by_value
	_array_clip_by_value  = cc_cpu_array_clip_by_value;

fn_array_add_by _array_add_by = cc_cpu_array_add_by;
fn_array_sub_by _array_sub_by = cc_cpu_array_sub_by;
fn_array_mul_by _array_mul_by = cc_cpu_array_mul_by;
fn_array_div_by _array_div_by = cc_cpu_array_div_by;

fn_array_add_ew _array_add_ew = cc_cpu_array_add_ew;
fn_array_sub_ew _array_sub_ew = cc_cpu_array_sub_ew;
fn_array_mul_ew _array_mul_ew = cc_cpu_array_mul_ew;
fn_array_div_ew _array_div_ew = cc_cpu_array_div_ew;

fn_array_dot_prod  _array_dot_prod = cc_cpu_array_dot_prod;
fn_array_sum       _array_sum      = cc_cpu_array_sum;
fn_array_mean      _array_mean     = cc_cpu_array_mean;

#define GLOBAL_FN_SET_ARRAY_CAST(dtype) \
fn_array_cast_ ## dtype _array_cast_ ## dtype = \
	cc_cpu_array_cast_ ## dtype;

GLOBAL_FN_SET_ARRAY_CAST  (uint8)
GLOBAL_FN_SET_ARRAY_CAST  (uint16)
GLOBAL_FN_SET_ARRAY_CAST  (uint32)
GLOBAL_FN_SET_ARRAY_CAST  (uint64)
GLOBAL_FN_SET_ARRAY_CAST  (int8)
GLOBAL_FN_SET_ARRAY_CAST  (int16)
GLOBAL_FN_SET_ARRAY_CAST  (int32)
GLOBAL_FN_SET_ARRAY_CAST  (int64)
GLOBAL_FN_SET_ARRAY_CAST  (float32)
GLOBAL_FN_SET_ARRAY_CAST  (float64)

fn_activation_relu    _activation_relu    = cc_cpu_activation_relu;
fn_activation_relu6   _activation_relu6   = cc_cpu_activation_relu6;
fn_activation_softmax _activation_softmax = cc_cpu_activation_softmax;

fn_conv2d          _conv2d          = cc_cpu_conv2d;
fn_max_pool2d      _max_pool2d      = cc_cpu_max_pool2d;
fn_max_pool2d      _avg_pool2d      = cc_cpu_avg_pool2d;
fn_batch_norm      _batch_norm      = cc_cpu_batch_norm;
