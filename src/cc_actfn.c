#include <string.h>

#include "cc_basic.h"
#include "cc_actfn.h"

#include "global_fn_cfg.h"
extern fn_activation_relu    _activation_relu;
extern fn_activation_relu6   _activation_relu6;
extern fn_activation_softmax _activation_softmax;

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
