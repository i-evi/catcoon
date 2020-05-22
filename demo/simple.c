#include "catcoon.h"

int main(int argc, const char *argv[])
{
	cc_tensor_t *tensor;
	cc_int32 shape[] = {3, 3, 3, 0};
	tensor = cc_create_tensor(shape, CC_FLOAT32, "tensor0");
	cc_print_tensor_property(tensor);
	cc_free_tensor(tensor);
	return 0;
}
