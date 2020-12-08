#include "catcoon.h"

int main(int argc, const char *argv[])
{
	cc_tensor_t *tensor;
	cc_ssize shape[] = {3, 3, 3, 0};
	tensor = cc_create(shape, CC_FLOAT32, "tensor0");
	cc_property(tensor);
	cc_free(tensor);
	return 0;
}
