#include <stdio.h>
#include <stdlib.h>

#include "catcoon.h"

#define INPUT_W 28
#define INPUT_H 28

const char *parameters_path = "./lenet_parameters.bin";

int main(int argc, const char *argv[])
{
	cc_int32 i, j;
	cc_float32 v;
	/* Image */
	utim_image_t *img, *img_read;
	/* Layers */
	cc_tensor_t *input,*l1, *l1_pool, *l2, *l2_pool, *l2_flat, *l3, *l4;
	/* Parameters */
	cc_tensor_t *conv1_w, *conv1_b,
		*conv2_w, *conv2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
	cc_int32 shape_flat[]    = {-1, 1, 1, 0};

	if (argc < 2) {
		fprintf(stderr, "usage: lenet [filename]\n");
		exit(255);
	}

	/* load parameters */
	cc_tsrmgr_import(parameters_path);
	conv1_w = cc_tsrmgr_get("conv1_w");
	conv1_b = cc_tsrmgr_get("conv1_b");
	conv2_w = cc_tsrmgr_get("conv2_w");
	conv2_b = cc_tsrmgr_get("conv2_b");
	fc1_w = cc_tsrmgr_get("fc1_w");
	fc1_b = cc_tsrmgr_get("fc1_b");
	fc2_w = cc_tsrmgr_get("fc2_w");
	fc2_b = cc_tsrmgr_get("fc2_b");

	img_read = utim_read(argv[1]);
	utim_img2gray(img_read);
	img = utim_resize(img_read, INPUT_H, INPUT_W, 0);
	input = cc_image2tensor(img, "input");
	utim_free_image(img);
	utim_free_image(img_read);
	input = cc_cast_tensor(input, CC_FLOAT32, "input");
	v = 255.;
	input = cc_tensor_by_scalar(input, '/', &v, "input");

	l1 = cc_conv2d(input, conv1_w, conv1_b, 1, 2, 0, "l1_conv");

	l1 = cc_relu(l1, NULL);

	l1_pool = cc_max_pool2d(l1, 2, "l1_pool");

	l2 = cc_conv2d(l1_pool, conv2_w, conv2_b, 1, 2, 0, "l2_conv");

	l2 = cc_relu(l2, NULL);

	l2_pool = cc_max_pool2d(l2, 2, "l2_pool");

	/* l2_flat = cc_fmap2d_flat(l2_pool, "l2_flat"); */
	l2_flat = cc_tensor_reshape(l2_pool, shape_flat);

	l3 = cc_fully_connected(l2_flat, fc1_w, fc1_b, "fc1");

	l3 = cc_relu(l3, NULL);

	l4 = cc_fully_connected(l3, fc2_w, fc2_b, "fc2");

	l4 = cc_softmax(l4, NULL);

	v = 0.;
	for (i = 0, j = 0; i < 10; ++i) {
		if (*((cc_float32*)l4->data + i) > v) {
			v = *((cc_float32*)l4->data + i);
			j = i;
		}
		printf("[%d]: %f\n", i, *((cc_float32*)l4->data + i));
	}
	printf("Result of \"%s\": [%d]\n", argv[1], j);
	cc_tsrmgr_list();
	cc_clear();
	return 0;
}
