#include <stdio.h>
#include <stdlib.h>

#include "catcoon.h"

#define INPUT_W 28
#define INPUT_H 28

const char *parameters_files[]={
	"conv1_w.bin",
	"conv1_b.bin",
	"conv2_w.bin",
	"conv2_b.bin",
	"fc1_w.bin",
	"fc1_b.bin",
	"fc2_w.bin",
	"fc2_b.bin",
};
const char *parameters_path = "/home/evi/catcoon-pytorch-model/lenet/dump";

int main(int argc, const char *argv[])
{
	cc_int32 i, j;
	cc_float32 v;
	char filepath[256];
	/* Image */
	utim_image_t *img, *img_read;
	/* Layers */
	cc_tensor_t *input,*l1, *l1_pool, *l2, *l2_pool, *l2_flat, *l3, *l4;
	/* Parameters */
	cc_tensor_t *conv1_w, *conv1_b,
		*conv2_w, *conv2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
	cc_int32 shape_conv1_w[] = {5, 5, 1, 32, 0};
	cc_int32 shape_conv1_b[] = {32, 0};
	cc_int32 shape_conv2_w[] = {5, 5, 32, 64, 0};
	cc_int32 shape_conv2_b[] = {64, 0};
	cc_int32 shape_flat[]    = {-1, 1, 1, 0};
	cc_int32 shape_fc1_w[]   = {1, 1, 3136, 128, 0};
	cc_int32 shape_fc1_b[]   = {128, 0};
	cc_int32 shape_fc2_w[]   = {1, 1, 128, 10, 0};
	cc_int32 shape_fc2_b[]   = {10, 0};

	if (argc < 2) {
		fprintf(stderr, "usage: lenet [filename]\n");
		exit(255);
	}

	img_read = utim_read(argv[1]);
	utim_img2gray(img_read);
	img = utim_resize(img_read, INPUT_H, INPUT_W, 0);
	input = cc_image2tensor(img, "input");
	utim_free_image(img);
	utim_free_image(img_read);
	input = cc_cast_tensor(input, CC_FLOAT32, "input");
	v = 255.;
	input = cc_tensor_by_scalar(input, '/', &v, "input");

	i = 0;
	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	conv1_w = cc_load_bin(filepath,
		shape_conv1_w, CC_FLOAT32, "conv1_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	conv1_b = cc_load_bin(filepath,
		shape_conv1_b, CC_FLOAT32, "conv1_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	conv2_w = cc_load_bin(filepath,
		shape_conv2_w, CC_FLOAT32, "conv2_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	conv2_b = cc_load_bin(filepath,
		shape_conv2_b, CC_FLOAT32, "conv2_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	fc1_w = cc_load_bin(filepath,
		shape_fc1_w, CC_FLOAT32, "fc1_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	fc1_b = cc_load_bin(filepath,
		shape_fc1_b, CC_FLOAT32, "fc1_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	fc2_w = cc_load_bin(filepath,
		shape_fc2_w, CC_FLOAT32, "fc2_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	fc2_b = cc_load_bin(filepath,
		shape_fc2_b, CC_FLOAT32, "fc2_b");

	l1 = cc_conv2d(input, conv1_w, conv1_b, 1, 2, 0, "l1_conv");

	l1 = cc_relu(l1, NULL);

	l1_pool = cc_max_pool2d(l1, 2, "l1_pool");

	l2 = cc_conv2d(l1_pool, conv2_w, conv2_b, 1, 2, 0, "l2_conv");

	l2 = cc_relu(l2, NULL);

	l2_pool = cc_max_pool2d(l2, 2, "l2_pool");

	/* l2_flat = cc_fmap2d_flat(l2_pool, "l2_flat"); */
	l2_flat = cc_tensor_reshape(l2_pool, shape_flat);

	/* l3 = cc_conv2d(l2_flat, fc1_w, fc1_b, 1, 0, 0, "fc1"); */
	l3 = cc_fully_connected(l2_flat, fc1_w, fc1_b, "fc1");

	l3 = cc_relu(l3, NULL);

	/* l4 = cc_conv2d(l3, fc2_w, fc2_b, 1, 0, 0, "fc2"); */
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
