#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parg.h"
#include "catcoon.h"

#define INPUT_W 224
#define INPUT_H 224

char image_path[128];
char parameters_path[128];

void arg_parser(int argc, char* const argv[]);
void vgg16(cc_tensor_t *in, cc_tensor_t **out);

int main(int argc, const char *argv[])
{
	cc_int32 i, j;
	cc_float32 v;
	cc_tensor_t *input, *output;
	arg_parser(argc, (char**)argv);
	cc_tsrmgr_import(parameters_path);
	input = cc_imread(image_path,
		INPUT_H, INPUT_W, CC_RESIZE_LINEAR, CC_RGB, "input");
	cc_image_norm(input, CC_IM_NORM_MINMAX_RGB);
	cc_image_norm(input, CC_IM_NORM_ZSCORE_RGB);
	vgg16(input, &output);
	cc_tsrmgr_list();
	v = 0.;
	for (i = 0, j = 0; i < 1000; ++i) {
		if (*((cc_float32*)output->data + i) > v) {
			v = *((cc_float32*)output->data + i);
			j = i;
		}
	}
	printf("Result: [%d]\n", j);
	cc_clear();
	return 0;
}

void arg_parser(int argc, char* const argv[])
{
	struct parg_state ps;
	int c;
	parg_init(&ps);
	while ((c = parg_getopt(&ps, argc, argv, "hp:v")) != -1) {
		switch (c) {
		case 1:
			strcpy(image_path, ps.optarg);
			break;
		case 'h':
			printf("Usage: [-h] -p parameters-path image-file\n");
			printf("-h: Displays this message\n");
			printf("-p: Choose parameters file path\n");
			exit(EXIT_SUCCESS);
			break;
		case 'p':
			strcpy(parameters_path, ps.optarg);
			break;
		case '?':
			if (ps.optopt == 'p') {
				printf("option -p requires an argument\n");
			} else {
				printf("unknown option -%c\n", ps.optopt);
			}
			exit(EXIT_FAILURE);
			break;
		default:
			printf("error: unhandled option -%c\n", c);
			exit(EXIT_FAILURE);
			break;
		}
	}
	if (!strlen(image_path) || !strlen(parameters_path)) {
		printf("error: incomplete argument\n");
		printf("run 'lenet -h' for help.\n");
		exit(EXIT_FAILURE);
	}
}

void vgg16(cc_tensor_t *in, cc_tensor_t **out)
{
	static cc_ssize __shape0[] = {-1, 1, 1, 0};
	static const char *p_namels[] = {
		"000.w", "000.b", "001.w", "001.b", "002.w", "002.b", "003.w",
		"003.b", "004.w", "004.b", "005.w", "005.b", "006.w", "006.b",
		"007.w", "007.b", "008.w", "008.b", "009.w", "009.b", "010.w",
		"010.b", "011.w", "011.b", "012.w", "012.b", "013.w", "013.b",
		"014.w", "014.b", "015.w", "015.b"};
	static cc_tensor_t *__pls[32];
	cc_tensor_t *l1, *l2, *l3, *l4, *l5, *l6, *l7, *l8, *l9, *l10, *l11,
	*l12, *l13, *l14, *l15, *l16, *l17, *l18, *l19, *l20, *l21, *l22, *l23,
	*l24, *l25, *l26, *l27, *l28, *l29, *l30, *l31, *l32, *l33, *l34, *l35,
	*l36, *l37;
	static int i;
	for (; i < 32; ++i) {
		__pls[i] = cc_tsrmgr_get(p_namels[i]);
	}
	l1 = cc_conv2d(in, __pls[0], __pls[1], 1, 1, 0, "vgg16/l1");
	l2 = cc_relu(l1, NULL);
	l3 = cc_conv2d(l2, __pls[2], __pls[3], 1, 1, 0, "vgg16/l3");
	l4 = cc_relu(l3, NULL);
	l5 = cc_max_pool2d(l4, 2, 2, 0, 0, "vgg16/l5");
	l6 = cc_conv2d(l5, __pls[4], __pls[5], 1, 1, 0, "vgg16/l6");
	l7 = cc_relu(l6, NULL);
	l8 = cc_conv2d(l7, __pls[6], __pls[7], 1, 1, 0, "vgg16/l8");
	l9 = cc_relu(l8, NULL);
	l10 = cc_max_pool2d(l9, 2, 2, 0, 0, "vgg16/l10");
	l11 = cc_conv2d(l10, __pls[8], __pls[9], 1, 1, 0, "vgg16/l11");
	l12 = cc_relu(l11, NULL);
	l13 = cc_conv2d(l12, __pls[10], __pls[11], 1, 1, 0, "vgg16/l13");
	l14 = cc_relu(l13, NULL);
	l15 = cc_conv2d(l14, __pls[12], __pls[13], 1, 1, 0, "vgg16/l15");
	l16 = cc_relu(l15, NULL);
	l17 = cc_max_pool2d(l16, 2, 2, 0, 0, "vgg16/l17");
	l18 = cc_conv2d(l17, __pls[14], __pls[15], 1, 1, 0, "vgg16/l18");
	l19 = cc_relu(l18, NULL);
	l20 = cc_conv2d(l19, __pls[16], __pls[17], 1, 1, 0, "vgg16/l20");
	l21 = cc_relu(l20, NULL);
	l22 = cc_conv2d(l21, __pls[18], __pls[19], 1, 1, 0, "vgg16/l22");
	l23 = cc_relu(l22, NULL);
	l24 = cc_max_pool2d(l23, 2, 2, 0, 0, "vgg16/l24");
	l25 = cc_conv2d(l24, __pls[20], __pls[21], 1, 1, 0, "vgg16/l25");
	l26 = cc_relu(l25, NULL);
	l27 = cc_conv2d(l26, __pls[22], __pls[23], 1, 1, 0, "vgg16/l27");
	l28 = cc_relu(l27, NULL);
	l29 = cc_conv2d(l28, __pls[24], __pls[25], 1, 1, 0, "vgg16/l29");
	l30 = cc_relu(l29, NULL);
	l31 = cc_max_pool2d(l30, 2, 2, 0, 0, "vgg16/l31");
	l32 = cc_reshape(l31, __shape0);
	l33 = cc_fully_connected(l32, __pls[26], __pls[27], "vgg16/l33");
	l34 = cc_relu(l33, NULL);
	l35 = cc_fully_connected(l34, __pls[28], __pls[29], "vgg16/l35");
	l36 = cc_relu(l35, NULL);
	l37 = cc_fully_connected(l36, __pls[30], __pls[31], "vgg16/l37");
	*out = cc_softmax(l37, NULL);
}
