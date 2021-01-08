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
void resnet18(cc_tensor_t *in, cc_tensor_t **out);

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
	resnet18(input, &output);
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

void resnet18(cc_tensor_t *in, cc_tensor_t **out)
{
	static cc_ssize _shape0[] = {-1, 1, 1, 0};
	cc_tensor_t *c_0, *c_1, *c_2, *f_00, *f_01, *f_02, *f_03, *f_04, *f_05,
	*f_06, *f_07, *f_08, *f_09, *f_10, *f_11, *f_12, *f_13, *f_14, *f_15,
	*f_16, *f_17, *f_18, *f_19, *f_20, *f_21, *f_22, *f_23, *f_24, *f_25,
	*f_26, *f_27, *f_28, *f_29, *f_30, *f_31, *f_32, *f_33, *f_34, *f_35,
	*f_36, *f_37, *f_38, *f_39, *f_40, *f_41, *f_42, *f_43, *f_44, *f_45,
	*f_46, *f_47, *f_48, *f_49, *f_50, *f_51, *f_52, *f_53, *f_block0,
	*f_block1, *f_block1_0, *f_block1_0i, *f_block1_1, *f_block2,
	*f_block2_0, *f_block2_0i, *f_block2_0t, *f_block2_1, *f_block3,
	*f_block3_0, *f_block3_0i, *f_block3_0t, *f_block3_1, *f_block4,
	*f_block4_0, *f_block4_0i, *f_block4_0t, *f_block4_1;
	static const char *p_namels[] = {
	"000.w", "000.b", "001.n", "002.w", "002.b", "003.n", "004.w", "004.b",
	"005.n", "006.w", "006.b", "007.n", "008.w", "008.b", "009.n", "010.w",
	"010.b", "011.n", "012.w", "012.b", "013.n", "014.w", "014.b", "015.n",
	"016.w", "016.b", "017.n", "018.w", "018.b", "019.n", "020.w", "020.b",
	"021.n", "022.w", "022.b", "023.n", "024.w", "024.b", "025.n", "026.w",
	"026.b", "027.n", "028.w", "028.b", "029.n", "030.w", "030.b", "031.n",
	"032.w", "032.b", "033.n", "034.w", "034.b", "035.n", "036.w", "036.b",
	"037.n", "038.w", "038.b", "039.n", "040.w", "040.b"};
	static cc_tensor_t *_p[62];
	static int i;
	for (; i < 62; ++i) {
		_p[i] = cc_tsrmgr_get(p_namels[i]);
	}
	f_00 = cc_conv2d(in, _p[0], _p[1], 2, 3, 0, "resnet18/f_00");
	f_01 = cc_batch_norm2d(f_00, _p[2], "resnet18/f_01");
	f_02 = cc_relu(f_01, NULL);
	f_block0 = cc_max_pool2d(f_02, 3, 2, 0, 0, "resnet18/f_block0");
	f_03 = cc_conv2d(f_block0, _p[3], _p[4], 1, 1, 0, "resnet18/f_03");
	f_04 = cc_batch_norm2d(f_03, _p[5], "resnet18/f_04");
	f_05 = cc_relu(f_04, NULL);
	f_06 = cc_conv2d(f_05, _p[6], _p[7], 1, 1, 0, "resnet18/f_06");
	f_block1_0i = cc_batch_norm2d(f_06, _p[8], "resnet18/f_block1_0i");
	f_07 = cc_elemwise(f_block1_0i, f_block0, '+', "resnet18/f_07");
	f_08 = cc_relu(f_07, NULL);
	f_block1_0 = f_08;
	f_09 = cc_conv2d(f_block1_0, _p[9], _p[10], 1, 1, 0, "resnet18/f_09");
	f_10 = cc_batch_norm2d(f_09, _p[11], "resnet18/f_10");
	f_11 = cc_relu(f_10, NULL);
	f_12 = cc_conv2d(f_11, _p[12], _p[13], 1, 1, 0, "resnet18/f_12");
	f_block1_1 = cc_batch_norm2d(f_12, _p[14], "resnet18/f_block1_1");
	f_13 = cc_elemwise(f_block1_1, f_block1_0, '+', "resnet18/f_13");
	f_14 = cc_relu(f_13, NULL);
	f_block1 = f_14;
	f_15 = cc_conv2d(f_block1, _p[15], _p[16], 2, 1, 0, "resnet18/f_15");
	f_16 = cc_batch_norm2d(f_15, _p[17], "resnet18/f_16");
	f_17 = cc_relu(f_16, NULL);
	f_18 = cc_conv2d(f_17, _p[18], _p[19], 1, 1, 0, "resnet18/f_18");
	f_block2_0i = cc_batch_norm2d(f_18, _p[20], "resnet18/f_block2_0i");
	f_block2_0t = cc_conv2d(f_block1, _p[21], _p[22], 2, 0, 0,
	"resnet18/f_block2_0t");
	f_19 = cc_batch_norm2d(f_block2_0t, _p[23], "resnet18/f_19");
	f_20 = cc_elemwise(f_19, f_block2_0i, '+', "resnet18/f_20");
	f_21 = cc_relu(f_20, NULL);
	f_block2_0 = f_21;
	f_22 = cc_conv2d(f_block2_0, _p[24], _p[25], 1, 1, 0, "resnet18/f_22");
	f_23 = cc_batch_norm2d(f_22, _p[26], "resnet18/f_23");
	f_24 = cc_relu(f_23, NULL);
	f_25 = cc_conv2d(f_24, _p[27], _p[28], 1, 1, 0, "resnet18/f_25");
	f_block2_1 = cc_batch_norm2d(f_25, _p[29], "resnet18/f_block2_1");
	f_26 = cc_elemwise(f_block2_1, f_block2_0, '+', "resnet18/f_26");
	f_27 = cc_relu(f_26, NULL);
	f_block2 = f_27;
	f_28 = cc_conv2d(f_block2, _p[30], _p[31], 2, 1, 0, "resnet18/f_28");
	f_29 = cc_batch_norm2d(f_28, _p[32], "resnet18/f_29");
	f_30 = cc_relu(f_29, NULL);
	f_31 = cc_conv2d(f_30, _p[33], _p[34], 1, 1, 0, "resnet18/f_31");
	f_block3_0i = cc_batch_norm2d(f_31, _p[35], "resnet18/f_block3_0i");
	f_block3_0t = cc_conv2d(f_block2, _p[36], _p[37], 2, 0, 0,
	"resnet18/f_block3_0t");
	f_32 = cc_batch_norm2d(f_block3_0t, _p[38], "resnet18/f_32");
	f_33 = cc_elemwise(f_32, f_block3_0i, '+', "resnet18/f_33");
	f_34 = cc_relu(f_33, NULL);
	f_block3_0 = f_34;
	f_35 = cc_conv2d(f_block3_0, _p[39], _p[40], 1, 1, 0, "resnet18/f_35");
	f_36 = cc_batch_norm2d(f_35, _p[41], "resnet18/f_36");
	f_37 = cc_relu(f_36, NULL);
	f_38 = cc_conv2d(f_37, _p[42], _p[43], 1, 1, 0, "resnet18/f_38");
	f_block3_1 = cc_batch_norm2d(f_38, _p[44], "resnet18/f_block3_1");
	f_39 = cc_elemwise(f_block3_1, f_block3_0, '+', "resnet18/f_39");
	f_40 = cc_relu(f_39, NULL);
	f_block3 = f_40;
	f_41 = cc_conv2d(f_block3, _p[45], _p[46], 2, 1, 0, "resnet18/f_41");
	f_42 = cc_batch_norm2d(f_41, _p[47], "resnet18/f_42");
	f_43 = cc_relu(f_42, NULL);
	f_44 = cc_conv2d(f_43, _p[48], _p[49], 1, 1, 0, "resnet18/f_44");
	f_block4_0i = cc_batch_norm2d(f_44, _p[50], "resnet18/f_block4_0i");
	f_block4_0t = cc_conv2d(f_block3, _p[51], _p[52], 2, 0, 0,
	"resnet18/f_block4_0t");
	f_45 = cc_batch_norm2d(f_block4_0t, _p[53], "resnet18/f_45");
	f_46 = cc_elemwise(f_45, f_block4_0i, '+', "resnet18/f_46");
	f_47 = cc_relu(f_46, NULL);
	f_block4_0 = f_47;
	f_48 = cc_conv2d(f_block4_0, _p[54], _p[55], 1, 1, 0, "resnet18/f_48");
	f_49 = cc_batch_norm2d(f_48, _p[56], "resnet18/f_49");
	f_50 = cc_relu(f_49, NULL);
	f_51 = cc_conv2d(f_50, _p[57], _p[58], 1, 1, 0, "resnet18/f_51");
	f_block4_1 = cc_batch_norm2d(f_51, _p[59], "resnet18/f_block4_1");
	f_52 = cc_elemwise(f_block4_1, f_block4_0, '+', "resnet18/f_52");
	f_53 = cc_relu(f_52, NULL);
	f_block4 = f_53;
	c_0 = cc_avg_pool2d(f_block4, 7, 7, 0, 0, "resnet18/c_0");
	c_1 = cc_reshape(c_0, _shape0);
	c_2 = cc_fully_connected(c_1, _p[60], _p[61], "resnet18/c_2");
	*out = cc_softmax(c_2, NULL);
}
