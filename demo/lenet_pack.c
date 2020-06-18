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
const char *parameters_pack = "lenet_parameters.bin";

int main(int argc, const char *argv[])
{
	cc_int32 i = 0;
	char filepath[256];
	cc_int32 shape_conv1_w[] = {5, 5, 1, 32, 0};
	cc_int32 shape_conv1_b[] = {32, 0};
	cc_int32 shape_conv2_w[] = {5, 5, 32, 64, 0};
	cc_int32 shape_conv2_b[] = {64, 0};
	cc_int32 shape_fc1_w[]   = {1, 1, 3136, 128, 0};
	cc_int32 shape_fc1_b[]   = {128, 0};
	cc_int32 shape_fc2_w[]   = {1, 1, 128, 10, 0};
	cc_int32 shape_fc2_b[]   = {10, 0};

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_conv1_w, CC_FLOAT32, "conv1_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_conv1_b, CC_FLOAT32, "conv1_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_conv2_w, CC_FLOAT32, "conv2_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_conv2_b, CC_FLOAT32, "conv2_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_fc1_w, CC_FLOAT32, "fc1_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_fc1_b, CC_FLOAT32, "fc1_b");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_fc2_w, CC_FLOAT32, "fc2_w");

	sprintf(filepath, "%s/%s",
		parameters_path, parameters_files[i++]);
	cc_load_bin(filepath,
		shape_fc2_b, CC_FLOAT32, "fc2_b");
	cc_tsrmgr_list();
	cc_tsrmgr_export(parameters_pack);
	cc_clear();
	return 0;
}
