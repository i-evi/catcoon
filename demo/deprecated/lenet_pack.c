#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parg.h"
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

char parameters_path[128];
char parameters_pack[128];

void arg_parser(int argc, char* const argv[])
{
	struct parg_state ps;
	int c;
	parg_init(&ps);
	while ((c = parg_getopt(&ps, argc, argv, "hp:o:v")) != -1) {
		switch (c) {
		case 'h':
			printf("Usage: [-h] -p pathname -o filename\n");
			printf("-h: Displays this message\n");
			printf("-p: Choose parameters file path\n");
			printf("-o: Output file name\n");
			exit(EXIT_SUCCESS);
			break;
		case 'p':
			strcpy(parameters_path, ps.optarg);
			break;
		case 'o':
			strcpy(parameters_pack, ps.optarg);
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
	if (!strlen(parameters_path) || !strlen(parameters_pack)) {
		printf("error: incomplete argument\n");
		printf("run 'lenet_pack -h' for help.\n");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, const char *argv[])
{
	cc_int32 i = 0;
	char filepath[256];
	cc_int32 shape_conv1_w[] = {32, 1, 5, 5, 0};
	cc_int32 shape_conv1_b[] = {32, 0};
	cc_int32 shape_conv2_w[] = {64, 32, 5, 5, 0};
	cc_int32 shape_conv2_b[] = {64, 0};
	cc_int32 shape_fc1_w[]   = {128, 3136, 1, 1, 0};
	cc_int32 shape_fc1_b[]   = {128, 0};
	cc_int32 shape_fc2_w[]   = {10, 128, 1, 1, 0};
	cc_int32 shape_fc2_b[]   = {10, 0};

	arg_parser(argc, (char**)argv);

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
