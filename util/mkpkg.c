#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "cJSON.h"
#include "parg.h"
#include "catcoon.h"
#include "util_vec.h"

enum composite_type {
	P_NOT_COMPOSITE = 0,
	P_NORMALIZATION,
};

static char *read_text_file(const char *pathname);
static void load_parameter(const char *path, cJSON *item);

static enum composite_type composite_type_chk(const char *name);

static void para_compose_norm(const char *name);

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
		printf("run 'mkpkg -h' for help.\n");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char const *argv[])
{
	char json_path[256];
	char *json_str;
	cJSON *json, *item;
	enum composite_type ctflag = 0;
	struct vector *vec_bn;
	arg_parser(argc, (char**)argv);
	sprintf(json_path, "%s/parameters.json", parameters_path);
	json_str = read_text_file(json_path);
	if (!json_str) {
		printf("error: missing \"parameters.json\"\n");
		return -1;
	}
	json = cJSON_Parse(json_str);
	item = json->child;
	vec_bn = vector_new(sizeof(char**));
	do {
		load_parameter(parameters_path, item);
		ctflag = composite_type_chk(item->string);
		if (ctflag) {
			switch (ctflag) {
			case P_NORMALIZATION:
				vector_push_back(vec_bn, &item->string);
				break;
			default:
				break;
			}
		}
		item = item->next;
	} while(item);
	for (int i = 0; i < vec_bn->size; ++i) {
		para_compose_norm(*(char**)vector_index(vec_bn, i));
	}
	cc_tsrmgr_list();
	cc_tsrmgr_export(parameters_pack);
	cc_clear();
	vector_del(vec_bn);
	free(json_str);
	cJSON_Delete(json);
	return 0;
}

cc_dtype resolve_dtype(const char *s)
{
	int tmp;
	cc_dtype dtype = 0;
	switch (s[0]) {
	case 'i':          /* intXX   */
		break;
	case 'u':          /* uintXX  */
		dtype |= CC_DT_FLAG_SIGN;
		break;
	case 'f':          /* floatXX */
		dtype |= CC_DT_FLAG_POINT;
		dtype |= CC_DT_FLAG_SIGN;
		break;
	default:
		assert(0); /* Unsupported */
		break;
	}
	while (!isdigit(*++s));
	tmp = atoi(s);
	while (tmp >>= 1)
		dtype++;
	return dtype;
}

static void resolve_shape(const char *s, cc_int32 *shape)
{
	do {
		if (*s == ',')
			s++;
		*shape++ =  atoi(s);
		s = strchr(s, ',');
	} while (s);
}

static void load_parameter(const char *path, cJSON *item)
{
	cc_dtype dtype;
	cc_int32 shape[32] = {0};
	char pathname[256];
	sprintf(pathname, "%s/%s", path, item->string);
	dtype = resolve_dtype(item->child->valuestring);
	resolve_shape(item->child->next->valuestring, shape);
	if (shape[0])
		cc_load_bin(pathname, shape, dtype, item->string);
}

static char *read_text_file(const char *pathname)
{
	int len;
	char *buf;
	FILE *fp = fopen(pathname, "r");
	if (!fp)
		return NULL;
	fseek(fp, 0, SEEK_END);
	len = ftell(fp);
	rewind(fp);
	assert((buf = (char*)calloc(len + 1, sizeof(char))));
	assert(fread(buf, len, 1, fp));
	fclose(fp);
	return buf;
}

static enum composite_type composite_type_chk(const char *name)
{
	static int mflag, vflag;
	static char cur_name[256];
	const char *pos;
	if (!cur_name[0]) {
		strcpy(cur_name, name);
		return P_NOT_COMPOSITE;
	} else {
		pos = (const char*)strrchr(name, '.');
		if (strncmp(cur_name, name, (pos - name))) {
			strcpy(cur_name, name);
			mflag = 0;
			vflag = 0;
			return P_NOT_COMPOSITE;
		} else {
			switch (pos[1]) {
			case 'w':
			case 'b':
				break;
			case 'm':
				mflag = 1;
				break;
			case 'v':
				vflag = 1;
				break;
			default:
				/* assert(0); */
				break;
			}
		}
		if (mflag && vflag) {
			mflag = 0;
			vflag = 0;
			cur_name[0] = '\0';
			return P_NORMALIZATION;
		}
	}
	return P_NOT_COMPOSITE;
}

static void para_compose_norm(const char *name)
{
	char buf[256];
	int i, len;
	/* cc_tensor_t *w, *b, *m, *v; */
	cc_tensor_t *tsr[CC_NORM_PARAMETERS];
	cc_int32 shape[] = {-1, 1, 1, 0};
	len = strlen(name);
	memcpy(buf, name, len);
	buf[len] = '\0';
	buf[len - 1] = 'w';
	tsr[CC_NORM_GAMMA] = cc_tensor_reshape(cc_tsrmgr_get(buf), shape);
	buf[len - 1] = 'b';
	tsr[CC_NORM_BETA]  = cc_tensor_reshape(cc_tsrmgr_get(buf), shape);
	buf[len - 1] = 'm';
	tsr[CC_NORM_MEAN]  = cc_tensor_reshape(cc_tsrmgr_get(buf), shape);
	buf[len - 1] = 'v';
	tsr[CC_NORM_VAR]   = cc_tensor_reshape(cc_tsrmgr_get(buf), shape);
	buf[len - 1] = 'e';
	tsr[CC_NORM_EPSILON] =cc_tsrmgr_get(buf);
	if (!tsr[CC_NORM_EPSILON]) {
		switch (*tsr[CC_NORM_GAMMA]->dtype) {
		case CC_FLOAT32:
			tsr[CC_NORM_EPSILON] =
				cc_create_tensor(tsr[CC_NORM_GAMMA]->shape,
					*tsr[CC_NORM_GAMMA]->dtype, buf);
			break;
		default: /* Unsupported dtype */
			assert(0);
			break;
		}
	}
	buf[len - 1] = 'n';
	cc_tensor_stack(tsr, CC_NORM_PARAMETERS, 1, buf);
	for (i = 0; i < CC_NORM_PARAMETERS; ++i) {
		cc_tsrmgr_del(tsr[i]->name);
	}
}
