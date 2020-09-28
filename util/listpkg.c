#include <stdio.h>
#include <stdlib.h>

#include "parg.h"
#include "catcoon.h"


int main(int argc, char const *argv[])
{
	if (argc != 2) {
		printf("Usage: listpkg package-file\n");
		exit(-1);
	}
	cc_tsrmgr_import(argv[1]);
	cc_tsrmgr_list();
	cc_clear();
	return 0;
}
