#include "catcoon.h"
#include "util_log.h"

void cc_print_info(void)
{
	int auto_tsrmgr_flag = 0;
#ifdef AUTO_TSRMGR
	auto_tsrmgr_flag = 1;
#endif
	utlog_format(UTLOG_INFO, "Catcoon Version: %.1f\n", CATCOON_VER);
	if (auto_tsrmgr_flag)
		utlog_format(UTLOG_INFO, "Auto tensor management: On\n");
	else
		utlog_format(UTLOG_INFO, "Auto tensor management: Off\n");
}
