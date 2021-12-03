#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util_log.h"
#ifdef UTLOG_LINUX_API_TIME
#include <sys/time.h>
#endif

static utlog_time_t start_time;
static FILE *user_log_ostream  = NULL;
static int   time_mode_flag_m  = UTLOG_USE_SYS_TIME;
static int   time_mode_flag_o  = UTLOG_USE_RUN_TIME;
static int   highlight_flag    = UTLOG_HIGHLIGHT_ON;
static int   error_action_flag = UTLOG_ERR_ACT_ABORT;

#if   defined (__GNUC__)
	void utlog_set_start_time(void) __attribute__((constructor));
#elif defined (_MSC_VER)
	#pragma data_seg(".CRT$XIU")
	void utlog_set_start_time(void);
	void (*$)(void) = utlog_set_start_time;
	#pragma data_seg
#else
	#define UTLOG_START_TIME_NOT_SET
	#warning "Unknown compiler, start_time may not be initialized"
#endif

#ifndef UTLOG_START_TIME_NOT_SET
void utlog_set_start_time(void)
{
	start_time = utlog_gettime();
}
#endif

void utlog_highlight_on(void)
{
	highlight_flag = UTLOG_HIGHLIGHT_ON;
}

void utlog_highlight_off(void)
{
	highlight_flag = UTLOG_HIGHLIGHT_OFF;
}

void utlog_set_ostream(void *stream)
{
	user_log_ostream = (FILE*)stream;
}

void *utlog_get_ostream(void)
{
	if (!user_log_ostream)
		return UTLOG_DEFAULT_OSTREAM;
	else
		return user_log_ostream;
}

void utlog_set_error_action(int act)
{
	error_action_flag = act;
}

const char *_log_clock_style_str(int logtype)
{
	switch (logtype) {
	case UTLOG_ERR:
		return "\033[;31m";
	case UTLOG_WARN:
		return "\033[;33m";
	case UTLOG_INFO:
		return "\033[0;36m";
	default:
		return NULL;
	}
}

#define UTLOG_FORMAT_RET \
if (logtype == UTLOG_ERR) {                   \
	switch (error_action_flag) {          \
	case UTLOG_ERR_ACT_ABORT:             \
		exit(-1); /* abort() */       \
		break;                        \
	case UTLOG_ERR_ACT_WARNING:           \
		/* print warning info only */ \
		break;                        \
	default:                              \
		break;                        \
	}                                     \
}                                             \
return;

void utlog_use_clk_time(void)
{
	time_mode_flag_m = UTLOG_USE_CLK_TIME;
}

void utlog_use_sys_time(void)
{
	time_mode_flag_m = UTLOG_USE_SYS_TIME;
}

void utlog_use_abs_time(void)
{
	time_mode_flag_o = UTLOG_USE_ABS_TIME;
}

void utlog_use_run_time(void)
{
	time_mode_flag_o = UTLOG_USE_RUN_TIME;
}

utlog_time_t utlog_gettime(void)
{
	utlog_time_t timestamp;
#ifdef UTLOG_LINUX_API_TIME
	struct timeval t;
#endif
	if (time_mode_flag_m == UTLOG_USE_SYS_TIME) {
#ifdef UTLOG_LINUX_API_TIME
		gettimeofday(&t, NULL);
		timestamp = t.tv_usec;
		timestamp = (timestamp / 1e6) + t.tv_sec;
#else
		timestamp = time(NULL);
#endif
	} else {
		timestamp = clock();
		timestamp /= UTLOG_CLK_FRAC;
	}
	return timestamp;
}

void utlog_tag(int logtype)
{
	FILE *log_ostream;
	utlog_time_t timestamp;
	if (time_mode_flag_o == UTLOG_USE_ABS_TIME) {
		timestamp = utlog_gettime();
	} else {
		if (time_mode_flag_m == UTLOG_USE_SYS_TIME)
			timestamp = utlog_gettime() - start_time;
		else
			timestamp = utlog_gettime();
	}
	if (user_log_ostream)
		log_ostream = user_log_ostream;
	else
		log_ostream = UTLOG_DEFAULT_OSTREAM;
	/* fprintf(log_ostream, "[%ld:%ld] : ", t, clk); */
	if (getenv("TERM") &&
		highlight_flag == UTLOG_HIGHLIGHT_ON) {
		fprintf(log_ostream, "[%s%08lf\033[0m]: ",
			_log_clock_style_str(logtype), timestamp);
	} else {
		fprintf(log_ostream, "[%08lf]: ", timestamp);
	}
}

void utlog_format(int logtype, const char *fmt, ...)
{
	va_list ap;
	utlog_tag(logtype);
	va_start(ap, fmt);
	vfprintf((FILE*)utlog_get_ostream(), fmt, ap);
	va_end(ap);
	UTLOG_FORMAT_RET;
}
