#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util_log.h"

static int highlight_flag = UTLOG_HIGHLIGHT_ON;
static FILE *user_log_ostream = NULL;
static int error_action_flag  = UTLOG_ERR_ACT_ABORT;

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

static int _account_symbol(char *s, char ch)
{
	int c = 0;
	while(*s)
		if(*s++ == ch)
			c++;
	return c;
}

static char *_find_symbol(char *s, char ch)
{
	if(*s == ch)
		return s;
	while(*s)
		if(*++s == ch)
 			return s;
	return NULL;
}

#define UTLOG_FORMAT_RET \
if (logtype == UTLOG_ERR) {                           \
	switch (error_action_flag) {                  \
		case UTLOG_ERR_ACT_ABORT:             \
			exit(-1); /* abort() */       \
			break;                        \
		case UTLOG_ERR_ACT_WARNING:           \
			/* print warning info only */ \
			break;                        \
		default:                              \
			break;                        \
	}                                             \
}                                                     \
return;

void utlog_format(int logtype, const char *fmt, ...)
{
	int n;
	char *p, *f, *pf;
	/* time_t t;
	 * time(&t);
	 */
	FILE *log_ostream;
	clock_t clk = clock();
	va_list ap;
	if (user_log_ostream)
		log_ostream = user_log_ostream;
	else
		log_ostream = UTLOG_DEFAULT_OSTREAM;
	f = pf = (char*)malloc(strlen(fmt) + 1);
	strcpy(f, fmt);
	/* fprintf(log_ostream, "[%ld:%ld] : ", t, clk); */
	if (getenv("TERM") &&
		highlight_flag == UTLOG_HIGHLIGHT_ON) {
		fprintf(log_ostream, "[%s%08ld\033[0m]: ",
				_log_clock_style_str(logtype),
			clk / UTLOG_CLK_FRAC);
	} else {
		fprintf(log_ostream, "[%08ld]: ",
			clk / UTLOG_CLK_FRAC);
	}
	va_start(ap, fmt);
	n = _account_symbol(f, '%');
	p = _find_symbol(f, '%');
	if(p == NULL){
		fprintf(log_ostream, "%s", f);
		free(pf);
		UTLOG_FORMAT_RET;
	}
	*p = '\0';
	fprintf(log_ostream, "%s", f);
	f = p;
	*f = '%';
	while(n-- > 1){
		p = _find_symbol(f + 1, '%');
		if(p == NULL){
			free(pf);
			UTLOG_FORMAT_RET;
		}
		*p = '\0';
		fprintf(log_ostream, f, va_arg(ap, void*));
		f = p;
		*f = '%';
	}
	fprintf(log_ostream, f, va_arg(ap, void*));
	free(pf);
	va_end(ap);
	UTLOG_FORMAT_RET;
}
