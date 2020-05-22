#ifndef _UTIL_LOG_H_
#define _UTIL_LOG_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define UTLOG_CLK_FRAC 1000

#define UTLOG_DEFAULT_OSTREAM stderr

enum utlog_type {
	UTLOG_ERR,
	UTLOG_WARN,
	UTLOG_INFO
};

enum utlog_err_act {
	UTLOG_ERR_ACT_ABORT,
	UTLOG_ERR_ACT_WARNING
};

enum utlog_highlight {
	UTLOG_HIGHLIGHT_ON,
	UTLOG_HIGHLIGHT_OFF
};


void  utlog_set_ostream(void *stream);
void *utlog_get_ostream(void);

void utlog_set_error_action(int act);

void utlog_highlight_on(void);
void utlog_highlight_off(void);

void utlog_format(int logtype, const char *fmt, ...);

#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_LOG_H_ */
