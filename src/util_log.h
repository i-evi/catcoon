#ifndef _UTIL_LOG_H_
#define _UTIL_LOG_H_

#ifdef __cplusplus
	extern "C" {
#endif

#ifdef __linux__
	#define UTLOG_LINUX_API_TIME
#endif

typedef double utlog_time_t;

#define UTLOG_CLK_FRAC CLOCKS_PER_SEC

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

enum utlog_time_mode {
	UTLOG_USE_SYS_TIME,
	UTLOG_USE_CLK_TIME,
	UTLOG_USE_ABS_TIME,
	UTLOG_USE_RUN_TIME
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

void utlog_use_clk_time(void);
void utlog_use_sys_time(void);
void utlog_use_abs_time(void);
void utlog_use_run_time(void);

utlog_time_t utlog_gettime(void);

void utlog_tag(int logtype);

void utlog_format(int logtype, const char *fmt, ...);

#define utlog_format_verbose(logtype, fmt, ...) \
	utlog_format(logtype, "[l_%d@<%s>] " fmt, \
	__LINE__, __FILE__, __VA_ARGS__);

#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_LOG_H_ */
