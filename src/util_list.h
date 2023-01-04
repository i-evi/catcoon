#ifndef _UTIL_LIST_H_
#define _UTIL_LIST_H_

#ifdef __cplusplus
	extern "C" {
#endif

#ifdef CONFIG_STD_C89
	#define LS_INLINE
#else
	#define LS_INLINE inline
#endif

#ifdef _MSC_VER
	#define LS_INT64 __int64
#else  /* 64bit integer for gcc, clang... */
	#define LS_INT64 long long
#endif

#include <stddef.h>

typedef unsigned char lsw_t;
typedef unsigned char lsu8_t;
typedef unsigned LS_INT64 lsiz_t;

enum {
	LSS_SUCCESS = 0,
	LSS_BAD_ID,
	LSS_MALLOC_ERR,
	LSS_DYN_ID_EXIST,
	LSS_BAD_OBJ,
	LSS_ARG_ILL,
	LSS_LNAME_ERR,
	LSS_FILE_ERR,
	LSS_SHM_ERR,
	LSS_ERR_LISTFULL,
	LSS_OBJ_NOFOUND,
	LSS_CCNT_MAX
};

#define SET_BIT(val, bitn) (val |=(1<<(bitn)))
#define CLR_BIT(val, bitn) (val&=~(1<<(bitn)))
#define GET_BIT(val, bitn) (val & (1<<(bitn)))

#define SET_FLAG(flag, bft) \
	if(bft & 0x0F)SET_BIT(flag, bft >> 4); else CLR_BIT(flag, bft >> 4);
#define TEST_FLAG(flag, bft) \
	!(((bft & 0x0F) << (bft >> 4)) ^ (flag & (1 << (bft >> 4))))

#define CONFLICT_COUNTER    lsu8_t
#define REC_LENGTH_TYPE     lsiz_t

typedef CONFLICT_COUNTER    ccnt_t;
typedef REC_LENGTH_TYPE     rlen_t;

#ifndef LIST_NAME_LEN
	#define LIST_NAME_LEN 128
#endif

#ifndef LIST_PATH_LEN
	#define LIST_PATH_LEN 256
#endif

struct list {
	char   *name;     /* name of LIST */
	lsu8_t *mem;      /* memory pool */
	lsu8_t **index;   /* index table */
	void *_;
	lsiz_t length;    /* length of memory pool */
	lsiz_t key;       /* key of shared mem */
	lsiz_t counter;   /* record counter */
	lsiz_t blen;      /* length of a record */
	lsiz_t nmemb;     /* nmemb */
  union {
	lsu8_t flag;      /* list's mode flag */
	lsiz_t _placeholder;
  };
};

#define LIST_INFO_LEN (sizeof(struct list) - sizeof(void *[4]))
#define LIST_INFO_OFFSET (sizeof(struct list) - LIST_INFO_LEN)

#define LIST_FLAG_DEFAULT           0x00
#define LIST_STATIC_MODE            0x00
#define LIST_DYNAMIC_MODE           0x01
#define LIST_HASH_NOT_SET           0x40
#define LIST_HASH_TABLE             0x41
#define LIST_RESIZABLE              0x50
#define LIST_UNRESIZABLE            0x51
#define LIST_SHARED_USER            0x60
#define LIST_SHARED_OWNER           0x61
#define LIST_NOT_SHARED             0x70
#define LIST_SHARED_MEM             0x71

#define LIST_ALIGNMENT_DFL 4 /* 32bits */

size_t list_set_alignment(size_t alignment);

struct list *list_new(lsiz_t nmemb, lsiz_t blen);

struct list *list_clone(struct list *ls);

void list_del(struct list *ls);

lsw_t list_resize(struct list *ls, lsiz_t nmemb);

lsw_t list_rename(struct list *ls, const char *name);

void *list_set_data(struct list *ls, lsiz_t id,
	const void *data, lsiz_t len);

void *list_alloc(struct list *ls, lsiz_t id, lsiz_t nbyte);

void *list_index(struct list *ls, lsiz_t id);

lsiz_t list_getlen(struct list *ls, lsiz_t id);

#define list_calc_dynlen(r) \
	(*((rlen_t*)((unsigned char*)r - sizeof(rlen_t))))

lsw_t list_erase(struct list *ls, lsiz_t id);

lsw_t list_swap(struct list *ls, lsiz_t id1, lsiz_t id2);

typedef void *(lsio_open) (const char *pathname, const char *mode);
typedef int   (lsio_close)(void *fp);
typedef long  (lsio_read) (void *buf, long size, void *fp);
typedef long  (lsio_write)(void *buf, long size, void *fp);
typedef long  (lsio_seek) (void *fp, long off, int whence);
typedef long  (lsio_tell) (void *fp);

struct lsio_struct {
	lsio_open  *open;
	lsio_close *close;
	lsio_read  *read;
	lsio_write *write;
	lsio_seek  *seek;
	lsio_tell  *tell;
};

struct lsio_struct *list_get_io_ctrl_struct(void);

void list_set_io_stdio(void);

/* Set `lsio` to seb ways */
extern void list_set_io_seb(void);
/* Set seb compression level */
extern int  list_get_seb_ctrl(void);
extern void list_set_seb_ctrl(int v);

/* Set `lsio` to zlib ways */
extern void list_set_io_zlib(void);
/* Set Zlib compression level */
extern int  list_get_zlib_ctrl(void);
extern void list_set_zlib_ctrl(int v);

lsw_t list_export(struct list *ls, const char *path);

void list_enable_io_auto_switch(void);
void list_disable_io_auto_switch(void);

struct list *list_import(const char *path);

/* struct list | name | mem */

struct list *list_new_shared(lsiz_t nmemb, lsiz_t blen, lsiz_t key);

struct list *list_link_shared(lsiz_t len, lsiz_t key);

#define list_shared_shm_len(list) \
	(sizeof(LIST) + LIST_NAME_LEN + list->length)

lsw_t list_del_shared(struct list *ls);

#define list_export_shared list_export_static
struct list *list_import_shared(const char *path, lsiz_t key);

#define list_set_unresizable(list) \
	SET_FLAG(list->flag, LIST_UNRESIZABLE)
#define list_set_resizable(list) \
	SET_FLAG(list->flag, LIST_RESIZABLE)

/*
 * Simple hash table, LIST_STATIC_MODE only 
 * For each record: | CONFLICT_COUNTER | Data |
 *
 * CONFLICT_COUNTER: Only used in hast table under static mode,
 * the number of conflicted elements is limited to 127
 * 0 1 2 3 4 5 6 7
 * x x x x x x x f
 * _____________ \__.Flag
 *              \___.CONFLICT_COUNTER
 */

#define CCNT_INC 0x02 /* (1 << 1)*/
#define CCNT_MAX 0x7F
#define CCNT_VAL(cnt) (cnt >> 1)

typedef lsiz_t (*__hashFx)(const void*, lsiz_t);
typedef int  (*__compFx)(const void*, const void*);

void list_set_hash_fn(__hashFx hfx);
void list_set_hash_seed(lsiz_t seed);

int list_hash_table_test_id(struct list *ls, lsiz_t id);

ccnt_t list_get_record_counter(struct list *ls, lsiz_t id);

struct list *list_new_hash_table(lsiz_t nmemb, lsiz_t blen);

lsw_t list_hash_id_calc(struct list *ls,
	const void *data, lsiz_t *hi, lsiz_t *id);

lsw_t list_hash_table_insert(struct list *ls,
	const void *record, lsiz_t len);

lsw_t list_hash_table_find(struct list *ls,
			const void *key, lsiz_t *id);

lsw_t list_hash_table_del(struct list *ls, const void *key);

void list_print_properties(struct list *ls, void *stream);

#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_LIST_H_ */
