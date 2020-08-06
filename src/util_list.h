#ifndef _LIST_H_
#define _LIST_H_

#ifdef __cplusplus
	extern "C" {
#endif

#ifndef byte
	#define byte unsigned char
#endif
#ifndef uint
	#define uint unsigned int
#endif

typedef unsigned char lsw_t;

/*========== operation status ===========*/ 
enum LSS_STATUS_ {
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

/*
 * CONFLICT_COUNTER: Only used in hast table under static mode, limited
 * the number of conflicted records to 127
 * 0 1 2 3 4 5 6 7
 * x x x x x x x f
 *                \__.KEPT THIS BIT
 */
#define CCNT_INC 0x02 /* (1 << 1)*/
#define CCNT_MAX 0x7F
#define CCNT_VAL(cnt) (cnt >> 1)

#define CONFLICT_COUNTER    byte
#define REC_LENGTH_TYPE     uint

typedef CONFLICT_COUNTER    ccnt_t;
typedef REC_LENGTH_TYPE     rlen_t;

/* Do not edit LIST_NAME_LEN unless necessary */
#ifndef LIST_NAME_LEN
	#define LIST_NAME_LEN 128
#endif

#ifndef LIST_PATH_LEN
	#define LIST_PATH_LEN 256
#endif

/*
 * Record on dynamic mode :
 * uint length | record |, when export :
 * uint length | record | id
 * Notice: length of a dynamic record will not count rlen_t
 *
 * index     record(Each record will allocate memory via "malloc")
 * index0----->rec0
 * index1----->rec1
 *        ...
 * indexX----->recX
 * X = list.scale - 1
 * ==============================================================
 * Record on static mode :
 * list.mem :
 * record 0 | record 1 | ... | record X
 * _________
 *          |
 *          |---->list.blen
 */

typedef struct {
	char *name;     /* name of LIST */
	byte *mem;      /* memory pool */
	byte **index;   /* index table */
	uint length;    /* length of memory pool */
	uint key;       /* key of shared mem */
	uint counter;   /* record counter */
	uint blen;      /* length of a record */
	uint scale;     /* scale */
	byte flag;      /* list's mode flag */
	byte status;    /* work status flag */
	byte placeholder[2];
} list_t;

struct list_info {
	uint length;    /* length of memory pool */
	uint key;       /* key of shared mem */
	uint counter;   /* record counter */
	uint blen;      /* length of a record */
	uint scale;     /* scale */
	byte flag;      /* list's mode flag */
	byte status;    /* work status flag */
	byte placeholder[2];
};

#define LIST_INFO_LEN sizeof(struct list_info)
#define LIST_INFO_OFFSET sizeof(list_t) - LIST_INFO_LEN

/*============== mode flag ==============*/
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

/*============= status flag =============*/
/* LIST_STATUS_ALRIGHT MUST BE 0x00 */
#define LIST_STATUS_ALRIGHT         0x00

#define LIST_MALLOC_ERROR           0x01
#define LIST_IMPORT_ERROR           0x02

list_t *list_new(uint scale, uint blen);
list_t *list_new_static(uint scale, uint blen);
list_t *list_new_dynamic(uint scale);

list_t *list_clone(list_t *list);
list_t *list_clone_static(list_t *list);
list_t *list_clone_dynamic(list_t *list);

lsw_t list_del(list_t *list);
lsw_t list_del_static(list_t *list);
lsw_t list_del_dynamic(list_t *list);

lsw_t list_resize(list_t *list, uint newScale);
lsw_t list_resize_static(list_t *list, uint newScale);
lsw_t list_resize_dynamic(list_t *list, uint newScale);

lsw_t list_set_name(list_t *list, const char *name);

lsw_t list_set_record(list_t *list, uint id,
	const void *record, uint len);
lsw_t list_set_static_record(list_t *list, uint id,
	const void *record, uint len);
lsw_t list_set_dynamic_record(list_t *list, uint id,
	const void *record, uint len);

int list_test_record(list_t *list, uint id, uint offset, uint nbyte);

void *list_alloc(list_t *list, uint id, uint nbyte);

void *list_get_record(list_t *list, uint id);
void *list_get_static_record(list_t *list, uint id);
void *list_get_dynamic_record(list_t *list, uint id);

uint list_get_record_len(list_t *list, uint id);
/* #define list_get_static_record_len */
#define list_get_dynamic_record_len(r) \
	(*((rlen_t*)((unsigned char*)r - sizeof(rlen_t))))

lsw_t list_del_record(list_t *list, uint id);
lsw_t list_del_static_record(list_t *list, uint id);
lsw_t list_del_dynamic_record(list_t *list, uint id);

lsw_t list_swap_record(list_t *list, uint id1, uint id2);
lsw_t list_swap_static_record(list_t *list, uint id1, uint id2);
lsw_t list_swap_dynamic_record(list_t *list, uint id1, uint id2);

#define IO_FILE   FILE*
#define IO_fopen  fopen
#define IO_fclose fclose
#define IO_fread  fread
#define IO_fwrite fwrite

/* #define ENABLE_ZLIB */

#ifdef ENABLE_ZLIB
	#define ZLIB_FILE   gzFile
	#define ZLIB_fopen  gzopen
	#define ZLIB_fclose gzclose
	#define ZLIB_fread(buf, size, count, fp) \
		gzread(fp, buf, size)
	#define ZLIB_fwrite(buf, size, count, fp) \
		gzwrite(fp, buf, size)
#endif /* ENABLE_ZLIB */

#define LIST_EXPORT_MODE_RAW "raw"

#ifdef ENABLE_ZLIB
	#define LIST_EXPORT_MODE_DEF "wb0"
#else
	#define LIST_EXPORT_MODE_DEF "wb"
#endif

lsw_t list_export(list_t *list, const char *path,
	const char *mode);

/*
 * Exported file structure of Static LIST
 * name_len | list->name | list | list->mem  
 */
lsw_t list_export_static(list_t *list, const char *path,
	const char *mode);

/*
 * Exported file structure of Dynamic LIST
 * name_len | list->name | list | *index[0:N]
 */
lsw_t list_export_dynamic(list_t *list, const char *path,
	const char *mode);

list_t *list_import(const char *path);

/*
 * Sharedlist is tlist's extension to create list_t at shared memory space.
 * Notice that sharedlist not support tlist which was created as Dynamic
 * mode, cuz that will create too many shm handles.
 */

/* #define ENABLE_SSHM */

/* list_t | name | mem */

#ifdef ENABLE_SSHM
	list_t *list_new_shared(uint scale, uint blen, uint key);
	list_t *list_link_shared(uint len, uint key);
	#define list_shared_shm_len(list) \
		(sizeof(LIST) + LIST_NAME_LEN + list->length)
	lsw_t list_del_shared(list_t *list);
	#define list_export_shared list_export_static
	list_t *list_import_shared(const char *path, uint key);
#endif /* ENABLE_SSHM */

#define list_set_unresizable(list) \
	SET_FLAG(list->flag, LIST_UNRESIZABLE)
#define list_set_resizable(list) \
	SET_FLAG(list->flag, LIST_RESIZABLE)
/*
 * Hash table functions, LIST_STATIC_MODE only - | Counter | Data |
 */
typedef uint (*__hashFx)(const void*, uint);
typedef int  (*__compFx)(const void*, const void*);

void list_set_hash_fn(__hashFx hfx);
void list_set_hash_seed(uint seed);

int list_hash_table_test_id(list_t *list, uint id);

ccnt_t list_get_record_counter(list_t *list, uint id);

list_t *list_new_hash_table(uint scale, uint blen);

lsw_t list_hash_id_calc(list_t *list,
	const void *data, uint *hi, uint *id);

lsw_t list_hash_table_insert(list_t *list,
	const void *record, uint len);

lsw_t list_hash_table_find(list_t *list, const void *key, uint *id);

lsw_t list_hash_table_del(list_t *list, const void *key);

/*
lsw_t list_calc_hash_id(list_t *list,
	void *data, uint len, uint seed, uint *id, __hashFx);

lsw_t list_search_by_hash(list_t *list,
		void *data, uint len, uint seed, __hashFx,
	__compFx, void *pattern, uint plen, uint *fid);
*/

/*================= Debug =================*/
#define list_ops operation_status
void list_print_info(list_t *list, void *stream);
void list_print_status(lsw_t stat, void *stream);

#undef byte
#undef uint

#ifdef __cplusplus
	}
#endif

#endif /* _LIST_H_ */

