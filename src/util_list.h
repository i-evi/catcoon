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

typedef unsigned char ls_status_t;

/*========== operation status ===========*/ 
#define LSS_STATUS_
#define LSS_SUCCESS         0x00
#define LSS_BAD_ID          0x01
#define LSS_MALLOC_ERR      0x02
#define LSS_DYN_ID_EXIST    0x03
#define LSS_BAD_OBJ         0x04
#define LSS_ARG_ILL         0x05
#define LSS_LNAME_ERR       0x06
#define LSS_FILE_ERR        0x07
#define LSS_SHM_ERR         0x08
#define LSS_ERR_LISTFULL    0x09
#define LSS_OBJ_NOFOUND     0x0A

#define SET_BIT(val, bitn) (val |=(1<<(bitn)))
#define CLR_BIT(val, bitn) (val&=~(1<<(bitn)))
#define GET_BIT(val, bitn) (val & (1<<(bitn)))

#define SET_FLAG(flag, bft) \
	if(bft & 0x0F)SET_BIT(flag, bft >> 4); else CLR_BIT(flag, bft >> 4);
#define TEST_FLAG(flag, bft) \
	!(((bft & 0x0F) << (bft >> 4)) ^ (flag & (1 << (bft >> 4))))

#define DYN_LENGTH_TYPE uint
typedef DYN_LENGTH_TYPE DynLenFlag;

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
 * Notice: length of a dynamic record will not count DynLenFlag
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

/* Tips:
 * Overload "list_new" under cpp
 * list_t *list_new(uint scale);
 * list_t *list_new(uint scale, uint blen);
 */
list_t *list_new_static(uint scale, uint blen);
list_t *list_new_dynamic(uint scale);

list_t *list_clone(list_t *list);
list_t *list_clone_static(list_t *list);
list_t *list_clone_dynamic(list_t *list);

ls_status_t list_del(list_t *list);
ls_status_t list_del_static(list_t *list);
ls_status_t list_del_dynamic(list_t *list);

ls_status_t list_resize(list_t *list, uint newScale);
ls_status_t list_resize_static(list_t *list, uint newScale);
ls_status_t list_resize_dynamic(list_t *list, uint newScale);

ls_status_t list_set_name(list_t *list, const char *name);

ls_status_t list_set_record(list_t *list, uint id,
	const void *record, uint len);
ls_status_t list_set_static_record(list_t *list, uint id,
	const void *record, uint len);
ls_status_t list_set_dynamic_record(list_t *list, uint id,
	const void *record, uint len);

int list_test_record(list_t *list, uint id, uint offset,
	uint nbyte);

void *list_alloc(list_t *list, uint id, uint nbyte);

void *list_get_record(list_t *list, uint id);
void *list_get_static_record(list_t *list, uint id);
void *list_get_dynamic_record(list_t *list, uint id);

uint list_get_record_len(list_t *list, uint id);
/* #define list_get_static_record_len */
#define list_get_dynamic_record_len(record) \
	(DynLenFlag)*(record - sizeof(DynLenFlag))

ls_status_t list_del_record(list_t *list, uint id);
ls_status_t list_del_static_record(list_t *list, uint id);
ls_status_t list_del_dynamic_record(list_t *list, uint id);

ls_status_t list_swap_record(list_t *list,
	uint id1, uint id2);
ls_status_t list_swap_static_record(list_t *list,
	uint id1, uint id2);
ls_status_t list_swap_dynamic_record(list_t *list,
	uint id1, uint id2);

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

ls_status_t list_export(list_t *list, const char *path,
	const char *mode);

/*
 * Exported file structure of Static LIST
 * name_len | list->name | list | list->mem  
 */
ls_status_t list_export_static(list_t *list, const char *path,
	const char *mode);

/*
 * Exported file structure of Dynamic LIST
 * name_len | list->name | list | *index[0:N]
 */
ls_status_t list_export_dynamic(list_t *list, const char *path,
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
	ls_status_t list_del_shared(list_t *list);
	#define list_export_shared list_export_static
	list_t *list_import_shared(const char *path, uint key);
#endif /* ENABLE_SSHM */

/*
 * simple hash function
 */
#define __hashFx uint hfx(void*, uint, uint)
#define __compFx int  cfx(void*, void*, uint)

#define list_set_unresizable(list) \
	SET_FLAG(list->flag, LIST_UNRESIZABLE)
#define list_set_resizable(list) \
	SET_FLAG(list->flag, LIST_RESIZABLE)

ls_status_t list_calc_hash_id(list_t *list,
	void *data, uint len, uint arg, uint *id, __hashFx);

ls_status_t list_search_by_hash(list_t *list,
		void *data, uint len, uint arg, __hashFx,
	__compFx, void *pattern, uint plen, uint *fid);

/*================= Debug =================*/
#define list_ops operation_status
void list_print_info(list_t *list, void *stream);
void operation_status(ls_status_t ops_stat);

#undef byte
#undef uint

#ifdef __cplusplus
	}
#endif

#endif /* _LIST_H_ */
