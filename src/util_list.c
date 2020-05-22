#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_list.h"

#ifdef ENABLE_SSHM
	#include "sshm.h"
#endif
#ifdef ENABLE_ZLIB
	/* #include <gzguts.h> */
	#include <zlib.h>
#endif

#ifndef byte
	#define byte unsigned char
#endif
#ifndef uint
	#define uint unsigned int
#endif

#define ASSERT_ON_BUILD(condition)\
	((void)sizeof(char[1 - 2*!!(condition)]))
/*=========== compile-time test ===========*/
#ifdef ENABLE_FOPS
void __________compile_time_test___________()
{
	/* LIST_INFO_LEN must be 32 */
	ASSERT_ON_BUILD(24 - LIST_INFO_LEN);
}
#endif

static uint _djb_hash(void *s, uint len, uint seed)
{
	uint hash = seed;
	uint i;
	for (i = 0; i < len; i++) {
		hash = (hash << 5) + hash + *((byte*)s + i);
	}
	return hash;
}

static int _is_little_endian()
{
	uint i = 1;
	unsigned char *c = (unsigned char*)&i;
	if (*c)
		return 1;
	return 0;
}

static int _is_gzfile(const char *filename)
{
	const byte gzhead[] = {0x1F, 0x8B, 0x08};
	byte buf[sizeof(gzhead)] = {0};
	int r;
	FILE *fp = fopen(filename, "rb");
	if (!fp)
		return 0;
	fread(buf, sizeof(gzhead), 1, fp);
	r = memcmp(gzhead, buf, sizeof(gzhead));
	fclose(fp);
	return !r;
}

static void _memswap(void *a, void *b, int len)
{
	unsigned char *p1 = (unsigned char*)a;
	unsigned char *p2 = (unsigned char*)b;
	while (len--) {
		*p1 ^= *p2;
		*p2 ^= *p1;
		*p1++ ^= *p2++;
	}
}

static void _memrev(void *i, int len)
{
	int j = len >> 1;
	unsigned char *c = (unsigned char*)i;
	for (len--; j > 0; j--) {
		*c ^= *(c + len);
		*(c + len) ^= *c;
		*c ^= *(c + len);
		c++, len -= 2;
	}
}

static void _switch_byte_order(list_t *list) {
	_memrev(&list->length , sizeof(uint));
	_memrev(&list->key    , sizeof(uint));
	_memrev(&list->counter, sizeof(uint));
	_memrev(&list->blen   , sizeof(uint));
	_memrev(&list->scale  , sizeof(uint));
}

list_t *list_new_static(uint scale, uint blen)
{
	list_t *list = (list_t*)malloc(sizeof(list_t));
	if (!scale || !blen || !list)
		return NULL;
	memset(list, 0, sizeof(list_t));
	SET_FLAG(list->flag, LIST_STATIC_MODE);
	list->length = scale * blen;
	list->mem = (byte*)malloc(list->length);
	if (!list->mem) {
		free(list);
		return NULL;
	}
	memset(list->mem, 0, list->length);
	list->blen    = blen;
	list->scale   = scale;
	list->counter = scale;
	return list;
}

list_t *list_new_dynamic(uint scale)
{
	list_t *list = (list_t*)malloc(sizeof(list_t));
	if (!scale || !list)
		return NULL;
	memset(list, 0, sizeof(list_t));
	SET_FLAG(list->flag, LIST_DYNAMIC_MODE);
	list->index = (byte**)malloc(scale * sizeof(byte*));
	if (!list->index) {
		free(list);
		return NULL;
	}
	memset(list->index, 0, scale * sizeof(byte*));
	list->scale = scale;
	return list;
}

list_t *list_clone(list_t *list)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_clone_dynamic(list);
	else /* LIST_STATIC_MODE */
		return list_clone_static(list);
}

list_t *list_clone_static(list_t *list)
{
	list_t *clone;
	if (!list)
		return NULL;
	clone = (list_t*)malloc(sizeof(list_t));
	if (!clone)
		return NULL;
	memcpy(clone, list, sizeof(list_t));
	if (list->name) {
		clone->name = (char*)malloc(LIST_NAME_LEN);
		if (!clone->name) {
			free(clone);
			return NULL;
		}
		strcpy(clone->name, list->name);
	} else {
		clone->name = NULL;
	}
	clone->mem = (byte*)malloc(list->length);
	if (!clone->mem) {
		if (clone->name)
			free(clone->name);
		free(clone);
		return NULL;
	}
	memcpy(clone->mem, list->mem, list->length);
	clone->index = NULL;
	return clone;
}

#define PROCESS_CLONE_DYN_INDEX(i) \
if (list->index[i]) {                                \
	s = list_set_record(clone, i,                \
		list->index[i] + sizeof(DynLenFlag), \
		*(DynLenFlag*)list->index[i]);       \
	if (s) {                                     \
		clone->status |= s;                  \
		return clone;                        \
	}                                            \
}

list_t *list_clone_dynamic(list_t *list)
{
	uint i;
	list_t *clone;
	ls_status_t s;
	if (!list)
		return NULL;
	clone = (list_t*)malloc(sizeof(list_t));
	if (!clone)
		return NULL;
	memcpy(clone, list, sizeof(list_t));
	if (list->name) {
		clone->name = (char*)malloc(LIST_NAME_LEN);
		if (!clone->name) {
			free(clone);
			return NULL;
		}
		strcpy(clone->name, list->name);
	} else {
		clone->name = NULL;
	}
	clone->index = (byte**)malloc(list->scale * sizeof(byte*));
	if (!clone->index) {
		if (clone->name)
			free(clone->name);
		free(clone);
		return NULL;
	}
	memset(clone->index, 0, list->scale * sizeof(byte*));
	clone->length  = 0;
	clone->counter = 0;
	for (i = 0; i < list->scale; ++i) {
		PROCESS_CLONE_DYN_INDEX(i);
	}
	return clone;
}

ls_status_t list_del(list_t *list)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_del_dynamic(list);
	else /* LIST_STATIC_MODE */
		return list_del_static(list);
}

ls_status_t list_del_static(list_t *list)
{
	if (!list)
		return LSS_SUCCESS;
	if (list->name)
		free(list->name);
	if (list->mem)
		free(list->mem);
	free(list);
	return LSS_SUCCESS;
}

ls_status_t list_del_dynamic(list_t *list)
{
	if (!list)
		return LSS_SUCCESS;
	if (list->name)
		free(list->name);
	while (list->scale--) {
		if (list->index[list->scale])
			free(list->index[list->scale]);
	}
	if (list->index)
		free(list->index);
	free(list);
	return LSS_SUCCESS;
}

ls_status_t list_set_name(list_t *list, const char *name)
{
	if (TEST_FLAG(list->flag, LIST_NOT_SHARED)) {
		if (!name) {
			if (list->name)
				free(list->name);
			list->name = NULL;
			return LSS_SUCCESS;
		}
		if (strlen(name) >= LIST_NAME_LEN)
			return LSS_ARG_ILL;
		if (list->name)
			free(list->name);
		list->name = (char*)malloc(LIST_NAME_LEN);
		if (!list->name)
			return LSS_MALLOC_ERR;
		strcpy(list->name, name);
	} else {
		memset((char*)list + sizeof(list_t), 0, LIST_NAME_LEN);
		strcpy((char*)list + sizeof(list_t), name);
	}
	return LSS_SUCCESS;
}


ls_status_t list_resize(list_t *list, uint newScale)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_resize_dynamic(list, newScale);
	else /* LIST_STATIC_MODE */
		return list_resize_static(list, newScale);
}

ls_status_t list_resize_static(list_t *list, uint newScale)
{
	byte *newMem;
	uint newLen = newScale * list->blen;
	if (TEST_FLAG(list->flag, LIST_UNRESIZABLE))
		return LSS_BAD_OBJ;
	newMem = (byte*)realloc(list->mem, newLen);
	if (!newMem)
		return LSS_MALLOC_ERR;
	if (newScale > list->scale) {
		memset(list->mem + list->scale * list->blen, 0,
			(newScale - list->scale) * list->blen);
	}
	list->mem     = newMem;
	list->length  = newLen;
	list->scale   = newScale;
	list->counter = newScale;
	return LSS_SUCCESS;
}

ls_status_t list_resize_dynamic(list_t *list, uint newScale)
{
	byte **newIndex;
	uint i = 0;
	if (TEST_FLAG(list->flag, LIST_UNRESIZABLE))
		return LSS_BAD_OBJ;
	if (newScale < list->scale) {
		i = newScale;
		for (; i < list->scale; ++i)
			list_del_dynamic_record(list, i);
	}
	newIndex = (byte**)realloc(list->index,
				newScale * sizeof(byte*));
	if (!newIndex)
		return LSS_MALLOC_ERR;
	list->index = newIndex;
	if (newScale > list->scale) {
		memset((byte*)newIndex + list->scale * sizeof(byte*), 0,
			(newScale - list->scale) * sizeof(byte*));
	}
	list->scale = newScale;
	return LSS_SUCCESS;
}

ls_status_t list_set_record(list_t *list, uint id,
	const void *record, uint len)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_set_dynamic_record(list, id, record, len);
	else /* LIST_STATIC_MODE */
		return list_set_static_record(list, id, record, len);
}

ls_status_t list_set_static_record(list_t *list, uint id,
	const void *record, uint len)
{
	if (id >= list->scale)
		return LSS_BAD_ID;
	len = list->blen < len ? list->blen : len;
	memcpy(list->mem + id * list->blen, record, len);
	return LSS_SUCCESS;
}

ls_status_t list_set_dynamic_record(list_t *list, uint id,
	const void *record, uint len)
{
	DynLenFlag *l;
	uint mlen = 0;
	if (id >= list->scale)
		return LSS_BAD_ID;
	if (list->index[id])
		return LSS_DYN_ID_EXIST;
	mlen = len + sizeof(DynLenFlag);
	list->index[id] = (byte*)malloc(mlen);
	if (!list->index[id])
		return LSS_MALLOC_ERR;
	memcpy(list->index[id] + sizeof(DynLenFlag), record, len);
	l = (DynLenFlag*)list->index[id];
	*l = len;
	list->counter++;
	list->length += mlen;
	return LSS_SUCCESS;
}

int list_test_record(list_t *list, uint id, uint offset,
	uint nbyte)
{
	byte *rec;
	uint c = 0;
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
		if (list->index[id])
			return 1;
		else
			return 0;
	} else { /* LIST_STATIC_MODE */
		rec = (byte*)list_get_static_record(list, id);
		while (nbyte--) {
			c += rec[offset + nbyte];
		}
		if (c)
			return 1;
		else
			return 0;
	}
}

void *list_alloc(list_t *list, uint id, uint nbyte)
{
	if (id >= list->scale)
		return NULL;
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
		if (list->index[id])
			list_del_dynamic_record(list, id);
		list->index[id] = (byte*)malloc(nbyte + sizeof(DynLenFlag));
		if (!list->index[id])
			return NULL;
		*(DynLenFlag*)list->index[id] = nbyte;
		list->length += (nbyte + sizeof(DynLenFlag));
		list->counter++;
		return list->index[id] + sizeof(DynLenFlag);
	} else { /* LIST_STATIC_MODE */
		if (nbyte > list->blen)
			return NULL;
		else
			return list_get_static_record(list, id);
	}
}

void *list_get_record(list_t *list, uint id)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_get_dynamic_record(list, id);
	else /* LIST_STATIC_MODE */
		return list_get_static_record(list, id);
}

void *list_get_static_record(list_t *list, uint id)
{
	if (id >= list->scale)
		return NULL;
	else
		return list->mem + id * list->blen;
}

void *list_get_dynamic_record(list_t *list, uint id)
{
	if (id >= list->scale)
		return NULL;
	else {
		if (list->index[id])
			return (list->index[id] + sizeof(DynLenFlag));
		else
			return NULL;
	}
}

uint list_get_record_len(list_t *list, uint id)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
		if (list->index[id])
			return *(DynLenFlag*)list->index[id];
		else
			return 0;
	}
	else /* LIST_STATIC_MODE */
		return list->blen;
}

ls_status_t list_del_record(list_t *list, uint id)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_del_dynamic_record(list, id);
	else /* LIST_STATIC_MODE */
		return list_del_static_record(list, id);
}

ls_status_t list_del_static_record(list_t *list, uint id)
{
	if (id >= list->scale)
		return LSS_BAD_ID;
	memset(list->mem + id * list->blen, 0, list->blen);
	return LSS_SUCCESS;
}

ls_status_t list_del_dynamic_record(list_t *list, uint id)
{
	if (id >= list->scale)
		return LSS_BAD_ID;
	if (list->index[id]) {
		list->length -= sizeof(DynLenFlag);
		list->length -= *(DynLenFlag*)list->index[id];
		list->counter--;
		free(list->index[id]);
		list->index[id] = NULL;
	}
	return LSS_SUCCESS;
}

ls_status_t list_swap_record(list_t *list,
	uint id1, uint id2)
{
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
		return list_swap_dynamic_record(list, id1, id2);
	else /* LIST_STATIC_MODE */
		return list_swap_static_record(list, id1, id2);
}

ls_status_t list_swap_static_record(list_t *list,
	uint id1, uint id2)
{
	if (id1 >= list->scale || id2 >= list->scale)
		return LSS_BAD_ID;
	_memswap(list->mem + id1 * list->blen,
		list->mem + id2 * list->blen, list->blen);
	return LSS_SUCCESS;	
}

ls_status_t list_swap_dynamic_record(list_t *list,
	uint id1, uint id2)
{
	byte *p;
	if (id1 >= list->scale || id2 >= list->scale)
		return LSS_BAD_ID;
	p = list->index[id1];
	list->index[id1] = list->index[id2];
	list->index[id2] = p;
	return LSS_SUCCESS;
}

#ifdef ENABLE_FOPS
	ls_status_t list_export(list_t *list, const char *path,
		const char *mode)
	{
		if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE))
			return list_export_dynamic(list, path, mode);
		else /* LIST_STATIC_MODE */
			return list_export_static(list, path, mode);
	}

	#define PROCESS_EXPORT_STATIC(PREFIX) \
	uint name_len = 0;                              \
	fp = PREFIX##fopen(path, mode);                 \
	if (!fp)                                        \
		return LSS_FILE_ERR;                    \
	if (list->name)                                 \
		name_len = strlen(list->name);          \
	if (!_is_little_endian()) {                     \
		_memrev(&name_len, sizeof(uint));       \
		_switch_byte_order(list);               \
	}                                               \
	PREFIX##fwrite(&name_len, sizeof(uint), 1, fp); \
	PREFIX##fwrite(list->name, name_len, 1, fp);    \
	PREFIX##fwrite((byte*)list + LIST_INFO_OFFSET,  \
			       	LIST_INFO_LEN, 1, fp);  \
	PREFIX##fwrite(list->mem, list->length, 1, fp); \
	if (!_is_little_endian()) {                     \
		_switch_byte_order(list);               \
	}

#ifdef ENABLE_ZLIB
	static ls_status_t _list_export_static_zlib(list_t *list,
		const char *path, const char *mode)
	{
		ZLIB_FILE fp;
		PROCESS_EXPORT_STATIC(ZLIB_);
		ZLIB_fclose(fp);
		return LSS_SUCCESS;
	}
#endif

	static ls_status_t _list_export_static_ansic(list_t *list,
		const char *path, const char *mode)
	{
		IO_FILE fp;
		PROCESS_EXPORT_STATIC(IO_);
		IO_fclose(fp);
		return LSS_SUCCESS;
	}

	ls_status_t list_export_static(list_t *list, const char *path,
		const char *mode) {
		if (!mode)
			mode = LIST_EXPORT_MODE_DEF;
#ifdef ENABLE_ZLIB
		if (strcmp(mode, LIST_EXPORT_MODE_RAW))
			return _list_export_static_zlib(list, path, mode);
		else
			return _list_export_static_ansic(list, path,
				LIST_EXPORT_MODE_DEF);
#else
		return _list_export_static_ansic(list, path,
			LIST_EXPORT_MODE_DEF);
#endif
	}

	#define PROCESS_EXPORT_DYN_RECORD(PREFIX) \
	if (list->index[sc]) {                                        \
		if (!_is_little_endian())                             \
			_memrev(list->index[sc], sizeof(DynLenFlag)); \
		PREFIX##fwrite(list->index[sc], sizeof(DynLenFlag) +  \
			(DynLenFlag)*list->index[sc], 1, fp);         \
		if (!_is_little_endian()) {                           \
			_memrev(list->index[sc], sizeof(DynLenFlag)); \
			_memrev(&sc, sizeof(uint));                   \
		}                                                     \
		PREFIX##fwrite(&sc, sizeof(uint), 1, fp);             \
		if (!_is_little_endian())                             \
			_memrev(&sc, sizeof(uint));                   \
	}

	#define PROCESS_EXPORT_DYNAMIC(PREFIX) \
	uint name_len = 0;                              \
	uint sc = list->scale;                          \
	fp = PREFIX##fopen(path, mode);                 \
	if (!fp)                                        \
		return LSS_FILE_ERR;                    \
	if (list->name)                                 \
		name_len = strlen(list->name);          \
	if (!_is_little_endian()) {                     \
		_memrev(&name_len, sizeof(uint));       \
		_switch_byte_order(list);               \
	}                                               \
	PREFIX##fwrite(&name_len, sizeof(uint), 1, fp); \
	PREFIX##fwrite(list->name, name_len, 1, fp);    \
	PREFIX##fwrite((byte*)list + LIST_INFO_OFFSET,  \
			       	LIST_INFO_LEN, 1, fp);  \
	while (sc--) {                                  \
		PROCESS_EXPORT_DYN_RECORD(PREFIX);      \
	}                                               \
	if (!_is_little_endian()) {                     \
		_switch_byte_order(list);               \
	}

#ifdef ENABLE_ZLIB
	static ls_status_t _list_export_dynamic_zlib(list_t *list,
		const char *path, const char *mode)
	{
		ZLIB_FILE fp;
		PROCESS_EXPORT_DYNAMIC(ZLIB_);
		ZLIB_fclose(fp);
		return LSS_SUCCESS;
	}
#endif

	static ls_status_t _list_export_dynamic_ansic(list_t *list,
		const char *path, const char *mode)
	{
		IO_FILE fp;
		PROCESS_EXPORT_DYNAMIC(IO_);
		IO_fclose(fp);
		return LSS_SUCCESS;
	}

	ls_status_t list_export_dynamic(list_t *list, const char *path,
		const char *mode)
	{
		if (!mode)
			mode = LIST_EXPORT_MODE_DEF;
#ifdef ENABLE_ZLIB
		if (strcmp(mode, LIST_EXPORT_MODE_RAW))
			return _list_export_dynamic_zlib(list, path, mode);
		else
			return _list_export_dynamic_ansic(list, path,
				LIST_EXPORT_MODE_DEF);
#else
		return _list_export_dynamic_ansic(list, path,
			LIST_EXPORT_MODE_DEF);
#endif
	}

	#define PROCESS_IMPORT_DYN_RECORD(PREFIX) \
	list->index = (byte**)                                   \
		malloc(list->scale * sizeof(byte*));             \
	if (!list->index) {                                      \
		free(list);                                      \
		free(name);                                      \
		PREFIX##fclose(fp);                              \
		return NULL;                                     \
	}                                                        \
	memset(list->index, 0, list->scale * sizeof(byte*));     \
	while (PREFIX##fread(&len, sizeof(DynLenFlag), 1, fp)) { \
		if (!_is_little_endian())                        \
			_memrev(&len, sizeof(DynLenFlag));       \
		tmp = (byte*)malloc(len + sizeof(DynLenFlag));   \
		if (!tmp) {                                      \
			list->status |= LIST_IMPORT_ERROR;       \
			list->status |= LIST_MALLOC_ERROR;       \
			PREFIX##fclose(fp);                      \
			return list;                             \
		}                                                \
		*(DynLenFlag*)tmp = len;                         \
		PREFIX##fread(tmp +                              \
			sizeof(DynLenFlag), len, 1, fp);         \
		PREFIX##fread(&id, sizeof(uint), 1, fp);         \
		if (!_is_little_endian())                        \
			_memrev(&id, sizeof(uint));              \
		list->index[id] = tmp;                           \
	}

	#define PROCESS_IMPORT_STATIC_RECORD(PREFIX) \
	list->mem = (byte*)malloc(list->length);        \
	if (!list->mem) {                               \
		free(list);                             \
		free(name);                             \
		PREFIX##fclose(fp);                     \
		return NULL;                            \
	}                                               \
	PREFIX##fread(list->mem, list->length, 1, fp);

	#define PROCESS_IMPORT_OPEN_FILE(PREFIX)    \
	char *name;                                    \
	byte *tmp;                                     \
	uint name_len  = 0;                            \
	uint id        = 0;                            \
	DynLenFlag len = 0;                            \
	fp = PREFIX##fopen(path, "rb");                \
	if (!fp)                                       \
		return NULL;                           \
	list = (list_t*)malloc(sizeof(list_t));        \
	if (!list) {                                   \
		PREFIX##fclose(fp);                    \
		return NULL;                           \
	}                                              \
	memset(list, 0, sizeof(list_t));               \
	PREFIX##fread(&name_len, sizeof(uint), 1, fp); \
	if (!_is_little_endian())                      \
		_memrev(&name_len, sizeof(uint));      \
	name_len++;                                    \
	name = (char*)malloc(name_len);                \
	if (!name) {                                   \
		free(list);                            \
		PREFIX##fclose(fp);                    \
		return NULL;                           \
	}                                              \
	memset(name, 0, name_len);                     \
	PREFIX##fread(name, name_len - 1, 1, fp);      \
	PREFIX##fread((byte*)list + LIST_INFO_OFFSET,  \
		LIST_INFO_LEN, 1, fp);                 \
	if (!_is_little_endian())                      \
		_switch_byte_order(list);              \
	list->name = name;

#ifdef ENABLE_ZLIB
	static list_t *_list_import_zlib(const char *path)
	{
		list_t *list;
		ZLIB_FILE fp;
		PROCESS_IMPORT_OPEN_FILE(ZLIB_);
		if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
			PROCESS_IMPORT_DYN_RECORD(ZLIB_);
		} else { /* LIST_STATIC_MODE */
			PROCESS_IMPORT_STATIC_RECORD(ZLIB_);
		}
		ZLIB_fclose(fp);
		return list;
	}
#endif

	static list_t *_list_import_ansic(const char *path)
	{
		list_t *list;
		FILE *fp;
		PROCESS_IMPORT_OPEN_FILE(IO_);
		if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
			PROCESS_IMPORT_DYN_RECORD(IO_);
		} else {  /* LIST_STATIC_MODE */
			PROCESS_IMPORT_STATIC_RECORD(IO_);
		}
		fclose(fp);
		return list;
	}

	list_t *list_import(const char *path)
	{
		if (_is_gzfile(path)) {
#ifdef ENABLE_ZLIB
			return _list_import_zlib(path);
#else
			return NULL;
#endif
		} else {
			return _list_import_ansic(path);
		}
		return NULL;
	}

#endif /* ENABLE_FOPS */

#ifdef ENABLE_SSHM
	list_t *list_new_shared(uint scale, uint blen, uint key)
	{
		byte *shm;
		uint shmlen  = 0;
		list_t *list = (list_t*)malloc(sizeof(list_t));
		if (!list)
			return NULL;
		memset(list, 0, sizeof(list_t));
		shmlen += LIST_NAME_LEN;
		shmlen += sizeof(list_t);
		SET_FLAG(list->flag, LIST_STATIC_MODE);
		SET_FLAG(list->flag, LIST_UNRESIZABLE);
		SET_FLAG(list->flag, LIST_SHARED_MEM);
		SET_FLAG(list->flag, LIST_SHARED_OWNER);
		list->length = scale * blen;
		shmlen += list->length;
		shm = (byte*)create_shm(shmlen, key);
		if (!shm) {
			free(list);
			return NULL;
		}
		memset(shm, 0, shmlen);
		memcpy(shm, list, sizeof(list_t));
		free(list);
		list          = (list_t*)shm;
		list->name    = (char*)shm + sizeof(list_t);
		list->mem     = shm + sizeof(list_t) + LIST_NAME_LEN;
		list->key     = key;
		list->blen    = blen;
		list->scale   = scale;
		list->counter = scale;
		return list;
	}

	list_t *list_link_shared(uint len, uint key)
	{
		byte *shm;
		list_t *list = (list_t*)malloc(sizeof(list_t));
		if (!list)
			return NULL;
		shm = (byte*)create_shm(len +
				sizeof(list_t) + LIST_NAME_LEN, key);
		if (!shm) {
			free(list);
			return NULL;
		}
		memcpy(list, shm, sizeof(list_t));
		list->name = (char*)shm + sizeof(list_t);
		list->mem = shm + sizeof(list_t) + LIST_NAME_LEN;
		SET_FLAG(list->flag, LIST_UNRESIZABLE);
		SET_FLAG(list->flag, LIST_SHARED_MEM);
		SET_FLAG(list->flag, LIST_SHARED_USER);
		return list;
	}

	ls_status_t list_del_shared(list_t *list)
	{
		uint shmlen;
		if (TEST_FLAG(list->flag, LIST_SHARED_USER))
			return LSS_ARG_ILL;
		shmlen = (sizeof(list_t) +
				LIST_NAME_LEN + list->length);
		if (free_shm(shmlen, list->key))
			return LSS_SHM_ERR;
		return LSS_SUCCESS;
	}

	#define PROCESS_IMPORT_SHARED(PREFIX) \
	byte *shm;                                            \
	char *name;                                           \
	uint name_len = 0;                                    \
	uint shmlen   = 0;                                    \
	fp = PREFIX##fopen(path, "rb");                       \
	if (!fp)                                              \
		return NULL;                                  \
	list = (list_t*)malloc(sizeof(list_t));               \
	if (!list)                                            \
		return NULL;                                  \
	PREFIX##fread(&name_len, sizeof(uint), 1, fp);        \
	if (!_is_little_endian())                             \
		_memrev(&name_len, sizeof(uint));             \
	name_len++;                                           \
	name = (char*)malloc(name_len);                       \
	if (!name) {                                          \
		free(list);                                   \
		return NULL;                                  \
	}                                                     \
	memset(name, 0, name_len);                            \
	PREFIX##fread(name, name_len - 1, 1, fp);             \
	PREFIX##fread(list, (sizeof(list_t)), 1, fp);         \
	if (!_is_little_endian())                             \
		_switch_byte_order(list);                     \
	shmlen = sizeof(list_t) + LIST_NAME_LEN;              \
	shmlen += list->length;                               \
	shm = (byte*)create_shm(shmlen, key);                 \
	if (!shm) {                                           \
		free(list);                                   \
		free(name);                                   \
		PREFIX##fclose(fp);                           \
		return NULL;                                  \
	}                                                     \
	PREFIX##fread(shm + sizeof(list_t) + LIST_NAME_LEN,   \
		       			list->length, 1, fp); \
	memcpy(shm, list, sizeof(list_t));                    \
	strcpy((char*)shm + sizeof(list_t), name);            \
	free(name);                                           \
	list->name = (char*)shm + sizeof(list_t);             \
	list->mem  = shm + sizeof(list_t) +LIST_NAME_LEN;

#ifdef ENABLE_ZLIB
	static list_t *_list_import_shm_zlib(const char *path, uint key)
	{
		list_t *list;
		ZLIB_FILE fp;
		PROCESS_IMPORT_SHARED(ZLIB_);
		ZLIB_fclose(fp);
		return list;
	}
#endif

	static list_t *_list_import_shm_ansic(const char *path, uint key)
	{
		list_t *list;
		FILE *fp;
		PROCESS_IMPORT_SHARED(IO_);
		fclose(fp);
		return list;
	}

	list_t *list_import_shared(const char *path, uint key)
	{
		if (_is_gzfile(path)) {
#ifdef ENABLE_ZLIB
			return _list_import_shm_zlib(path, key);
#else
			return NULL;
#endif
		} else {
			return _list_import_shm_ansic(path, key);
		}
		return NULL;
	}
#endif /* ENABLE_SSHM */

ls_status_t list_calc_hash_id(list_t *list,
	void *data, uint len, uint arg, uint *id, __hashFx)
{
	uint nhash, cid;
	if (hfx == NULL)
		hfx = _djb_hash;
	nhash = hfx(data, len, arg) % list->scale;
	cid = nhash;
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
		while (list->index[cid]) {
			cid++;
			if (cid == list->scale)
				cid = 0;
			if (cid == nhash)
				return LSS_ERR_LISTFULL;
		}
	} else {  /* LIST_STATIC_MODE */
		while (list_test_record(list, cid, 0, list->blen)) {
			cid++;
			if (cid == list->scale)
				cid = 0;
			if (cid == nhash)
				return LSS_ERR_LISTFULL;
		}
	}
	*id = cid;
	return LSS_SUCCESS;
}

ls_status_t list_search_by_hash(list_t *list,
		void *data, uint len, uint arg, __hashFx,
	__compFx, void *pattern, uint plen, uint *fid)
{
	uint nhash, id;
	if (hfx == NULL)
		hfx = _djb_hash;
	nhash = hfx(data, len, arg) % list->scale;
	id = nhash;
	if (TEST_FLAG(list->flag, LIST_DYNAMIC_MODE)) {
		while (list->index[id]) {
			if (!cfx(list->index[id] +
				sizeof(DynLenFlag), pattern, plen)) {
				*fid = id;
				return LSS_SUCCESS;
			}
			id++;
			if (id == list->scale)
				id = 0;
			if (id == nhash)
				return LSS_OBJ_NOFOUND;
		}
	} else { /* LIST_STATIC_MODE */
		while (list_test_record(list, id, 0, list->blen)) {
			if (!cfx(list->index[id], pattern, plen)) {
				*fid = id;
				return LSS_SUCCESS;
			}
			id++;
			if (id == list->scale)
				id = 0;
			if (id == nhash)
				return LSS_OBJ_NOFOUND;
		}
	}
	return LSS_OBJ_NOFOUND;
}

void list_print_info(list_t *list, void *stream)
{
	FILE *fp = stream ? (FILE*)stream : stdout;
	if (!list) {
		fprintf(fp, "[ERROR] : Bad object, stopped!\n");
		return;
	}
	if (!list->status)
		fprintf(fp, "[Status] : List status: OK!\n");
	else{
		fprintf(fp, "[Status] : Something wrong, stopped!\n");
		return;
	}
	if (list->name)
		fprintf(fp, "[%s]\n", list->name);
	else
		fprintf(fp, "Not named list.\n");
	if (TEST_FLAG(list->flag, LIST_STATIC_MODE))
		fprintf(fp, "[MODE]         = STATIC\n");
	else
		fprintf(fp, "[MODE]         = DYNAMIC\n");
	fprintf(fp, "[length]       = %d\n", list->length);
	fprintf(fp, "[scale]        = %d\n", list->scale);
	fprintf(fp, "[record]       = %d\n", list->counter);
	fprintf(fp, "[block length] = %d\n", list->blen);
}

void operation_status(ls_status_t ops_stat)
{
	FILE *fp = stderr;
	switch (ops_stat) {
		case LSS_SUCCESS:
			fprintf(fp, "Operation Succeeded!\n");
			break;
		case LSS_BAD_ID:
			fprintf(fp, "Failed: Bad ID\n");
			break;
		case LSS_MALLOC_ERR:
			fprintf(fp, "Failed: Mem Alloc Failed\n");
			break;
		case LSS_DYN_ID_EXIST:
			fprintf(fp, "Failed: Record Exist\n");
			break;
		case LSS_BAD_OBJ:
			fprintf(fp, "Failed: Bad Object\n");
			break;
		case LSS_ARG_ILL:
			fprintf(fp, "Failed: Arg Illegal\n");
			break;
		case LSS_LNAME_ERR:
			fprintf(fp, "Failed: Bad LIST name\n");
			break;
		case LSS_FILE_ERR:
			fprintf(fp, "Failed: File Access Err\n");
			break;
		case LSS_SHM_ERR:
			fprintf(fp, "Failed: Shared Memory Err\n");
			break;
		case LSS_ERR_LISTFULL:
			fprintf(fp, "Failed: Full list\n");
			break;
		case LSS_OBJ_NOFOUND:
			fprintf(fp, "Failed: Object Not Found\n");
			break;
		default :
			fprintf(fp, "Failed: Unknown Error\n");
			break;
	}
}
