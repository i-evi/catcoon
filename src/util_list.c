#define _UTIL_LIST_C_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "util_list.h"

#ifdef ENABLE_SSHM
	#include "sshm.h"
#endif

#define ASSERT_ON_BUILD(condition)\
	((void)sizeof(char[1 - 2*!!(condition)]))

void __________compile_time_test___________()
{
#ifdef ENABLE_FOPS
	ASSERT_ON_BUILD((24 - LIST_INFO_LEN) & (48 - LIST_INFO_LEN));
#endif
}

#ifdef ENABLE_ALIGNMENT
static size_t ls_alignment_order = LIST_ALIGNMENT_DFL;

struct c_aligned_memory {
	void *loc;
	size_t order;
};

static LS_INLINE
size_t c_aligned_calc(size_t size, size_t order)
{
	size_t u = size >> order;
	(u << order) < size ? ++u : u;
	return (++u << order) + sizeof(struct c_aligned_memory);
}

static void *c_aligned_calloc(size_t nmemb, size_t size, size_t order)
{
	size_t u, v;
	struct c_aligned_memory *s;
	void *m = calloc(1, c_aligned_calc(size * nmemb, order));
	if (!m)
		return NULL;
	u = ((size_t)m) + sizeof(struct c_aligned_memory);
	v = u >> order;
	u > (v << order) ? ++v : v;
	v = v << order;
	s = (struct c_aligned_memory*)(v - sizeof(struct c_aligned_memory));
	s->loc = m;
	s->order = order;
	return (void*)v;
}

static void *c_aligned_malloc(size_t size, size_t order)
{
	size_t u, v;
	struct c_aligned_memory *s;
	void *m = malloc(c_aligned_calc(size, order));
	if (!m)
		return NULL;
	u = ((size_t)m) + sizeof(struct c_aligned_memory);
	v = u >> order;
	u > (v << order) ? ++v : v;
	v = v << order;
	s = (struct c_aligned_memory*)(v - sizeof(struct c_aligned_memory));
	s->loc = m;
	s->order = order;
	return (void*)v;
}

static void *c_aligned_realloc(void *ptr, size_t size)
{
	void *nptr;
	size_t off;
	struct c_aligned_memory *s;
	if (!ptr)
		return NULL;
	s = (struct c_aligned_memory*)(
		((size_t)ptr) - sizeof(struct c_aligned_memory));
	off = ((size_t)ptr) - ((size_t)s->loc);
	nptr = realloc(s->loc, c_aligned_calc(size, s->order));
	if (!nptr)
		return NULL;
	ptr = (void*)(((size_t)(nptr)) + off);
	s = (struct c_aligned_memory*)(
		((size_t)ptr) - sizeof(struct c_aligned_memory));
	s->loc = nptr;
	return ptr;
}

static void c_aligned_free(void *ptr)
{
	struct c_aligned_memory *s;
	if (!ptr)
		return;
	s = (struct c_aligned_memory*)(
		((size_t)ptr) - sizeof(struct c_aligned_memory));
	free(s->loc);
}

#define LS_A_MALLOC(size) (c_aligned_malloc(size, ls_alignment_order))
#define LS_A_CALLOC(nmemb, size) \
(c_aligned_calloc(nmemb, size, ls_alignment_order))
#define LS_A_REALLOC(ptr, size) (c_aligned_realloc(ptr, size))
#define LS_A_FREE(ptr) (c_aligned_free(ptr))
#else
#define LS_A_MALLOC(size) (malloc(size))
#define LS_A_CALLOC(nmemb, size) (calloc(nmemb, size))
#define LS_A_REALLOC(ptr, size) (realloc(ptr, size))
#define LS_A_FREE(ptr) (free(ptr))
#endif /* ENABLE_ALIGNMENT */

#define LS_ALLOC(type) ((type*)calloc(1, sizeof(type)))
#define LS_CALLOC(nmemb, size) (calloc(nmemb, size))
#define LS_MALLOC(size) (malloc(size))
#define LS_REALLOC(ptr, size) (realloc(ptr, size))
#define LS_FREE(ptr) (free(ptr))

size_t list_set_alignment(size_t alignment)
{
	size_t i;
	for (i = 0; i < sizeof(size_t) * 8; ++i) {
		if (alignment & (1 << i))
			break;
	}
	assert(!((((size_t)1) << i) - alignment));
#ifdef ENABLE_ALIGNMENT
	ls_alignment_order = i;
#endif
	return i;
}

static lsw_t _gstate;

static uint _djb_hash(const void *s, uint len, uint seed)
{
	uint hash = seed;
	uint i;
	for (i = 0; i < len; i++) {
		hash = (hash << 5) + hash + *((byte*)s + i);
	}
	return hash;
}

static int _strcmp(const void *a, const void *b)
{
	return strcmp((const char*)a, (const char*)b);
}

static uint _string_hash(const void *s, uint seed)
{
	return _djb_hash(s, strlen((const char*)s), seed);
}

static uint _seed;
static __hashFx _hfx = _string_hash;
static __compFx _cmp = _strcmp;

static LS_INLINE
int _is_little_endian(void)
{
	uint i = 1;
	unsigned char *c = (unsigned char*)&i;
	if (*c)
		return 1;
	return 0;
}

static LS_INLINE
void _memswap(void *a, void *b, int len)
{
	unsigned char *p1 = (unsigned char*)a;
	unsigned char *p2 = (unsigned char*)b;
	while (len--) {
		*p1 ^= *p2;
		*p2 ^= *p1;
		*p1++ ^= *p2++;
	}
}

static LS_INLINE
void _memrev(void *i, int len)
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

static LS_INLINE
void _switch_byte_order(struct list *ls) {
	_memrev(&ls->length , sizeof(uint));
	_memrev(&ls->key    , sizeof(uint));
	_memrev(&ls->counter, sizeof(uint));
	_memrev(&ls->blen   , sizeof(uint));
	_memrev(&ls->nmemb  , sizeof(uint));
}

static LS_INLINE
struct list *list_new_static(uint nmemb, uint blen)
{
	struct list *ls;
	if (!nmemb || !blen)
		return NULL;
	ls = LS_ALLOC(struct list);
	if (!ls)
		return NULL;
	SET_FLAG(ls->flag, LIST_STATIC_MODE);
	ls->length = nmemb * blen;
	ls->mem = (byte*)LS_A_CALLOC(ls->length, sizeof(byte));
	if (!ls->mem) {
		LS_FREE(ls);
		return NULL;
	}
	ls->blen    = blen;
	ls->nmemb   = nmemb;
	ls->counter = nmemb;
	return ls;
}

static LS_INLINE
struct list *list_new_dynamic(uint nmemb)
{
	struct list *ls;
	if (!nmemb)
		return NULL;
	ls = LS_ALLOC(struct list);
	if (!ls)
		return NULL;
	SET_FLAG(ls->flag, LIST_DYNAMIC_MODE);
	ls->index = (byte**)LS_CALLOC(nmemb, sizeof(byte*));
	if (!ls->index) {
		LS_FREE(ls);
		return NULL;
	}
	ls->nmemb = nmemb;
	return ls;
}

struct list *list_new(uint nmemb, uint blen)
{
	if (!blen)
		return list_new_dynamic(nmemb);
	return list_new_static(nmemb, blen);
}

static LS_INLINE
struct list *list_clone_static(struct list *ls)
{
	struct list *clone;
	if (!ls)
		return NULL;
	clone = LS_ALLOC(struct list);
	if (!clone)
		return NULL;
	memcpy(clone, ls, sizeof(struct list));
	if (ls->name) {
		clone->name = (char*)LS_MALLOC(LIST_NAME_LEN);
		if (!clone->name) {
			LS_FREE(clone);
			return NULL;
		}
		strcpy(clone->name, ls->name);
	} else {
		clone->name = NULL;
	}
	clone->mem = (byte*)LS_MALLOC(ls->length);
	if (!clone->mem) {
		if (clone->name)
			LS_FREE(clone->name);
		LS_FREE(clone);
		return NULL;
	}
	memcpy(clone->mem, ls->mem, ls->length);
	clone->index = NULL;
	return clone;
}

#define PROCESS_CLONE_DYN_INDEX(i) \
if (ls->index[i]) {                            \
	if (!list_set_data(clone, i,           \
		ls->index[i] + sizeof(rlen_t), \
		*(rlen_t*)ls->index[i])) {     \
		return clone;                  \
	}                                      \
}

static LS_INLINE
struct list *list_clone_dynamic(struct list *ls)
{
	uint i;
	struct list *clone;
	if (!ls)
		return NULL;
	clone = LS_ALLOC(struct list);
	if (!clone)
		return NULL;
	memcpy(clone, ls, sizeof(struct list));
	if (ls->name) {
		clone->name = (char*)LS_MALLOC(LIST_NAME_LEN);
		if (!clone->name) {
			LS_FREE(clone);
			return NULL;
		}
		strcpy(clone->name, ls->name);
	} else {
		clone->name = NULL;
	}
	clone->index = (byte**)LS_MALLOC(ls->nmemb * sizeof(byte*));
	if (!clone->index) {
		if (clone->name)
			LS_FREE(clone->name);
		LS_FREE(clone);
		return NULL;
	}
	memset(clone->index, 0, ls->nmemb * sizeof(byte*));
	clone->length  = 0;
	clone->counter = 0;
	for (i = 0; i < ls->nmemb; ++i) {
		PROCESS_CLONE_DYN_INDEX(i);
	}
	return clone;
}

struct list *list_clone(struct list *ls)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_clone_dynamic(ls);
	else /* LIST_STATIC_MODE */
		return list_clone_static(ls);
}

static LS_INLINE
void list_del_static(struct list *ls)
{
	if (!ls)
		return;
	if (ls->name)
		LS_FREE(ls->name);
	if (ls->mem)
		LS_A_FREE(ls->mem);
	LS_FREE(ls);
}

static LS_INLINE
void list_del_dynamic(struct list *ls)
{
	if (!ls)
		return;
	if (ls->name)
		LS_FREE(ls->name);
	while (ls->nmemb--) {
		if (ls->index[ls->nmemb])
			LS_A_FREE(ls->index[ls->nmemb]);
	}
	if (ls->index)
		LS_FREE(ls->index);
	LS_FREE(ls);
}

void list_del(struct list *ls)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		list_del_dynamic(ls);
	else /* LIST_STATIC_MODE */
		list_del_static(ls);
}

lsw_t list_rename(struct list *ls, const char *name)
{
	if (TEST_FLAG(ls->flag, LIST_NOT_SHARED)) {
		if (!name) {
			if (ls->name)
				LS_FREE(ls->name);
			ls->name = NULL;
			return LSS_SUCCESS;
		}
		if (strlen(name) >= LIST_NAME_LEN)
			return LSS_ARG_ILL;
		if (ls->name)
			LS_FREE(ls->name);
		ls->name = (char*)LS_MALLOC(LIST_NAME_LEN);
		if (!ls->name)
			return LSS_MALLOC_ERR;
		strcpy(ls->name, name);
	} else { /* LIST_SHARED_ */
		strcpy(ls->name, name);
	}
	return LSS_SUCCESS;
}

static LS_INLINE
lsw_t list_resize_static(struct list *ls, uint nmemb)
{
	byte *new_mem;
	uint new_len = nmemb * ls->blen;
	if (TEST_FLAG(ls->flag, LIST_UNRESIZABLE))
		return LSS_BAD_OBJ;
	new_mem = (byte*)LS_A_REALLOC(ls->mem, new_len);
	if (!new_mem)
		return LSS_MALLOC_ERR;
	if (nmemb > ls->nmemb) {
		memset(ls->mem + ls->nmemb * ls->blen, 0,
			(nmemb - ls->nmemb) * ls->blen);
	}
	ls->mem     = new_mem;
	ls->length  = new_len;
	ls->nmemb   = nmemb;
	ls->counter = nmemb;
	return LSS_SUCCESS;
}

static LS_INLINE
lsw_t list_resize_dynamic(struct list *ls, uint nmemb)
{
	byte **new_index;
	uint i = 0;
	if (TEST_FLAG(ls->flag, LIST_UNRESIZABLE))
		return LSS_BAD_OBJ;
	if (nmemb < ls->nmemb) {
		i = nmemb;
		for (; i < ls->nmemb; ++i)
			list_erase(ls, i);
	}
	new_index = (byte**)LS_REALLOC(ls->index, nmemb * sizeof(byte*));
	if (!new_index)
		return LSS_MALLOC_ERR;
	ls->index = new_index;
	if (nmemb > ls->nmemb) {
		memset((byte*)new_index + ls->nmemb * sizeof(byte*), 0,
			(nmemb - ls->nmemb) * sizeof(byte*));
	}
	ls->nmemb = nmemb;
	return LSS_SUCCESS;
}

lsw_t list_resize(struct list *ls, uint nmemb)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_resize_dynamic(ls, nmemb);
	else /* LIST_STATIC_MODE */
		return list_resize_static(ls, nmemb);
}

static LS_INLINE
void *list_set_data_sta(struct list *ls,
	       uint id, const void *data, uint len)
{
	void *ptr;
	if (id >= ls->nmemb) {
		_gstate = LSS_BAD_ID;
		return NULL;
	}
	len = ls->blen < len ? ls->blen : len;
	ptr = ls->mem + id * ls->blen;
	memcpy(ptr, data, len);
	return ptr;
}

static LS_INLINE
void *list_set_data_dyn(struct list *ls,
	uint id, const void *data, uint len)
{
	uint mlen = 0;
	void *ptr;
	if (id >= ls->nmemb) {
		_gstate = LSS_BAD_ID;
		return NULL;
	}
	if (ls->index[id]) {
		_gstate = LSS_DYN_ID_EXIST;
		return NULL;
	}
	mlen = len + sizeof(rlen_t);
	ls->index[id] = (byte*)LS_A_MALLOC(mlen);
	if (!ls->index[id]) {
		_gstate = LSS_MALLOC_ERR;
		return NULL;
	}
	ptr = ls->index[id] + sizeof(rlen_t);
	memcpy(ptr, data, len);
	(*(rlen_t*)ls->index[id]) = len;
	ls->counter++;
	ls->length += mlen;
	return ptr;
}

void *list_set_data(struct list *ls, uint id,
	const void *record, uint len)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_set_data_dyn(ls, id, record, len);
	else /* LIST_STATIC_MODE */
		return list_set_data_sta(ls, id, record, len);
}

void *list_alloc(struct list *ls, uint id, uint nbyte)
{
	if (id >= ls->nmemb)
		return NULL;
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE)) {
		if (ls->index[id])
			list_erase(ls, id);
		ls->index[id] = (byte*)LS_A_MALLOC(nbyte + sizeof(rlen_t));
		if (!ls->index[id])
			return NULL;
		*(rlen_t*)ls->index[id] = nbyte;
		ls->length += (nbyte + sizeof(rlen_t));
		ls->counter++;
		return ls->index[id] + sizeof(rlen_t);
	} else { /* LIST_STATIC_MODE */
		if (nbyte > ls->blen)
			return NULL;
		else
			return ls->mem + id * ls->blen;
	}
}

static LS_INLINE
void *list_index_sta(struct list *ls, uint id)
{
	if (id >= ls->nmemb)
		return NULL;
	else
		return ls->mem + id * ls->blen;
}

static LS_INLINE
void *list_index_dyn(struct list *ls, uint id)
{
	if (id >= ls->nmemb)
		return NULL;
	else {
		if (ls->index[id])
			return (ls->index[id] + sizeof(rlen_t));
		else
			return NULL;
	}
}

void *list_index(struct list *ls, uint id)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_index_dyn(ls, id);
	else /* LIST_STATIC_MODE */
		return list_index_sta(ls, id);
}

static LS_INLINE
lsw_t list_erase_sta(struct list *ls, uint id)
{
	if (id >= ls->nmemb)
		return LSS_BAD_ID;
	memset(ls->mem + id * ls->blen, 0, ls->blen);
	return LSS_SUCCESS;
}

uint list_getlen(struct list *ls, uint id)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE)) {
		if (ls->index[id])
			return *(rlen_t*)ls->index[id];
		else
			return 0;
	}
	else /* LIST_STATIC_MODE */
		return ls->blen;
}

static LS_INLINE
lsw_t list_erase_dyn(struct list *ls, uint id)
{
	if (id >= ls->nmemb)
		return LSS_BAD_ID;
	if (ls->index[id]) {
		ls->length -= sizeof(rlen_t);
		ls->length -= *(rlen_t*)ls->index[id];
		ls->counter--;
		LS_A_FREE(ls->index[id]);
		ls->index[id] = NULL;
	}
	return LSS_SUCCESS;
}

lsw_t list_erase(struct list *ls, uint id)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_erase_dyn(ls, id);
	else /* LIST_STATIC_MODE */
		return list_erase_sta(ls, id);
}

static LS_INLINE
lsw_t list_swap_sta(struct list *ls, uint id1, uint id2)
{
	if (id1 >= ls->nmemb || id2 >= ls->nmemb)
		return LSS_BAD_ID;
	_memswap(ls->mem + id1 * ls->blen,
		ls->mem + id2 * ls->blen, ls->blen);
	return LSS_SUCCESS;
}

static LS_INLINE
lsw_t list_swap_dyn(struct list *ls, uint id1, uint id2)
{
	byte *p;
	if (id1 >= ls->nmemb || id2 >= ls->nmemb)
		return LSS_BAD_ID;
	p = ls->index[id1];
	ls->index[id1] = ls->index[id2];
	ls->index[id2] = p;
	return LSS_SUCCESS;
}

lsw_t list_swap(struct list *ls,
	uint id1, uint id2)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_swap_dyn(ls, id1, id2);
	else /* LIST_STATIC_MODE */
		return list_swap_sta(ls, id1, id2);
}

#ifdef ENABLE_FOPS

static void *std_open(const char *pathname, const char *mode)
{
	return fopen(pathname, mode);
}

static int std_close(void *fp)
{
	return fclose((FILE*)fp);
}

static long std_read(void *buf, long size, void *fp)
{
	return fread(buf, size, 1, (FILE*)fp);
}

static long std_write(void *buf, long size, void *fp)
{
	return fwrite(buf, size, 1, (FILE*)fp);
}

static struct lsio_struct lsio = {
	/* .open  = */ std_open,
	/* .close = */ std_close,
	/* .read  = */ std_read,
	/* .write = */ std_write,
	/* .seek  = */ NULL,
	/* .tell  = */ NULL
};

void list_set_io_stdio(void)
{
	lsio.open  = std_open;
	lsio.close = std_close;
	lsio.read  = std_read;
	lsio.write = std_write;
}

static const unsigned char sebheader[] = {'s', 'e', 'b'};
#ifdef ENABLE_SEB
#include "seb.h"

#define SEB_ENC_CTRL_DFL 1

static int _seb_encode_ctrl = SEB_ENC_CTRL_DFL;

void list_set_seb_ctrl(int v)
{
	_seb_encode_ctrl = v;
}

static void *_seb_open(const char *pathname, const char *mode)
{
	seb_global_parameter(SEB_GLOBAL_ENCODE_CTRL, _seb_encode_ctrl);
	return sebfopen(pathname, mode);
}

static int _seb_close(void *fp)
{
	sebfclose((sebFILE*)fp);
	return 0;
}

static long _seb_read(void *buf, long size, void *fp)
{
	return sebfread(buf, size, 1, (sebFILE*)fp);
}

static long _seb_write(void *buf, long size, void *fp)
{
	return sebfwrite(buf, size, 1, (sebFILE*)fp);
}

void list_set_io_seb(void)
{
	lsio.open  = _seb_open;
	lsio.close = _seb_close;
	lsio.read  = _seb_read;
	lsio.write = _seb_write;
}
#endif /* ENABLE_SEB */

static const unsigned char gzheader[] = {0x1F, 0x8B, 0x08};
#ifdef ENABLE_ZLIB
#include <zlib.h>

#define ZLIB_COMPRESS_LV_DFL 7

static int _zlib_compress_level = ZLIB_COMPRESS_LV_DFL;

void list_set_zlib_ctrl(int v)
{
	_zlib_compress_level = v;
}

static void *_zlib_open(const char *pathname, const char *mode)
{
	char open_r[] = "rb";
	char open_w[] = "wb";
	char open_wl[8];
	if (*mode == 'r') {
		return gzopen(pathname, open_r);
	} else if (*mode == 'w') {
		sprintf(open_wl, "%s%d", open_w, _zlib_compress_level);
		return gzopen(pathname, open_wl);
	} else {
		/* Unsupported mode */
		return NULL;
	}
}

static int _zlib_close(void *fp)
{
	return gzclose((gzFile)fp);
}

static long _zlib_read(void *buf, long size, void *fp)
{
	return gzread((gzFile)fp, buf, size);
}

static long _zlib_write(void *buf, long size, void *fp)
{
	return gzwrite((gzFile)fp, buf, size);
}

void list_set_io_zlib(void)
{
	lsio.open  = _zlib_open;
	lsio.close = _zlib_close;
	lsio.read  = _zlib_read;
	lsio.write = _zlib_write;
}
#endif /* ENABLE_ZLIB */

/*
 * Exported file structure of Static LIST
 * name_len | list->name | list | list->mem  
 */
static lsw_t list_export_static(
	struct list *ls, const char *path)
{
	void *fp;
	uint name_len = 0;
	fp = lsio.open(path, "wb");
	if (!fp)
		return LSS_FILE_ERR;
	if (ls->name)
		name_len = strlen(ls->name);
	if (!_is_little_endian()) {
		_memrev(&name_len, sizeof(uint));
		_switch_byte_order(ls);
	}
	lsio.write(&name_len, sizeof(uint), fp);
	lsio.write(ls->name, name_len, fp);
	lsio.write((byte*)ls +
		LIST_INFO_OFFSET, LIST_INFO_LEN, fp);
	lsio.write(ls->mem, ls->length, fp);
	if (!_is_little_endian()) {
		_switch_byte_order(ls);
	}
	lsio.close(fp);
	return LSS_SUCCESS;
}

#define PROCESS_EXPORT_DYN \
if (ls->index[sc]) {                                    \
	if (!_is_little_endian())                       \
		_memrev(ls->index[sc], sizeof(rlen_t)); \
	lsio.write(ls->index[sc], sizeof(rlen_t) +      \
		*(rlen_t*)ls->index[sc], fp);           \
	if (!_is_little_endian()) {                     \
		_memrev(ls->index[sc], sizeof(rlen_t)); \
		_memrev(&sc, sizeof(uint));             \
	}                                               \
	lsio.write(&sc, sizeof(uint), fp);              \
	if (!_is_little_endian())                       \
		_memrev(&sc, sizeof(uint));             \
}

/*
 * Exported file structure of Dynamic LIST
 * name_len | list->name | list | *index[0:N]
 */
static lsw_t list_export_dynamic(
	struct list *ls, const char *path)
{
	void *fp;
	uint name_len = 0;
	uint sc = ls->nmemb;
	fp = lsio.open(path, "wb");
	if (!fp)
		return LSS_FILE_ERR;
	if (ls->name)
		name_len = strlen(ls->name);
	if (!_is_little_endian()) {
		_memrev(&name_len, sizeof(uint));
		_switch_byte_order(ls);
	}
	lsio.write(&name_len, sizeof(uint), fp);
	lsio.write(ls->name, name_len, fp);
	lsio.write((byte*)ls +
		LIST_INFO_OFFSET, LIST_INFO_LEN, fp);
	while (sc--) {
		PROCESS_EXPORT_DYN;
	}
	if (!_is_little_endian()) {
		_switch_byte_order(ls);
	}
	lsio.close(fp);
	return LSS_SUCCESS;
}

lsw_t list_export(struct list *ls, const char *path)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return list_export_dynamic(ls, path);
	else /* LIST_STATIC_MODE */
		return list_export_static(ls, path);
}

#define PROCESS_IMPORT_DYN_RECORD \
	ls->index = (byte**)                                    \
		LS_MALLOC(ls->nmemb * sizeof(byte*));           \
	if (!ls->index) {                                       \
		LS_FREE(ls);                                    \
		LS_FREE(name);                                  \
		lsio.close(fp);                                 \
		return NULL;                                    \
	}                                                       \
	memset(ls->index, 0, ls->nmemb * sizeof(byte*));        \
	while (lsio.read(&len, sizeof(rlen_t), fp)) {           \
		if (!_is_little_endian())                       \
			_memrev(&len, sizeof(rlen_t));          \
		tmp = (byte*)LS_A_MALLOC(len + sizeof(rlen_t)); \
		if (!tmp) {                                     \
			lsio.close(fp);                         \
			return ls;                              \
		}                                               \
		*(rlen_t*)tmp = len;                            \
		lsio.read(tmp + sizeof(rlen_t), len, fp);       \
		lsio.read(&id, sizeof(uint), fp);               \
		if (!_is_little_endian())                       \
			_memrev(&id, sizeof(uint));             \
		ls->index[id] = tmp;                            \
	}

#define PROCESS_IMPORT_STATIC_RECORD \
	ls->mem = (byte*)LS_A_MALLOC(ls->length); \
	if (!ls->mem) {                           \
		LS_FREE(ls);                      \
		LS_FREE(name);                    \
		lsio.close(fp);                   \
		return NULL;                      \
	}                                         \
	lsio.read(ls->mem, ls->length, fp);

#define PROCESS_IMPORT_OPEN_FILE \
	fp = lsio.open(path, "rb");                \
	if (!fp)                                   \
		return NULL;                       \
	ls = LS_ALLOC(struct list);                \
	if (!ls) {                                 \
		lsio.close(fp);                    \
		return NULL;                       \
	}                                          \
	lsio.read(&name_len, sizeof(uint), fp);    \
	if (!_is_little_endian())                  \
		_memrev(&name_len, sizeof(uint));  \
	if (name_len) {                            \
		name_len++;                        \
		name = (char*)LS_MALLOC(name_len); \
		if (!name) {                       \
			LS_FREE(ls);               \
			lsio.close(fp);            \
			return NULL;               \
		}                                  \
		memset(name, 0, name_len);         \
		lsio.read(name, name_len - 1, fp); \
	}                                          \
	lsio.read((byte*)ls + LIST_INFO_OFFSET,    \
		LIST_INFO_LEN, fp);                \
	if (!_is_little_endian())                  \
		_switch_byte_order(ls);            \
	ls->name = name;

static int _allow_io_auto_switch = 1;

void list_enable_io_auto_switch(void)
{
	_allow_io_auto_switch = 1;
}

void list_disable_io_auto_switch(void)
{
	_allow_io_auto_switch = 0;
}

#define F_MAGIC_LEN 4
static void _list_import_io_auto_switch(const char *pathname)
{
	unsigned char buf[F_MAGIC_LEN] = {0};
	FILE *fp = fopen(pathname, "rb");
	assert(fp);
	fread(buf, F_MAGIC_LEN, 1, fp);
	if (!memcmp(sebheader, buf, sizeof(sebheader))) {
#ifdef ENABLE_SEB
		list_set_io_seb();
#else
		assert(0);
#endif /* ENABLE_SEB */
	} else if (!memcmp(gzheader, buf, sizeof(gzheader))) {
#ifdef ENABLE_ZLIB
		list_set_io_zlib();
#else
		assert(0);
#endif /* ENABLE_ZLIB */
	} else {
		list_set_io_stdio();
	}
	fclose(fp);
}

struct list *list_import(const char *path)
{
	struct list *ls;
	struct lsio_struct lsio_bak;
	void *fp;
	char *name = NULL;
	byte *tmp;
	uint name_len, id;
	rlen_t len;
	if (_allow_io_auto_switch) {
		lsio_bak = lsio;
		_list_import_io_auto_switch(path);
	}
	PROCESS_IMPORT_OPEN_FILE;
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE)) {
		PROCESS_IMPORT_DYN_RECORD;
	} else {  /* LIST_STATIC_MODE */
		PROCESS_IMPORT_STATIC_RECORD;
	}
	lsio.close(fp);
	if (_allow_io_auto_switch)
		lsio = lsio_bak;
	return ls;
}

struct lsio_struct *list_get_io_ctrl_struct(void)
{
	return &lsio;
}
#else  /* DISABLE FILE OPS  */
struct lsio_struct *list_get_io_ctrl_struct(void)
{
	return NULL;
}
#endif /* ENABLE_FOPS */

#ifdef ENABLE_SSHM
struct list *list_new_shared(uint nmemb, uint blen, uint key)
{
	byte *shm;
	uint shmlen = 0;
	struct list *ls = LS_ALLOC(struct list);
	if (!ls)
		return NULL;
	memset(ls, 0, sizeof(struct list));
	shmlen += LIST_NAME_LEN;
	shmlen += sizeof(struct list);
	SET_FLAG(ls->flag, LIST_STATIC_MODE);
	SET_FLAG(ls->flag, LIST_UNRESIZABLE);
	SET_FLAG(ls->flag, LIST_SHARED_MEM);
	SET_FLAG(ls->flag, LIST_SHARED_OWNER);
	ls->length = nmemb * blen;
	shmlen += ls->length;
	shm = (byte*)create_shm(shmlen, key);
	if (!shm) {
		LS_FREE(ls);
		return NULL;
	}
	memset(shm, 0, shmlen);
	memcpy(shm, ls, sizeof(struct list));
	LS_FREE(ls);
	ls      = (struct list*)shm;
	ls->mem = shm + sizeof(struct list) + LIST_NAME_LEN;
	ls->key = key;
	ls->blen    = blen;
	ls->nmemb   = nmemb;
	ls->counter = nmemb;
	ls->name    = (char*)shm + sizeof(struct list);
	return ls;
}

struct list *list_link_shared(uint len, uint key)
{
	byte *shm;
	struct list *ls = LS_ALLOC(struct list);
	if (!ls)
		return NULL;
	shm = (byte*)create_shm(len +
		sizeof(struct list) + LIST_NAME_LEN, key);
	if (!shm) {
		LS_FREE(ls);
		return NULL;
	}
	memcpy(ls, shm, sizeof(struct list));
	ls->name = (char*)shm + sizeof(struct list);
	ls->mem = shm + sizeof(struct list) + LIST_NAME_LEN;
	SET_FLAG(ls->flag, LIST_UNRESIZABLE);
	SET_FLAG(ls->flag, LIST_SHARED_MEM);
	SET_FLAG(ls->flag, LIST_SHARED_USER);
	return ls;
}

lsw_t list_del_shared(struct list *ls)
{
	uint shmlen;
	if (TEST_FLAG(ls->flag, LIST_SHARED_USER))
		return LSS_ARG_ILL;
	shmlen = (sizeof(struct list) +
			LIST_NAME_LEN + ls->length);
	if (free_shm(shmlen, ls->key))
		return LSS_SHM_ERR;
	return LSS_SUCCESS;
}

#endif /* ENABLE_SSHM */

#if defined(ENABLE_SSHM) && defined(ENABLE_FOPS)
#define PROCESS_IMPORT_SHARED \
	byte *shm;                                      \
	char *name;                                     \
	uint name_len = 0;                              \
	uint shmlen   = 0;                              \
	fp = lsio.open(path, "rb");                     \
	if (!fp)                                        \
		return NULL;                            \
	ls = LS_ALLOC(struct list);                     \
	if (!ls)                                        \
		return NULL;                            \
	lsio.read(&name_len, sizeof(uint), fp);         \
	if (!_is_little_endian())                       \
		_memrev(&name_len, sizeof(uint));       \
	name_len++;                                     \
	name = (char*)LS_MALLOC(name_len);              \
	if (!name) {                                    \
		LS_FREE(ls);                            \
		return NULL;                            \
	}                                               \
	memset(name, 0, name_len);                      \
	lsio.read(name, name_len - 1, fp);              \
	lsio.read(ls, (sizeof(struct list)), fp);       \
	if (!_is_little_endian())                       \
		_switch_byte_order(ls);                 \
	shmlen = sizeof(struct list) + LIST_NAME_LEN;   \
	shmlen += ls->length;                           \
	shm = (byte*)create_shm(shmlen, key);           \
	if (!shm) {                                     \
		LS_FREE(ls);                            \
		LS_FREE(name);                          \
		lsio.close(fp);                         \
		return NULL;                            \
	}                                               \
	lsio.read(shm + sizeof(struct list) +           \
			LIST_NAME_LEN, ls->length, fp); \
	memcpy(shm, ls, sizeof(struct list));           \
	strcpy((char*)shm + sizeof(struct list), name); \
	LS_FREE(name);                                  \
	ls->name = (char*)shm + sizeof(struct list);    \
	ls->mem  = shm + sizeof(struct list) + LIST_NAME_LEN;

struct list *
list_import_shared(const char *path, uint key)
{
	struct list *ls;
	void *fp;
	PROCESS_IMPORT_SHARED;
	lsio.close(fp);
	return ls;
}
#endif /* ENABLE_SSHM && ENABLE_FOPS */

ccnt_t list_get_record_counter(struct list *ls, uint id)
{
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE)) {
		if (ls->index[id])
			return 1;
		else
			return 0;
	} else { /* LIST_STATIC_MODE */
		if (TEST_FLAG(ls->flag, LIST_HASH_TABLE))
			return (*(ccnt_t*)(ls->mem + id * ls->blen)) >> 1;
	}
	return 1;
}

int list_hash_table_test_id(struct list *ls, uint id)
{
	return (*(ccnt_t*)(ls->mem + id * ls->blen)) & 1;
}

struct list *list_new_hash_table(uint nmemb, uint blen)
{
	struct list *ls;
        if (!blen) /* support LIST_STATIC_MODE only */
		return NULL;
	ls = list_new_static(nmemb, blen + sizeof(ccnt_t));
	if (!ls)
		return NULL;
	SET_FLAG(ls->flag, LIST_UNRESIZABLE);
	SET_FLAG(ls->flag, LIST_HASH_TABLE);
	return ls;
}

lsw_t list_hash_id_calc(struct list *ls,
	const void *data, uint *hi, uint *id)
{
	uint nhash, cid;
	nhash = _hfx(data, _seed) % ls->nmemb;
	*hi = nhash;
	cid = nhash;
	while (list_hash_table_test_id(ls, cid)) {
		cid++;
		if (cid == ls->nmemb)
			cid = 0;
		if (cid  == nhash)
			return LSS_ERR_LISTFULL;
	}
	*id = cid;
	return LSS_SUCCESS;
}

lsw_t list_hash_table_insert(struct list *ls,
	const void *record, uint len)
{
	uint hi, id, dlen;
	ccnt_t *hrec, *rrec;
	lsw_t stat;
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return LSS_BAD_OBJ;
	if ((stat = list_hash_id_calc(ls, record, &hi, &id)))
		return stat;
	hrec = (ccnt_t*)(ls->mem + hi * ls->blen);
	rrec = (ccnt_t*)(ls->mem + id * ls->blen);
	dlen = ls->blen - sizeof(ccnt_t);
	if (CCNT_VAL(*hrec) == CCNT_MAX)
		return LSS_CCNT_MAX;
	memcpy(rrec + 1, record, len > dlen ? dlen : len);
	(*hrec) += CCNT_INC;
	(*rrec) |= 1; /* SET FLAG */
	return LSS_SUCCESS;
}

lsw_t list_hash_table_find(struct list *ls,
		const void *key, uint *id)
{
	uint nhash, cid;
	byte *rec;
	ccnt_t nrec;
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return LSS_BAD_OBJ;
	nhash = _hfx(key, _seed) % ls->nmemb;
	cid = nhash;
	rec  = ls->mem + cid * ls->blen;
	nrec = (*(ccnt_t*)(ls->mem + cid * ls->blen)) >> 1;
	if (!nrec)
		return LSS_OBJ_NOFOUND;
	while (nrec--) {
		if (_hfx(rec + sizeof(ccnt_t),
			_seed) % ls->nmemb == nhash) {
			if (!_cmp(rec + sizeof(ccnt_t), key)) {
				*id = cid;
				return LSS_SUCCESS;
			}
		}
		cid++;
		if (cid == nhash)
			return LSS_OBJ_NOFOUND;
		if (cid == ls->nmemb) {
			cid = 0;
			rec = ls->mem;
		} else {
			rec += ls->blen;
		}
	}
	return LSS_SUCCESS;
}

lsw_t list_hash_table_del(struct list *ls, const void *key)
{
	uint nhash, id;
	byte *rec, *csr;
	ccnt_t nrec;
	if (TEST_FLAG(ls->flag, LIST_DYNAMIC_MODE))
		return LSS_BAD_OBJ;
	nhash = _hfx(key, _seed) % ls->nmemb;
	rec   = ls->mem + nhash * ls->blen;
	csr   = rec;
	nrec  = (*(ccnt_t*)rec) >> 1;
	if (!nrec)
		return LSS_SUCCESS;
	id = nhash;
	while (1) {
		if (_hfx(csr + sizeof(ccnt_t),
			_seed) % ls->nmemb == nhash) {
			if (!_cmp(csr + sizeof(ccnt_t), key)) {
				nrec--;
				if (!nrec)
					break;
			}
		}
		id++;
		if (id == nhash) {
			if (nrec)
				return LSS_OBJ_NOFOUND;
			return LSS_BAD_OBJ;
		}
		if (id == ls->nmemb) {
			id  = 0;
			csr = ls->mem;
		} else {
			csr += ls->blen;
		}
	}
	(*(ccnt_t*)csr) ^= 1; /* CLR FLAG */
	(*(ccnt_t*)rec) -= CCNT_INC;
	return LSS_SUCCESS;
}

void list_print_properties(struct list *ls, void *stream)
{
	FILE *fp = stream ? (FILE*)stream : stdout;
	if (!ls) {
		fprintf(fp, "[ERROR] : Bad object, stopped!\n");
		return;
	}
	if (ls->name)
		fprintf(fp, "[%s]\n", ls->name);
	else
		fprintf(fp, "Not named list.\n");
	if (TEST_FLAG(ls->flag, LIST_STATIC_MODE))
		fprintf(fp, "[MODE]         = STATIC\n");
	else
		fprintf(fp, "[MODE]         = DYNAMIC\n");
	fprintf(fp, "[length]       = %lld\n", ls->length);
	fprintf(fp, "[nmemb]        = %lld\n", ls->nmemb);
	fprintf(fp, "[record]       = %lld\n", ls->counter);
	fprintf(fp, "[block length] = %lld\n", ls->blen);
}

#ifdef LS_TYPE_BYTE
	#undef byte
#endif
#ifdef LS_TYPE_UINT
	#undef uint
#endif
#undef _UTIL_LIST_C_
