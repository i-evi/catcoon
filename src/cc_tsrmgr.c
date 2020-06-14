#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_rbt.h"
#include "util_log.h"
#include "cc_assert.h"
#include "cc_tsrmgr.h"

static int global_counter;
static rbt_t *global_tsrmgr_table;

struct pair {
	char *key;
	void *dat;
};

static void *getkey(struct rbt_node *node)
{
	struct pair *pair = (struct pair*)node->data;
	return pair->key;
}

static int compare(void *a, void *b)
{
	return strcmp((char*)a, (char*)b);
}

static void free_pair(struct pair *pair)
{
	/* if (pair) */
	cc_free_tensor((cc_tensor_t*)pair->dat);
	free(pair);
}

static struct pair *mkpair(cc_tensor_t *tensor)
{
	struct pair *pair = (struct pair*)malloc(sizeof(struct pair));
	cc_assert_ptr(pair->dat = tensor);
	cc_assert_ptr(pair->key = (char*)tensor->name);
	return pair;
}

void cc_tsrmgr_init(void)
{
	if (global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: already initialized\n");
		return;
	}
	global_tsrmgr_table = new_rbt(getkey, compare);
	return;
}

void cc_tsrmgr_clear(void)
{
	rbt_iterator *it;
	struct pair *pair;
	if (!global_tsrmgr_table) return;
	it = new_rbt_iterator(global_tsrmgr_table);
	while (rbt_iterator_has_next(it)) {
		pair = (struct pair*)rbt_iterator_next(it);
		free_pair(pair);
	}
	free_rbt_iterator(it);
	free_rbt(global_tsrmgr_table);
	global_tsrmgr_table = NULL;
	global_counter = 0;
	return;
}

int cc_tsrmgr_status(void)
{
	return global_tsrmgr_table ? 1 : 0;
}

void cc_tsrmgr_reg(cc_tensor_t *tensor)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(tensor->name);
#endif
	if (!global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	if (!strlen(tensor->name)) {
		utlog_format(UTLOG_ERR, "cc_tsrmgr: illegal tensor name\n");
		return;
	}
	ptr = (struct pair*)
		rbt_insert(global_tsrmgr_table, mkpair(tensor));
	if (ptr) {
		utlog_format(UTLOG_ERR,
			"cc_tsrmgr: tensor \"%s\" is exist\n",
			tensor->name);
		if (tensor == ptr->dat)
			free(ptr);
		else
			free_pair(ptr);
	} else {
		global_counter++;
	}
}

void cc_tsrmgr_replace(cc_tensor_t *tensor)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(tensor->name);
#endif
	if (!global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	if (!strlen(tensor->name)) {
		utlog_format(UTLOG_ERR, "cc_tsrmgr: illegal tensor name\n");
		return;
	}
	ptr = (struct pair*)
		rbt_insert(global_tsrmgr_table, mkpair(tensor));
	if (ptr) {
		if (tensor == ptr->dat)
			free(ptr);
		else
			free_pair(ptr);
	} else {
		global_counter++;
	}
}

void cc_tsrmgr_del(const char *name)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(name);
#endif
	if (!global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	ptr = (struct pair*)
		rbt_delete(global_tsrmgr_table, (void*)name);
	if (ptr) {
		free_pair(ptr);
		global_counter--;
	}
}

cc_tensor_t *cc_tsrmgr_get(const char *name)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(name);
#endif
	if (!global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return NULL;
	}
	ptr = (struct pair*)
		rbt_search(global_tsrmgr_table, (void*)name);
	if (ptr)
		return (cc_tensor_t*)ptr->dat;
	return NULL;
}

#define BUFLEN 128
#define BUFLIM 100
static void _print_tensor_property(cc_tensor_t *tensor)
{
	char buf[BUFLEN];
	char *bptr = buf;
	const cc_int32 *sptr;
	if (!tensor) {
		utlog_format(UTLOG_WARN, "invalid tensor\n");
		return;
	}
	sptr = tensor->shape;
	while (*sptr) {
		bptr += sprintf(bptr, "%d, ", *sptr++);
		if ((bptr - buf) > BUFLIM) {
			sprintf(bptr, "..., ");
			break;
		}
	}
	buf[strlen(buf) - 2] = '\0'; /* clear ", " */
	fprintf((FILE*)utlog_get_ostream(),
		"dtype: \"%s\", shape: [%s]\n",
		cc_dtype_to_string(*tensor->dtype), buf);
}

#define FMTLEN 32
static void _print_pair(struct pair *pair)
{
	char fmt[FMTLEN];
	sprintf(fmt, "<0x%%0%dlx>: \"%%s\", ", (int)(sizeof(void*) << 1));
	fprintf((FILE*)utlog_get_ostream(), fmt, pair->dat,
			((cc_tensor_t*)pair->dat)->name);
		_print_tensor_property((cc_tensor_t*)pair->dat);
}

static void _cc_tsrmgr_list_traversal(struct rbt_node *n)
{
	if (n != rbt_nil()) {
		_cc_tsrmgr_list_traversal(n->left);
		_print_pair((struct pair*)n->data);
		_cc_tsrmgr_list_traversal(n->right);
	}
}

void cc_tsrmgr_list(void)
{
	if (!global_tsrmgr_table) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	utlog_format(UTLOG_INFO,
		"cc_tsrmgr: handling %d tensor(s)\n", global_counter);
	_cc_tsrmgr_list_traversal(global_tsrmgr_table->root);
}

list_t *cc_tsrmgr_pack()
{
	list_t *pack;
	cc_uint8 *dptr;
	cc_int32 len, off;
	cc_tensor_t *csr;
	cc_int32 i, idc = 0;
	rbt_iterator *it = new_rbt_iterator(global_tsrmgr_table);
	cc_assert_ptr(pack = list_new_dynamic(global_counter));
	while (rbt_iterator_has_next(it)) {
		csr = (cc_tensor_t*)
			((struct pair*)rbt_iterator_next(it))->dat;
		len = strlen(csr->name) + csr->container->length + 1;
		cc_assert_ptr(dptr = (cc_uint8*)list_alloc(pack, idc, len));
		strcpy((char*)dptr, csr->name);
		off = strlen(csr->name) + 1;
		for (i = 0; i < csr->container->counter; ++i) {
			/* Ref: util_list.h */
			len = *(DynLenFlag*)
				csr->container->index[i] + sizeof(DynLenFlag);
			memcpy(dptr + off, csr->container->index[i], len);
			off += len;
		}
		idc++;
	}
	free_rbt_iterator(it);
	return pack;
}

void cc_tsrmgr_unpack(list_t *tls)
{
	const char *name;
	cc_tensor_t *t;
	list_t *container;
	cc_uint8 *dptr, *rptr;
	cc_int32 i, j, off, len;
	for (i = 0; i < tls->counter; ++i) {
		cc_assert_ptr(
			container = list_new_dynamic(CC_TENSOR_ITEMS));
		rptr = (cc_uint8*)list_get_record(tls, i);
		name = (const char*)rptr;
		cc_assert_zero(list_set_name(container, name));
		off = strlen(name) + 1;
		for (j = 0; j < CC_TENSOR_ITEMS; ++j) {
			/* Ref: util_list.h */
			len = *(DynLenFlag*)(rptr + off);
			dptr = rptr + off + sizeof(DynLenFlag);
			cc_assert_zero(
				list_set_record(container, j, dptr, len));
			off += (len + sizeof(DynLenFlag));
		}
		cc_tsrmgr_del(name);
		cc_assert_alloc(
			t = (cc_tensor_t*)malloc(sizeof(cc_tensor_t)));
		t->container = container;
		t->name = container->name;
		t->data = (cc_uint8*)
			list_get_record(t->container, CC_TENSOR_DATA);
		cc_assert_ptr(
			t->dtype = (cc_dtype*)
				list_get_record(container, CC_TENSOR_DTYPE));
		cc_assert_ptr(
			t->shape = (cc_int32*)
				list_get_record(container, CC_TENSOR_SHAPE));
		cc_tsrmgr_reg(t);
	}
}
