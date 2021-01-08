#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_rbt.h"
#include "util_log.h"
#include "cc_macro.h"
#include "cc_assert.h"
#include "cc_tsrmgr.h"

static struct cc_tsrmgr_struct {
	int          counter;
	struct rbtree *table;
} *g_tsrmgr; /* global_tsrmgr_table */

struct pair {
	char *key;
	void *dat;
};

static void *getkey(struct rbt_node *node)
{
	struct pair *pair = (struct pair*)node->data;
	return pair->key;
}

static int compare(const void *a, const void *b)
{
	return strcmp((const char*)a, (const char*)b);
}

static void free_pair(struct pair *pair)
{
	/* if (pair) */
	cc_free((cc_tensor_t*)pair->dat);
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
	if (g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: already initialized\n");
		return;
	}
	g_tsrmgr = CC_ALLOC(struct cc_tsrmgr_struct);
	g_tsrmgr->table = rbt_new(getkey, compare);
	return;
}

static void _tsrmgr_clear(struct rbt_node *n)
{
	if (n != rbt_nil()) {
		_tsrmgr_clear(n->left);
		free_pair((struct pair*)n->data);
		_tsrmgr_clear(n->right);
	}
}

void cc_tsrmgr_clear(void)
{
	if (!g_tsrmgr)
		return;
	_tsrmgr_clear(g_tsrmgr->table->root);
	rbt_del(g_tsrmgr->table);
	free(g_tsrmgr);
	g_tsrmgr = NULL;
	return;
}

int cc_tsrmgr_status(void)
{
	return g_tsrmgr ? 1 : 0;
}

void cc_tsrmgr_reg(cc_tensor_t *tensor)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(tensor->name);
#endif
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	if (!strlen(tensor->name)) {
		utlog_format(UTLOG_ERR, "cc_tsrmgr: illegal tensor name\n");
		return;
	}
	ptr = (struct pair*)
		rbt_insert(g_tsrmgr->table, mkpair(tensor));
	if (ptr) {
		utlog_format(UTLOG_ERR,
			"cc_tsrmgr: tensor \"%s\" is exist\n",
			tensor->name);
		if (tensor == ptr->dat)
			free(ptr);
		else
			free_pair(ptr);
	} else {
		g_tsrmgr->counter++;
	}
}

void cc_tsrmgr_replace(cc_tensor_t *tensor)
{
	struct pair *ptr;
	if (!tensor->name)
		return;    /* Nothing happen */
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	if (!strlen(tensor->name)) {
		utlog_format(UTLOG_ERR, "cc_tsrmgr: illegal tensor name\n");
		return;
	}
	ptr = (struct pair*)
		rbt_insert(g_tsrmgr->table, mkpair(tensor));
	if (ptr) {
		if (tensor == ptr->dat)
			free(ptr);
		else
			free_pair(ptr);
	} else {
		g_tsrmgr->counter++;
	}
}

void cc_tsrmgr_del(const char *name)
{
	struct pair *ptr;
#ifdef ENABLE_CC_ASSERT
	cc_assert_ptr(name);
#endif
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	ptr = (struct pair*)
		rbt_erase(g_tsrmgr->table, (void*)name);
	if (ptr) {
		free_pair(ptr);
		g_tsrmgr->counter--;
	}
}

cc_tensor_t *cc_tsrmgr_get(const char *name)
{
	struct pair *ptr;
	if (!name)
		return NULL;
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return NULL;
	}
	ptr = (struct pair*)
		rbt_find(g_tsrmgr->table, (void*)name);
	if (ptr)
		return (cc_tensor_t*)ptr->dat;
	return NULL;
}

#define BUFLEN 128
#define BUFLIM 100
static void _print_property(cc_tensor_t *tensor)
{
	char buf[BUFLEN];
	char *bptr = buf;
	const cc_ssize *sptr;
	if (!tensor) {
		utlog_format(UTLOG_WARN, "invalid tensor\n");
		return;
	}
	sptr = tensor->shape;
	while (*sptr) {
		bptr += sprintf(bptr, "%lld, ", *sptr++);
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

static void _print_pair(struct pair *pair)
{
	fprintf((FILE*)utlog_get_ostream(), "<%p>: \"%s\", ",
		pair->dat, ((cc_tensor_t*)pair->dat)->name);
		_print_property((cc_tensor_t*)pair->dat);
}

static void _cc_tsrmgr_list(struct rbt_node *n)
{
	if (n != rbt_nil()) {
		_cc_tsrmgr_list(n->left);
		_print_pair((struct pair*)n->data);
		_cc_tsrmgr_list(n->right);
	}
}

void cc_tsrmgr_list(void)
{
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return;
	}
	utlog_format(UTLOG_INFO,
		"cc_tsrmgr: handling %d tensor(s)\n", g_tsrmgr->counter);
	_cc_tsrmgr_list(g_tsrmgr->table->root);
}

struct tsrmgr_pack_state {
	struct list *pack;
	cc_tensor_t *csr;
	cc_uint8 *dptr;
	cc_ssize len, off;
	cc_usize idc;
};

static struct tsrmgr_pack_state _ps;

static void _tsrmgr_pack(struct rbt_node *n)
{
	cc_usize i;
	if (n != rbt_nil()) {
		_tsrmgr_pack(n->left);
		_ps.csr = (cc_tensor_t*)
			((struct pair*)n->data)->dat;
		_ps.len = strlen(_ps.csr->name) +
			_ps.csr->container->length + 1;
		cc_assert_ptr(_ps.dptr =
			(cc_uint8*)list_alloc(_ps.pack, _ps.idc, _ps.len));
		strcpy((char*)_ps.dptr, _ps.csr->name);
		_ps.off = strlen(_ps.csr->name) + 1;
		for (i = 0; i < _ps.csr->container->counter; ++i) {
			/* Ref: util_list.h */
			_ps.len = *(rlen_t*)
				_ps.csr->container->index[i] + sizeof(rlen_t);
			memcpy(_ps.dptr + _ps.off,
				_ps.csr->container->index[i], _ps.len);
			_ps.off += _ps.len;
		}
		_ps.idc++;
		_tsrmgr_pack(n->right);
	}
}

struct list *cc_tsrmgr_pack()
{
	if (!g_tsrmgr) {
		utlog_format(UTLOG_WARN, "cc_tsrmgr: not initialized\n");
		return NULL;
	}
	memset(&_ps, 0, sizeof(struct tsrmgr_pack_state));
	cc_assert_ptr(_ps.pack = list_new(g_tsrmgr->counter, 0));
	_tsrmgr_pack(g_tsrmgr->table->root);
	return _ps.pack;
}

void cc_tsrmgr_unpack(struct list *tls)
{
	const char *name;
	cc_tensor_t *t;
	struct list *container;
	cc_uint8 *dptr, *rptr;
	cc_ssize j, off, len;
	cc_usize i;
	if (!cc_tsrmgr_status())
		cc_tsrmgr_init();
	for (i = 0; i < tls->counter; ++i) {
		cc_assert_ptr(
			container = list_new(CC_TENSOR_ITEMS, 0));
		rptr = (cc_uint8*)list_index(tls, i);
		name = (const char*)rptr;
		cc_assert_zero(list_rename(container, name));
		off = strlen(name) + 1;
		for (j = 0; j < CC_TENSOR_ITEMS; ++j) {
			/* Ref: util_list.h */
			len = *(rlen_t*)(rptr + off);
			dptr = rptr + off + sizeof(rlen_t);
			cc_assert_ptr(
				list_set_data(container, j, dptr, len));
			off += (len + sizeof(rlen_t));
		}
		cc_tsrmgr_del(name);
		cc_assert_alloc(t = CC_ALLOC(cc_tensor_t));
		t->container = container;
		t->name = container->name;
		t->data = (cc_uint8*)
			list_index(t->container, CC_TENSOR_DATA);
		cc_assert_ptr(
			t->dtype = (cc_dtype*)
				list_index(container, CC_TENSOR_DTYPE));
		cc_assert_ptr(
			t->shape = (cc_ssize*)
				list_index(container, CC_TENSOR_SHAPE));
		cc_tsrmgr_reg(t);
	}
}

void cc_tsrmgr_export(const char *filename)
{
	struct list *pack;
	cc_assert_ptr(pack = cc_tsrmgr_pack());
	cc_assert_zero(list_export(pack, filename));
	list_del(pack);
}

void cc_tsrmgr_import(const char *filename)
{
	struct list *pack;
	cc_assert_ptr(pack = list_import(filename));
	cc_tsrmgr_unpack(pack);
	list_del(pack);
}

void cc_tsrmgr_gc(enum cc_tsrmgr_ctrl ctrl)
{
	switch (ctrl) {
	case CC_GC_CLEAN:
		break;
	default:
		/* Do nothing */
		break;
	}
}
