#include <stdio.h>
#include <stdlib.h>

#include "util_rbt.h"

static struct rbt_node node_nil = {
	NULL,           /* .data   = NULL,           */
	&node_nil,      /* .parent = &node_nil,      */
	&node_nil,      /* .left   = &node_nil,      */
	&node_nil,      /* .right  = &node_nil,      */
	RBT_COLOR_BLACK /* .color  = RBT_COLOR_BLACK */
};

static void* __pointer(struct rbt_node *node) {
	return node->data;
}

static int __compare(void *a, void *b) {
	return (unsigned char*)a - (unsigned char*)b;
}

struct rbt_node *rbt_nil(void)
{
	return &node_nil;
}

rbt_t *new_rbt(void*(*get_key)(struct rbt_node *), 
			int (*compare)(void*, void*))
{
	rbt_t *t = (rbt_t*)malloc(sizeof(rbt_t));
	t->root = &node_nil;
	t->get_key = get_key;
	t->compare = compare;
	t->get_key = t->get_key ? t->get_key : __pointer;
	t->compare = t->compare ? t->compare : __compare;
	return t;
}

static struct rbt_node *new_rbt_node(void *d)
{
	struct rbt_node *z = (struct rbt_node*)
		malloc(sizeof(struct rbt_node));
	z->data   = d;
	z->parent = &node_nil;
	z->left   = &node_nil;
	z->right  = &node_nil;
	z->color  = RBT_COLOR_RED;
	return z;
}

static void left_rotate(rbt_t *t, struct rbt_node *node)
{
	struct rbt_node *y = node->right;
	node->right = y->left;
	if (y->left != &node_nil)
		y->left->parent = node;
	y->parent = node->parent;
	if(node->parent == &node_nil)
		t->root = y;
	else if(node == node->parent->left)
		node->parent->left = y;
	else
		node->parent->right = y;
	y->left = node;
	node->parent = y;
}

static void right_rotate(rbt_t *t, struct rbt_node *node)
{
	struct rbt_node *y = node->left;
	node->left = y->right;
	if(y->right != &node_nil)
		y->right->parent = node;  
	y->parent = node->parent;
	if(node->parent == &node_nil)
		t->root = y;
	else if(node == node->parent->right)
		node->parent->right = y;
	else
		node->parent->left = y;
	y->right = node;
	node->parent = y;
}

static void rbt_insert_fixup(rbt_t *t, struct rbt_node *z)
{
	struct rbt_node *y;
	while (z->parent->color == RBT_COLOR_RED) {
		if (z->parent == z->parent->parent->left) {
			y = z->parent->parent->right;
		if (y->color == RBT_COLOR_RED) {
			z->parent->color = RBT_COLOR_BLACK;
			y->color = RBT_COLOR_BLACK;
			z->parent->parent->color = RBT_COLOR_RED;
			z = z->parent->parent;
		} else {
			if (z == z->parent->right) {
				z = z->parent;
				left_rotate(t, z);
			}
			z->parent->color = RBT_COLOR_BLACK;
			z->parent->parent->color = RBT_COLOR_RED;
			right_rotate(t, z->parent->parent);
		}
		} else {
			y = z->parent->parent->left;
			if (y->color == RBT_COLOR_RED) {
				z->parent->color = RBT_COLOR_BLACK;
				y->color = RBT_COLOR_BLACK;
				z->parent->parent->color = RBT_COLOR_RED;
				z = z->parent->parent;
			} else {
				if (z == z->parent->left) {
					z = z->parent;
					right_rotate(t, z);
				}
				z->parent->color = RBT_COLOR_BLACK;
				z->parent->parent->color = RBT_COLOR_RED;
				left_rotate(t, z->parent->parent);
			}
		}
	}
	t->root->color = RBT_COLOR_BLACK;
}

void *rbt_insert(rbt_t *t, void *d)
{
	void *holder;
	struct rbt_node *y = &node_nil, *x = t->root;
	struct rbt_node *z = new_rbt_node(d);
	while (x != &node_nil) {
		y = x;
		if (!t->compare(t->get_key(z), t->get_key(x))) {
			holder = x->data;
			free(z);
			x->data = d;
			return holder;
		}
		if (t->compare(t->get_key(z), t->get_key(x)) < 0)
			x = x->left;
		else
			x = x->right;
	}
	z->parent = y;
	if (y == &node_nil)
		t->root = z;
	else {
		if (t->compare(t->get_key(z), t->get_key(y)) < 0)
		y->left = z;
	else
		y->right = z;
	}
	rbt_insert_fixup(t, z);
	return NULL;
}

static struct rbt_node *__search_rbt_node(rbt_t *t, void *key)
{
	struct rbt_node *z = t->root;
	while (z != &node_nil) {
		if (t->compare(t->get_key(z), key) == 0)
			return z;
		if (t->compare(t->get_key(z), key) < 0)
			z = z->right;
		else
			z = z->left;
	}
	return NULL;
}

void *rbt_search(rbt_t *t, void *key)
{
	struct rbt_node *z = __search_rbt_node(t, key);
	if (z)
		return z->data;
	return NULL;
}

static void __free_rbt(struct rbt_node *root)
{
	struct rbt_node *l, *r;
	if (root == &node_nil)
		return;
	l = root->left;
	r = root->right;
	free(root);
	__free_rbt(l);
	__free_rbt(r);
}

void free_rbt(rbt_t *t)
{
	__free_rbt(t->root);
	free(t);
}

static void __rb_transplant(rbt_t *t,
	struct rbt_node *u, struct rbt_node *v)
{
	if (u->parent == &node_nil)
		t->root = v;
	else if (u == u->parent->left)
		u->parent->left = v;
	else
		u->parent->right = v;
	v->parent = u->parent;
}

static struct rbt_node *__rb_tree_minimum(struct rbt_node *z)
{
	for(; z->left != &node_nil; z = z->left);
	return z;
}

static void __rb_tree_delete_fixup(rbt_t *t, struct rbt_node *x)
{
	struct rbt_node *w;
 	while (x != t->root && x->color == RBT_COLOR_BLACK) {
		if (x == x->parent->left) {
			w = x->parent->right;
			if (w->color == RBT_COLOR_RED) {
				w->color = RBT_COLOR_BLACK;
				x->parent->color = RBT_COLOR_RED;
				left_rotate(t, x->parent);
				w = x->parent->right;
			}
			if (w->left->color == RBT_COLOR_BLACK &&
				w->right->color == RBT_COLOR_BLACK){
				w->color = RBT_COLOR_RED;
				x = x->parent;
			} else {
				if (w->right->color == RBT_COLOR_BLACK) {
					w->left->color = RBT_COLOR_BLACK;
					w->color = RBT_COLOR_RED;
					right_rotate(t, w);
					w = w->parent->right;
				}
				w->color = x->parent->color;
				x->parent->color = RBT_COLOR_BLACK;
				w->right->color = RBT_COLOR_BLACK;
				left_rotate(t, x->parent);
				x = t->root;
			}
		} else {
			w = x->parent->left;
			if (w->color == RBT_COLOR_RED) {
				w->color = RBT_COLOR_BLACK;
				x->parent->color = RBT_COLOR_RED;
				right_rotate(t, x->parent);
				w = x->parent->left;
			}
			if (w->right->color == RBT_COLOR_BLACK &&
				w->left->color == RBT_COLOR_BLACK) {
				w->color = RBT_COLOR_RED;
				x = x->parent;
			} else {
				if (w->left->color == RBT_COLOR_BLACK) {
					w->right->color = RBT_COLOR_BLACK;
					w->color = RBT_COLOR_RED;
					left_rotate(t, w);
					w = w->parent->left;
				}
				w->color = x->parent->color;
				x->parent->color = RBT_COLOR_BLACK;
				w->left->color = RBT_COLOR_BLACK;
				right_rotate(t, x->parent);
				x = t->root;
			}
		}
	}
	x->color = RBT_COLOR_BLACK;
}

void *rbt_delete(rbt_t *t, void *key)
{
	struct rbt_node *y, *z, *x, *hold_node_to_delete;
	unsigned char y_original_color;
	void *node_to_return;
	hold_node_to_delete = y = z = __search_rbt_node(t, key);
	if (y == NULL) /* Node not exist */
		return NULL;
	node_to_return = y->data;
	y_original_color = y->color;
	if (z->left == &node_nil) {
		x = z->right;
		__rb_transplant(t, z, z->right);
	} else if (z->right == &node_nil) {
		x = z->left;
		__rb_transplant(t, z, z->left);
	} else {
		y = __rb_tree_minimum(z->right);
		y_original_color = y->color;
		x = y->right;
		if (y->parent == z)
			x->parent = y;
		else {
			__rb_transplant(t, y, y->right);
			y->right = z->right;
			y->right->parent = y;
		}
		__rb_transplant(t, z, y);
		y->left = z->left;
		y->left->parent = y;
		y->color = z->color;
	}
	if (y_original_color == RBT_COLOR_BLACK)
		__rb_tree_delete_fixup(t, x);
	free(hold_node_to_delete);
	return node_to_return;
}

rbt_iterator *new_rbt_iterator(rbt_t *t)
{
	struct rbt_node *aux = t->root;
	rbt_iterator *it = (rbt_iterator*)malloc(sizeof(rbt_iterator));
	while(aux->left != &node_nil || aux->right != &node_nil) {
		while(aux->left != &node_nil)
			aux = aux->left;
		if(aux->right != &node_nil)
			aux = aux->right;
	}
	it->current = aux;
	return it;
}

int rbt_iterator_has_next(rbt_iterator *it)
{
	if(it->current != &node_nil)
		return 1;
	return 0;
}

void *rbt_iterator_next(rbt_iterator *it)
{
	struct rbt_node *aux;
	struct rbt_node *tn = it->current;
	if (tn->parent != &node_nil &&
		tn->parent->right != &node_nil &&
		tn->parent->right != tn) {
		aux = tn->parent->right;
		while (aux->left != &node_nil || aux->right != &node_nil) {
			while(aux->left != &node_nil)
				aux = aux->left;
			if (aux->right != &node_nil)
				aux = aux->right;
			}
			it->current = aux;
		}
	else
		it->current = it->current->parent;
	return tn->data;
}

void free_rbt_iterator(rbt_iterator *it)
{
	free(it);
}
