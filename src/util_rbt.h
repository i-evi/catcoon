#ifndef _UTIL_RBT_H_
#define _UTIL_RBT_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define RBT_COLOR_RED   1
#define RBT_COLOR_BLACK 2

struct rbt_node {
	void            *data;
	struct rbt_node *parent;
	struct rbt_node *left;
	struct rbt_node *right;
	unsigned char    color;
};

struct rbtree {
	struct rbt_node *root;
	void *(*get_key)(struct rbt_node*);
	int (*compare)(const void*, const void*);
};

struct rbtree *rbt_new(void*(*get_key)(struct rbt_node *), 
		int (*compare)(const void*, const void*));

void rbt_del(struct rbtree *t);

struct rbt_node *rbt_nil(void);

void *rbt_insert(struct rbtree *t, void *d);
void *rbt_find(struct rbtree *t, void *key);
void *rbt_erase(struct rbtree *t, void *key);


#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_RBT_H_ */
