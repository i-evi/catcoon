#ifndef _RBT_H_
#define _RBT_H_

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

typedef struct {
	struct rbt_node *root;
	void *(*get_key)(struct rbt_node*);
	int (*compare)(void*, void*);
} rbt_t;

typedef struct {
	struct rbt_node *current;
} rbt_iterator;

rbt_t *new_rbt(void*(*get_key)(struct rbt_node *), 
			int (*compare)(void*, void*));

void free_rbt(rbt_t *t);

struct rbt_node *rbt_nil(void);

void *rbt_insert(rbt_t *t, void *d);

void *rbt_delete(rbt_t *t, void *key);

void *rbt_search(rbt_t *t, void *key);

rbt_iterator *new_rbt_iterator(rbt_t *t);

int rbt_iterator_has_next(rbt_iterator *it);

void *rbt_iterator_next(rbt_iterator *it);

void free_rbt_iterator(rbt_iterator* it);

#ifdef __cplusplus
	}
#endif

#endif /* _RBT_H_ */
