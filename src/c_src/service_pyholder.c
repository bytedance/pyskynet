#include "skynet.h"
#include "skynet_server.h"
#include "skynet_modify/skynet_py.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct pyholder {
	int data;
};

struct pyholder *
pyholder_create(void) {
	struct pyholder * inst = skynet_malloc(sizeof(*inst));
	return inst;
}

void
pyholder_release(struct pyholder * inst) {
	skynet_free(inst);
}

static int
pyholder_cb(struct skynet_context * context, void *ud, int type, int session, uint32_t source, const void * data, size_t sz) {
    struct SkynetPyMessage msg;
  msg.type = type;
	msg.session = session;
	msg.source = source;
  msg.data = (void*)data;
  msg.size = sz;
	skynet_py_queue_push(&msg);
	// return 1 means reserve message @ skynet_server.c ctx->cb
	// free data by python
	return 1;
}

int
pyholder_init(struct pyholder * inst, struct skynet_context *ctx, const char * parm) {
	skynet_callback(ctx, inst, pyholder_cb);
	return 0;
}
