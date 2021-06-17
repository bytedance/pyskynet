
#include "skynet.h"
#include "skynet_server.h"
#include "skynet_malloc.h"
#include "skynet_env.h"

#include "skynet_modify/skynet_py.h"

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <signal.h>

struct SkynetPyGlobal G_SKYNET_PY;

// code just like skynet_mq.c
void skynet_py_queue_push(struct SkynetPyMessage *message){
	struct SkynetPyQueue *q = &(G_SKYNET_PY.recv_queue);
	int uv_async_busy = 0;
	SPIN_LOCK(q)
	// push into queue
	q->queue[q->tail] = *message;
	if (++ q->tail >= q->cap) {
		q->tail = 0;
	}

	if (q->head == q->tail) {
	    struct SkynetPyMessage *new_queue = skynet_malloc(sizeof(struct SkynetPyMessage) * q->cap * 2);
	    int i;
	    for (i=0;i<q->cap;i++) {
		    new_queue[i] = q->queue[(q->head + i) % q->cap];
	    }
	    q->head = 0;
	    q->tail = q->cap;
	    q->cap *= 2;

	    skynet_free(q->queue);
	    q->queue = new_queue;
	}


	uv_async_busy = G_SKYNET_PY.uv_async_busy;
	SPIN_UNLOCK(q)

	// if uv in python is not busy, schedule it again
	if(!uv_async_busy){
		G_SKYNET_PY.uv_async_busy = 1;
	    G_SKYNET_PY.uv_async_send(G_SKYNET_PY.uv_async_handle);
	}
}

// code just like skynet_mq.c
int skynet_py_queue_pop(struct SkynetPyMessage *message){
	int ret = 1;
	struct SkynetPyQueue *q = &(G_SKYNET_PY.recv_queue);
	SPIN_LOCK(q)

	if (q->head != q->tail) {
		*message = q->queue[q->head++];
		ret = 0;
		int head = q->head;
		int tail = q->tail;
		int cap = q->cap;

		if (head >= cap) {
			q->head = head = 0;
		}
		int length = tail - head;
		if (length < 0) {
			length += cap;
		}
	}else {
		// if python pop all message and know queue is empty, set no busy
		G_SKYNET_PY.uv_async_busy = 0;
	}
	SPIN_UNLOCK(q)

	// ret == 1 means empty
	return ret;
}

int skynet_py_send(uint32_t lua_destination, int type, int session, void* msg, size_t sz){
    int real_session = skynet_send(G_SKYNET_PY.holder_context, G_SKYNET_PY.holder_address, lua_destination, type, session, msg, sz);
    skynet_py_wakeup();
    return real_session;
}

int skynet_py_sendname(const char *lua_destination, int type, int session, void* msg, size_t sz){
    int real_session = skynet_sendname(G_SKYNET_PY.holder_context, G_SKYNET_PY.holder_address, lua_destination, type, session, msg, sz);
    skynet_py_wakeup();
    return real_session;
}

void skynet_py_decref_python(void * pyobj) {
	struct SkynetPyMessage msg;
    msg.type = PTYPE_DECREF_PYTHON;
	msg.session = 0;
	msg.source = 0;
    msg.data = pyobj;
    msg.size = 0;
	skynet_py_queue_push(&msg);
}

static int sigign() {
	struct sigaction sa;
	sa.sa_handler = SIG_IGN;
	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGPIPE, &sa, 0);
	return 0;
}

void skynet_py_init(int (*p_uv_async_send)(void *), void * p_uv_async_t){
    // init queue
	struct SkynetPyQueue *q = &(G_SKYNET_PY.recv_queue);
	q->cap = 64;
	q->head = 0;
	q->tail = 0;
	q->queue = skynet_malloc(sizeof(struct SkynetPyMessage) * q->cap);

    // init uv
	G_SKYNET_PY.uv_async_send = p_uv_async_send;
	G_SKYNET_PY.uv_async_handle = p_uv_async_t;
	G_SKYNET_PY.uv_async_busy = 0;
	G_SKYNET_PY.holder_context = NULL;
	G_SKYNET_PY.holder_address = 0;

	SPIN_INIT(q);

	skynet_globalinit();
	skynet_env_init();

	sigign();

#ifdef LUA_CACHELIB
	// init the lock of code cache
	luaL_initcodecache();
#endif

}

// if pyholder not started, return 0
uint32_t skynet_py_address() {
	return G_SKYNET_PY.holder_address;
}
