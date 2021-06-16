
#pragma once

#include "skynet.h"
#include "skynet_imp.h"
#include "skynet_mq.h"
#include "spinlock.h"
#include <lua.h>
#include <Python.h>

#define PTYPE_FOREIGN_REMOTE 254
#define PTYPE_FOREIGN 255
#define PTYPE_DECREF_PYTHON 257 // bigger than 255 for diff with PTYPE in skynet.h

struct SkynetPyMessage {
	int type;
	int session;
	uint32_t source;
	void * data;
	size_t size;
};

struct SkynetPyQueue {
	struct spinlock lock;
	struct SkynetPyMessage *queue;
	int cap;
	int head;
	int tail;
};

struct SkynetPyGlobal {
	// gevent item
	void *uv_async_handle;
	int (*uv_async_send)(void *);
	int uv_async_busy;   // means python is busy with queue, don't need send async call
	struct SkynetPyQueue recv_queue;  // queue
	// holder item
	uint32_t holder_address;
	struct skynet_context * holder_context;
	// temp malloc when start
	void *temp_monitor;
	void *temp_pids;
	void *temp_wps;
};

extern struct SkynetPyGlobal G_SKYNET_PY;

void skynet_py_queue_push(struct SkynetPyMessage* message); // return session
int skynet_py_queue_pop(struct SkynetPyMessage* message); // return if empty 1 else 0
int skynet_py_send(uint32_t lua_destination, int type, int session, void* msg, size_t sz);
int skynet_py_sendname(const char *lua_destination, int type, int session, void* msg, size_t sz);

void skynet_py_decref_python(void * pyobj); // decref python object, called by foreign

void skynet_py_init(int (*p_uv_async_send)(void *), void * p_uv_async_t);
void skynet_py_start(struct skynet_config * config);
void skynet_py_wakeup();
void skynet_py_exit();

/* function for env */
void skynet_py_setlenv(const char *key, const char *value_str, size_t sz);
const char *skynet_py_getlenv(const char *key, size_t *sz);
const char *skynet_py_nextenv(const char *key);
