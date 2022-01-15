
#pragma once

#include "foreign_seri/SeriMode.h"
#define BLOCK_SIZE 128

class WriteBlock {
protected:
	char *mBuffer;
	int64_t mCapacity;
	int64_t mLen;
public:
	WriteBlock(): mBuffer((char*)foreign_malloc(BLOCK_SIZE)), mCapacity(BLOCK_SIZE), mLen(0) {
		// TODO use foreign_malloc or skynet_malloc ???
	}
	~WriteBlock() {
		free_buffer();
	}
	void free_buffer() {
		if(mBuffer!=NULL) {
			foreign_free(mBuffer);
			mBuffer = NULL;
		}
	}
	void push(const void* vData, int64_t vSize) {
		int64_t newCapacity = mCapacity;
		while(newCapacity < mLen + vSize) {
			newCapacity += newCapacity / 2;
		}
		if(newCapacity != mCapacity) {
			char *newBuffer = reinterpret_cast<char*>(foreign_malloc(newCapacity));
			memcpy(newBuffer, mBuffer, mLen);
			foreign_free(mBuffer);
			mBuffer = newBuffer;
		}
		memcpy(mBuffer + mLen, vData, vSize);
		mLen += vSize;
	}
	void wb_nil() {
		uint8_t n = TYPE_NIL;
		push(&n, 1);
	}
	void wb_boolean(int v) {
		uint8_t n = COMBINE_TYPE(TYPE_BOOLEAN , v ? 1 : 0);
		push(&n, 1);
	}
	void wb_integer(lua_Integer v) {
		int type = TYPE_NUMBER;
		if (v == 0) {
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_ZERO);
			push(&n, 1);
		} else if (v != (int32_t)v) {
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_QWORD);
			int64_t v64 = v;
			push(&n, 1);
			push(&v64, sizeof(v64));
		} else if (v < 0) {
			int32_t v32 = (int32_t)v;
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_DWORD);
			push(&n, 1);
			push(&v32, sizeof(v32));
		} else if (v<0x100) {
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_BYTE);
			push(&n, 1);
			uint8_t byte = (uint8_t)v;
			push(&byte, sizeof(byte));
		} else if (v<0x10000) {
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_WORD);
			push(&n, 1);
			uint16_t word = (uint16_t)v;
			push(&word, sizeof(word));
		} else {
			uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_DWORD);
			push(&n, 1);
			uint32_t v32 = (uint32_t)v;
			push(&v32, sizeof(v32));
		}
	}
	void wb_real(double v) {
		uint8_t n = COMBINE_TYPE(TYPE_NUMBER , TYPE_NUMBER_REAL);
		push(&n, 1);
		push(&v, sizeof(v));
	}
	void wb_pointer(void *v) {
		uint8_t n = TYPE_USERDATA;
		push(&n, 1);
		push(&v, sizeof(v));
	}
	void wb_string(const char *str, int len) {
		if (len < MAX_COOKIE) {
			uint8_t n = COMBINE_TYPE(TYPE_SHORT_STRING, len);
			push(&n, 1);
			if (len > 0) {
				push(str, len);
			}
		} else {
			uint8_t n;
			if (len < 0x10000) {
				n = COMBINE_TYPE(TYPE_LONG_STRING, 2);
				push(&n, 1);
				uint16_t x = (uint16_t) len;
				push(&x, 2);
			} else {
				n = COMBINE_TYPE(TYPE_LONG_STRING, 4);
				push(&n, 1);
				uint32_t x = (uint32_t) len;
				push(&x, 4);
			}
			push(str, len);
		}
	}
	void push_uint(npy_intp v) {
		static const int B = 128;
		uint8_t data = v | B;
		while (v >= B) {
			data = v | B;
			push(&data, 1);
			v >>= 7;
		}
		data = (uint8_t)v;
		push(&data, 1);
	}
};
