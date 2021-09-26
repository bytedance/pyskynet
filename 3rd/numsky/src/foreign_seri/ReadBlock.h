
#pragma once

#include "foreign_seri/SeriMode.h"

class ReadBlock {
protected:
	char * mBuffer;
	int64_t mLen; // left length
	int64_t mCur; // cur length
public:
	ReadBlock(char* vBuffer, int64_t vSize) : mBuffer(vBuffer), mLen(vSize), mCur(0) {
	}
	char* rb_read(int64_t vSize) {
		if (mLen < vSize) {
			return NULL;
		}
		char * ptr = mBuffer + mCur;
		mCur += vSize;
		mLen -= vSize;
		return ptr;
	}
};
