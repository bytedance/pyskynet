
#pragma once

#include <stdexcept>

#include "foreign_seri/SeriMode.h"

class ReadBlock {
protected:
	const char * mBuffer;
	int64_t mLen; // left length
	int64_t mCur; // cur length
public:
	ReadBlock() : mBuffer(NULL), mLen(0), mCur(0) {}
	void set_buffer(const char* vBuffer, int64_t vLen) {
		mBuffer = vBuffer;
		mLen = vLen;
		mCur = 0;
	}
	const char* read(int64_t vSize) {
		if (mLen < vSize) {
			throw std::out_of_range("read buffer not enough");
		}
		const char * ptr = mBuffer + mCur;
		mCur += vSize;
		mLen -= vSize;
		return ptr;
	}
	template <class T> T read(){
		int64_t nSize = sizeof(T);
		if (mLen < nSize) {
			throw std::out_of_range("read buffer not enough");
		}
		char * ptr = const_cast<char*>(mBuffer + mCur);
		mCur += nSize;
		mLen -= nSize;
		return *reinterpret_cast<T*>(ptr);
	}
	int64_t get_integer(int cookie) {
		switch (cookie) {
		case TYPE_NUMBER_ZERO:
			return 0;
		case TYPE_NUMBER_BYTE: {
			return read<uint8_t>();
		}
		case TYPE_NUMBER_WORD: {
			return read<uint16_t>();
		}
		case TYPE_NUMBER_DWORD: {
			return read<int32_t>();
		}
		case TYPE_NUMBER_QWORD: {
			return read<int64_t>();
		}
		default:
			throw std::domain_error("invalid cookie in read buffer");
			return 0;
		}
	}
	double get_real() {
		return read<double>();
	}
	void *get_pointer() {
		return read<void*>();
	}
	const char *get_string(int len) {
		return read(len);
	}
};
