#include <string.h>
#include "skynet_foreign/skynet_foreign.h"
#include "skynet_foreign/numsky.h"

/* regard iter->dataptr as the start ptr of a sub array, get the ndim of sub array*/
int numsky_nditer_sub_ndim(struct numsky_nditer *iter) {
	if(iter->nd <= 0) {
		return 0;
	} else {
		for(int i=iter->nd-1;i>=0;i--) {
			if(iter->coordinates[i] != 0) {
				return iter->nd - 1 - i;
			}
		}
		return iter->nd;
	}
}


