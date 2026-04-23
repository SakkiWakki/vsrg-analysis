#pragma once

#include "../../../../overlay/widgets/overlay_shm.h"

#ifdef __cplusplus
extern "C" {
#endif

int              shm_consumer_ensure(void);
int              shm_consumer_read(VsrgOverlayShm *out);
VsrgOverlayShm  *shm_consumer_writable(void);

#ifdef __cplusplus
}
#endif
