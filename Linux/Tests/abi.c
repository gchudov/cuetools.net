#include <scsi/sg.h>
#include <stddef.h>
#include <stdio.h>
int main(void) {
    printf("sg_io_hdr_t size=%zu data=%zu residual=%zu\n", sizeof(sg_io_hdr_t), offsetof(sg_io_hdr_t, dxferp), offsetof(sg_io_hdr_t, resid));
    return sizeof(sg_io_hdr_t) != (sizeof(void*) == 8 ? 88 : 64) || offsetof(sg_io_hdr_t, dxferp) != 16 || offsetof(sg_io_hdr_t, resid) != (sizeof(void*) == 8 ? 72 : 52);
}
